"""SQL Agent: schema-aware natural-language-to-SQL generation and validation."""

import json
import re
import urllib.error
import urllib.request
from typing import Any

import duckdb
import pandas as pd

from core.config import DEFAULT_OLLAMA_MODEL, DEFAULT_OLLAMA_URL, OLLAMA_TIMEOUT_SECONDS
from core.ingestion import detect_datetime_columns
from core.logging_config import get_logger
from core.sql_engine import get_table_schema, validate_readonly_sql

logger = get_logger(__name__)


def quote_identifier(name: str) -> str:
    """Escape an identifier for DuckDB double-quoted SQL."""
    return '"' + str(name).replace('"', '""') + '"'


def _quote_sql_literal(value: str) -> str:
    """Escape a string literal safely for single-quoted SQL."""
    return "'" + str(value).replace("'", "''") + "'"


def _dim_expr(df: pd.DataFrame, col: str) -> str:
    """Return a whitespace-trimmed expression for text dimensions without mutating df."""
    qcol = quote_identifier(col)
    if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
        return f"TRIM(CAST({qcol} AS VARCHAR))"
    return qcol


def _match_column(question_lower: str, candidates: list[str]) -> str | None:
    """Find the best matching column name mentioned in the question."""
    for col in sorted(candidates, key=lambda c: len(str(c)), reverse=True):
        col_l = str(col).lower()
        col_spaced = col_l.replace("_", " ")
        if re.search(rf"\b{re.escape(col_l)}\b", question_lower) or (
            col_spaced != col_l and re.search(rf"\b{re.escape(col_spaced)}\b", question_lower)
        ):
            return col
    return None


def _extract_limit(question_lower: str, default: int = 5, max_limit: int = 100) -> int:
    match = re.search(r"\b(?:top|bottom|first|last|limit)\s+(\d+)\b", question_lower)
    if match:
        return max(1, min(int(match.group(1)), max_limit))
    return default


def _detect_agg(question_lower: str) -> tuple[str, str]:
    if any(k in question_lower for k in ["average", "avg", "mean"]):
        return "AVG", "avg"
    if any(k in question_lower for k in ["count", "how many", "number of"]):
        return "COUNT", "count"
    if any(k in question_lower for k in ["minimum", "min ", "lowest", "smallest"]):
        return "MIN", "min"
    if any(k in question_lower for k in ["maximum", "max ", "highest", "largest"]):
        return "MAX", "max"
    return "SUM", "total"


def _extract_where_clauses(question: str, df: pd.DataFrame) -> list[str]:
    """
    Safely extract numeric comparisons (e.g. 'revenue > 500') and categorical value
    filters (e.g. 'in North', 'where region = East') from the question using exact
    schema values and escaped literals.
    """
    clauses: list[str] = []
    q_lower = question.lower()
    numeric_cols = [str(c) for c in df.select_dtypes(include="number").columns]
    cat_cols = [str(c) for c in df.columns if str(c) not in numeric_cols]

    # 1. Numeric filter: "<col> (>|>=|<|<=|=|above|over|below|under) <number>"
    for col in sorted(numeric_cols, key=lambda c: len(c), reverse=True):
        col_pat = re.escape(col.lower()).replace("_", "[_ ]")
        m = re.search(
            rf"\b{col_pat}\b\s*(>=|<=|>|<|=|above|over|greater than|more than|below|under|less than)\s*(-?\d+(?:\.\d+)?)",
            q_lower,
        )
        if m:
            op_raw, num_val = m.group(1), m.group(2)
            op_map = {
                ">": ">", ">=": ">=", "<": "<", "<=": "<=", "=": "=",
                "above": ">", "over": ">", "greater than": ">", "more than": ">",
                "below": "<", "under": "<", "less than": "<",
            }
            op = op_map[op_raw]
            clauses.append(f"{quote_identifier(col)} {op} {float(num_val) if '.' in num_val else int(num_val)}")

    # 2. Categorical value filter: match known low-cardinality values preceded by in/for/where/=
    for col in cat_cols:
        if df[col].nunique(dropna=True) > 50:
            continue
        unique_vals = [str(v).strip() for v in df[col].dropna().unique() if str(v).strip()]
        for val in sorted(unique_vals, key=len, reverse=True):
            val_l = val.lower()
            if re.search(rf"\b(?:in|for|where|=|is)\s+['\"]?{re.escape(val_l)}['\"]?\b", q_lower):
                clauses.append(f"LOWER(TRIM(CAST({quote_identifier(col)} AS VARCHAR))) = {_quote_sql_literal(val_l)}")
                break

    return clauses


def _generate_deterministic_sql(question: str, df: pd.DataFrame) -> str:
    """Schema-aware deterministic NL-to-SQL translator supporting filtering, grouping, aggregation, sorting, and limits."""
    q = question.lower().strip()
    if not q:
        raise ValueError("Please enter a natural-language question.")

    cols = [str(c) for c in df.columns]
    if not cols:
        raise ValueError("Dataset has no columns.")

    numeric_cols = [str(c) for c in df.select_dtypes(include="number").columns]
    datetime_cols = detect_datetime_columns(df)
    categorical_cols = [c for c in cols if c not in numeric_cols and c not in datetime_cols]

    default_metric = next(
        (c for c in numeric_cols if c.lower() in {"revenue", "sales", "amount", "total", "price", "profit"}),
        numeric_cols[0] if numeric_cols else None,
    )
    default_region = next(
        (c for c in categorical_cols if c.lower() in {"region", "country", "state", "category", "segment"}),
        categorical_cols[0] if categorical_cols else None,
    )
    default_product = next(
        (c for c in categorical_cols if c.lower() in {"product", "product_name", "item", "name"}),
        categorical_cols[0] if categorical_cols else None,
    )

    where_clauses = _extract_where_clauses(question, df)
    where_sql = (" WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    # 1. Row count
    if any(term in q for term in ["how many rows", "row count", "number of rows", "count rows", "total rows"]):
        return f"SELECT COUNT(*) AS row_count FROM data{where_sql}"

    # 2. Missing values per column
    if "missing" in q and ("column" in q or "values" in q or "null" in q):
        parts = [
            f"COUNT(*) FILTER (WHERE {quote_identifier(c)} IS NULL) AS {quote_identifier(f'{c}_missing')}"
            for c in cols
        ]
        return "SELECT " + ", ".join(parts) + f" FROM data{where_sql}"

    # 3. Preview / filtered records
    if any(term in q for term in ["show all", "preview", "show records", "sample rows", "first rows", "filter"]):
        limit = _extract_limit(q, default=100)
        return f"SELECT * FROM data{where_sql} LIMIT {limit}"

    # 4. Unique / distinct values
    if any(term in q for term in ["unique", "distinct"]):
        target = _match_column(q, cols) or (categorical_cols[0] if categorical_cols else cols[0])
        d_expr = _dim_expr(df, target)
        return (
            f"SELECT {d_expr} AS {quote_identifier(target)}, COUNT(*) AS record_count "
            f"FROM data{where_sql} GROUP BY {d_expr} ORDER BY record_count DESC LIMIT 100"
        )

    # 5. Time-series / trend query
    if datetime_cols and any(term in q for term in ["over time", "trend", "by date", "by month", "monthly", "daily"]):
        date_col = _match_column(q, datetime_cols) or datetime_cols[0]
        metric_col = _match_column(q, numeric_cols) or default_metric
        agg_fn, agg_prefix = _detect_agg(q)
        date_expr = quote_identifier(date_col)
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            date_expr = f"TRY_CAST({quote_identifier(date_col)} AS TIMESTAMP)"
        if "month" in q:
            period_expr = f"DATE_TRUNC('month', {date_expr})"
            period_alias = "month"
        else:
            period_expr = f"DATE_TRUNC('day', {date_expr})"
            period_alias = "period"
        combined_where = (
            f" WHERE {date_expr} IS NOT NULL" + (" AND " + " AND ".join(where_clauses) if where_clauses else "")
        )
        if metric_col and agg_fn != "COUNT":
            alias = re.sub(r"\W+", "_", f"{agg_prefix}_{metric_col}").strip("_")
            return (
                f"SELECT {period_expr} AS {period_alias}, {agg_fn}({quote_identifier(metric_col)}) AS {alias} "
                f"FROM data{combined_where} "
                f"GROUP BY {period_alias} ORDER BY {period_alias} ASC"
            )
        return (
            f"SELECT {period_expr} AS {period_alias}, COUNT(*) AS record_count "
            f"FROM data{combined_where} "
            f"GROUP BY {period_alias} ORDER BY {period_alias} ASC"
        )

    # 6. Top / Bottom N ranking
    if any(term in q for term in ["top", "bottom", "highest", "lowest", "best-selling", "best selling", "worst"]):
        limit = _extract_limit(q, default=5)
        order_dir = "ASC" if any(t in q for t in ["bottom", "lowest", "worst", "smallest"]) else "DESC"
        group_col = _match_column(q, categorical_cols) or default_product or default_region
        metric_col = _match_column(q, numeric_cols) or default_metric
        if group_col and metric_col:
            agg_fn, agg_prefix = _detect_agg(q)
            alias = re.sub(r"\W+", "_", f"{agg_prefix}_{metric_col}").strip("_")
            g_expr = _dim_expr(df, group_col)
            return (
                f"SELECT {g_expr} AS {quote_identifier(group_col)}, {agg_fn}({quote_identifier(metric_col)}) AS {alias} "
                f"FROM data{where_sql} GROUP BY {g_expr} "
                f"ORDER BY {alias} {order_dir} LIMIT {limit}"
            )
        if metric_col:
            return f"SELECT * FROM data{where_sql} ORDER BY {quote_identifier(metric_col)} {order_dir} LIMIT {limit}"

    # 7. Group-by ("by <column>")
    by_match = re.search(r"\bby\s+([a-z0-9_ ]+)", q)
    if by_match:
        dim_col = _match_column(by_match.group(1), cols)
        if not dim_col and any(t in q for t in ["by region", "by category", "by segment"]):
            dim_col = default_region
        metric_col = _match_column(q, numeric_cols) or default_metric
        if dim_col and metric_col:
            agg_fn, agg_prefix = _detect_agg(q)
            alias = re.sub(r"\W+", "_", f"{agg_prefix}_{metric_col}").strip("_")
            d_expr = _dim_expr(df, dim_col)
            return (
                f"SELECT {d_expr} AS {quote_identifier(dim_col)}, {agg_fn}({quote_identifier(metric_col)}) AS {alias} "
                f"FROM data{where_sql} GROUP BY {d_expr} ORDER BY {alias} DESC"
            )
        if dim_col:
            d_expr = _dim_expr(df, dim_col)
            return (
                f"SELECT {d_expr} AS {quote_identifier(dim_col)}, COUNT(*) AS record_count "
                f"FROM data{where_sql} GROUP BY {d_expr} ORDER BY record_count DESC"
            )

    # 8. Single-column aggregate (sum / avg / min / max / count)
    if numeric_cols and any(
        k in q for k in ["average", "avg", "mean", "total", "sum", "minimum", "min", "maximum", "max"]
    ):
        metric_col = _match_column(q, numeric_cols) or default_metric
        agg_fn, agg_prefix = _detect_agg(q)
        alias = re.sub(r"\W+", "_", f"{agg_prefix}_{metric_col}").strip("_")
        return f"SELECT {agg_fn}({quote_identifier(metric_col)}) AS {alias} FROM data{where_sql}"

    # 9. Pure filtered query fallback when WHERE clauses were matched
    if where_clauses:
        limit = _extract_limit(q, default=100)
        return f"SELECT * FROM data{where_sql} LIMIT {limit}"

    raise ValueError(
        "Could not map that question to a schema-aware SQL query. Try questions like: "
        "'how many rows', 'total revenue by region', 'top 5 products by revenue in North', "
        "'monthly revenue trend', or 'show records where revenue > 500'."
    )


def _generate_sql_with_ollama(
    question: str,
    df: pd.DataFrame,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    ollama_model: str = DEFAULT_OLLAMA_MODEL,
) -> str | None:
    """Attempt local Ollama NL-to-SQL generation; return None if unavailable."""
    schema = get_table_schema(df)
    schema_lines = "\n".join(f"- \"{s['column']}\" ({s['dtype']})" for s in schema)
    prompt = (
        "You are a DuckDB SQL expert. Convert the user's question into a single read-only "
        "SELECT or WITH query against the table named `data`.\n"
        "Rules:\n"
        "1. Use ONLY double-quoted column names from the schema below.\n"
        "2. Return ONLY raw SQL with no markdown fences or explanation.\n"
        "3. Never use INSERT, UPDATE, DELETE, DROP, COPY, or file-reading functions.\n\n"
        f"Schema for table `data`:\n{schema_lines}\n\n"
        f"Question: {question}\nSQL:"
    )
    payload = json.dumps({
        "model": ollama_model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0},
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{ollama_url.rstrip('/')}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT_SECONDS) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            raw_sql = str(body.get("response", "")).strip()
            raw_sql = re.sub(r"^```(?:sql)?\s*|\s*```$", "", raw_sql, flags=re.IGNORECASE).strip()
            return raw_sql or None
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        logger.info("Ollama unavailable or failed (%s); falling back to deterministic SQL.", exc)
        return None


def _validate_against_duckdb(df: pd.DataFrame, sql: str) -> None:
    """Run read-only keyword checks and DuckDB EXPLAIN validation without full execution."""
    normalized = validate_readonly_sql(sql)
    con = duckdb.connect(database=":memory:")
    try:
        con.register("data", df.head(0))
        con.execute(f"EXPLAIN {normalized}")
    except duckdb.Error as exc:
        raise ValueError(f"Generated SQL failed schema validation: {exc}") from exc
    finally:
        con.close()


def generate_sql(
    question: str,
    df: pd.DataFrame,
    use_ollama: bool = False,
    ollama_model: str = DEFAULT_OLLAMA_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
) -> str:
    """Generate and validate a read-only DuckDB SQL query for a natural-language question."""
    if use_ollama:
        llm_sql = _generate_sql_with_ollama(
            question, df, ollama_url=ollama_url, ollama_model=ollama_model
        )
        if llm_sql:
            try:
                _validate_against_duckdb(df, llm_sql)
                logger.info("Generated validated SQL via local Ollama (%s).", ollama_model)
                return validate_readonly_sql(llm_sql)
            except ValueError as exc:
                logger.warning("Ollama SQL rejected (%s); falling back to deterministic rules.", exc)

    sql = _generate_deterministic_sql(question, df)
    _validate_against_duckdb(df, sql)
    return validate_readonly_sql(sql)


def generate_and_validate_sql(
    question: str,
    df: pd.DataFrame,
    use_ollama: bool = False,
    ollama_model: str = DEFAULT_OLLAMA_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
) -> dict[str, Any]:
    """Structured SQL Agent entry point returning SQL, source, validation status, and schema."""
    schema = get_table_schema(df)
    source = "deterministic"
    if use_ollama:
        llm_sql = _generate_sql_with_ollama(
            question, df, ollama_url=ollama_url, ollama_model=ollama_model
        )
        if llm_sql:
            try:
                _validate_against_duckdb(df, llm_sql)
                return {
                    "question": question,
                    "sql": validate_readonly_sql(llm_sql),
                    "source": f"ollama:{ollama_model}",
                    "valid": True,
                    "error": None,
                    "schema": schema,
                }
            except ValueError as exc:
                logger.warning("Ollama SQL rejected (%s); falling back to deterministic.", exc)

    sql = _generate_deterministic_sql(question, df)
    _validate_against_duckdb(df, sql)
    return {
        "question": question,
        "sql": validate_readonly_sql(sql),
        "source": source,
        "valid": True,
        "error": None,
        "schema": schema,
    }
