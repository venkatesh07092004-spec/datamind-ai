"""Read-only DuckDB SQL validation and execution engine."""

import re

import duckdb
import pandas as pd

from core.config import MAX_QUERY_ROWS
from core.logging_config import get_logger

logger = get_logger(__name__)

_FORBIDDEN_SQL_PATTERN = re.compile(
    r"(?i)\b("
    r"insert|update|delete|drop|alter|create|copy|attach|detach|install|load|"
    r"pragma|call|export|import|replace|truncate|execute|vacuum|checkpoint|"
    r"set|reset|use|merge|grant|revoke|"
    r"read_csv|read_csv_auto|read_parquet|read_json|read_json_auto|read_text|"
    r"glob|getenv|duckdb_secrets|http_get|system"
    r")\b"
)


def strip_sql_comments(query: str) -> str:
    """Remove single-line (-- ...) and block (/* ... */) SQL comments."""
    no_block = re.sub(r"/\*.*?\*/", " ", query, flags=re.DOTALL)
    no_line = re.sub(r"--[^\r\n]*", " ", no_block)
    return no_line.strip()


def validate_readonly_sql(query: str) -> str:
    """Validate that a SQL query is a single read-only SELECT or WITH statement."""
    if not query or not query.strip():
        raise ValueError("SQL query cannot be empty.")

    cleaned = strip_sql_comments(query).rstrip(";").strip()
    if not re.match(r"(?is)^(select|with)\b", cleaned):
        raise ValueError("Only SELECT or WITH queries are permitted.")
    if ";" in cleaned:
        raise ValueError("Multiple SQL statements are not permitted.")
    if _FORBIDDEN_SQL_PATTERN.search(cleaned):
        raise ValueError("The query contains a disallowed SQL keyword or external I/O function.")
    return cleaned


def get_table_schema(df: pd.DataFrame) -> list[dict[str, str]]:
    """Return DuckDB-compatible column names and inferred types for a DataFrame."""
    return [{"column": str(col), "dtype": str(dtype)} for col, dtype in df.dtypes.items()]


def execute_readonly_query(
    df: pd.DataFrame,
    query: str,
    max_rows: int = MAX_QUERY_ROWS,
) -> pd.DataFrame:
    """Validate and execute a single read-only SELECT/WITH query against an in-memory DuckDB table."""
    normalized = validate_readonly_sql(query)
    con = duckdb.connect(database=":memory:")
    try:
        con.register("data", df)
        # Dry-run EXPLAIN catches syntax/column errors before materializing results.
        con.execute(f"EXPLAIN {normalized}")
        limited_query = f"SELECT * FROM ({normalized}) AS _dm_subquery LIMIT {int(max_rows)}"
        result = con.execute(limited_query).df()
        logger.info("Executed read-only SQL query returning %d rows.", len(result))
        return result
    except duckdb.Error as exc:
        logger.warning("DuckDB query error: %s", exc)
        raise ValueError(f"SQL execution failed: {exc}") from exc
    finally:
        con.close()
