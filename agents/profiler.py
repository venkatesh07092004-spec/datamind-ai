"""Data Profiler Agent: structural profiling, semantic classification, anomaly/pattern flags, and cleaning suggestions."""

from typing import Any

import pandas as pd

from core.ingestion import classify_column_semantic, detect_datetime_columns
from core.logging_config import get_logger

logger = get_logger(__name__)


def profile_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """Compute a comprehensive structural and quality profile for a DataFrame."""
    rows = int(df.shape[0])
    cols = int(df.shape[1])
    total_cells = max(rows * cols, 1)

    numeric_df = df.select_dtypes(include="number")
    numeric_cols = [str(c) for c in numeric_df.columns]
    datetime_cols = detect_datetime_columns(df)
    categorical_cols = [
        str(c) for c in df.columns if str(c) not in numeric_cols and str(c) not in datetime_cols
    ]
    semantic_types = {
        str(c): classify_column_semantic(df[c], str(c)) for c in df.columns
    }

    missing_values = {str(c): int(n) for c, n in df.isna().sum().items()}
    missing_cells = int(sum(missing_values.values()))
    missing_pct = round((missing_cells / total_cells) * 100.0, 2)
    duplicate_rows = int(df.duplicated().sum())
    cardinality = {str(c): int(df[c].nunique(dropna=True)) for c in df.columns}

    constant_columns = [str(c) for c in df.columns if cardinality[str(c)] <= 1]
    high_missing_columns = [
        str(c) for c in df.columns if rows > 0 and (missing_values[str(c)] / rows) >= 0.2
    ]
    high_cardinality_columns = [
        str(c)
        for c in categorical_cols
        if rows >= 10 and cardinality[str(c)] >= max(10, int(0.8 * rows))
    ]

    skewed_columns: dict[str, float] = {}
    for col in numeric_cols:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(series) >= 8 and series.std(ddof=1) > 0:
            skew_val = float(series.skew())
            if abs(skew_val) >= 1.5:
                skewed_columns[col] = round(skew_val, 3)

    unusual_patterns: list[str] = []
    if constant_columns:
        unusual_patterns.append(
            f"Constant or single-value column(s): {', '.join(constant_columns)}."
        )
    if high_missing_columns:
        unusual_patterns.append(
            f"High missingness (>=20%) in: {', '.join(high_missing_columns)}."
        )
    if high_cardinality_columns:
        unusual_patterns.append(
            f"High-cardinality text column(s) (possible IDs): {', '.join(high_cardinality_columns)}."
        )
    if skewed_columns:
        skew_desc = ", ".join(f"{k} (skew={v})" for k, v in skewed_columns.items())
        unusual_patterns.append(f"Strongly skewed numeric column(s): {skew_desc}.")
    if duplicate_rows > 0:
        dup_pct = round((duplicate_rows / max(rows, 1)) * 100.0, 2)
        unusual_patterns.append(f"{duplicate_rows} duplicate rows ({dup_pct}% of rows).")

    dup_penalty = min(25.0, (duplicate_rows / max(rows, 1)) * 100.0 * 0.5)
    missing_penalty = min(50.0, missing_pct * 1.5)
    const_penalty = min(15.0, len(constant_columns) * 5.0)
    quality_score = round(max(0.0, 100.0 - missing_penalty - dup_penalty - const_penalty), 1)

    summary_text = (
        f"Dataset has {rows:,} rows and {cols} columns "
        f"({len(numeric_cols)} numeric, {len(categorical_cols)} categorical, {len(datetime_cols)} datetime). "
        f"Missing cells: {missing_cells:,} ({missing_pct}%), duplicate rows: {duplicate_rows:,}. "
        f"Overall quality score: {quality_score}/100."
    )

    logger.info("Profiled dataset: %d rows x %d cols, quality=%.1f", rows, cols, quality_score)

    return {
        "rows": rows,
        "columns": cols,
        "column_names": [str(c) for c in df.columns],
        "data_types": {str(c): str(t) for c, t in df.dtypes.items()},
        "semantic_types": semantic_types,
        "missing_values": missing_values,
        "missing_cells": missing_cells,
        "missing_pct": missing_pct,
        "duplicate_rows": duplicate_rows,
        "cardinality": cardinality,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "datetime_columns": datetime_cols,
        "constant_columns": constant_columns,
        "high_missing_columns": high_missing_columns,
        "high_cardinality_columns": high_cardinality_columns,
        "skewed_columns": skewed_columns,
        "unusual_patterns": unusual_patterns,
        "quality_score": quality_score,
        "numeric_summary": numeric_df.describe().round(3).to_dict() if not numeric_df.empty else {},
        "memory_mb": round(float(df.memory_usage(deep=True).sum()) / (1024 ** 2), 3),
        "summary_text": summary_text,
    }


def cleaning_suggestions(df: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Generate semantic-aware, user-reviewable cleaning recommendations,
    each with a confidence score (0.0-1.0) and clear explanation.
    """
    suggestions: list[dict[str, Any]] = []
    rows = max(len(df), 1)

    for col in df.columns:
        col_str = str(col)
        sem = classify_column_semantic(df[col], col_str)
        missing = int(df[col].isna().sum())
        if missing:
            pct = round((missing / rows) * 100.0, 1)
            issue = "high_missing_rate" if pct >= 20.0 else "missing_values"
            conf = 0.96 if pct >= 20.0 else 0.90
            suggestions.append({
                "column": col_str,
                "semantic_type": sem["semantic_type"],
                "issue": issue,
                "count": missing,
                "confidence": conf,
                "explanation": f"{missing} of {rows} rows ({pct}%) are null in {sem['semantic_type']} column '{col_str}'.",
                "suggestion": (
                    f"{pct}% missing. Consider domain-specific imputation or dropping if non-essential."
                    if pct >= 20.0
                    else "Review whether to fill (e.g., median for numeric) or remove missing values."
                ),
            })

        if sem["semantic_type"] == "numeric_text":
            suggestions.append({
                "column": col_str,
                "semantic_type": sem["semantic_type"],
                "issue": "numeric_formatted_as_text",
                "count": int(df[col].notna().sum()),
                "confidence": sem["confidence"],
                "explanation": sem["reason"],
                "suggestion": (
                    f"'{col_str}' appears to hold numeric values with currency/unit symbols; "
                    "parse symbols before numeric modeling if needed."
                ),
            })

    duplicates = int(df.duplicated().sum())
    if duplicates:
        dup_pct = round((duplicates / rows) * 100.0, 1)
        suggestions.append({
            "column": "*",
            "semantic_type": "table",
            "issue": "duplicate_rows",
            "count": duplicates,
            "confidence": 0.98,
            "explanation": f"Found {duplicates} exact duplicate row(s) ({dup_pct}% of dataset).",
            "suggestion": "Review duplicate rows before removing them.",
        })

    for col in df.select_dtypes(include=["object", "string"]).columns:
        col_str = str(col)
        trimmed = df[col].dropna().astype(str).str.strip()
        changed = int((trimmed != df.loc[trimmed.index, col].astype(str)).sum())
        if changed:
            suggestions.append({
                "column": col_str,
                "semantic_type": classify_column_semantic(df[col], col_str)["semantic_type"],
                "issue": "leading_trailing_whitespace",
                "count": changed,
                "confidence": 0.95,
                "explanation": f"{changed} non-null text value(s) in '{col_str}' have leading or trailing spaces.",
                "suggestion": "Trim leading and trailing whitespace in text values.",
            })

    for col in df.columns:
        if df[col].nunique(dropna=False) <= 1:
            col_str = str(col)
            suggestions.append({
                "column": col_str,
                "semantic_type": classify_column_semantic(df[col], col_str)["semantic_type"],
                "issue": "constant_column",
                "count": rows,
                "confidence": 0.99,
                "explanation": f"Column '{col_str}' has at most 1 unique value across all {rows} rows.",
                "suggestion": "Column carries zero variance; consider removing.",
            })

    for col in detect_datetime_columns(df):
        if not pd.api.types.is_datetime64_any_dtype(df[col]):
            sem = classify_column_semantic(df[col], str(col))
            suggestions.append({
                "column": str(col),
                "semantic_type": "datetime",
                "issue": "potential_datetime_text",
                "count": int(df[col].notna().sum()),
                "confidence": sem["confidence"],
                "explanation": sem["reason"],
                "suggestion": "Parse text column as datetime to enable time-series analysis and forecasting.",
            })

    return suggestions
