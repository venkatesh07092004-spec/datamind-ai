"""User-controlled dataset cleaning and safe CSV export utilities."""

from typing import Any

import pandas as pd

from core.ingestion import detect_datetime_columns
from core.logging_config import get_logger

logger = get_logger(__name__)

_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def sanitize_csv_cell(value: Any) -> Any:
    """Neutralize spreadsheet formula injection in exported string cells."""
    if isinstance(value, str) and value.startswith(_FORMULA_PREFIXES):
        return "'" + value
    return value


def dataframe_to_safe_csv_bytes(df: pd.DataFrame) -> bytes:
    """Serialize a DataFrame to UTF-8 CSV bytes with formula-injection protection."""
    safe_df = df.copy()
    for col in safe_df.select_dtypes(include=["object", "string"]).columns:
        safe_df[col] = safe_df[col].map(sanitize_csv_cell)
    return safe_df.to_csv(index=False).encode("utf-8")


def clean_dataframe(
    df: pd.DataFrame,
    trim_text: bool = False,
    drop_duplicates: bool = False,
    fill_numeric_median: bool = False,
    parse_dates: bool = False,
    drop_constant_columns: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """Return a cleaned copy of the DataFrame and a human-readable transformation log."""
    result = df.copy()
    log: list[str] = []

    if trim_text:
        for col in result.select_dtypes(include=["object", "string"]).columns:
            before = result[col].copy()
            result[col] = result[col].map(lambda x: x.strip() if isinstance(x, str) else x)
            changed = int((before.fillna("\u0000") != result[col].fillna("\u0000")).sum())
            if changed:
                log.append(f"Trimmed whitespace in '{col}' ({changed} values).")

    if parse_dates:
        for col in detect_datetime_columns(result):
            if pd.api.types.is_datetime64_any_dtype(result[col]):
                continue
            parsed = pd.to_datetime(result[col], errors="coerce", format="mixed")
            if parsed.notna().sum() > 0:
                result[col] = parsed
                log.append(f"Parsed '{col}' as datetime ({int(parsed.notna().sum())} values).")

    if fill_numeric_median:
        for col in result.select_dtypes(include="number").columns:
            count = int(result[col].isna().sum())
            if count:
                median = result[col].median()
                if pd.notna(median):
                    result[col] = result[col].fillna(median)
                    log.append(f"Filled {count} missing values in '{col}' with median {median}.")

    if drop_duplicates:
        count = int(result.duplicated().sum())
        if count:
            result = result.drop_duplicates().reset_index(drop=True)
            log.append(f"Removed {count} duplicate rows.")

    if drop_constant_columns:
        constant_cols = [c for c in result.columns if result[c].nunique(dropna=False) <= 1]
        if constant_cols:
            result = result.drop(columns=constant_cols)
            log.append(f"Removed constant column(s): {', '.join(map(str, constant_cols))}.")

    if not log:
        log.append("No transformations were applied.")

    logger.info("Applied cleaning pipeline: %s", "; ".join(log))
    return result, log
