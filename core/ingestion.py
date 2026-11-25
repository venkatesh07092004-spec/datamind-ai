"""Secure dataset ingestion, validation, and semantic column classification utilities."""

from io import BytesIO
from pathlib import Path
import re
from typing import Any

import pandas as pd

from core.config import MAX_COLUMNS, MAX_FILE_SIZE_MB, MAX_ROWS
from core.logging_config import get_logger

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}

_NON_DATE_NAME_TOKENS = {
    "size", "price", "cost", "amount", "revenue", "sales", "rating", "score",
    "qty", "quantity", "count", "total", "age", "weight", "height", "width",
    "length", "percent", "pct", "ratio", "discount", "tax", "profit", "margin",
    "installs", "reviews", "version", "rank", "index", "id",
}

_DATE_NAME_TOKENS = {
    "date", "time", "datetime", "timestamp", "updated", "created", "modified",
    "dob", "birth", "day", "month", "year", "period",
}

_EXPLICIT_DATE_PATTERN = re.compile(
    r"^(?:"
    r"\d{4}[-/]\d{1,2}[-/]\d{1,2}(?:[T\s]\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)?"
    r"|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}(?:\s+\d{1,2}:\d{2}(?::\d{2})?)?"
    r"|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*[\s,.-]+\d{1,2}(?:[\s,.-]+\d{2,4})?"
    r"|\d{1,2}[\s,.-]+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*(?:[\s,.-]+\d{2,4})?"
    r")$",
    re.IGNORECASE,
)

_NUMERIC_TEXT_PATTERN = re.compile(
    r"^\s*[$€£¥₹]?\s*-?\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*[kKmMbB%+]?\s*$"
    r"|^\s*[$€£¥₹]?\s*-?\d+(?:\.\d+)?\s*[kKmMbB%+]?\s*$"
)


def _read_bytes_with_limit(file: Any, max_file_size_mb: float) -> bytes:
    """Read bytes from an uploaded or file-like object while enforcing a size cap."""
    max_bytes = int(max_file_size_mb * 1024 * 1024)
    size_attr = getattr(file, "size", None)
    if isinstance(size_attr, (int, float)) and size_attr > max_bytes:
        raise ValueError(
            f"File size ({size_attr / (1024 * 1024):.2f} MB) exceeds the {max_file_size_mb:.0f} MB limit."
        )

    data = file.getvalue() if hasattr(file, "getvalue") else file.read()
    if len(data) > max_bytes:
        raise ValueError(
            f"File size ({len(data) / (1024 * 1024):.2f} MB) exceeds the {max_file_size_mb:.0f} MB limit."
        )
    if len(data) == 0:
        raise ValueError("The uploaded file is empty.")
    return data


def load_dataframe(
    file: Any,
    filename: str | None = None,
    max_file_size_mb: float = MAX_FILE_SIZE_MB,
) -> pd.DataFrame:
    """Load a Streamlit UploadedFile or file-like object safely into a DataFrame."""
    name = filename or getattr(file, "name", "upload.csv")
    suffix = Path(name).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError("Unsupported file type. Upload CSV or Excel (.xlsx/.xls).")

    data = _read_bytes_with_limit(file, max_file_size_mb=max_file_size_mb)
    logger.info("Loading dataset '%s' (%d bytes, type=%s)", Path(name).name, len(data), suffix)

    if suffix == ".csv":
        try:
            return pd.read_csv(BytesIO(data))
        except UnicodeDecodeError:
            logger.warning("UTF-8 decode failed for '%s'; falling back to latin-1.", Path(name).name)
            return pd.read_csv(BytesIO(data), encoding="latin-1")
    return pd.read_excel(BytesIO(data))


def validate_dataframe(
    df: pd.DataFrame,
    max_columns: int = MAX_COLUMNS,
    max_rows: int = MAX_ROWS,
) -> None:
    """Validate basic structural and resource constraints on a loaded DataFrame."""
    if df.empty:
        raise ValueError("The uploaded dataset has no rows.")
    if len(df.columns) == 0:
        raise ValueError("The uploaded dataset has no columns.")
    if len(df.columns) > max_columns:
        raise ValueError(f"Too many columns ({len(df.columns)}). Limit: {max_columns}.")
    if len(df) > max_rows:
        raise ValueError(f"Too many rows ({len(df):,}). Limit: {max_rows:,}.")
    if not df.columns.is_unique:
        raise ValueError("Column names must be unique. Rename duplicate columns and upload again.")
    if any(not str(c).strip() for c in df.columns):
        raise ValueError("Column names cannot be blank.")


def _name_tokens(col_name: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(col_name).lower()))


def classify_column_semantic(
    series: pd.Series,
    col_name: str,
    sample_size: int = 200,
) -> dict[str, Any]:
    """
    Classify a column semantically using its name, dtype, and actual values.
    Prevents numeric/measurement columns (Size, Price, Rating, Revenue, Quantity)
    from being misclassified as datetime columns.
    """
    tokens = _name_tokens(col_name)
    has_non_date_name = bool(tokens & _NON_DATE_NAME_TOKENS)
    has_date_name = bool(tokens & _DATE_NAME_TOKENS)

    if pd.api.types.is_datetime64_any_dtype(series):
        return {
            "semantic_type": "datetime",
            "confidence": 1.0,
            "reason": "Column already has a datetime64 dtype.",
        }

    if pd.api.types.is_bool_dtype(series):
        return {
            "semantic_type": "categorical",
            "confidence": 0.98,
            "reason": "Boolean dtype.",
        }

    if pd.api.types.is_numeric_dtype(series):
        return {
            "semantic_type": "numeric",
            "confidence": 0.99,
            "reason": f"Numeric dtype ({series.dtype}).",
        }

    non_null = series.dropna().astype(str).str.strip()
    if non_null.empty:
        return {
            "semantic_type": "categorical",
            "confidence": 0.5,
            "reason": "Column contains only missing values.",
        }

    sample = non_null.head(sample_size)

    # Check numeric-like text (e.g. "$4.99", "19M", "10,000+", "4.5")
    numeric_match_ratio = float(sample.str.fullmatch(_NUMERIC_TEXT_PATTERN).mean())
    if numeric_match_ratio >= 0.75:
        return {
            "semantic_type": "numeric_text",
            "confidence": round(min(0.98, 0.65 + 0.33 * numeric_match_ratio), 2),
            "reason": f"{numeric_match_ratio * 100:.0f}% of sampled values match numeric/currency/unit patterns.",
        }

    # Check explicit date patterns + successful datetime parsing
    if not (has_non_date_name and not has_date_name):
        pattern_ratio = float(sample.str.fullmatch(_EXPLICIT_DATE_PATTERN).mean())
        if pattern_ratio >= 0.8:
            parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
            valid = parsed.dropna()
            if not valid.empty:
                in_range = valid.dt.year.between(1900, 2100).mean()
                parse_ratio = float(len(valid) / len(sample))
                if parse_ratio >= 0.8 and in_range >= 0.9:
                    conf = 0.85 * parse_ratio + (0.12 if has_date_name else 0.05)
                    return {
                        "semantic_type": "datetime",
                        "confidence": round(min(0.99, conf), 2),
                        "reason": (
                            f"{parse_ratio * 100:.0f}% of values match explicit date patterns and parse cleanly"
                            + (" (name also suggests a date/time)." if has_date_name else ".")
                        ),
                    }

    # Otherwise categorical vs. identifier/free text
    n_total = len(non_null)
    n_unique = int(non_null.nunique())
    unique_ratio = n_unique / max(n_total, 1)
    if n_total >= 10 and unique_ratio >= 0.85 and ("id" in tokens or "name" in tokens or n_unique > 25):
        return {
            "semantic_type": "identifier",
            "confidence": 0.85,
            "reason": f"High uniqueness ({unique_ratio * 100:.0f}% unique values).",
        }

    return {
        "semantic_type": "categorical",
        "confidence": 0.9,
        "reason": f"Text/categorical values ({n_unique} unique across {n_total} non-null rows).",
    }


def detect_datetime_columns(df: pd.DataFrame, sample_size: int = 200) -> list[str]:
    """Identify columns that are already datetime or semantically verified as date strings."""
    detected: list[str] = []
    for col in df.columns:
        info = classify_column_semantic(df[col], str(col), sample_size=sample_size)
        if info["semantic_type"] == "datetime":
            detected.append(str(col))
    return detected
