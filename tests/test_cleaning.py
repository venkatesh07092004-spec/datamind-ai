import pandas as pd
from core.cleaning import clean_dataframe, dataframe_to_safe_csv_bytes, sanitize_csv_cell


def test_trim_and_drop_duplicates():
    df = pd.DataFrame({"name": [" Alice ", "Alice"], "score": [1, 1]})
    cleaned, log = clean_dataframe(df, trim_text=True, drop_duplicates=True)
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["name"] == "Alice"
    assert len(log) >= 1


def test_median_fill():
    df = pd.DataFrame({"score": [1.0, None, 3.0]})
    cleaned, _ = clean_dataframe(df, fill_numeric_median=True)
    assert cleaned["score"].isna().sum() == 0
    assert cleaned.loc[1, "score"] == 2.0


def test_drop_constant_and_parse_dates():
    df = pd.DataFrame({
        "d": ["2026-01-01", "2026-01-02", "2026-01-03"],
        "const": [42, 42, 42],
    })
    cleaned, _ = clean_dataframe(df, parse_dates=True, drop_constant_columns=True)
    assert "const" not in cleaned.columns
    assert pd.api.types.is_datetime64_any_dtype(cleaned["d"])


def test_csv_formula_injection_sanitized():
    assert sanitize_csv_cell("=CMD()") == "'=CMD()"
    assert sanitize_csv_cell("+1+1") == "'+1+1"
    df = pd.DataFrame({"note": ["=HYPERLINK(\"http://bad\")", "normal"]})
    csv_text = dataframe_to_safe_csv_bytes(df).decode("utf-8")
    assert "'=HYPERLINK" in csv_text
