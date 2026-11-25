from io import BytesIO
import pandas as pd
import pytest

from core.ingestion import detect_datetime_columns, load_dataframe, validate_dataframe


def test_load_csv_and_validate():
    csv_bytes = BytesIO(b"date,region,revenue\n2026-01-01,North,100\n2026-01-02,South,200\n")
    df = load_dataframe(csv_bytes, "test.csv")
    validate_dataframe(df)
    assert df.shape == (2, 3)
    assert "date" in detect_datetime_columns(df)


def test_reject_unsupported_extension():
    with pytest.raises(ValueError, match="Unsupported file type"):
        load_dataframe(BytesIO(b"abc"), "malware.exe")


def test_reject_oversized_file():
    big_bytes = BytesIO(b"a,b\n" + b"1,2\n" * 500)
    with pytest.raises(ValueError, match="exceeds"):
        load_dataframe(big_bytes, "big.csv", max_file_size_mb=0.0001)


def test_validate_rejects_empty_and_duplicate_cols():
    with pytest.raises(ValueError):
        validate_dataframe(pd.DataFrame())
    df_dup = pd.DataFrame([[1, 2]], columns=["a", "a"])
    with pytest.raises(ValueError, match="unique"):
        validate_dataframe(df_dup)
