import pandas as pd

from agents.profiler import cleaning_suggestions, profile_dataset
from core.cleaning import clean_dataframe
from core.ingestion import classify_column_semantic, detect_datetime_columns


def test_profile_counts_and_quality():
    df = pd.DataFrame({
        "x": [1, 2, None, 4, 5, 6, 7, 8, 9, 10],
        "label": ["a", "b", "b", "a", "b", "a", "b", "a", "b", "a"],
        "const": [1] * 10,
    })
    profile = profile_dataset(df)
    assert profile["rows"] == 10
    assert profile["columns"] == 3
    assert profile["missing_values"]["x"] == 1
    assert profile["duplicate_rows"] == 0
    assert "const" in profile["constant_columns"]
    assert 0 <= profile["quality_score"] <= 100
    assert len(profile["unusual_patterns"]) >= 1


def test_duplicate_and_constant_suggestions_include_confidence_and_explanation():
    df = pd.DataFrame({"x": [1, 1], "txt": [" hi ", " hi "]})
    suggestions = cleaning_suggestions(df)
    issues = {s["issue"] for s in suggestions}
    assert "duplicate_rows" in issues
    assert "leading_trailing_whitespace" in issues
    assert "constant_column" in issues
    for s in suggestions:
        assert 0.0 <= s["confidence"] <= 1.0
        assert isinstance(s["explanation"], str) and len(s["explanation"]) > 5


def test_semantic_classification_does_not_misclassify_size_price_rating_revenue_quantity():
    df = pd.DataFrame({
        "App Name": ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"],
        "Category": ["Tools", "Games", "Tools", "Finance", "Games"],
        "Size": ["19M", "14M", "8.7M", "25M", "12M"],
        "Price": ["$0", "$4.99", "$1.99", "$0", "$2.99"],
        "Rating": [4.5, 4.2, 3.9, 4.8, 4.1],
        "Revenue": ["1,200", "3,450", "890", "5,100", "2,300"],
        "Quantity": ["10", "25", "5", "40", "15"],
        "Last Updated": [
            "January 7, 2018",
            "January 15, 2018",
            "August 1, 2018",
            "June 8, 2018",
            "June 20, 2018",
        ],
    })

    dt_cols = detect_datetime_columns(df)
    assert dt_cols == ["Last Updated"]

    assert classify_column_semantic(df["Size"], "Size")["semantic_type"] == "numeric_text"
    assert classify_column_semantic(df["Price"], "Price")["semantic_type"] == "numeric_text"
    assert classify_column_semantic(df["Rating"], "Rating")["semantic_type"] == "numeric"
    assert classify_column_semantic(df["Revenue"], "Revenue")["semantic_type"] == "numeric_text"
    assert classify_column_semantic(df["Quantity"], "Quantity")["semantic_type"] == "numeric_text"
    assert classify_column_semantic(df["Category"], "Category")["semantic_type"] == "categorical"
    assert classify_column_semantic(df["Last Updated"], "Last Updated")["semantic_type"] == "datetime"

    suggestions = cleaning_suggestions(df)
    dt_suggestions = [s for s in suggestions if s["issue"] == "potential_datetime_text"]
    assert [s["column"] for s in dt_suggestions] == ["Last Updated"]
    assert dt_suggestions[0]["confidence"] >= 0.85

    # Applying parse_dates only parses Last Updated and leaves Size/Price/Revenue/Quantity untouched
    cleaned, _ = clean_dataframe(df, parse_dates=True)
    assert pd.api.types.is_datetime64_any_dtype(cleaned["Last Updated"])
    assert cleaned["Size"].tolist() == df["Size"].tolist()
    assert cleaned["Price"].tolist() == df["Price"].tolist()
    # Original df remains unmodified
    assert not pd.api.types.is_datetime64_any_dtype(df["Last Updated"])
