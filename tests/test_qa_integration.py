from io import BytesIO
import pandas as pd
import pytest

from agents.profiler import cleaning_suggestions, profile_dataset
from agents.reviewer import review_findings
from agents.statistician import forecast_series, iqr_outliers
from agents.visualization import make_chart
from core.cleaning import clean_dataframe
from core.ingestion import load_dataframe, validate_dataframe
from orchestration.graph import run_analysis_workflow


def test_workflow_schema_1_sales_timeseries():
    """Schema 1: Sales time-series dataset with SQL, anomalies, forecasting, and PDF."""
    dates = pd.date_range("2026-01-01", periods=14, freq="W")
    df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "region": ["North", "South"] * 7,
        "product": ["Laptop", "Monitor"] * 7,
        "revenue": [1000 + i * 100 for i in range(14)],
        "quantity": [2 + (i % 4) for i in range(14)],
    })
    state = run_analysis_workflow(df, question="total revenue by region", horizon=3)
    assert state["status"] == "completed"
    assert state["profile"]["rows"] == 14
    assert state["sql_result"] is not None
    assert state["forecast_result"]["status"] == "completed"
    assert state["review_result"]["status"] == "approved"
    assert state["pdf_bytes"].startswith(b"%PDF")


def test_workflow_schema_2_app_catalog_with_dirty_data():
    """Schema 2: App catalog with Size, Price, Rating, Last Updated, missing values, and duplicates."""
    df = pd.DataFrame({
        "App Name": ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta", "Iota", "Kappa", "Alpha"],
        "Category": ["Tools", "Games", "Tools", "Finance", "Games", "Tools", "Finance", "Games", "Tools", "Games", "Tools"],
        "Size": ["19M", "14M", "8.7M", "25M", "12M", "30M", "11M", "9.5M", "18M", "22M", "19M"],
        "Price": ["$0", "$4.99", "$1.99", "$0", "$2.99", "$0", "$9.99", "$0", "$1.49", "$0", "$0"],
        "Rating": [4.5, 4.2, None, 4.8, 4.1, 4.6, 3.8, 4.4, 4.7, 4.3, 4.5],
        "Last Updated": [
            "January 7, 2018", "January 15, 2018", "February 1, 2018", "March 8, 2018",
            "April 20, 2018", "May 5, 2018", "June 12, 2018", "July 19, 2018",
            "August 3, 2018", "September 14, 2018", "January 7, 2018",
        ],
    })
    profile = profile_dataset(df)
    assert profile["datetime_columns"] == ["Last Updated"]
    assert "Size" not in profile["datetime_columns"]
    assert "Price" not in profile["datetime_columns"]
    assert profile["duplicate_rows"] == 1
    assert profile["missing_values"]["Rating"] == 1

    cleaned, log = clean_dataframe(df, drop_duplicates=True, fill_numeric_median=True, parse_dates=True)
    assert len(cleaned) == 10
    assert cleaned["Rating"].isna().sum() == 0
    assert pd.api.types.is_datetime64_any_dtype(cleaned["Last Updated"])
    assert len(log) >= 3

    state = run_analysis_workflow(cleaned, question="how many rows")
    assert state["status"] == "completed"
    assert int(state["sql_result"].iloc[0, 0]) == 10


def test_workflow_schema_3_purely_categorical_survey():
    """Schema 3: Purely categorical dataset with no numeric or datetime columns."""
    df = pd.DataFrame({
        "respondent_id": [f"R{i}" for i in range(1, 13)],
        "department": ["Eng", "Sales", "HR", "Ops"] * 3,
        "satisfaction": ["High", "Medium", "High", "Low"] * 3,
    })
    state = run_analysis_workflow(df, question="unique department")
    assert state["status"] == "completed"
    assert state["isolation_forest_result"]["status"] == "skipped"
    assert state["forecast_result"]["status"] == "skipped"
    assert state["sql_result"] is not None
    assert state["review_result"]["status"] == "approved"


def test_edge_cases_missing_columns_invalid_files_and_short_forecast():
    df = pd.DataFrame({"date": ["2026-01-01", "2026-01-02"], "val": [10, 20]})
    with pytest.raises(ValueError, match="Unknown column"):
        iqr_outliers(df, "nonexistent_col")
    with pytest.raises(ValueError, match="Unknown column"):
        make_chart(df, "nonexistent_col")
    with pytest.raises(ValueError, match="empty"):
        load_dataframe(BytesIO(b""), "empty.csv")

    fc = forecast_series(df, "date", "val")
    assert fc["status"] == "skipped"

    # Reviewer numerical verification against computed context
    profile = profile_dataset(df)
    res = review_findings(
        df,
        [
            {"claim": "2 rows", "evidence": "profile", "metric": "profile.rows", "expected_value": 2, "sample_size": 10},
            {"claim": "99 rows", "evidence": "profile", "metric": "profile.rows", "expected_value": 99, "sample_size": 10},
        ],
        computed_context={"profile": profile},
    )
    assert len(res["approved"]) == 1
    assert len(res["rejected"]) == 1
