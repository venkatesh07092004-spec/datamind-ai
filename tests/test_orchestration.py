import pandas as pd

from orchestration.graph import run_analysis_workflow


def _sample_sales_df() -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=16, freq="W")
    return pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "region": ["North", "South", "East", "West"] * 4,
        "product": ["Laptop", "Monitor", "Keyboard", "Mouse"] * 4,
        "revenue": [1000 + i * 75 for i in range(16)],
        "quantity": [2 + (i % 5) for i in range(16)],
    })


def test_full_langgraph_workflow_with_sql_and_forecast():
    df = _sample_sales_df()
    state = run_analysis_workflow(
        df,
        question="total revenue by region",
        horizon=4,
    )

    assert state["validation_ok"] is True
    assert state["status"] == "completed"
    assert state["profile"]["rows"] == 16
    assert state["sql_result"] is not None
    assert len(state["sql_result"]) == 4
    assert state["forecast_result"]["status"] == "completed"
    assert len(state["charts"]) >= 2
    assert state["review_result"]["status"] == "approved"
    assert len(state["pdf_bytes"]) > 100

    agents_run = [ev["agent"] for ev in state["timeline"]]
    assert "Ingestion Validator" in agents_run
    assert "Data Profiler" in agents_run
    assert "SQL Agent" in agents_run
    assert "Statistical Analyst" in agents_run
    assert "Visualization Agent" in agents_run
    assert "Insight Reviewer" in agents_run
    assert "Report Builder" in agents_run


def test_workflow_handles_invalid_dataset_gracefully():
    empty_df = pd.DataFrame()
    state = run_analysis_workflow(empty_df)

    assert state["validation_ok"] is False
    assert state["status"] == "failed"
    assert len(state["errors"]) >= 1
    assert state["pdf_bytes"] == b""


def test_workflow_handles_sql_failure_without_crashing():
    df = _sample_sales_df()
    state = run_analysis_workflow(df, question="xyzzy unmappable nonsense query 12345")

    assert state["validation_ok"] is True
    assert state["sql_error"] is not None
    assert state["status"] == "completed_with_warnings"
    assert state["review_result"]["status"] == "approved"
    assert len(state["pdf_bytes"]) > 100


def test_workflow_reviewer_rejection_and_bounded_retry_loop():
    df = _sample_sales_df()
    fabricated = [
        {
            "claim": "Dataset has 99999 rows.",
            "evidence": "profiler.profile_dataset",
            "metric": "profile.rows",
            "expected_value": 99999,
            "sample_size": 16,
        },
        {
            "claim": "Quantity causes revenue to rise.",
            "evidence": "correlation",
            "sample_size": 16,
        },
    ]
    state = run_analysis_workflow(
        df,
        max_retries=1,
        injected_findings=fabricated,
    )

    # First pass rejects fabricated findings, Correction Loop rebuilds from verified evidence,
    # second review pass approves strictly verified findings and terminates at retry_count == 1.
    assert state["retry_count"] == 1
    assert state["review_result"]["status"] == "approved"
    assert len(state["review_result"]["rejected"]) == 0
    agents_run = [ev["agent"] for ev in state["timeline"]]
    assert "Correction Loop" in agents_run
