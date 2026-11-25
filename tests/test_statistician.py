import numpy as np
import pandas as pd
import pytest

from agents.statistician import (
    correlation_analysis,
    detrended_isolation_forest_outliers,
    forecast_series,
    iqr_outliers,
    isolation_forest_outliers,
    numeric_summary,
    run_statistical_tests,
)


def test_iqr_and_isolation_forest_explanations():
    rng = np.random.default_rng(42)
    normal_vals = rng.normal(100, 5, size=30).tolist()
    df = pd.DataFrame({
        "metric_a": normal_vals + [500.0],
        "metric_b": normal_vals + [900.0],
    })
    iqr = iqr_outliers(df, "metric_a")
    assert iqr["count"] >= 1
    assert "disclaimer" in iqr

    iso = isolation_forest_outliers(df, contamination=0.05, random_state=42)
    assert iso["status"] == "completed"
    assert iso["count"] >= 1
    assert "why_flagged" in iso["rows"][0]
    assert "disclaimer" in iso

    with pytest.raises(ValueError, match="Contamination"):
        isolation_forest_outliers(df, contamination=0.9)

    df_trend = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=20, freq="D"),
        "val": [100 + i * 10 + (200 if i == 10 else 0) for i in range(20)],
    })
    det = detrended_isolation_forest_outliers(df_trend, columns=["val"], date_col="date", contamination=0.05)
    assert det["status"] == "completed"
    assert det["count"] >= 1


def test_summary_cis_correlation_and_statistical_tests():
    df = pd.DataFrame({
        "group": ["A"] * 10 + ["B"] * 10,
        "tier": ["low", "high"] * 10,
        "x": list(range(1, 11)) + list(range(21, 31)),
        "y": [v * 2 for v in range(1, 21)],
    })
    summary = numeric_summary(df)
    assert "ci_95_low" in summary.columns
    assert "ci_95_high" in summary.columns

    corr = correlation_analysis(df)
    assert not corr["matrix"].empty
    assert corr["pairs"][0]["significant"] is True
    assert corr["pairs"][0]["ci_95_low"] is not None

    tests = run_statistical_tests(df)
    assert len(tests["normality"]) >= 1
    assert len(tests["group_comparisons"]) >= 1
    assert tests["group_comparisons"][0]["significant_at_0_05"] is True
    assert len(tests["categorical_associations"]) == 1


def test_small_sample_and_chi_square_warning():
    df = pd.DataFrame({
        "cat1": ["a", "a", "b", "b"],
        "cat2": ["x", "y", "x", "y"],
        "val": [1.0, 2.0, 3.0, 4.0],
    })
    tests = run_statistical_tests(df)
    assert tests["categorical_associations"][0]["assumption_met"] is False
    assert tests["categorical_associations"][0]["warning"] is not None


def test_forecast_chronological_holdout_and_no_leakage():
    dates = pd.date_range("2026-01-01", periods=16, freq="W")
    values = [100 + i * 10 for i in range(16)]
    df = pd.DataFrame({"date": dates, "revenue": values})

    fcst = forecast_series(df, date_col="date", target_col="revenue", horizon=4)
    assert fcst["status"] == "completed"
    assert fcst["test_periods"] >= 2
    assert fcst["selected_model"] == "ridge_linear_trend"
    assert fcst["beats_baseline"] is True
    assert len(fcst["forecast"]) == 4

    # Verify no data leakage: modifying only the final holdout test rows must NOT change
    # the Naive or Moving Average holdout predictions computed from training rows.
    df_perturbed = df.copy()
    df_perturbed.loc[df_perturbed.index[-2:], "revenue"] = 999999.0
    fcst_perturbed = forecast_series(df_perturbed, date_col="date", target_col="revenue", horizon=4)
    assert fcst_perturbed["test_comparison"][0]["naive"] == fcst["test_comparison"][0]["naive"]
    assert fcst_perturbed["test_comparison"][0]["moving_avg"] == fcst["test_comparison"][0]["moving_avg"]


def test_forecast_skips_insufficient_history():
    df = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=5, freq="D"),
        "revenue": [10, 20, 30, 40, 50],
    })
    fcst = forecast_series(df, date_col="date", target_col="revenue")
    assert fcst["status"] == "skipped"
    assert len(fcst["limitations"]) >= 1
