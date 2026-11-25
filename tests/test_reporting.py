import pandas as pd

from agents.profiler import profile_dataset
from agents.statistician import forecast_series, iqr_outliers, isolation_forest_outliers, run_statistical_tests
from core.reporting import make_pdf_report


def test_make_pdf_report_backward_compatible():
    df = pd.DataFrame({"x": [1, 2, 3]})
    profile = profile_dataset(df)
    pdf = make_pdf_report(profile, ["Dataset has 3 rows."], "Check row count")
    assert pdf.startswith(b"%PDF")
    assert len(pdf) > 500


def test_make_pdf_report_full_sections():
    dates = pd.date_range("2026-01-01", periods=12, freq="W")
    df = pd.DataFrame({
        "date": dates,
        "region": ["North", "South"] * 6,
        "revenue": [100 + i * 25 for i in range(12)],
        "units": [5 + i for i in range(12)],
    })
    profile = profile_dataset(df)
    iqr = iqr_outliers(df, "revenue")
    iso = isolation_forest_outliers(df, contamination=0.1)
    tests = run_statistical_tests(df)
    fcst = forecast_series(df, "date", "revenue", horizon=3)

    pdf = make_pdf_report(
        profile=profile,
        insights=["Verified revenue trend."],
        question="Analyze weekly revenue",
        cleaning_log=["Trimmed whitespace in 'region'."],
        stat_tests=tests,
        iqr_result=iqr,
        iso_result=iso,
        forecast_result=fcst,
        rejected_findings=[{"reason": "Unsupported causal claim."}],
        chart_titles=["Revenue Over Time", "Forecast: revenue"],
    )
    assert pdf.startswith(b"%PDF")
    assert len(pdf) > 1500
