import pandas as pd

from agents.statistician import forecast_series, isolation_forest_outliers
from agents.visualization import (
    make_anomaly_chart,
    make_box_chart,
    make_chart,
    make_correlation_heatmap,
    make_forecast_chart,
    make_scatter_chart,
    make_timeseries_chart,
    recommend_charts,
)


def test_recommendations_and_charts():
    dates = pd.date_range("2026-01-01", periods=12, freq="W")
    df = pd.DataFrame({
        "date": dates,
        "region": ["North", "South"] * 6,
        "revenue": [100 + i * 15 for i in range(12)],
        "units": [10 + i for i in range(12)],
    })
    recs = recommend_charts(df)
    assert len(recs) >= 3

    fig_hist = make_chart(df, "revenue")
    assert fig_hist is not None

    fig_hm, exp_hm = make_correlation_heatmap(df)
    assert fig_hm is not None and "Pearson" in exp_hm

    fig_sc, _ = make_scatter_chart(df, "units", "revenue", "region")
    assert fig_sc is not None

    fig_box, _ = make_box_chart(df, "revenue", "region")
    assert fig_box is not None

    fig_ts, _ = make_timeseries_chart(df, "date", "revenue")
    assert fig_ts is not None

    fcst = forecast_series(df, "date", "revenue", horizon=3)
    fig_fc, _ = make_forecast_chart(fcst)
    assert fig_fc is not None

    iso = isolation_forest_outliers(df, contamination=0.1)
    fig_anom, exp_anom = make_anomaly_chart(df, iso)
    assert fig_anom is not None and "Isolation Forest" in exp_anom
