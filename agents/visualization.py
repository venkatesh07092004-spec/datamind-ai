"""Visualization Agent: chart recommendations, dark-themed Plotly builders, and explanations."""

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from core.ingestion import detect_datetime_columns

_DARK_LAYOUT = {
    "template": "plotly_dark",
    "paper_bgcolor": "rgba(15, 23, 42, 0.0)",
    "plot_bgcolor": "rgba(15, 23, 42, 0.45)",
    "font": {"color": "#E2E8F0", "family": "Inter, system-ui, sans-serif"},
    "margin": {"l": 40, "r": 25, "t": 50, "b": 40},
}


def apply_dark_theme(fig: go.Figure) -> go.Figure:
    """Apply consistent dark AI dashboard styling to a Plotly figure."""
    fig.update_layout(
        **_DARK_LAYOUT,
        hovermode="closest",
        legend={"bgcolor": "rgba(15, 23, 42, 0.6)", "bordercolor": "rgba(148, 163, 184, 0.2)", "borderwidth": 1},
    )
    fig.update_xaxes(gridcolor="rgba(148, 163, 184, 0.12)", zerolinecolor="rgba(148, 163, 184, 0.2)")
    fig.update_yaxes(gridcolor="rgba(148, 163, 184, 0.12)", zerolinecolor="rgba(148, 163, 184, 0.2)")
    return fig


def make_anomaly_chart(
    df: pd.DataFrame,
    iso_result: dict[str, Any],
    x_col: str | None = None,
    y_col: str | None = None,
) -> tuple[go.Figure | None, str]:
    """Visualize normal records vs. Isolation Forest flagged anomalies."""
    features = iso_result.get("features", [])
    if not features:
        return None, "No numeric features available to plot anomalies."

    x = x_col if x_col in df.columns else features[0]
    y = y_col if y_col in df.columns else (features[1] if len(features) > 1 else features[0])

    flagged_rows = pd.DataFrame(iso_result.get("rows", []))
    plot_df = df[[x, y]].copy() if x != y else pd.DataFrame({x: df[x], "row_index": df.index})
    y_axis = y if x != y else "row_index"

    plot_df["status"] = "Normal"
    if not flagged_rows.empty and x in flagged_rows.columns:
        # Match flagged records back by index if preserved, or by value tuple
        if x != y and y in flagged_rows.columns:
            flagged_pairs = set(zip(flagged_rows[x], flagged_rows[y]))
            is_anom = [
                (vx, vy) in flagged_pairs for vx, vy in zip(plot_df[x], plot_df[y_axis])
            ]
        else:
            flagged_vals = set(flagged_rows[x])
            is_anom = [vx in flagged_vals for vx in plot_df[x]]
        plot_df.loc[is_anom, "status"] = "Anomaly"

    fig = px.scatter(
        plot_df,
        x=x,
        y=y_axis,
        color="status",
        color_discrete_map={"Normal": "#38BDF8", "Anomaly": "#F43F5E"},
        title=f"Anomaly Detection View: {y_axis} vs. {x}",
        labels={x: x, y_axis: y_axis, "status": "Record Status"},
        opacity=0.85,
    )
    explanation = (
        f"Highlights {iso_result.get('count', 0)} Isolation Forest flagged record(s) in red "
        f"against {len(df):,} total records."
    )
    return apply_dark_theme(fig), explanation


def make_chart(df: pd.DataFrame, column: str) -> go.Figure:
    """Generate a single-column distribution or frequency Plotly chart."""
    if column not in df.columns:
        raise ValueError(f"Unknown column: {column}")
    series = df[column]
    if pd.api.types.is_numeric_dtype(series):
        fig = px.histogram(
            df,
            x=column,
            title=f"Distribution of {column}",
            color_discrete_sequence=["#38BDF8"],
        )
    else:
        counts = (
            df[column]
            .astype(str)
            .value_counts(dropna=False)
            .head(20)
            .rename_axis(column)
            .reset_index(name="count")
        )
        fig = px.bar(
            counts,
            x=column,
            y="count",
            title=f"Top values of {column}",
            color_discrete_sequence=["#818CF8"],
        )
    return apply_dark_theme(fig)


def make_correlation_heatmap(df: pd.DataFrame) -> tuple[go.Figure | None, str]:
    """Build an interactive correlation heatmap for numeric columns."""
    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        return None, "At least two numeric columns are required for a correlation heatmap."
    corr = numeric.corr(numeric_only=True).round(3)
    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Numeric Correlation Matrix",
    )
    explanation = (
        f"Displays pairwise Pearson correlations across {numeric.shape[1]} numeric columns "
        "(-1 indicates strong inverse linear association, +1 indicates strong positive association)."
    )
    return apply_dark_theme(fig), explanation


def make_scatter_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: str | None = None,
) -> tuple[go.Figure, str]:
    """Build a scatter plot comparing two numeric variables."""
    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError("Selected scatter columns do not exist in dataset.")
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color_col if color_col and color_col in df.columns else None,
        title=f"{y_col} vs. {x_col}",
        opacity=0.82,
    )
    explanation = f"Compares '{y_col}' against '{x_col}' across {len(df):,} rows to reveal linear or nonlinear patterns."
    return apply_dark_theme(fig), explanation


def make_box_chart(
    df: pd.DataFrame,
    numeric_col: str,
    category_col: str | None = None,
) -> tuple[go.Figure, str]:
    """Build a box plot showing spread, median, and potential outliers."""
    if numeric_col not in df.columns:
        raise ValueError(f"Unknown column: {numeric_col}")
    fig = px.box(
        df,
        x=category_col if category_col and category_col in df.columns else None,
        y=numeric_col,
        color=category_col if category_col and category_col in df.columns else None,
        title=f"Spread & Outliers of {numeric_col}" + (f" by {category_col}" if category_col else ""),
    )
    explanation = (
        f"Shows median, interquartile range (IQR), and outlier points for '{numeric_col}'"
        + (f" grouped by '{category_col}'." if category_col else ".")
    )
    return apply_dark_theme(fig), explanation


def make_timeseries_chart(df: pd.DataFrame, date_col: str, value_col: str) -> tuple[go.Figure, str]:
    """Build an aggregated time-series line chart."""
    work = df[[date_col, value_col]].copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce", format="mixed")
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.dropna().groupby(date_col, as_index=False)[value_col].sum().sort_values(date_col)
    fig = px.line(
        work,
        x=date_col,
        y=value_col,
        markers=True,
        title=f"{value_col} Over Time ({date_col})",
        color_discrete_sequence=["#22D3EE"],
    )
    explanation = f"Tracks aggregated '{value_col}' chronologically across {len(work)} distinct time periods."
    return apply_dark_theme(fig), explanation


def make_forecast_chart(forecast_result: dict[str, Any]) -> tuple[go.Figure, str]:
    """Plot historical series alongside forecast values and 95% residual bounds."""
    hist = pd.DataFrame(forecast_result.get("history", []))
    fcst = pd.DataFrame(forecast_result.get("forecast", []))
    target = forecast_result.get("target_col", "value")
    model = forecast_result.get("selected_model", "model")

    fig = go.Figure()
    if not hist.empty:
        fig.add_trace(
            go.Scatter(
                x=hist["date"],
                y=hist["actual"],
                mode="lines+markers",
                name="Historical Actual",
                line={"color": "#38BDF8", "width": 2.5},
            )
        )
    if not fcst.empty:
        fig.add_trace(
            go.Scatter(
                x=fcst["date"],
                y=fcst["upper_95"],
                mode="lines",
                line={"width": 0},
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=fcst["date"],
                y=fcst["lower_95"],
                mode="lines",
                line={"width": 0},
                fill="tonexty",
                fillcolor="rgba(167, 139, 250, 0.2)",
                name="95% Residual Band",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=fcst["date"],
                y=fcst["forecast"],
                mode="lines+markers",
                name=f"Forecast ({model})",
                line={"color": "#A78BFA", "width": 2.5, "dash": "dash"},
            )
        )
    fig.update_layout(title=f"Forecast for {target} ({model})")
    explanation = (
        f"Shows historical '{target}' values and {len(fcst)}-period forecast generated by '{model}' "
        "selected via chronological holdout RMSE."
    )
    return apply_dark_theme(fig), explanation


def explain_chart(chart_type: str, column: str = "", target: str = "") -> str:
    """Return a concise explanation of what a given chart type represents."""
    descriptions = {
        "histogram": f"Histogram showing the frequency distribution and skewness of '{column}'.",
        "bar": f"Bar chart ranking the most frequent categories in '{column}'.",
        "heatmap": "Correlation heatmap showing linear association strength between numeric features.",
        "scatter": f"Scatter plot showing the relationship between '{column}' and '{target}'.",
        "box": f"Box plot highlighting median, IQR spread, and outliers for '{column}'.",
        "timeseries": f"Time-series line chart showing '{target}' aggregated chronologically by '{column}'.",
    }
    return descriptions.get(chart_type, "Interactive Plotly visualization of the active dataset.")


def recommend_charts(df: pd.DataFrame, profile: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Recommend suitable chart specifications based on dataset schema."""
    numeric_cols = [str(c) for c in df.select_dtypes(include="number").columns]
    datetime_cols = (
        profile.get("datetime_columns", [])
        if profile is not None
        else detect_datetime_columns(df)
    )
    cat_cols = [
        str(c)
        for c in df.columns
        if str(c) not in numeric_cols and str(c) not in datetime_cols and 2 <= df[c].nunique(dropna=True) <= 25
    ]

    recs: list[dict[str, Any]] = []
    if datetime_cols and numeric_cols:
        recs.append({
            "chart_type": "timeseries",
            "columns": [datetime_cols[0], numeric_cols[0]],
            "title": f"{numeric_cols[0]} Trend Over {datetime_cols[0]}",
            "reason": "Datetime and numeric columns detected.",
        })
    if len(numeric_cols) >= 2:
        recs.append({
            "chart_type": "heatmap",
            "columns": numeric_cols[:8],
            "title": "Numeric Correlation Heatmap",
            "reason": "Multiple numeric columns available for pairwise correlation.",
        })
        recs.append({
            "chart_type": "scatter",
            "columns": [numeric_cols[0], numeric_cols[1]],
            "title": f"{numeric_cols[1]} vs. {numeric_cols[0]}",
            "reason": "Inspect bivariate relationship between top numeric columns.",
        })
    if numeric_cols and cat_cols:
        recs.append({
            "chart_type": "box",
            "columns": [numeric_cols[0], cat_cols[0]],
            "title": f"{numeric_cols[0]} by {cat_cols[0]}",
            "reason": "Compare numeric distribution across categories.",
        })
    if numeric_cols:
        recs.append({
            "chart_type": "histogram",
            "columns": [numeric_cols[0]],
            "title": f"Distribution of {numeric_cols[0]}",
            "reason": "Inspect spread and skewness of primary numeric column.",
        })
    if cat_cols:
        recs.append({
            "chart_type": "bar",
            "columns": [cat_cols[0]],
            "title": f"Category Counts for {cat_cols[0]}",
            "reason": "Inspect category balance.",
        })
    return recs
