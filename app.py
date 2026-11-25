from pathlib import Path
import pandas as pd
import streamlit as st

from agents.profiler import cleaning_suggestions
from agents.sql_agent import generate_and_validate_sql
from agents.statistician import forecast_series, iqr_outliers, isolation_forest_outliers
from agents.visualization import (
    make_anomaly_chart,
    make_box_chart,
    make_chart,
    make_correlation_heatmap,
    make_forecast_chart,
    make_scatter_chart,
    make_timeseries_chart,
)
from core.cleaning import clean_dataframe, dataframe_to_safe_csv_bytes
from core.config import DEFAULT_OLLAMA_MODEL, MAX_FILE_SIZE_MB, MAX_QUERY_ROWS, MAX_ROWS
from core.ingestion import load_dataframe, validate_dataframe
from core.reporting import make_pdf_report
from core.sql_engine import execute_readonly_query
from orchestration.graph import run_analysis_workflow

st.set_page_config(
    page_title="DataMind AI · Autonomous Multi-Agent Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Subtle futuristic dark-theme styling with blue & violet accents
st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at top right, rgba(99, 102, 241, 0.08), transparent 40%),
                    radial-gradient(circle at top left, rgba(56, 189, 248, 0.07), transparent 35%),
                    #0B1120;
        color: #F8FAFC;
    }
    .dm-header {
        padding: 0.75rem 1.1rem;
        border-radius: 12px;
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.85), rgba(15, 23, 42, 0.92));
        border: 1px solid rgba(56, 189, 248, 0.25);
        margin-bottom: 1rem;
    }
    .dm-header h1 {
        margin: 0;
        font-size: 1.75rem;
        font-weight: 700;
        background: linear-gradient(90deg, #38BDF8, #A78BFA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .dm-header p {
        margin: 0.25rem 0 0 0;
        color: #94A3B8;
        font-size: 0.9rem;
    }
    .dm-kpi {
        background: rgba(17, 24, 39, 0.85);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-top: 2px solid #38BDF8;
        border-radius: 10px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.5rem;
    }
    .dm-kpi.violet {
        border-top-color: #A78BFA;
    }
    .dm-kpi-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94A3B8;
    }
    .dm-kpi-value {
        font-size: 1.45rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-top: 0.2rem;
    }
    .dm-kpi-sub {
        font-size: 0.78rem;
        color: #CBD5E1;
        margin-top: 0.15rem;
    }
    .dm-badge {
        display: inline-block;
        padding: 0.15rem 0.55rem;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
    }
    .dm-badge-completed { background: rgba(16, 185, 129, 0.18); color: #34D399; border: 1px solid rgba(52, 211, 153, 0.35); }
    .dm-badge-running { background: rgba(56, 189, 248, 0.18); color: #38BDF8; border: 1px solid rgba(56, 189, 248, 0.35); }
    .dm-badge-skipped { background: rgba(148, 163, 184, 0.16); color: #CBD5E1; border: 1px solid rgba(148, 163, 184, 0.3); }
    .dm-badge-needs_review { background: rgba(245, 158, 11, 0.18); color: #FBBF24; border: 1px solid rgba(251, 191, 36, 0.35); }
    .dm-badge-failed { background: rgba(239, 68, 68, 0.18); color: #F87171; border: 1px solid rgba(248, 113, 113, 0.35); }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="dm-header">
        <h1>DataMind AI · Autonomous Multi-Agent Data Analyst</h1>
        <p>LangGraph Orchestration · Local-First Analytics · Read-Only DuckDB SQL · Evidence-Verified Insights</p>
    </div>
    """,
    unsafe_allow_html=True,
)


def _render_kpi(label: str, value: str, sub: str = "", violet: bool = False) -> None:
    cls = "dm-kpi violet" if violet else "dm-kpi"
    st.markdown(
        f"""
        <div class="{cls}">
            <div class="dm-kpi-label">{label}</div>
            <div class="dm-kpi-value">{value}</div>
            <div class="dm-kpi-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _reset_dataset_state(new_df: pd.DataFrame, label: str) -> None:
    """Reset all dataset-scoped session state so results from a previous dataset are never reused."""
    st.session_state.original_df = new_df.copy()
    st.session_state.working_df = new_df.copy()
    st.session_state.cleaning_log = []
    st.session_state.current_upload = label
    st.session_state.workflow_version = st.session_state.get("workflow_version", 0) + 1
    for stale_key in ("workflow_state", "workflow_key", "last_sql", "last_sql_source", "last_sql_result"):
        st.session_state.pop(stale_key, None)


with st.sidebar:
    st.markdown("### Control Center")
    use_sample = st.button("Load Sample Sales Dataset", use_container_width=True)
    if use_sample:
        sample_path = Path("sample_data/sales.csv")
        if sample_path.exists():
            sample_df = pd.read_csv(sample_path)
            _reset_dataset_state(sample_df, "sample_data/sales.csv")
            st.rerun()

    st.divider()
    st.markdown("### Local AI & Limits")
    use_ollama = st.checkbox("Use optional local Ollama for SQL", value=False)
    ollama_model = st.text_input("Ollama model", value=DEFAULT_OLLAMA_MODEL, disabled=not use_ollama)
    st.caption(
        f"Limits: {MAX_FILE_SIZE_MB:.0f} MB file · {MAX_ROWS:,} rows · {MAX_QUERY_ROWS:,} SQL rows"
    )
    st.caption("100% CPU-friendly · No paid API key required.")
    st.warning("Use only authorized data. Avoid uploading sensitive PII.")

uploaded = st.file_uploader("Upload a CSV or Excel dataset", type=["csv", "xlsx", "xls"])

if uploaded is not None:
    upload_token = f"{uploaded.name}:{getattr(uploaded, 'size', 0)}:{getattr(uploaded, 'file_id', '')}"
    if st.session_state.get("last_handled_upload_token") != upload_token:
        try:
            with st.spinner("Loading and validating dataset..."):
                raw_df = load_dataframe(uploaded, uploaded.name)
                validate_dataframe(raw_df)
            _reset_dataset_state(raw_df, uploaded.name)
            st.session_state.last_handled_upload_token = upload_token
        except Exception as exc:
            st.error(f"Could not load dataset: {exc}")
            st.stop()

if "working_df" not in st.session_state:
    st.info("Upload a CSV/Excel file above or click **Load Sample Sales Dataset** in the sidebar to begin.")
    st.stop()

df = st.session_state.working_df
if df.shape[0] > 250_000:
    st.warning("Dataset exceeds 250,000 rows; computations may run slower on limited-memory machines.")

workflow_key = f"{st.session_state.get('current_upload')}:{st.session_state.get('workflow_version', 0)}:{len(df)}:{len(df.columns)}"
if st.session_state.get("workflow_key") != workflow_key:
    with st.spinner("Running 5-agent LangGraph analysis workflow..."):
        st.session_state.workflow_state = run_analysis_workflow(df)
        st.session_state.workflow_key = workflow_key

with st.sidebar:
    if st.button("Re-run Multi-Agent Workflow", use_container_width=True):
        st.session_state.workflow_version = st.session_state.get("workflow_version", 0) + 1
        st.rerun()

wf = st.session_state.workflow_state
profile = wf.get("profile", {})
iso_wf = wf.get("isolation_forest_result", {})

# Top Responsive KPI Row
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    _render_kpi("Dataset Size", f"{profile.get('rows', 0):,}", f"{profile.get('columns', 0)} columns")
with k2:
    _render_kpi("Data Quality", f"{profile.get('quality_score', 0)}/100", f"{profile.get('duplicate_rows', 0):,} duplicates", violet=True)
with k3:
    _render_kpi("Missing Values", f"{profile.get('missing_cells', 0):,}", f"{profile.get('missing_pct', 0.0)}% of cells")
with k4:
    anom_cnt = iso_wf.get("count", 0) if iso_wf.get("status") == "completed" else 0
    _render_kpi("Anomalies Flagged", f"{anom_cnt:,}", "Isolation Forest", violet=True)
with k5:
    wf_status = str(wf.get("status", "completed")).replace("_", " ").title()
    _render_kpi("Analysis Status", wf_status, f"Source: {st.session_state.get('current_upload', 'upload')}")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Overview",
    "Data Cleaning",
    "Analytics and Anomalies",
    "Forecasting",
    "Ask with SQL",
    "Insights and Reports",
])

with tab1:
    st.subheader("Dataset Overview & Schema Profile")
    if profile.get("unusual_patterns"):
        with st.expander("Unusual Patterns Detected by Data Profiler", expanded=True):
            for pat in profile["unusual_patterns"]:
                st.write(f"- {pat}")

    st.markdown("#### Data Preview (Top 100 Rows)")
    st.dataframe(df.head(100), use_container_width=True)

    c_left, c_right = st.columns(2)
    with c_left:
        st.markdown("#### Column Profile")
        col_names = profile.get("column_names", [])
        sem_map = profile.get("semantic_types", {})
        type_df = pd.DataFrame({
            "column": col_names,
            "data_type": [profile["data_types"][c] for c in col_names],
            "semantic_type": [sem_map.get(c, {}).get("semantic_type", "unknown") for c in col_names],
            "missing": [profile["missing_values"][c] for c in col_names],
            "unique_values": [profile.get("cardinality", {}).get(c, 0) for c in col_names],
        })
        st.dataframe(type_df, use_container_width=True)
    with c_right:
        st.markdown("#### Numeric Summary (with 95% CI)")
        summary = wf.get("numeric_summary", pd.DataFrame())
        if isinstance(summary, pd.DataFrame) and not summary.empty:
            st.dataframe(summary, use_container_width=True)
        else:
            st.info("No numeric columns detected in this dataset.")

with tab2:
    st.subheader("User-Approved Data Cleaning")
    suggestions = cleaning_suggestions(df)
    if suggestions:
        sugg_df = pd.DataFrame(suggestions)
        if "confidence" in sugg_df.columns:
            sugg_df["confidence"] = sugg_df["confidence"].map(lambda v: f"{float(v) * 100:.0f}%")
        st.dataframe(sugg_df, use_container_width=True)
    else:
        st.success("No structural cleaning issues detected by automated checks.")
    st.caption("Review each recommendation's confidence and explanation before applying; original uploaded data is never modified.")

    col_a, col_b = st.columns(2)
    with col_a:
        trim = st.checkbox("Trim leading/trailing whitespace in text values")
        fill = st.checkbox("Fill missing numeric values with column median")
    with col_b:
        drop = st.checkbox("Remove duplicate rows")
        parse_dt = st.checkbox("Parse detected datetime text columns")

    if st.button("Apply Selected Cleaning", type="primary"):
        cleaned, log = clean_dataframe(
            df,
            trim_text=trim,
            fill_numeric_median=fill,
            drop_duplicates=drop,
            parse_dates=parse_dt,
        )
        st.session_state.working_df = cleaned
        st.session_state.cleaning_log.extend(log)
        st.session_state.workflow_version = st.session_state.get("workflow_version", 0) + 1
        st.rerun()

    if st.session_state.cleaning_log:
        st.markdown("#### Transformation Log")
        for item in st.session_state.cleaning_log:
            st.write(f"- {item}")

    exp1, exp2 = st.columns(2)
    with exp1:
        st.download_button(
            "Download Cleaned Dataset (CSV)",
            data=dataframe_to_safe_csv_bytes(df),
            file_name="datamind_cleaned.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with exp2:
        if st.button("Reset to Original Upload", use_container_width=True):
            st.session_state.working_df = st.session_state.original_df.copy()
            st.session_state.cleaning_log = []
            st.session_state.workflow_version = st.session_state.get("workflow_version", 0) + 1
            st.rerun()

with tab3:
    st.subheader("Statistical Analysis, Visualizations & Anomaly Detection")
    numeric_cols = list(df.select_dtypes(include="number").columns)
    cat_cols = [
        str(c) for c in df.columns
        if str(c) not in numeric_cols and 2 <= df[c].nunique(dropna=True) <= 25
    ]
    if not numeric_cols:
        st.info("No numeric columns are available for statistical analysis.")
    else:
        selected = st.selectbox("Inspect numeric column (IQR & Distribution)", numeric_cols)
        out = iqr_outliers(df, selected)
        st.metric("Potential IQR Outliers", out["count"])
        if out["count"]:
            st.write(f"IQR bounds: {out['lower_bound']:.3f} to {out['upper_bound']:.3f}")
            st.dataframe(pd.DataFrame(out["rows"]), use_container_width=True)
        else:
            st.info("No IQR outliers detected for this column.")
        st.caption(out.get("disclaimer", ""))

        v1, v2 = st.columns(2)
        with v1:
            st.plotly_chart(make_chart(df, selected), use_container_width=True)
        with v2:
            box_fig, box_exp = make_box_chart(df, selected, cat_cols[0] if cat_cols else None)
            st.plotly_chart(box_fig, use_container_width=True)
            st.caption(box_exp)

        if len(numeric_cols) >= 2:
            sc_fig, sc_exp = make_scatter_chart(
                df, numeric_cols[0], numeric_cols[1], cat_cols[0] if cat_cols else None
            )
            st.plotly_chart(sc_fig, use_container_width=True)
            st.caption(sc_exp)

        st.markdown("#### Multivariate Isolation Forest Anomaly Detection")
        contam_mode = st.radio("Contamination mode", ["auto", "custom"], horizontal=True)
        contam_val: str | float = "auto"
        if contam_mode == "custom":
            contam_val = st.slider("Expected anomaly fraction", min_value=0.01, max_value=0.30, value=0.05, step=0.01)
        iso = isolation_forest_outliers(df, contamination=contam_val)
        if iso.get("status") == "completed":
            st.metric("Flagged Anomalous Records", f"{iso['count']} ({iso['rate'] * 100:.2f}%)")
            st.caption(iso.get("disclaimer", ""))
            anom_fig, anom_exp = make_anomaly_chart(df, iso)
            if anom_fig is not None:
                st.plotly_chart(anom_fig, use_container_width=True)
                st.caption(anom_exp)
            if iso.get("rows"):
                anom_df = pd.DataFrame(iso["rows"])
                st.dataframe(anom_df, use_container_width=True)
                st.download_button(
                    "Export Flagged Anomalies (CSV)",
                    data=dataframe_to_safe_csv_bytes(anom_df),
                    file_name="datamind_anomalies.csv",
                    mime="text/csv",
                )
        else:
            st.info(iso.get("reason", "Isolation Forest skipped."))

        st.markdown("#### Correlation & Hypothesis Tests")
        hm_fig, hm_exp = make_correlation_heatmap(df)
        if hm_fig is not None:
            st.plotly_chart(hm_fig, use_container_width=True)
            st.caption(hm_exp)

        corr_pairs = wf.get("correlation_result", {}).get("pairs", [])
        if corr_pairs:
            st.dataframe(pd.DataFrame(corr_pairs), use_container_width=True)

        stat_tests = wf.get("statistical_tests", {})
        if stat_tests.get("normality"):
            with st.expander("Normality Tests (Shapiro-Wilk / D'Agostino-Pearson)", expanded=False):
                st.dataframe(pd.DataFrame(stat_tests["normality"]), use_container_width=True)
        if stat_tests.get("group_comparisons"):
            with st.expander("Group Comparison Tests (t-test / Mann-Whitney / ANOVA / Kruskal)", expanded=True):
                st.dataframe(pd.DataFrame(stat_tests["group_comparisons"]), use_container_width=True)
        if stat_tests.get("categorical_associations"):
            with st.expander("Categorical Independence (Chi-square)", expanded=False):
                st.dataframe(pd.DataFrame(stat_tests["categorical_associations"]), use_container_width=True)

with tab4:
    st.subheader("Chronological Time-Series Forecasting")
    datetime_cols = profile.get("datetime_columns", [])
    if not datetime_cols or not numeric_cols:
        st.info("Forecasting requires at least one datetime column and one numeric column.")
    else:
        fc1, fc2, fc3 = st.columns(3)
        f_date = fc1.selectbox("Date column", datetime_cols, key="fc_date")
        f_target = fc2.selectbox("Target metric", numeric_cols, key="fc_target")
        f_horizon = fc3.slider("Forecast horizon (periods)", min_value=2, max_value=12, value=6, key="fc_horizon")

        ts_fig, ts_exp = make_timeseries_chart(df, f_date, f_target)
        st.plotly_chart(ts_fig, use_container_width=True)
        st.caption(ts_exp)

        fcst = forecast_series(df, f_date, f_target, horizon=f_horizon)
        if fcst.get("status") == "completed":
            st.success(
                f"Selected Model: **{fcst['selected_model']}** · "
                f"Holdout Periods: {fcst['test_periods']} · "
                f"Beats Baselines: {fcst['beats_baseline']}"
            )
            eval_df = pd.DataFrame(fcst["evaluations"]).T.reset_index().rename(columns={"index": "model"})
            st.markdown("#### Chronological Holdout Evaluation")
            st.dataframe(eval_df, use_container_width=True)

            fc_fig, fc_exp = make_forecast_chart(fcst)
            st.plotly_chart(fc_fig, use_container_width=True)
            st.caption(fc_exp)

            fc_table = pd.DataFrame(fcst["forecast"])
            st.markdown("#### Forecast Values")
            st.dataframe(fc_table, use_container_width=True)
            st.download_button(
                "Export Forecast Results (CSV)",
                data=dataframe_to_safe_csv_bytes(fc_table),
                file_name="datamind_forecast.csv",
                mime="text/csv",
            )
            for lim in fcst.get("limitations", []):
                st.caption(f"• {lim}")
        else:
            st.warning(fcst.get("reason", "Forecasting skipped due to insufficient history."))

with tab5:
    st.subheader("Ask a Question with Read-Only DuckDB SQL")
    st.write("Schema-aware NL-to-SQL with filtering, grouping, sorting, pre-execution validation, and optional local Ollama.")
    question = st.text_input(
        "Your question",
        placeholder="e.g. top 5 products by revenue in North",
        key="sql_question",
    )
    if st.button("Generate and Run SQL", type="primary"):
        try:
            with st.spinner("Generating and validating SQL..."):
                res = generate_and_validate_sql(
                    question,
                    df,
                    use_ollama=use_ollama,
                    ollama_model=ollama_model,
                )
                sql = res["sql"]
                result = execute_readonly_query(df, sql)
            st.session_state.last_sql = sql
            st.session_state.last_sql_source = res["source"]
            st.session_state.last_sql_result = result
        except Exception as exc:
            st.error(str(exc))

    if st.session_state.get("last_sql") and st.session_state.get("last_sql_result") is not None:
        st.caption(f"Query source: `{st.session_state.get('last_sql_source', 'deterministic')}`")
        st.code(st.session_state.last_sql, language="sql")
        st.dataframe(st.session_state.last_sql_result, use_container_width=True)
        st.download_button(
            "Download Query Results (CSV)",
            data=dataframe_to_safe_csv_bytes(st.session_state.last_sql_result),
            file_name="datamind_query_results.csv",
            mime="text/csv",
        )

with tab6:
    st.subheader("Multi-Agent Execution Timeline, Reviewer Findings & PDF Report")
    timeline = wf.get("timeline", [])
    if timeline:
        st.markdown("#### Multi-Agent Execution Timeline")
        for ev in timeline:
            status = ev.get("status", "completed")
            badge_cls = f"dm-badge dm-badge-{status}"
            st.markdown(
                f"**{ev['agent']}** · {ev['task']} &nbsp; "
                f"<span class='{badge_cls}'>{status}</span> &nbsp; "
                f"`{ev['duration_ms']} ms` — {ev['summary']}",
                unsafe_allow_html=True,
            )

    if wf.get("errors"):
        st.markdown("#### Workflow Warnings / Errors")
        for err in wf["errors"]:
            st.error(err)

    review = wf.get("review_result", {})
    st.markdown("#### Insight Reviewer Verification")
    approved_claims: list[str] = []
    for item in review.get("approved", []):
        approved_claims.append(item["claim"])
        st.success(f"✓ {item['claim']} (Evidence: {item.get('evidence')})")
    rejected_list = review.get("rejected", [])
    if rejected_list:
        for rej in rejected_list:
            st.warning(f"✗ Rejected: {rej.get('reason')}")
    else:
        st.info("No findings were rejected by the Insight Reviewer.")

    st.divider()
    st.markdown("#### Export Full Analysis Report (PDF)")
    report_question = st.text_input("Question or analysis goal (optional)", key="report_question")
    chart_titles = [c.get("title", "") for c in wf.get("charts", []) if c.get("title")]
    pdf = make_pdf_report(
        profile=profile,
        insights=approved_claims,
        question=report_question,
        cleaning_log=st.session_state.cleaning_log,
        stat_tests=wf.get("statistical_tests"),
        iqr_result=wf.get("iqr_result"),
        iso_result=wf.get("isolation_forest_result"),
        forecast_result=wf.get("forecast_result"),
        rejected_findings=rejected_list,
        chart_titles=chart_titles,
    )
    st.download_button(
        "Download Full PDF Report",
        data=pdf,
        file_name="datamind_analysis_report.pdf",
        mime="application/pdf",
        type="primary",
    )
