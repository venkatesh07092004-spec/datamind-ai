"""LangGraph multi-agent orchestration for DataMind AI."""

import time
from typing import Any

import pandas as pd
from langgraph.graph import END, StateGraph

from agents.profiler import cleaning_suggestions, profile_dataset
from agents.reviewer import build_verified_findings, review_findings
from agents.sql_agent import generate_and_validate_sql
from agents.statistician import (
    correlation_analysis,
    forecast_series,
    iqr_outliers,
    isolation_forest_outliers,
    numeric_summary,
    run_statistical_tests,
)
from agents.visualization import (
    make_chart,
    make_correlation_heatmap,
    make_forecast_chart,
    make_timeseries_chart,
    recommend_charts,
)
from core.config import DEFAULT_OLLAMA_MODEL, DEFAULT_OLLAMA_URL
from core.ingestion import validate_dataframe
from core.logging_config import get_logger
from core.reporting import make_pdf_report
from core.sql_engine import execute_readonly_query
from orchestration.state import AnalystState, TimelineEvent

logger = get_logger(__name__)


def _append_timeline(
    state: AnalystState,
    agent: str,
    task: str,
    status: str,
    start_ts: float,
    summary: str,
) -> list[TimelineEvent]:
    events = list(state.get("timeline", []))
    events.append({
        "agent": agent,
        "task": task,
        "status": status,
        "duration_ms": round((time.perf_counter() - start_ts) * 1000.0, 2),
        "summary": summary,
    })
    return events


def _append_error(state: AnalystState, message: str) -> list[str]:
    errs = list(state.get("errors", []))
    errs.append(message)
    return errs


def validate_dataset_node(state: AnalystState) -> dict[str, Any]:
    """Validate input DataFrame before any agent runs."""
    t0 = time.perf_counter()
    df = state.get("df")
    try:
        if df is None or not isinstance(df, pd.DataFrame):
            raise ValueError("No valid DataFrame supplied to workflow.")
        validate_dataframe(df)
        return {
            "validation_ok": True,
            "errors": list(state.get("errors", [])),
            "retry_count": int(state.get("retry_count", 0)),
            "max_retries": int(state.get("max_retries", 1)),
            "timeline": _append_timeline(
                state, "Ingestion Validator", "Validate dataset", "completed", t0,
                f"Validated {df.shape[0]:,} rows x {df.shape[1]} columns.",
            ),
        }
    except Exception as exc:
        msg = f"Dataset validation failed: {exc}"
        logger.warning(msg)
        return {
            "validation_ok": False,
            "status": "failed",
            "errors": _append_error(state, msg),
            "timeline": _append_timeline(
                state, "Ingestion Validator", "Validate dataset", "failed", t0, msg
            ),
        }


def profile_data_node(state: AnalystState) -> dict[str, Any]:
    """Run the Data Profiler Agent."""
    t0 = time.perf_counter()
    df = state["df"]
    try:
        profile = profile_dataset(df)
        suggestions = cleaning_suggestions(df)
        return {
            "profile": profile,
            "cleaning_suggestions": suggestions,
            "timeline": _append_timeline(
                state, "Data Profiler", "Profile dataset & quality", "completed", t0,
                f"Quality {profile['quality_score']}/100; {len(suggestions)} cleaning suggestion(s).",
            ),
        }
    except Exception as exc:
        msg = f"Data Profiler failed: {exc}"
        logger.exception(msg)
        return {
            "profile": {},
            "cleaning_suggestions": [],
            "errors": _append_error(state, msg),
            "timeline": _append_timeline(
                state, "Data Profiler", "Profile dataset & quality", "failed", t0, msg
            ),
        }


def plan_analysis_node(state: AnalystState) -> dict[str, Any]:
    """Build deterministic analysis plan from profile and user inputs."""
    t0 = time.perf_counter()
    profile = state.get("profile", {})
    numeric_cols = profile.get("numeric_columns", [])
    datetime_cols = profile.get("datetime_columns", [])
    question = str(state.get("question", "")).strip()

    selected_col = state.get("selected_column")
    if not selected_col or selected_col not in profile.get("column_names", []):
        selected_col = numeric_cols[0] if numeric_cols else (
            profile.get("column_names", [None])[0]
        )

    date_col = state.get("date_col")
    if not date_col or date_col not in datetime_cols:
        date_col = datetime_cols[0] if datetime_cols else None

    target_col = state.get("target_col")
    if not target_col or target_col not in numeric_cols:
        target_col = numeric_cols[0] if numeric_cols else None

    plan = {
        "run_sql": bool(question),
        "run_stats": bool(numeric_cols),
        "run_forecast": bool(date_col and target_col),
        "selected_column": selected_col,
        "date_col": date_col,
        "target_col": target_col,
    }
    return {
        "analysis_plan": plan,
        "timeline": _append_timeline(
            state, "Orchestrator Planner", "Plan analytical workflow", "completed", t0,
            f"SQL={plan['run_sql']}, Stats={plan['run_stats']}, Forecast={plan['run_forecast']}.",
        ),
    }


def sql_analysis_node(state: AnalystState) -> dict[str, Any]:
    """Run the SQL Agent for natural-language queries against read-only DuckDB."""
    t0 = time.perf_counter()
    df = state["df"]
    question = str(state.get("question", "")).strip()
    if not question:
        return {
            "sql_query": None,
            "sql_source": None,
            "sql_result": None,
            "sql_error": None,
            "timeline": _append_timeline(
                state, "SQL Agent", "Natural-language SQL", "skipped", t0, "No question provided."
            ),
        }

    try:
        res = generate_and_validate_sql(
            question,
            df,
            use_ollama=bool(state.get("use_ollama", False)),
            ollama_model=str(state.get("ollama_model", DEFAULT_OLLAMA_MODEL)),
            ollama_url=str(state.get("ollama_url", DEFAULT_OLLAMA_URL)),
        )
        result_df = execute_readonly_query(df, res["sql"])
        return {
            "sql_query": res["sql"],
            "sql_source": res["source"],
            "sql_result": result_df,
            "sql_error": None,
            "timeline": _append_timeline(
                state, "SQL Agent", "Generate & execute SQL", "completed", t0,
                f"Executed ({res['source']}) -> {len(result_df)} row(s).",
            ),
        }
    except Exception as exc:
        msg = f"SQL Agent error: {exc}"
        logger.warning(msg)
        return {
            "sql_query": None,
            "sql_source": None,
            "sql_result": None,
            "sql_error": str(exc),
            "errors": _append_error(state, msg),
            "timeline": _append_timeline(
                state, "SQL Agent", "Generate & execute SQL", "failed", t0, str(exc)
            ),
        }


def statistical_analysis_node(state: AnalystState) -> dict[str, Any]:
    """Run the Statistical Analysis Agent (summary, IQR, Isolation Forest, correlations, tests, forecast)."""
    t0 = time.perf_counter()
    df = state["df"]
    plan = state.get("analysis_plan", {})
    selected_col = plan.get("selected_column")
    date_col = plan.get("date_col")
    target_col = plan.get("target_col")

    try:
        num_sum = numeric_summary(df)
        iqr_res = (
            iqr_outliers(df, selected_col)
            if selected_col and selected_col in df.columns
            else {}
        )
        iso_res = isolation_forest_outliers(
            df,
            contamination=state.get("contamination", "auto"),
        )
        corr_res = correlation_analysis(df)
        stat_tests = run_statistical_tests(df)
        fcst_res = (
            forecast_series(df, date_col, target_col, horizon=int(state.get("horizon", 6)))
            if plan.get("run_forecast") and date_col and target_col
            else {"status": "skipped", "reason": "Dataset has no datetime + numeric pair."}
        )
        return {
            "numeric_summary": num_sum,
            "iqr_result": iqr_res,
            "isolation_forest_result": iso_res,
            "correlation_result": corr_res,
            "statistical_tests": stat_tests,
            "forecast_result": fcst_res,
            "timeline": _append_timeline(
                state, "Statistical Analyst", "Stats, anomalies & forecast", "completed", t0,
                f"IQR outliers={iqr_res.get('count', 0)}, IsolationForest={iso_res.get('count', 0)}, "
                f"Forecast={fcst_res.get('status', 'skipped')}.",
            ),
        }
    except Exception as exc:
        msg = f"Statistical Analyst failed: {exc}"
        logger.exception(msg)
        return {
            "errors": _append_error(state, msg),
            "timeline": _append_timeline(
                state, "Statistical Analyst", "Stats, anomalies & forecast", "failed", t0, msg
            ),
        }


def visualization_node(state: AnalystState) -> dict[str, Any]:
    """Run the Visualization Agent to recommend and build interactive Plotly charts."""
    t0 = time.perf_counter()
    df = state["df"]
    profile = state.get("profile", {})
    plan = state.get("analysis_plan", {})
    fcst_res = state.get("forecast_result", {})

    try:
        recs = recommend_charts(df, profile=profile)
        charts: list[dict[str, Any]] = []

        selected_col = plan.get("selected_column")
        if selected_col and selected_col in df.columns:
            charts.append({
                "title": f"Distribution of {selected_col}",
                "chart_type": "distribution",
                "figure": make_chart(df, selected_col),
                "explanation": f"Shows distribution/frequencies of '{selected_col}'.",
            })

        hm_fig, hm_exp = make_correlation_heatmap(df)
        if hm_fig is not None:
            charts.append({
                "title": "Numeric Correlation Matrix",
                "chart_type": "heatmap",
                "figure": hm_fig,
                "explanation": hm_exp,
            })

        date_col = plan.get("date_col")
        target_col = plan.get("target_col")
        if date_col and target_col:
            ts_fig, ts_exp = make_timeseries_chart(df, date_col, target_col)
            charts.append({
                "title": f"{target_col} Over Time",
                "chart_type": "timeseries",
                "figure": ts_fig,
                "explanation": ts_exp,
            })

        if fcst_res and fcst_res.get("status") == "completed":
            fc_fig, fc_exp = make_forecast_chart(fcst_res)
            charts.append({
                "title": f"Forecast: {target_col}",
                "chart_type": "forecast",
                "figure": fc_fig,
                "explanation": fc_exp,
            })

        return {
            "chart_recommendations": recs,
            "charts": charts,
            "timeline": _append_timeline(
                state, "Visualization Agent", "Recommend & render charts", "completed", t0,
                f"Generated {len(charts)} interactive chart(s) ({len(recs)} recommended).",
            ),
        }
    except Exception as exc:
        msg = f"Visualization Agent failed: {exc}"
        logger.exception(msg)
        return {
            "chart_recommendations": [],
            "charts": [],
            "errors": _append_error(state, msg),
            "timeline": _append_timeline(
                state, "Visualization Agent", "Recommend & render charts", "failed", t0, msg
            ),
        }


def insight_review_node(state: AnalystState) -> dict[str, Any]:
    """Run the Insight Reviewer Agent to verify claims against computed evidence."""
    t0 = time.perf_counter()
    df = state["df"]
    profile = state.get("profile", {})
    iqr_res = state.get("iqr_result", {})
    iso_res = state.get("isolation_forest_result", {})
    corr_res = state.get("correlation_result", {})
    stat_tests = state.get("statistical_tests", {})
    fcst_res = state.get("forecast_result", {})

    try:
        candidates = state.get("candidate_findings")
        if candidates is None:
            # First pass: generate evidence-backed findings + include any injected findings for verification
            base_review = build_verified_findings(
                df,
                profile=profile,
                iqr_summary=iqr_res,
                iso_summary=iso_res,
                corr_summary=corr_res,
                stat_tests=stat_tests,
                forecast_summary=fcst_res,
            )
            candidates = list(base_review["approved"]) + list(state.get("injected_findings", []))

        context = {
            "profile": profile,
            "iqr": iqr_res,
            "isolation_forest": iso_res,
            "correlation": corr_res,
            "forecast": fcst_res,
        }
        review = review_findings(df, candidates, computed_context=context)
        feedback = [item["reason"] for item in review["rejected"]]
        status_label = "completed" if not review["rejected"] else "needs_review"

        return {
            "candidate_findings": candidates,
            "review_result": review,
            "review_status": review["status"],
            "reviewer_feedback": feedback,
            "timeline": _append_timeline(
                state, "Insight Reviewer", "Verify evidence & claims", status_label, t0,
                f"Approved {len(review['approved'])}, Rejected {len(review['rejected'])}.",
            ),
        }
    except Exception as exc:
        msg = f"Insight Reviewer failed: {exc}"
        logger.exception(msg)
        return {
            "review_result": {"approved": [], "rejected": [], "status": "rejected", "note": msg},
            "review_status": "rejected",
            "errors": _append_error(state, msg),
            "timeline": _append_timeline(
                state, "Insight Reviewer", "Verify evidence & claims", "failed", t0, msg
            ),
        }


def revise_findings_node(state: AnalystState) -> dict[str, Any]:
    """
    Correct or drop unsupported findings when the reviewer requests revisions.
    Rebuilds strictly from computed evidence so the feedback loop terminates cleanly.
    """
    t0 = time.perf_counter()
    retry_count = int(state.get("retry_count", 0)) + 1
    df = state["df"]
    profile = state.get("profile", {})

    verified = build_verified_findings(
        df,
        profile=profile,
        iqr_summary=state.get("iqr_result"),
        iso_summary=state.get("isolation_forest_result"),
        corr_summary=state.get("correlation_result"),
        stat_tests=state.get("statistical_tests"),
        forecast_summary=state.get("forecast_result"),
    )
    return {
        "candidate_findings": verified["approved"],
        "retry_count": retry_count,
        "timeline": _append_timeline(
            state, "Correction Loop", "Revise rejected findings", "completed", t0,
            f"Retry #{retry_count}: retained {len(verified['approved'])} strictly verified finding(s).",
        ),
    }


def final_report_node(state: AnalystState) -> dict[str, Any]:
    """Assemble final PDF report and workflow status."""
    t0 = time.perf_counter()
    if not state.get("validation_ok", False):
        return {
            "pdf_bytes": b"",
            "status": "failed",
            "timeline": _append_timeline(
                state, "Report Builder", "Assemble final report", "skipped", t0,
                "Skipped because dataset validation failed.",
            ),
        }

    profile = state.get("profile", {})
    review = state.get("review_result", {})
    approved = review.get("approved", [])
    rejected = review.get("rejected", [])
    insights = [item["claim"] for item in approved if isinstance(item, dict) and "claim" in item]
    chart_titles = [c.get("title", "") for c in state.get("charts", []) if c.get("title")]
    question = str(state.get("question", "")).strip()

    try:
        pdf_bytes = make_pdf_report(
            profile=profile,
            insights=insights,
            question=question,
            stat_tests=state.get("statistical_tests"),
            iqr_result=state.get("iqr_result"),
            iso_result=state.get("isolation_forest_result"),
            forecast_result=state.get("forecast_result"),
            rejected_findings=rejected,
            chart_titles=chart_titles,
        )
        final_status = "completed_with_warnings" if state.get("errors") else "completed"
        return {
            "pdf_bytes": pdf_bytes,
            "status": final_status,
            "timeline": _append_timeline(
                state, "Report Builder", "Assemble final PDF report", "completed", t0,
                f"Generated PDF ({len(pdf_bytes):,} bytes) with {len(insights)} verified insight(s).",
            ),
        }
    except Exception as exc:
        msg = f"Report generation failed: {exc}"
        logger.exception(msg)
        return {
            "pdf_bytes": b"",
            "status": "completed_with_warnings",
            "errors": _append_error(state, msg),
            "timeline": _append_timeline(
                state, "Report Builder", "Assemble final PDF report", "failed", t0, msg
            ),
        }


def _route_after_validation(state: AnalystState) -> str:
    return "profile_data" if state.get("validation_ok", False) else "final_report"


def _route_after_plan(state: AnalystState) -> str:
    plan = state.get("analysis_plan", {})
    return "sql_analysis" if plan.get("run_sql", False) else "statistical_analysis"


def _route_after_review(state: AnalystState) -> str:
    review = state.get("review_result", {})
    rejected = review.get("rejected", [])
    retry_count = int(state.get("retry_count", 0))
    max_retries = int(state.get("max_retries", 1))
    if rejected and retry_count < max_retries:
        return "revise_findings"
    return "final_report"


def build_analysis_graph():
    """Compile and return the LangGraph multi-agent StateGraph."""
    graph = StateGraph(AnalystState)

    graph.add_node("validate_dataset", validate_dataset_node)
    graph.add_node("profile_data", profile_data_node)
    graph.add_node("plan_analysis", plan_analysis_node)
    graph.add_node("sql_analysis", sql_analysis_node)
    graph.add_node("statistical_analysis", statistical_analysis_node)
    graph.add_node("visualization", visualization_node)
    graph.add_node("insight_review", insight_review_node)
    graph.add_node("revise_findings", revise_findings_node)
    graph.add_node("final_report", final_report_node)

    graph.set_entry_point("validate_dataset")
    graph.add_conditional_edges(
        "validate_dataset",
        _route_after_validation,
        {"profile_data": "profile_data", "final_report": "final_report"},
    )
    graph.add_edge("profile_data", "plan_analysis")
    graph.add_conditional_edges(
        "plan_analysis",
        _route_after_plan,
        {"sql_analysis": "sql_analysis", "statistical_analysis": "statistical_analysis"},
    )
    graph.add_edge("sql_analysis", "statistical_analysis")
    graph.add_edge("statistical_analysis", "visualization")
    graph.add_edge("visualization", "insight_review")
    graph.add_conditional_edges(
        "insight_review",
        _route_after_review,
        {"revise_findings": "revise_findings", "final_report": "final_report"},
    )
    graph.add_edge("revise_findings", "insight_review")
    graph.add_edge("final_report", END)

    return graph.compile()


def run_analysis_workflow(
    df: pd.DataFrame,
    question: str = "",
    selected_column: str | None = None,
    date_col: str | None = None,
    target_col: str | None = None,
    horizon: int = 6,
    contamination: float | str = "auto",
    use_ollama: bool = False,
    ollama_model: str = DEFAULT_OLLAMA_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    max_retries: int = 1,
    injected_findings: list[dict[str, Any]] | None = None,
) -> AnalystState:
    """Execute the full LangGraph multi-agent workflow and return the final AnalystState."""
    app = build_analysis_graph()
    initial_state: AnalystState = {
        "df": df,
        "question": question,
        "selected_column": selected_column,
        "date_col": date_col,
        "target_col": target_col,
        "horizon": horizon,
        "contamination": contamination,
        "use_ollama": use_ollama,
        "ollama_model": ollama_model,
        "ollama_url": ollama_url,
        "max_retries": max_retries,
        "retry_count": 0,
        "errors": [],
        "timeline": [],
    }
    if injected_findings is not None:
        initial_state["injected_findings"] = injected_findings

    return app.invoke(initial_state)
