"""Typed shared state definitions for the DataMind AI LangGraph workflow."""

from typing import Any, TypedDict

import pandas as pd


class TimelineEvent(TypedDict):
    """Execution record for a single agent or workflow step."""

    agent: str
    task: str
    status: str  # completed | skipped | failed | needs_review
    duration_ms: float
    summary: str


class AnalystState(TypedDict, total=False):
    """Shared typed state passed across all nodes in the LangGraph workflow."""

    # Inputs
    df: pd.DataFrame
    question: str
    selected_column: str | None
    date_col: str | None
    target_col: str | None
    horizon: int
    contamination: float | str
    use_ollama: bool
    ollama_model: str
    ollama_url: str
    max_retries: int
    injected_findings: list[dict[str, Any]]

    # Validation & Profiling
    validation_ok: bool
    profile: dict[str, Any]
    cleaning_suggestions: list[dict[str, Any]]

    # Planning
    analysis_plan: dict[str, Any]

    # SQL Agent outputs
    sql_query: str | None
    sql_source: str | None
    sql_result: pd.DataFrame | None
    sql_error: str | None

    # Statistical Analysis Agent outputs
    numeric_summary: pd.DataFrame
    iqr_result: dict[str, Any]
    isolation_forest_result: dict[str, Any]
    correlation_result: dict[str, Any]
    statistical_tests: dict[str, Any]
    forecast_result: dict[str, Any]

    # Visualization Agent outputs
    chart_recommendations: list[dict[str, Any]]
    charts: list[dict[str, Any]]

    # Insight Reviewer Agent outputs
    candidate_findings: list[dict[str, Any]]
    review_result: dict[str, Any]
    review_status: str
    retry_count: int
    reviewer_feedback: list[str]

    # Final Report & Workflow Metadata
    pdf_bytes: bytes
    status: str
    errors: list[str]
    timeline: list[TimelineEvent]
