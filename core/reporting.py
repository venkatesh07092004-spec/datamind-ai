"""Professional PDF report builder using ReportLab."""

from datetime import datetime
from io import BytesIO
from typing import Any
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def _styled_table(data: list[list[str]], col_widths: list[int]) -> Table:
    table = Table(data, colWidths=col_widths, repeatRows=1)
    table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E293B")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#F8FAFC")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
            ("PADDING", (0, 0), (-1, -1), 6),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ])
    )
    return table


def make_pdf_report(
    profile: dict[str, Any],
    insights: list[str],
    question: str = "",
    cleaning_log: list[str] | None = None,
    stat_tests: dict[str, Any] | None = None,
    iqr_result: dict[str, Any] | None = None,
    iso_result: dict[str, Any] | None = None,
    forecast_result: dict[str, Any] | None = None,
    rejected_findings: list[dict[str, Any]] | None = None,
    chart_titles: list[str] | None = None,
) -> bytes:
    """
    Generate a comprehensive, evidence-based PDF analysis report.
    Preserves backward compatibility with (profile, insights, question) callers.
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=42,
        leftMargin=42,
        topMargin=42,
        bottomMargin=42,
    )
    styles = getSampleStyleSheet()
    subtitle_style = ParagraphStyle(
        "SubtitleMuted",
        parent=styles["Normal"],
        textColor=colors.HexColor("#475569"),
        fontSize=9.5,
    )
    body = styles["BodyText"]

    story: list[Any] = [
        Paragraph("DataMind AI — Autonomous Multi-Agent Analysis Report", styles["Title"]),
        Paragraph(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} · Local-First Evidence Report",
            subtitle_style,
        ),
        Spacer(1, 12),
    ]

    # 1. Executive Summary
    story.append(Paragraph("1. Executive Summary", styles["Heading2"]))
    if question:
        story.append(Paragraph(f"<b>Analysis Goal / Question:</b> {escape(str(question))}", body))
    story.append(
        Paragraph(
            escape(
                str(
                    profile.get(
                        "summary_text",
                        f"Analyzed dataset with {profile.get('rows', 'N/A')} rows and {profile.get('columns', 'N/A')} columns.",
                    )
                )
            ),
            body,
        )
    )
    story.append(Spacer(1, 10))

    # 2. Dataset Overview & Data Quality
    story.append(Paragraph("2. Dataset Overview & Data Quality", styles["Heading2"]))
    overview = [
        ["Metric", "Value"],
        ["Rows", str(profile.get("rows", "N/A"))],
        ["Columns", str(profile.get("columns", "N/A"))],
        ["Missing cells", f"{profile.get('missing_cells', 'N/A')} ({profile.get('missing_pct', 0.0)}%)"],
        ["Duplicate rows", str(profile.get("duplicate_rows", "N/A"))],
        ["Data quality score", f"{profile.get('quality_score', 'N/A')} / 100"],
        ["Memory footprint (MB)", str(profile.get("memory_mb", "N/A"))],
    ]
    story.extend([_styled_table(overview, [200, 310]), Spacer(1, 10)])

    if profile.get("unusual_patterns"):
        story.append(Paragraph("<b>Unusual Patterns Flagged:</b>", body))
        for pat in profile["unusual_patterns"]:
            story.append(Paragraph(f"• {escape(str(pat))}", body))
        story.append(Spacer(1, 8))

    # 3. Cleaning Operations Performed
    story.append(Paragraph("3. Cleaning Operations Performed", styles["Heading2"]))
    if cleaning_log:
        for item in cleaning_log:
            story.append(Paragraph(f"• {escape(str(item))}", body))
    else:
        story.append(Paragraph("No manual cleaning transformations were applied to the active dataset.", body))
    story.append(Spacer(1, 10))

    # 4. Statistical Analysis & Hypothesis Tests
    story.append(Paragraph("4. Statistical Analysis & Hypothesis Tests", styles["Heading2"]))
    group_tests = (stat_tests or {}).get("group_comparisons", [])
    if group_tests:
        gt_rows = [["Test", "Variables", "Statistic", "p-value", "Significant (p<0.05)"]]
        for gt in group_tests[:8]:
            gt_rows.append([
                str(gt.get("test", "")),
                f"{gt.get('numeric', '')} by {gt.get('categorical', '')}",
                str(gt.get("statistic", "")),
                str(gt.get("p_value", "")),
                "Yes" if gt.get("significant_at_0_05") else "No",
            ])
        story.extend([_styled_table(gt_rows, [110, 150, 75, 75, 100]), Spacer(1, 8)])
    else:
        story.append(Paragraph("No eligible group comparison hypothesis tests were executed.", body))
    story.append(Spacer(1, 8))

    # 5. Anomaly Detection Results
    story.append(Paragraph("5. Anomaly Detection Results", styles["Heading2"]))
    if iqr_result and iqr_result.get("column"):
        story.append(
            Paragraph(
                f"• <b>IQR Univariate ({escape(str(iqr_result['column']))}):</b> "
                f"{iqr_result.get('count', 0)} potential outlier(s) outside "
                f"[{iqr_result.get('lower_bound')}, {iqr_result.get('upper_bound')}].",
                body,
            )
        )
    if iso_result and iso_result.get("status") == "completed":
        story.append(
            Paragraph(
                f"• <b>Isolation Forest Multivariate:</b> {iso_result.get('count', 0)} anomalous record(s) "
                f"({round(float(iso_result.get('rate', 0.0)) * 100, 2)}%) across features "
                f"{escape(', '.join(iso_result.get('features', [])))}.",
                body,
            )
        )
    elif not iqr_result:
        story.append(Paragraph("No anomaly detection results were recorded.", body))
    story.append(Spacer(1, 10))

    # 6. Forecasting Results & Evaluation Metrics
    story.append(Paragraph("6. Forecasting Results & Evaluation Metrics", styles["Heading2"]))
    if forecast_result and forecast_result.get("status") == "completed":
        story.append(
            Paragraph(
                f"Target: <b>{escape(str(forecast_result.get('target_col')))}</b> | "
                f"Selected Model: <b>{escape(str(forecast_result.get('selected_model')))}</b> | "
                f"Holdout Periods: {forecast_result.get('test_periods')} | "
                f"Beats Baselines: {forecast_result.get('beats_baseline')}",
                body,
            )
        )
        evals = forecast_result.get("evaluations", {})
        eval_rows = [["Model", "MAE", "RMSE", "MAPE (%)"]]
        for m_name, metrics in evals.items():
            eval_rows.append([
                str(m_name),
                str(metrics.get("mae", "N/A")),
                str(metrics.get("rmse", "N/A")),
                str(metrics.get("mape", "N/A")),
            ])
        story.extend([Spacer(1, 4), _styled_table(eval_rows, [200, 100, 100, 110]), Spacer(1, 8)])
    else:
        reason = (forecast_result or {}).get("reason", "Forecasting was not applicable to this dataset.")
        story.append(Paragraph(escape(str(reason)), body))
    story.append(Spacer(1, 8))

    # 7. Generated Charts Summary
    story.append(Paragraph("7. Generated Visualizations", styles["Heading2"]))
    if chart_titles:
        for title in chart_titles:
            story.append(Paragraph(f"• {escape(str(title))}", body))
    else:
        story.append(Paragraph("Interactive Plotly charts are available in the DataMind AI dashboard.", body))
    story.append(Spacer(1, 10))

    # 8. Reviewer-Approved & Rejected Insights
    story.append(Paragraph("8. Reviewer-Approved & Rejected Insights", styles["Heading2"]))
    if insights:
        for item in insights:
            story.append(Paragraph(f"✓ {escape(str(item))}", body))
    else:
        story.append(Paragraph("No approved observations were added.", body))
    if rejected_findings:
        story.append(Spacer(1, 6))
        story.append(Paragraph("<b>Rejected / Flagged Claims:</b>", body))
        for rej in rejected_findings:
            reason = rej.get("reason", "Unsupported claim")
            story.append(Paragraph(f"✗ {escape(str(reason))}", body))
    story.append(Spacer(1, 10))

    # 9. Limitations & Recommendations
    story.append(Paragraph("9. Limitations & Recommendations", styles["Heading2"]))
    story.append(
        Paragraph(
            "• Statistical outliers and Isolation Forest flags indicate rarity, not proof of data errors.<br/>"
            "• Correlation and group-test results show statistical association, not causal proof.<br/>"
            "• Validate business definitions, missingness mechanisms, and external factors before acting on forecasts.",
            body,
        )
    )

    doc.build(story)
    return buffer.getvalue()
