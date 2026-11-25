"""Build the complete MCA Final-Year Submission Package for DataMind AI."""

import json
from pathlib import Path
from typing import Any

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from pptx.dml.color import RGBColor as PptxRGB
from pptx.util import Inches as PptxInches, Pt as PptxPt
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Image as RlImage, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

PROJECT_TITLE = "DataMind AI: Autonomous Multi-Agent Data Analyst"


def _load_font(size: int = 16) -> ImageFont.ImageFont:
    for candidate in ["arial.ttf", "seguiui.ttf", "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_box(draw: ImageDraw.ImageDraw, xy: tuple[int, int, int, int], title: str, sub: str, border: str, fill: str = "#111827") -> None:
    draw.rounded_rectangle(xy, radius=10, fill=fill, outline=border, width=3)
    f_title = _load_font(18)
    f_sub = _load_font(13)
    x0, y0, x1, y1 = xy
    draw.text((x0 + 14, y0 + 12), title, fill="#F8FAFC", font=f_title)
    draw.text((x0 + 14, y0 + 38), sub, fill="#94A3B8", font=f_sub)


def _draw_arrow(draw: ImageDraw.ImageDraw, p1: tuple[int, int], p2: tuple[int, int], color: str = "#38BDF8") -> None:
    draw.line([p1, p2], fill=color, width=3)
    x2, y2 = p2
    if abs(p2[0] - p1[0]) >= abs(p2[1] - p1[1]):
        direction = 1 if p2[0] >= p1[0] else -1
        draw.polygon([(x2, y2), (x2 - 10 * direction, y2 - 6), (x2 - 10 * direction, y2 + 6)], fill=color)
    else:
        direction = 1 if p2[1] >= p1[1] else -1
        draw.polygon([(x2, y2), (x2 - 6, y2 - 10 * direction), (x2 + 6, y2 - 10 * direction)], fill=color)


def generate_diagrams(fig_dir: Path) -> dict[str, Path]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    f_hdr = _load_font(22)

    # Figure 1: System Architecture & LangGraph Workflow
    img1 = Image.new("RGB", (1100, 620), "#0B1120")
    d1 = ImageDraw.Draw(img1)
    d1.text((30, 20), f"{PROJECT_TITLE} — LangGraph System Architecture", fill="#38BDF8", font=f_hdr)
    _draw_box(d1, (40, 80, 320, 160), "1. Ingestion Validator", "CSV/Excel, <=50MB, Schema", "#38BDF8")
    _draw_box(d1, (390, 80, 690, 160), "2. Data Profiler Agent", "Semantic types, quality score", "#38BDF8")
    _draw_box(d1, (760, 80, 1050, 160), "3. Orchestrator Planner", "Route SQL / Stats / Forecast", "#A78BFA")
    _draw_arrow(d1, (320, 120), (390, 120))
    _draw_arrow(d1, (690, 120), (760, 120))

    _draw_box(d1, (760, 240, 1050, 320), "4. SQL Agent (DuckDB)", "Read-only SELECT, EXPLAIN", "#38BDF8")
    _draw_box(d1, (390, 240, 690, 320), "5. Statistical Analyst", "IQR, IsolationForest, Forecast", "#38BDF8")
    _draw_box(d1, (40, 240, 320, 320), "6. Visualization Agent", "Dark Plotly charts + captions", "#A78BFA")
    _draw_arrow(d1, (905, 160), (905, 240))
    _draw_arrow(d1, (760, 280), (690, 280))
    _draw_arrow(d1, (390, 280), (320, 280))

    _draw_box(d1, (40, 420, 380, 510), "7. Insight Reviewer Agent", "Verify metrics, n>=5, causality", "#FBBF24")
    _draw_box(d1, (440, 420, 720, 510), "8. Correction Loop", "Bounded retry (max_retries=1)", "#F43F5E")
    _draw_box(d1, (780, 420, 1050, 510), "9. Report Builder", "9-section PDF + Safe CSVs", "#34D399")
    _draw_arrow(d1, (180, 320), (180, 420))
    _draw_arrow(d1, (380, 450), (440, 450), "#F43F5E")
    _draw_arrow(d1, (440, 485), (380, 485), "#FBBF24")
    _draw_arrow(d1, (380, 500), (780, 500), "#34D399")
    p1 = fig_dir / "fig1_system_architecture.png"
    img1.save(p1)

    # Figure 2: Data-Flow & Security Boundary
    img2 = Image.new("RGB", (1100, 420), "#0B1120")
    d2 = ImageDraw.Draw(img2)
    d2.text((30, 20), "Data-Flow & Read-Only Security Boundaries", fill="#A78BFA", font=f_hdr)
    _draw_box(d2, (40, 100, 300, 190), "User Upload", "CSV / XLSX (<=50 MB)", "#38BDF8")
    _draw_box(d2, (360, 100, 660, 190), "original_df (Immutable)", "Preserved untouched", "#34D399")
    _draw_box(d2, (720, 100, 1050, 190), "working_df (In-Memory)", "User-approved cleaning only", "#38BDF8")
    _draw_arrow(d2, (300, 145), (360, 145))
    _draw_arrow(d2, (660, 145), (720, 145))

    _draw_box(d2, (40, 260, 340, 350), "NL Question / Ollama", "Optional local LLM fallback", "#A78BFA")
    _draw_box(d2, (400, 260, 720, 350), "SQL Guard + DuckDB", "Strip comments, EXPLAIN, LIMIT", "#FBBF24")
    _draw_box(d2, (780, 260, 1050, 350), "Sanitized Exports", "Formula-safe CSV & PDF", "#34D399")
    _draw_arrow(d2, (340, 305), (400, 305))
    _draw_arrow(d2, (720, 305), (780, 305))
    p2 = fig_dir / "fig2_data_flow_security.png"
    img2.save(p2)

    # Figure 3: Multi-Agent Interaction & Evidence Verification
    img3 = Image.new("RGB", (1100, 420), "#0B1120")
    d3 = ImageDraw.Draw(img3)
    d3.text((30, 20), "Multi-Agent Shared State & Evidence Verification Loop", fill="#38BDF8", font=f_hdr)
    _draw_box(d3, (40, 140, 280, 240), "AnalystState", "TypedDict shared state", "#A78BFA")
    _draw_box(d3, (340, 90, 680, 180), "Specialist Agents", "Profiler, SQL, Stats, Viz", "#38BDF8")
    _draw_box(d3, (340, 220, 680, 310), "Insight Reviewer", "32-claim + 16-unseen audited", "#FBBF24")
    _draw_box(d3, (760, 140, 1050, 240), "Verified Outputs", "Timeline, PDF, CSVs", "#34D399")
    _draw_arrow(d3, (280, 175), (340, 135))
    _draw_arrow(d3, (510, 180), (510, 220))
    _draw_arrow(d3, (680, 250), (760, 195), "#34D399")
    p3 = fig_dir / "fig3_agent_interaction.png"
    img3.save(p3)

    return {"fig1": p1, "fig2": p2, "fig3": p3}


def build_research_paper_docx(bench: dict[str, Any], figs: dict[str, Path], out_path: Path) -> None:
    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)

    t = doc.add_heading(PROJECT_TITLE, level=0)
    t.runs[0].font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)

    meta = doc.add_paragraph(
        "Author: [Author Name] | Roll No: [Roll Number]\n"
        "Guide: [Project Guide Name] | Department: [Department of Computer Applications / University]\n"
        "Program: Master of Computer Applications (MCA) — Final-Year Project"
    )
    meta.runs[0].italic = True

    doc.add_heading("Abstract", level=1)
    doc.add_paragraph(
        "DataMind AI: Autonomous Multi-Agent Data Analyst is a local-first, CPU-friendly analytics platform "
        "orchestrated with LangGraph across five specialist agents: Data Profiler, SQL Agent, Statistical "
        "Analysis Agent, Visualization Agent, and Insight Reviewer. Unlike single-prompt LLM wrappers that "
        "can hallucinate numbers or execute unsafe SQL, DataMind AI enforces deterministic Python/SciPy/DuckDB "
        "computation, read-only SQL validation, chronological holdout forecasting, and automated evidence "
        "verification. Across five domain-simulated benchmark datasets, the system achieved 100.0% SQL "
        "execution accuracy (23/23 questions on both uncleaned and cleaned datasets), 100.0% accuracy on both "
        "the 32-claim Insight Reviewer suite and a separate 16-claim unseen adversarial suite, and F1=0.889 on "
        "stationary clinical anomaly detection."
    )

    doc.add_heading("1. Introduction and Objectives", level=1)
    doc.add_paragraph(
        "Exploratory data analysis (EDA) requires coordinating schema inspection, cleaning, SQL querying, "
        "statistical hypothesis testing, anomaly detection, forecasting, and reporting. The objectives of "
        f"{PROJECT_TITLE} are: (1) automate end-to-end EDA via a modular 5-agent LangGraph workflow; "
        "(2) preserve user data integrity with non-destructive cleaning and semantic column classification; "
        "(3) enforce read-only DuckDB SQL security; and (4) audit every generated claim against computed evidence."
    )

    doc.add_heading("2. System Architecture and Five Specialist Agents", level=1)
    doc.add_paragraph(
        "The workflow coordinates nine LangGraph nodes sharing a typed AnalystState dictionary: "
        "validate_dataset -> profile_data -> plan_analysis -> sql_analysis -> statistical_analysis -> "
        "visualization -> insight_review -> (bounded revise_findings loop) -> final_report."
    )
    doc.add_picture(str(figs["fig1"]), width=Inches(6.2))
    doc.add_paragraph("Figure 1: System Architecture and LangGraph Multi-Agent Workflow.")
    doc.add_picture(str(figs["fig2"]), width=Inches(6.2))
    doc.add_paragraph("Figure 2: Data-Flow and Read-Only Security Boundaries.")
    doc.add_picture(str(figs["fig3"]), width=Inches(6.2))
    doc.add_paragraph("Figure 3: Multi-Agent Shared State and Bounded Reviewer Feedback Loop.")

    doc.add_heading("3. Experimental Environment and Datasets", level=1)
    hw = bench["hardware"]
    doc.add_paragraph(
        f"Hardware: {hw['processor']}, {hw['cpu_cores']} CPU cores, {hw['total_ram_gb']} GB RAM, "
        f"OS: {hw['os']}, Python {hw['python_version']}, GPU required: {hw['gpu_required']}. "
        "Five deterministic synthetic domain-simulated datasets were evaluated: retail_sales (120x5), "
        "app_catalog (82x6), healthcare_vitals (100x6), ecommerce_traffic (90x5), and hr_survey (60x5)."
    )

    doc.add_heading("4. Experimental Results", level=1)
    doc.add_heading("4.1 Performance vs. Equivalent Single-Process Baseline", level=2)
    tbl1 = doc.add_table(rows=1, cols=7)
    tbl1.style = "Table Grid"
    for i, h in enumerate(["Dataset", "Rows x Cols", "Ingest (ms)", "Profile (ms)", "Multi-Agent (ms)", "Peak Mem (MB)", "Equiv Baseline (ms)"]):
        tbl1.rows[0].cells[i].text = h
    for name, d in bench["datasets"].items():
        row = tbl1.add_row().cells
        row[0].text = name
        row[1].text = f"{d['rows']} x {d['columns']}"
        row[2].text = str(d["ingestion_ms"])
        row[3].text = str(d["profiling_ms"])
        row[4].text = str(d["multi_agent_workflow_ms"])
        row[5].text = str(d["multi_agent_peak_memory_mb"])
        row[6].text = str(d["equivalent_baseline"]["elapsed_ms"])

    doc.add_heading("4.2 SQL Accuracy, Anomaly Detection & Forecasting", level=2)
    doc.add_paragraph(
        f"SQL Execution Accuracy across {bench['summary']['total_sql_questions']} predefined questions: "
        f"{bench['summary']['sql_uncleaned_accuracy_pct']}% on uncleaned datasets and "
        f"{bench['summary']['sql_cleaned_accuracy_pct']}% on cleaned datasets (enabled by TRIM-resilient SQL grouping).\n"
        "Anomaly Detection (contamination=0.05, random_state=42): healthcare_vitals achieved Precision=0.800, "
        "Recall=1.000, F1=0.889; retail_sales achieved Precision=0.500, Recall=1.000, F1=0.667 for both "
        "standard and detrended Isolation Forest.\n"
        "Forecasting (6-period chronological holdout): Ridge Linear Trend beat Naive and Moving Average "
        "baselines on ecommerce_traffic (RMSE=27.682, MAPE=1.91%) and app_catalog (RMSE=0.567, MAPE=13.19%), "
        "while Moving Average won on retail_sales (RMSE=1.354) and healthcare_vitals (RMSE=19.292)."
    )

    doc.add_heading("4.3 Insight Reviewer Evaluation (32-Claim & Separate 16-Claim Unseen)", level=2)
    doc.add_paragraph(
        "On the 32-claim structured evaluation suite, the Insight Reviewer achieved 100.0% accuracy (32/32) "
        "across all 5 datasets (10/10 verified facts approved; 8/8 numerical mismatches, 6/6 causal overreach, "
        "4/4 n<5 small-sample, and 4/4 missing-evidence claims rejected). On the separate 16-claim unseen "
        "adversarial and boundary suite, it also achieved 100.0% accuracy (16/16)."
    )

    doc.add_heading("5. Application Interface (Screenshots)", level=1)
    doc.add_paragraph(
        "Figure 4: [Insert Live Screenshot — DataMind AI Dark Dashboard Overview & KPI Cards]\n"
        "Figure 5: [Insert Live Screenshot — Analytics, Isolation Forest & Chronological Forecasting Tabs]\n"
        "Figure 6: [Insert Live Screenshot — Ask with SQL & Multi-Agent Execution Timeline]"
    )

    doc.add_heading("6. Limitations and Future Work", level=1)
    doc.add_paragraph(
        "Measured Limitations: (1) Benchmark datasets are synthetic domain simulations (60–120 rows); "
        "(2) Forecasting is univariate and requires >=8 chronological periods; (3) Fixed contamination=0.05 "
        "caps precision when true anomaly prevalence is below 5%; (4) PDF reports summarize charts in tables/text "
        "to remain CPU-only.\n"
        "Future Work: Multi-table SQL joins, seasonal ARIMA/Prophet optional plugins, and persistent workspace history."
    )

    doc.add_heading("7. References", level=1)
    doc.add_paragraph("[Add Formal Academic Citations for LangGraph, DuckDB, Isolation Forest (Liu et al.), SciPy, and Streamlit]")

    # Add footer page numbering to DOCX
    section = doc.sections[0]
    footer_p = section.footer.paragraphs[0]
    footer_p.text = f"{PROJECT_TITLE}  |  Page "
    fld_simple = OxmlElement("w:fldSimple")
    fld_simple.set(qn("w:instr"), "PAGE")
    footer_p._p.append(fld_simple)

    doc.save(out_path)


def _draw_pdf_page_number(canvas_obj: Any, doc_obj: Any) -> None:
    canvas_obj.saveState()
    canvas_obj.setFont("Helvetica", 8.5)
    canvas_obj.setFillColor(colors.HexColor("#64748B"))
    canvas_obj.drawRightString(570, 24, f"{PROJECT_TITLE}  |  Page {doc_obj.page}")
    canvas_obj.restoreState()


def build_research_paper_pdf(bench: dict[str, Any], figs: dict[str, Path], out_path: Path) -> None:
    doc = SimpleDocTemplate(str(out_path), pagesize=letter, rightMargin=42, leftMargin=42, topMargin=42, bottomMargin=42)
    styles = getSampleStyleSheet()
    body = styles["BodyText"]
    story: list[Any] = [
        Paragraph(PROJECT_TITLE, styles["Title"]),
        Paragraph("Author: [Author Name] | Roll No: [Roll Number] | Guide: [Guide Name]", styles["Normal"]),
        Spacer(1, 10),
        Paragraph("Abstract", styles["Heading2"]),
        Paragraph(
            "DataMind AI: Autonomous Multi-Agent Data Analyst is a local-first, CPU-friendly analytics platform "
            "orchestrated via LangGraph across five specialist agents (Data Profiler, SQL Agent, Statistical "
            "Analysis Agent, Visualization Agent, and Insight Reviewer). Across 5 domain-simulated datasets, "
            "the system achieved 100.0% SQL execution accuracy (23/23), 100.0% on both 32-claim and 16-claim "
            "Insight Reviewer suites, and F1=0.889 on clinical anomaly detection.",
            body,
        ),
        Spacer(1, 8),
        Paragraph("System Architecture & Data-Flow Diagrams", styles["Heading2"]),
        RlImage(str(figs["fig1"]), width=480, height=270),
        Paragraph("Figure 1: System Architecture and LangGraph Multi-Agent Workflow.", body),
        Spacer(1, 6),
        RlImage(str(figs["fig2"]), width=480, height=183),
        Paragraph("Figure 2: Data-Flow and Read-Only Security Boundaries.", body),
        Spacer(1, 8),
        Paragraph("Experimental Results Summary", styles["Heading2"]),
    ]

    t_data = [["Dataset", "Rows x Cols", "Ingest (ms)", "Multi-Agent (ms)", "Peak Mem (MB)", "Equiv Base (ms)", "SQL Acc", "Rev (32/16)"]]
    for name, d in bench["datasets"].items():
        t_data.append([
            name,
            f"{d['rows']}x{d['columns']}",
            str(d["ingestion_ms"]),
            str(d["multi_agent_workflow_ms"]),
            str(d["multi_agent_peak_memory_mb"]),
            str(d["equivalent_baseline"]["elapsed_ms"]),
            f"{d['sql_accuracy_cleaned']['accuracy_pct']}%",
            f"{d['reviewer_evaluation']['accuracy_pct']}% / {d['reviewer_unseen_evaluation']['accuracy_pct']}%",
        ])
    tbl = Table(t_data, colWidths=[82, 55, 52, 68, 62, 65, 48, 78])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E293B")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 4),
    ]))
    story.extend([
        tbl,
        Spacer(1, 8),
        Paragraph("Application Screenshots (Manual Insertion Required)", styles["Heading2"]),
        Paragraph("Figure 4: [Insert Live Screenshot — Overview & KPI Cards]", body),
        Paragraph("Figure 5: [Insert Live Screenshot — Analytics, Anomaly Detection & Forecasting]", body),
        Paragraph("Figure 6: [Insert Live Screenshot — Ask with SQL & Execution Timeline]", body),
        Spacer(1, 8),
        Paragraph("Limitations & References", styles["Heading2"]),
        Paragraph("Synthetic 60–120 row benchmarks; univariate forecasting (>=8 periods); [Add Formal Citations].", body),
    ])
    doc.build(story, onFirstPage=_draw_pdf_page_number, onLaterPages=_draw_pdf_page_number)


def _add_dark_slide(prs: Presentation, title: str, bullets: list[str], img_path: Path | None = None) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = PptxRGB(0x0B, 0x11, 0x20)

    title_box = slide.shapes.add_textbox(PptxInches(0.6), PptxInches(0.4), PptxInches(12.1), PptxInches(0.8))
    tf = title_box.text_frame
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = PptxPt(28)
    p.font.bold = True
    p.font.color.rgb = PptxRGB(0x38, 0xBD, 0xF8)

    if img_path is not None:
        body_box = slide.shapes.add_textbox(PptxInches(0.6), PptxInches(1.35), PptxInches(5.5), PptxInches(5.5))
        slide.shapes.add_picture(str(img_path), PptxInches(6.3), PptxInches(1.4), width=PptxInches(6.4))
    else:
        body_box = slide.shapes.add_textbox(PptxInches(0.6), PptxInches(1.35), PptxInches(12.0), PptxInches(5.5))

    btf = body_box.text_frame
    btf.word_wrap = True
    for idx, b in enumerate(bullets):
        bp = btf.paragraphs[0] if idx == 0 else btf.add_paragraph()
        bp.text = f"• {b}"
        bp.font.size = PptxPt(17)
        bp.font.color.rgb = PptxRGB(0xE2, 0xE8, 0xF0)
        bp.space_after = PptxPt(10)


def build_presentation(bench: dict[str, Any], figs: dict[str, Path], out_path: Path) -> None:
    prs = Presentation()
    prs.slide_width = PptxInches(13.333)
    prs.slide_height = PptxInches(7.5)

    slides_spec: list[tuple[str, list[str], Path | None]] = [
        (PROJECT_TITLE, ["MCA Final-Year Project Presentation", "Author: [Author Name] | Roll No: [Roll Number]", "Guide: [Guide Name] | Department: [Department / University]"], None),
        ("Problem Statement & Motivation", ["Manual EDA across profiling, cleaning, SQL, stats, and reporting is slow and error-prone.", "Naive LLM wrappers risk numerical hallucinations and unsafe SQL execution.", "Need a local-first, CPU-friendly, evidence-verified multi-agent analyst."], None),
        ("Project Objectives", ["Coordinate five specialist agents via LangGraph StateGraph.", "Preserve user data with non-destructive cleaning and semantic column detection.", "Enforce read-only DuckDB SQL with injection and resource guards.", "Verify every insight against computed evidence before PDF/CSV export."], None),
        ("System Architecture (LangGraph)", ["9-node StateGraph with shared typed AnalystState.", "Conditional routing for validation, SQL queries, and bounded reviewer retries.", "Graceful degradation when optional modules or Ollama are unavailable."], figs["fig1"]),
        ("Five Specialist Agents", ["Data Profiler: structure, missingness, quality score, semantic types.", "SQL Agent: schema-aware DuckDB SELECT + optional local Ollama.", "Statistical Analyst: 95% CIs, IQR, Isolation Forest, hypothesis tests, forecasting.", "Visualization Agent: dark-themed Plotly charts + explanations.", "Insight Reviewer: numerical, sample-size (n>=5), and causality verification."], None),
        ("Multi-Agent Interaction & Reviewer Loop", ["Reviewer checks metric values against computed_context within tolerance.", "Rejects unsupported causal claims, missing evidence, and small samples.", "Bounded Correction Loop (max_retries=1) prevents infinite cycles."], figs["fig3"]),
        ("Semantic Column Detection & Safe Cleaning", ["Protects Size, Price, Rating, Revenue, Quantity from date misclassification.", "Strict regex + >=80% parse ratio for true date columns (e.g. Last Updated).", "Confidence scores (0–100%) and explanations for every recommendation.", "original_df is never modified without explicit user approval."], None),
        ("Read-Only DuckDB SQL & Security Boundaries", ["Strips SQL comments, blocks DDL/DML and file I/O (read_csv, glob).", "Pre-validates via DuckDB EXPLAIN and caps results at 10,000 rows.", "TRIM-resilient grouping handles dirty text without mutating source data.", "CSV exports neutralize spreadsheet formula injection (=, +, -, @)."], figs["fig2"]),
        ("Statistical Analysis & Anomaly Detection", ["Descriptive stats with 95% t-CIs; Pearson/Spearman with Fisher-z CIs.", "Normality-checked tests: Shapiro-Wilk, Welch's t-test, Mann-Whitney, ANOVA, Kruskal, Chi-square.", "Isolation Forest with configurable contamination and why_flagged SD explanations."], None),
        ("Chronological Forecasting & Leakage Guard", ["Strict chronological train/test holdout split (no data leakage).", "Compares Ridge Linear Trend against Naive and Moving Average baselines.", "Reports MAE, RMSE, MAPE, and 95% residual prediction intervals."], None),
        ("Experimental Setup & 5-Domain Datasets", [f"Hardware: {bench['hardware']['cpu_cores']} CPU cores, {bench['hardware']['total_ram_gb']} GB RAM, Python {bench['hardware']['python_version']}.", "100% CPU-only execution; no GPU or paid API keys required.", "5 domain-simulated datasets: retail_sales, app_catalog, healthcare_vitals, ecommerce_traffic, hr_survey."], None),
        ("Performance vs. Equivalent Baseline", ["Multi-Agent workflow: 569.74–4,244.78 ms; Peak traced memory: 0.925–21.352 MB.", "Equivalent single-process baseline: 230.48–2,143.84 ms (no LangGraph/Viz/Reviewer/PDF).", "Overhead (~0.55–2.34 s) buys full orchestration, charts, auditing, and PDF export."], None),
        ("SQL Accuracy & Anomaly Detection Results", ["23-Question SQL Benchmark: 100.0% (23/23) on both uncleaned and cleaned datasets.", "healthcare_vitals Isolation Forest: Precision=0.800, Recall=1.000, F1=0.889.", "retail_sales (standard & detrended IF): Precision=0.500, Recall=1.000, F1=0.667."], None),
        ("Forecasting & Insight Reviewer Results", ["Ridge beat baselines on ecommerce_traffic (RMSE=27.682, MAPE=1.91%) and app_catalog (RMSE=0.567).", "Moving Average won on retail_sales (RMSE=1.354) and healthcare_vitals (RMSE=19.292).", "Insight Reviewer: 100.0% (32/32) structured suite + 100.0% (16/16) unseen adversarial suite."], None),
        ("Live Dashboard & Demo Walkthrough", ["Dark-themed 6-tab Streamlit UI with 5 KPI cards and Execution Timeline.", "42 automated pytest unit/integration/security tests passing.", "[Insert Live Screenshot of DataMind AI Dashboard Here]"], None),
        ("Limitations, Future Work & Conclusion", ["Limitations: synthetic benchmarks (60–120 rows), univariate forecasting (>=8 periods).", "Future Work: multi-table SQL joins, seasonal ARIMA/Prophet plugins.", "Conclusion: Reliable, auditable, local-first multi-agent data analysis."], None),
    ]

    for title, bullets, img in slides_spec:
        _add_dark_slide(prs, title, bullets, img)

    prs.save(out_path)


def main() -> None:
    sub_dir = Path("submission_package")
    sub_dir.mkdir(parents=True, exist_ok=True)
    bench = json.loads(Path("experiments/results/benchmark_results.json").read_text(encoding="utf-8"))
    figs = generate_diagrams(sub_dir / "figures")
    build_research_paper_docx(bench, figs, sub_dir / "DataMind_AI_Research_Paper.docx")
    build_research_paper_pdf(bench, figs, sub_dir / "DataMind_AI_Research_Paper.pdf")
    build_presentation(bench, figs, sub_dir / "DataMind_AI_MCA_Presentation.pptx")

    import zipfile
    zip_path = sub_dir / "DataMind_AI_MCA_Submission_Package.zip"
    files_to_zip = [
        sub_dir / "DataMind_AI_Research_Paper.docx",
        sub_dir / "DataMind_AI_Research_Paper.pdf",
        sub_dir / "DataMind_AI_MCA_Presentation.pptx",
        sub_dir / "DataMind_AI_Viva_QA.md",
        sub_dir / "DataMind_AI_Installation_and_Demo_Guide.md",
        figs["fig1"],
        figs["fig2"],
        figs["fig3"],
    ]
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fp in files_to_zip:
            zf.write(fp, arcname=fp.relative_to(sub_dir))
    print(f"Built DOCX, PDF, PPTX, PNG figures, and ZIP ({zip_path}) in submission_package/.")


if __name__ == "__main__":
    main()
