# DataMind AI — Autonomous Multi-Agent Data Analyst

**MCA Final-Year Project** · Local-First, CPU-Friendly, Evidence-Verified Analytics Platform

DataMind AI is an autonomous multi-agent data analysis platform built with **Python, Streamlit, LangGraph, DuckDB, Pandas, SciPy, scikit-learn, Plotly, and ReportLab**. It transforms raw CSV and Excel datasets into profiled, cleaned, queried, statistically tested, forecasted, and evidence-reviewed business reports—without requiring paid cloud APIs or a GPU.

---

## Objectives
1. Automate end-to-end exploratory data analysis using five coordinated specialist agents.
2. Enforce strict security boundaries: read-only DuckDB SQL, SQL injection prevention, resource limits, and zero execution of uploaded data.
3. Prevent hallucinated insights by verifying every analytical claim against computed evidence, sample sizes, and statistical assumptions.
4. Provide a responsive, dark-themed AI analytics dashboard with full CSV and PDF exportability.

---

## Key Features
- **CSV & Excel Ingestion**: UTF-8/`latin-1` CSV and `.xlsx`/`.xls` upload with pre-load file size caps (`50 MB`), row/column limits, and datetime detection.
- **User-Approved Data Cleaning**: Semantic column classification (`numeric`, `numeric_text`, `datetime`, `categorical`, `identifier`) with confidence scores and explanations for missing values, duplicates, whitespace, constant columns, and date parsing; non-destructive `working_df` vs. `original_df` with formula-injection-safe CSV exports.
- **LangGraph Multi-Agent Orchestration**: Shared typed `AnalystState`, conditional routing, graceful error handling, bounded reviewer feedback/retry loop, and a live **Agent Execution Timeline**.
- **Schema-Aware Read-Only NL-to-SQL**: Supports filtering, grouping, aggregations, time trends, sorting, and limits over DuckDB `:memory:`, with `EXPLAIN` pre-validation and optional local **Ollama** LLM support (with deterministic fallback).
- **Advanced Statistical Analysis & Anomaly Detection**: Descriptive stats with 95% confidence intervals, Pearson/Spearman correlations with p-values & Fisher-z CIs, assumption-checked hypothesis tests (Shapiro-Wilk, Welch's t-test, Mann-Whitney U, ANOVA, Kruskal-Wallis, Chi-square), IQR outliers, and **Isolation Forest** multivariate anomaly detection with per-record `why_flagged` explanations.
- **Leakage-Free Chronological Forecasting**: Strict chronological train/test holdout split comparing **Naive**, **Moving Average**, and **Ridge Linear Trend** models using `MAE`, `RMSE`, and `MAPE`.
- **Interactive Dark-Themed Visualizations & PDF Reports**: Plotly distributions, box plots, scatter plots, correlation heatmaps, anomaly charts, forecast charts, and 9-section ReportLab PDF reports.

---

## System Architecture

```text
Dataset Validation ──▶ Data Profiler ──▶ Orchestrator Planner
                                                │
                       ┌────────────────────────┴────────────────────────┐
                       ▼ (if SQL question)                               │
                   SQL Agent (DuckDB Read-Only)                          │
                       │                                                 │
                       └────────────────────────┬────────────────────────┘
                                                ▼
                                     Statistical Analyst
                                  (Stats, Anomalies, Forecast)
                                                │
                                                ▼
                                       Visualization Agent
                                      (Dark Plotly Charts)
                                                │
                                                ▼
                                        Insight Reviewer ◀───┐
                                   (Evidence & Claim Audit)  │ (Bounded Retry
                                                │            │  if rejected)
                                                ├────────────┘
                                                ▼
                                         Report Builder
                                     (PDF + Timeline + CSV)
```

---

## Five Specialist Agents
1. **Data Profiler Agent (`agents/profiler.py`)**: Profiles shape, types, missingness, cardinality, skewness, constant/ID-like columns, quality score (`0–100`), and cleaning suggestions.
2. **SQL Agent (`agents/sql_agent.py`)**: Translates natural-language questions into schema-aware DuckDB `SELECT` queries, validates via keyword blocklisting + `EXPLAIN`, and optionally uses local Ollama.
3. **Statistical Analysis Agent (`agents/statistician.py`)**: Computes 95% CIs, IQR & Isolation Forest anomalies, correlation p-values, hypothesis tests, and chronological holdout forecasts.
4. **Visualization Agent (`agents/visualization.py`)**: Recommends and builds dark-themed Plotly charts with plain-English explanations.
5. **Insight Reviewer Agent (`agents/reviewer.py`)**: Audits claims against computed metrics, sample-size thresholds (`n >= 5`), and causal-language boundaries, rejecting unsupported conclusions.

---

## Technology Stack
- **UI & Visualization**: Streamlit, Plotly
- **Orchestration**: LangGraph (`StateGraph`)
- **Data & SQL**: Pandas, NumPy, DuckDB, OpenPyXL
- **Statistics & ML**: SciPy, scikit-learn (`IsolationForest`, `Ridge`, `StandardScaler`)
- **Reporting & Testing**: ReportLab, pytest
- **Optional Local LLM**: Ollama (`http://localhost:11434`)

---

## Installation & Launch (Windows PowerShell)

```powershell
# 1. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# 2. Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 3. Launch the Streamlit dashboard
streamlit run app.py
```

If PowerShell blocks script activation, run directly:
```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local URL shown in the terminal (default: `http://localhost:8501`).

---

## Sample Dataset & Demonstration Workflow
1. Launch the app and click **Load Sample Sales Dataset** in the sidebar (loads `sample_data/sales.csv`, 33 rows × 6 columns).
2. **Overview**: Inspect KPI cards (`Dataset Size`, `Data Quality`, `Missing Values`, `Anomalies Flagged`, `Analysis Status`), schema profile, and 95% CIs.
3. **Data Cleaning**: Review suggestions, apply optional transformations, or export the cleaned CSV.
4. **Analytics and Anomalies**: Inspect IQR outliers, adjust Isolation Forest contamination, view `why_flagged` anomaly explanations, correlation heatmap, and hypothesis tests.
5. **Forecasting**: Select `date` and `revenue`, inspect chronological holdout metrics (Naive vs. Moving Average vs. Ridge), and export the forecast CSV.
6. **Ask with SQL**: Ask `"top 5 products by revenue in North"` or `"monthly revenue trend"` and inspect the validated SQL + result table.
7. **Insights and Reports**: Review the **Multi-Agent Execution Timeline**, approved/rejected findings, and download the full **PDF Report**.

---

## Running Tests & Reproducible Benchmarks
```powershell
# Run full unit, integration, security, and accuracy test suite
.venv\Scripts\python.exe -m pytest -v

# Run 5-domain reproducible benchmark suite
.venv\Scripts\python.exe -m experiments.run_benchmarks
```

See [docs/architecture.md](docs/architecture.md) for Mermaid system architecture, data-flow, and agent-interaction diagrams, [docs/research_paper_results.md](docs/research_paper_results.md) for the complete verified experimental evaluation, [docs/mca_demo_guide.md](docs/mca_demo_guide.md) for the step-by-step live demo walkthrough, and [experiments/results/benchmark_summary.md](experiments/results/benchmark_summary.md) for measured metrics across 5 domains (`retail_sales`, `app_catalog`, `healthcare_vitals`, `ecommerce_traffic`, `hr_survey`).

---

## Project Structure
```text
datamind-ai/
├── app.py                      # Dark-themed 6-tab Streamlit dashboard
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git exclusions
├── .streamlit/config.toml      # Dark theme configuration
├── agents/
│   ├── profiler.py             # Data Profiler Agent
│   ├── sql_agent.py            # Schema-aware SQL Agent (+ optional Ollama)
│   ├── statistician.py         # Statistical, Anomaly & Forecasting Agent
│   ├── visualization.py        # Plotly Visualization Agent
│   └── reviewer.py             # Evidence-based Insight Reviewer Agent
├── core/
│   ├── config.py               # Resource limits & Ollama defaults
│   ├── logging_config.py       # Structured logging
│   ├── ingestion.py            # Secure CSV/Excel loader & validator
│   ├── cleaning.py             # Cleaning pipeline & safe CSV export
│   ├── sql_engine.py           # Read-only DuckDB validation & execution
│   └── reporting.py            # 9-section ReportLab PDF generator
├── orchestration/
│   ├── state.py                # Typed AnalystState & TimelineEvent
│   └── graph.py                # LangGraph StateGraph & retry loop
├── sample_data/
│   └── sales.csv               # Sample sales dataset (33 rows)
└── tests/                      # Unit, integration, security & accuracy tests
```

---

## Limitations & Future Enhancements
- **Univariate Forecasting**: Current forecasting uses univariate chronological models (Naive, Moving Average, Ridge) and requires at least 8 distinct time periods.
- **PDF Chart Rendering**: PDF reports summarize generated charts in text/tables to avoid heavy headless-browser dependencies on CPU-only laptops.
- **Future Enhancements**: Multi-table SQL joins, seasonal ARIMA/Prophet optional plugins, and persistent workspace history.
