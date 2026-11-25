# DataMind AI — System Architecture & Data-Flow Diagrams

## 1. Multi-Agent System Architecture (LangGraph)

```mermaid
flowchart TD
    UI["Streamlit Dark Dashboard (app.py)"] -->|"CSV / Excel Upload"| ING["Ingestion & Validation (core/ingestion.py)"]
    ING -->|"Validated DataFrame"| VAL["1. validate_dataset_node"]
    VAL -->|"Valid"| PROF["2. profile_data_node (Data Profiler Agent)"]
    VAL -->|"Invalid"| REP["9. final_report_node (Failed)"]
    PROF -->|"Profile + Semantic Types + Quality Score"| PLAN["3. plan_analysis_node (Orchestrator Planner)"]
    PLAN -->|"NL Question Present"| SQL["4. sql_analysis_node (SQL Agent + Read-Only DuckDB)"]
    PLAN -->|"No Question"| STAT["5. statistical_analysis_node (Statistical Analyst)"]
    SQL --> STAT
    STAT -->|"Stats, Isolation Forest, Tests, Forecasts"| VIZ["6. visualization_node (Plotly Visualization Agent)"]
    VIZ --> REV["7. insight_review_node (Insight Reviewer Agent)"]
    REV -->|"Rejected & retry < max_retries"| COR["8. revise_findings_node (Correction Loop)"]
    COR --> REV
    REV -->|"Approved or max_retries reached"| REP
    REP -->|"PDF + Timeline + Safe CSV"| UI
```

## 2. Data-Flow & Security Boundary Diagram

```mermaid
flowchart LR
    U["User Upload (.csv / .xlsx)"] -->|"Size <= 50MB, Rows <= 500k"| DF["working_df (In-Memory Copy)"]
    U -->|"Preserved Read-Only"| ORIG["original_df"]
    DF -->|"Read-Only Registration"| DUCK["DuckDB :memory: (SELECT / WITH only)"]
    NL["User NL Question"] -->|"Schema-Aware Rules / Optional Ollama"| VALSQL["validate_readonly_sql + EXPLAIN"]
    VALSQL --> DUCK
    DF -->|"Numeric Features"| ML["SciPy + scikit-learn (IsolationForest / Ridge)"]
    DUCK --> REV["Insight Reviewer (Evidence & Numerical Audit)"]
    ML --> REV
    REV -->|"Formula-Safe CSV / ReportLab PDF"| EXP["User Exports"]
```

## 3. Agent-Interaction Sequence Diagram

```mermaid
sequenceDiagram
    participant User as User / Streamlit UI
    participant Val as Ingestion Validator
    participant Prof as Data Profiler
    participant Plan as Orchestrator Planner
    participant SQL as SQL Agent (DuckDB)
    participant Stat as Statistical Analyst
    participant Viz as Visualization Agent
    participant Rev as Insight Reviewer
    participant Rep as Report Builder

    User->>Val: Upload CSV/Excel (or Sample)
    Val->>Prof: Validated DataFrame
    Prof->>Plan: Profile + Semantic Types + Quality Score
    alt NL Question Provided
        Plan->>SQL: Generate & Validate Read-Only SQL
        SQL->>Stat: SQL Results + Schema
    else No NL Question
        Plan->>Stat: Analysis Plan
    end
    Stat->>Viz: Stats, Anomalies, Tests, Forecast
    Viz->>Rev: Interactive Plotly Charts + Candidate Findings
    alt Findings Rejected (retry_count < max_retries)
        Rev->>Rev: Bounded Correction Loop (rebuild from verified evidence)
    end
    Rev->>Rep: Approved & Rejected Findings
    Rep->>User: 9-Section PDF + Timeline + Safe CSV Exports
```
