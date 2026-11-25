# DataMind AI: Autonomous Multi-Agent Data Analyst — Viva Q&A

**Project Title**: `DataMind AI: Autonomous Multi-Agent Data Analyst`  
**Candidate**: `[Author Name]` (`[Roll Number]`) · **Guide**: `[Guide Name]`

---

### 1. Architecture & Multi-Agent Design
**Q1. Why did you choose a multi-agent architecture over a single monolithic script or single LLM prompt?**  
A monolithic LLM prompt mixes schema inspection, SQL generation, statistical math, and narrative generation—leading to numerical hallucinations and brittle error handling. Separating responsibilities across five specialist agents (**Data Profiler**, **SQL Agent**, **Statistical Analysis Agent**, **Visualization Agent**, and **Insight Reviewer**) ensures deterministic Python/SciPy/DuckDB computations and an independent verification gate.

**Q2. How does LangGraph coordinate the five agents?**  
`orchestration/graph.py` defines a `StateGraph(AnalystState)` with 9 nodes (`validate_dataset`, `profile_data`, `plan_analysis`, `sql_analysis`, `statistical_analysis`, `visualization`, `insight_review`, `revise_findings`, `final_report`). Conditional edges skip downstream work on invalid datasets, skip `sql_analysis` when no question is supplied, and route rejected insights through a bounded correction loop.

**Q3. How do you prevent infinite loops in the Insight Reviewer feedback cycle?**  
`AnalystState` tracks `retry_count` (starting at `0`) and `max_retries` (default `1`). When `insight_review` rejects a claim and `retry_count < max_retries`, it routes to `revise_findings`, which rebuilds findings strictly from verified `computed_context` and increments `retry_count`. Once approved or `retry_count >= max_retries`, the graph routes to `final_report` and `END`.

---

### 2. Data Ingestion, Semantic Detection & Cleaning
**Q4. How do you prevent columns like `Size`, `Price`, `Rating`, or `Revenue` from being misclassified as dates?**  
`classify_column_semantic()` in `core/ingestion.py` combines column-name token rules (`_NON_DATE_NAME_TOKENS`), numeric/currency text regexes (`_NUMERIC_TEXT_PATTERN`), explicit date regexes (`_EXPLICIT_DATE_PATTERN`), and `pd.to_datetime` parse ratios (`>= 80%` with years in `1900–2100`). Thus `"19M"` or `"$4.99"` is classified as `numeric_text`, while `"January 7, 2018"` in `Last Updated` is classified as `datetime`.

**Q5. Does DataMind AI ever automatically modify the uploaded dataset?**  
No. `original_df` is preserved untouched in `st.session_state`. Cleaning operations run only when the user explicitly selects checkboxes and clicks **Apply Selected Cleaning**, updating `working_df` with a full transformation log.

---

### 3. SQL Agent & Security
**Q6. How do you protect against SQL injection and unauthorized file access in DuckDB?**  
`core/sql_engine.py` strips `--` and `/* ... */` comments, enforces a single `SELECT` or `WITH` statement, blocks `;` stacked queries and DDL/DML/file-I/O keywords (`INSERT`, `DELETE`, `DROP`, `COPY`, `ATTACH`, `PRAGMA`, `read_csv`, `read_parquet`, `glob`, `getenv`), dry-runs `EXPLAIN` against `df.head(0)`, executes in-memory (`:memory:`), and caps output at `10,000` rows.

**Q7. How did you fix whitespace issues in uncleaned SQL queries without altering the dataset?**  
`_dim_expr()` in `agents/sql_agent.py` wraps text grouping/distinct/filter expressions in `TRIM(CAST(col AS VARCHAR))` while preserving the original column alias (`AS "Category"`), raising uncleaned `app_catalog` SQL accuracy from `50.0%` to `100.0%` without mutating `original_df`.

---

### 4. Statistics, Anomaly Detection & Forecasting
**Q8. How does Isolation Forest explain why a record was flagged, and what is its limitation?**  
`isolation_forest_outliers()` standardizes numeric features (`StandardScaler`), computes `decision_function` anomaly scores, and formats `why_flagged` from the top standardized feature deviations (e.g., `systolic_bp (+3.20 SD)`). Limitation: fixed `contamination=0.05` caps precision when true anomaly prevalence is below `5%`, and anomalies indicate statistical rarity rather than guaranteed errors.

**Q9. How do you prevent data leakage in chronological forecasting?**  
`forecast_series()` sorts chronologically, holds out the final `test_size` periods (`max(2, min(horizon, n // 4))`), and fits `Ridge` and baselines (**Naive**, **Moving Average**) strictly on the training prefix (`train_df`). Perturbing holdout test rows in `test_forecast_chronological_holdout_and_no_leakage` confirms zero change to baseline holdout predictions.

---

### 5. Evaluation & Research Contribution
**Q10. Summarize your verified benchmark results.**  
Across 5 domain-simulated datasets (`12` CPU cores, `15.7 GB` RAM, CPU-only): `100.0%` SQL accuracy (`23/23` uncleaned and cleaned), `100.0%` Insight Reviewer accuracy on both the `32`-claim structured suite and `16`-claim unseen suite, `F1=0.889` on `healthcare_vitals` anomaly detection, and `42/42` automated `pytest` tests passing.
