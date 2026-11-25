# DataMind AI — MCA Project Demonstration & Viva Guide

## 1. Pre-Demo Setup (2 Minutes Before Evaluation)
```powershell
cd "c:\Users\VENKATESH\OneDrive\Desktop\DataMind AI\datamind-ai-starter\datamind-ai"
.venv\Scripts\python.exe -m pytest -v
.venv\Scripts\python.exe -m streamlit run app.py
```
- Confirm `42 passed` in `pytest` and open `http://localhost:8501` in the browser.

## 2. Step-by-Step Live Application Walkthrough
1. **Load Sample Dataset**: Click **Load Sample Sales Dataset** in the left sidebar (`sample_data/sales.csv`, 33 rows × 6 columns).
2. **Overview Tab**:
   - Point out the 5 KPI cards (`Dataset Size`, `Data Quality`, `Missing Values`, `Anomalies Flagged`, `Analysis Status`).
   - Show the **Column Profile** with semantic types (`datetime`, `numeric`, `categorical`) and the **Numeric Summary** with 95% confidence intervals.
3. **Data Cleaning Tab**:
   - Show the confidence-scored cleaning recommendations and explanations.
   - Demonstrate that `original_df` is never modified automatically; apply whitespace trimming or reset to original.
   - Download `datamind_cleaned.csv` (formula-injection protected).
4. **Analytics and Anomalies Tab**:
   - Show IQR univariate outliers, interactive distribution/box/scatter Plotly charts, and **Isolation Forest** with configurable contamination and `why_flagged` explanations.
   - Expand **Group Comparison Tests** and **Correlation Heatmap** to show assumption-checked SciPy tests (Shapiro-Wilk, Welch's t-test/ANOVA/Kruskal-Wallis).
5. **Forecasting Tab**:
   - Select `date` and `revenue`, set horizon to `6`, and show the chronological holdout comparison table (**Naive** vs. **Moving Average** vs. **Ridge Linear Trend**) with `MAE`, `RMSE`, `MAPE`, and 95% residual bands.
6. **Ask with SQL Tab**:
   - Enter `"top 5 products by revenue in North"` -> show generated `TRIM`-resilient read-only DuckDB SQL and result table.
   - Mention SQL injection protection (`DROP`/`DELETE`/`read_csv` blocked, `EXPLAIN` pre-validated).
7. **Insights and Reports Tab**:
   - Show the **Multi-Agent Execution Timeline** (agent name, task, color-coded status badge, latency in ms).
   - Show **Insight Reviewer** approved vs. rejected findings.
   - Click **Download Full PDF Report** and open the generated 9-section PDF.
