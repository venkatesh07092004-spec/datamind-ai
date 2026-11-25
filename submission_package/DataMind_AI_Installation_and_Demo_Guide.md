# DataMind AI: Autonomous Multi-Agent Data Analyst — Installation & Demonstration Guide

**Project Title**: `DataMind AI: Autonomous Multi-Agent Data Analyst`

---

## 1. System Requirements
- **OS**: Windows 10/11 (or macOS/Linux)
- **Python**: Python 3.10 or newer (verified on Python `3.14.10`)
- **Hardware**: CPU-only laptop/desktop (`4+ GB` RAM recommended; no GPU or paid API key required)

## 2. Installation (Windows PowerShell)
```powershell
cd "c:\Users\VENKATESH\OneDrive\Desktop\DataMind AI\datamind-ai-starter\datamind-ai"

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Running Automated Tests & Benchmarks
```powershell
# Run full 42-test automated suite
.venv\Scripts\python.exe -m pytest -v

# Re-run 5-domain reproducible benchmarks
.venv\Scripts\python.exe -m experiments.run_benchmarks
```

## 4. Launching the Dashboard
```powershell
.venv\Scripts\python.exe -m streamlit run app.py
```
Open `http://localhost:8501` in your browser.

## 5. Step-by-Step Demonstration Walkthrough
1. Click **Load Sample Sales Dataset** in the sidebar (`sample_data/sales.csv`, 33 rows × 6 columns) or upload any `.csv` / `.xlsx` / `.xls` file (`<= 50 MB`).
2. **Overview**: Inspect the 5 KPI cards, semantic column profile, and numeric summary with 95% confidence intervals.
3. **Data Cleaning**: Review confidence-scored recommendations, apply selected operations (original upload stays untouched), and export `datamind_cleaned.csv`.
4. **Analytics and Anomalies**: Inspect IQR outliers, interactive Plotly charts, Isolation Forest (`why_flagged` explanations + `datamind_anomalies.csv` export), correlation heatmap, and SciPy hypothesis tests.
5. **Forecasting**: Select `date` and `revenue`, review chronological holdout metrics (Naive vs. Moving Average vs. Ridge), and export `datamind_forecast.csv`.
6. **Ask with SQL**: Query `"top 5 products by revenue in North"` or `"monthly revenue trend"` using read-only DuckDB (or optional local Ollama).
7. **Insights and Reports**: Inspect the Multi-Agent Execution Timeline, Reviewer-approved/rejected findings, and download the 9-section PDF report.
