# Experimental Evaluation & Results

## 1. Experimental Environment
All experiments were executed locally on a CPU-only workstation without GPU acceleration or paid external LLM APIs:
- **Operating System**: `Windows-11-10.0.26200-SP0`
- **Python Runtime**: `3.14.10`
- **Processor / Cores**: `Intel64 Family 6 Model 186 Stepping 2, GenuineIntel` (`12` logical cores)
- **Physical RAM**: `15.7 GB` (`gpu_required = false`)
- **Reproducibility**: All datasets, JSON metrics, Markdown summaries, and Plotly HTML charts were generated deterministically (`random_state=42`, `np.random.default_rng(42)`) via `.venv\Scripts\python.exe -m experiments.run_benchmarks`.

## 2. Dataset Descriptions (Synthetic Domain-Simulated)
1. **`retail_sales` (120 rows × 5 cols)**: Daily sales time series (`date`, `region`, `product`, `quantity`, `revenue`) with 3 injected multi-feature spikes and 3 missing `quantity` values.
2. **`app_catalog` (82 rows × 6 cols)**: Dirty product catalog (`App Name`, `Category`, `Size`, `Price`, `Rating`, `Last Updated`) with 2 duplicate rows, leading/trailing whitespace in `Category` (`" Tools "`), and 3 missing `Rating` values.
3. **`healthcare_vitals` (100 rows × 6 cols)**: Patient vitals (`visit_date`, `department`, `age`, `bmi`, `systolic_bp`, `heart_rate`) with 4 injected clinical outliers.
4. **`ecommerce_traffic` (90 rows × 5 cols)**: Daily web traffic (`date`, `channel`, `sessions`, `conversions`, `revenue`) with a steady upward trend.
5. **`hr_survey` (60 rows × 5 cols)**: Purely categorical/ordinal survey (`employee_id`, `department`, `role_level`, `satisfaction`, `remote_preference`).

## 3. Evaluation Methodology
- **Timing & Memory**: Measured using `time.perf_counter()` (milliseconds) and Python `tracemalloc` peak traced memory (MB).
- **Equivalent Baseline Scope**: The single-process baseline executes `profile_dataset`, `generate_and_validate_sql` + `execute_readonly_query`, `iqr_outliers`, `isolation_forest_outliers`, `correlation_analysis`, `run_statistical_tests`, and `forecast_series`. The full LangGraph workflow additionally executes `validate_dataframe`, `cleaning_suggestions`, `visualization_node` (Plotly figures), `insight_review_node` (evidence verification), and `final_report_node` (ReportLab PDF generation).
- **SQL Evaluation**: 23 predefined natural-language questions across all 5 datasets, evaluated separately on uncleaned and cleaned DataFrames.
- **Reviewer Evaluation**: Evaluated on two separate suites—a 32-claim structured suite and a 16-claim unseen adversarial/boundary suite.

## 4. Performance Comparison (Multi-Agent vs. Equivalent Baseline)
| Dataset | Rows × Cols | Ingest (ms) | Profile (ms) | Multi-Agent (ms) | Multi-Agent Mem (MB) | Equiv. Baseline (ms) | Baseline Mem (MB) |
|---|---|---:|---:|---:|---:|---:|---:|
| `retail_sales` | 120 × 5 | 6.81 | 120.08 | 4,244.78 | 21.352 | 1,906.06 | 0.427 |
| `app_catalog` | 82 × 6 | 4.40 | 100.33 | 3,330.44 | 1.301 | 1,785.37 | 0.455 |
| `healthcare_vitals` | 100 × 6 | 6.91 | 98.16 | 2,894.29 | 1.471 | 1,863.96 | 0.436 |
| `ecommerce_traffic` | 90 × 5 | 5.25 | 43.29 | 3,374.97 | 1.786 | 2,143.84 | 0.441 |
| `hr_survey` | 60 × 5 | 6.91 | 67.68 | 781.15 | 0.925 | 230.48 | 0.090 |

## 5. SQL Execution Accuracy (23 Questions)
- **Uncleaned Datasets**: **100.0% (23/23)**
- **Cleaned Datasets**: **100.0% (23/23)**
- Applying `TRIM(CAST(col AS VARCHAR))` in SQL `GROUP BY` / `DISTINCT` / `WHERE` clauses resolved whitespace fragmentation (`" Tools "` vs. `"Tools"`) on `app_catalog` without mutating `original_df`.

## 6. Anomaly Detection Evaluation (`contamination=0.05`, `random_state=42`)
| Dataset | Method | Ground Truth | Flagged | TP | FP | FN | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `retail_sales` | Standard `IsolationForest` | 3 | 6 | 3 | 3 | 0 | 0.500 | 1.000 | 0.667 |
| `retail_sales` | Detrended `IsolationForest` | 3 | 6 | 3 | 3 | 0 | 0.500 | 1.000 | 0.667 |
| `healthcare_vitals` | Standard `IsolationForest` | 4 | 5 | 4 | 1 | 0 | 0.800 | 1.000 | 0.889 |

## 7. Forecasting Evaluation (6-Period Chronological Holdout)
| Dataset (Target) | Naive RMSE | Moving Avg RMSE | Ridge Trend RMSE | Selected Model | Beats Baselines? | MAPE (%) |
|---|---:|---:|---:|---|---|---:|
| `ecommerce_traffic` (`sessions`) | 47.531 | 56.650 | **27.682** | `ridge_linear_trend` | Yes | 1.91% |
| `app_catalog` (`Rating`) | 0.751 | 0.708 | **0.567** | `ridge_linear_trend` | Yes | 13.19% |
| `retail_sales` (`quantity`) | 2.121 | **1.354** | 3.182 | `moving_average_baseline` | No | 4.70% |
| `healthcare_vitals` (`age`) | 26.636 | **19.292** | 20.292 | `moving_average_baseline` | No | 51.10%* |

*\*MAPE Suitability*: MAPE is unsuitable when target values are zero/near-zero or when evaluating non-temporal cross-sectional attributes such as patient `age` (`51.10%` MAPE), where `MAE` and `RMSE` remain interpretable.

## 8. Insight Reviewer Evaluation
- **32-Claim Structured Suite**: **100.0% (32/32)** on all 5 datasets (10/10 verified facts approved; 8/8 numerical mismatches, 6/6 causal overreach, 4/4 `n < 5` small-sample, and 4/4 missing-evidence claims rejected).
- **16-Claim Unseen Adversarial & Boundary Suite** (reported separately): **100.0% (16/16)** on all 5 datasets (5/5 valid boundary/RCT/tolerance approvals; 11/11 out-of-tolerance, `n=4`, non-dict, whitespace-only, missing-metric, and negative-sample rejections).

## 9. Limitations & Threats to Validity
- **Synthetic Benchmarks**: The 5 benchmark datasets are deterministic domain simulations (`60–120` rows); real-world enterprise datasets with higher noise, missingness, or millions of rows may exhibit different latency and accuracy profiles.
- **Univariate Forecasting**: Current forecasting models do not incorporate external covariates or multi-seasonality.
- **Contamination Sensitivity**: Fixed `contamination=0.05` caps maximum precision when true anomaly prevalence is lower than `5%`.

## 10. Generated Result Charts
- [execution_time_comparison.html](../experiments/results/execution_time_comparison.html)
- [sql_and_reviewer_accuracy.html](../experiments/results/sql_and_reviewer_accuracy.html)
- [forecast_rmse_comparison.html](../experiments/results/forecast_rmse_comparison.html)
