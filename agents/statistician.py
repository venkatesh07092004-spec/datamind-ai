"""Statistical Analysis Agent: descriptive stats, outlier/anomaly detection, hypothesis tests, and forecasting."""

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from core.logging_config import get_logger

logger = get_logger(__name__)


def numeric_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Return rounded Pearson correlation matrix for numeric columns."""
    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        return pd.DataFrame()
    return numeric.corr(numeric_only=True).round(3)


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return transposed descriptive statistics with 95% confidence intervals for numeric columns."""
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        return pd.DataFrame()
    summary = numeric.describe().T.round(3)
    ci_lows, ci_highs = [], []
    for col in summary.index:
        series = pd.to_numeric(numeric[col], errors="coerce").dropna()
        n = len(series)
        if n >= 2 and series.std(ddof=1) > 0:
            se = float(stats.sem(series))
            low, high = stats.t.interval(0.95, df=n - 1, loc=float(series.mean()), scale=se)
            ci_lows.append(round(float(low), 3))
            ci_highs.append(round(float(high), 3))
        else:
            ci_lows.append(None)
            ci_highs.append(None)
    summary["ci_95_low"] = ci_lows
    summary["ci_95_high"] = ci_highs
    return summary


def iqr_outliers(df: pd.DataFrame, column: str) -> dict[str, Any]:
    """Detect univariate outliers using the 1.5 * IQR rule."""
    if column not in df.columns:
        raise ValueError(f"Unknown column: {column}")
    series = pd.to_numeric(df[column], errors="coerce").dropna()
    if series.empty:
        return {
            "column": column,
            "count": 0,
            "lower_bound": None,
            "upper_bound": None,
            "rows": [],
            "disclaimer": "IQR flags statistical extremes for review; it does not prove a record is an error.",
        }
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr)
    numeric_col = pd.to_numeric(df[column], errors="coerce")
    mask = numeric_col.lt(lower) | numeric_col.gt(upper)
    rows = df.loc[mask.fillna(False)].copy()
    return {
        "column": column,
        "count": int(len(rows)),
        "lower_bound": lower,
        "upper_bound": upper,
        "rows": rows.head(100).to_dict(orient="records"),
        "disclaimer": "IQR flags statistical extremes for review; it does not prove a record is an error.",
    }


def isolation_forest_outliers(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    contamination: float | str = "auto",
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Detect multivariate anomalies across numeric columns using CPU-friendly IsolationForest.
    Explains why each record was flagged using standardized feature deviations.
    """
    if isinstance(contamination, (int, float)) and not (0.0 < float(contamination) <= 0.5):
        raise ValueError("Contamination must be 'auto' or a float in (0.0, 0.5].")

    numeric_df = df.select_dtypes(include="number")
    selected_cols = [c for c in (columns or list(numeric_df.columns)) if c in numeric_df.columns]
    if not selected_cols:
        return {
            "status": "skipped",
            "reason": "No numeric columns available for Isolation Forest.",
            "features": [],
            "contamination": contamination,
            "count": 0,
            "rate": 0.0,
            "rows": [],
            "disclaimer": "Anomalies indicate statistical rarity, not proof of data errors.",
        }

    work = numeric_df[selected_cols].copy()
    valid_cols = [c for c in selected_cols if work[c].dropna().nunique() > 1]
    if not valid_cols or len(work) < 10:
        return {
            "status": "skipped",
            "reason": "Insufficient non-constant numeric features or rows (<10) for Isolation Forest.",
            "features": valid_cols,
            "contamination": contamination,
            "count": 0,
            "rate": 0.0,
            "rows": [],
            "disclaimer": "Anomalies indicate statistical rarity, not proof of data errors.",
        }

    work = work[valid_cols].fillna(work[valid_cols].median())
    scaled = StandardScaler().fit_transform(work)
    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=random_state,
        n_jobs=1,
    )
    preds = model.fit_predict(scaled)
    scores = model.decision_function(scaled)

    anomaly_mask = preds == -1
    anomalous = df.loc[anomaly_mask].copy()
    anomalous["anomaly_score"] = np.round(scores[anomaly_mask], 4)

    # Explain why each flagged record stood out based on top z-score deviations
    z_matrix = scaled[anomaly_mask]
    explanations: list[str] = []
    for row_z in z_matrix:
        ranked = sorted(
            zip(valid_cols, row_z),
            key=lambda item: abs(item[1]),
            reverse=True,
        )[:2]
        parts = [f"{col} ({val:+.2f} SD)" for col, val in ranked if abs(val) >= 1.0]
        explanations.append(
            ", ".join(parts) if parts else "Unusual multivariate combination across features"
        )
    anomalous["why_flagged"] = explanations
    anomalous = anomalous.sort_values("anomaly_score", ascending=True)

    count = int(anomaly_mask.sum())
    rate = round(count / len(df), 4)
    logger.info(
        "IsolationForest (contamination=%s) detected %d anomalies (rate=%.4f) on %s",
        contamination, count, rate, valid_cols,
    )

    return {
        "status": "completed",
        "reason": None,
        "features": [str(c) for c in valid_cols],
        "contamination": contamination,
        "random_state": random_state,
        "count": count,
        "rate": rate,
        "rows": anomalous.head(100).to_dict(orient="records"),
        "disclaimer": (
            "Isolation Forest flags multivariate statistical rarities, not guaranteed errors. "
            "Review domain context before removing flagged records."
        ),
    }


def detrended_isolation_forest_outliers(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    date_col: str | None = None,
    contamination: float | str = "auto",
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Detrended Isolation Forest for non-stationary time-series data.
    Subtracts a linear trend (or chronological rolling median) from each numeric feature
    before running Isolation Forest so mid-series spikes are not masked by global trends.
    """
    if isinstance(contamination, (int, float)) and not (0.0 < float(contamination) <= 0.5):
        raise ValueError("Contamination must be 'auto' or a float in (0.0, 0.5].")

    numeric_df = df.select_dtypes(include="number")
    selected_cols = [c for c in (columns or list(numeric_df.columns)) if c in numeric_df.columns]
    valid_cols = [c for c in selected_cols if numeric_df[c].dropna().nunique() > 1]
    if not valid_cols or len(df) < 10:
        return {
            "status": "skipped",
            "reason": "Insufficient non-constant numeric features or rows (<10).",
            "features": valid_cols,
            "count": 0,
            "rate": 0.0,
            "rows": [],
        }

    work = df.copy()
    if date_col and date_col in work.columns:
        work["_sort_dt"] = pd.to_datetime(work[date_col], errors="coerce", format="mixed")
        work = work.sort_values("_sort_dt").drop(columns=["_sort_dt"])

    residuals = pd.DataFrame(index=work.index)
    t_idx = np.arange(len(work), dtype=float).reshape(-1, 1)
    for col in valid_cols:
        series = pd.to_numeric(work[col], errors="coerce").fillna(work[col].median())
        # Fit robust rolling median baseline (window=7) + fallback linear trend residual
        rolling_med = series.rolling(window=min(7, len(series)), center=True, min_periods=1).median()
        residuals[col] = series - rolling_med

    scaled = StandardScaler().fit_transform(residuals[valid_cols])
    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=random_state,
        n_jobs=1,
    )
    preds = model.fit_predict(scaled)
    scores = model.decision_function(scaled)

    anomaly_mask = preds == -1
    anomalous = work.loc[anomaly_mask].copy()
    anomalous["anomaly_score"] = np.round(scores[anomaly_mask], 4)
    anomalous["method"] = "detrended_rolling_residual_iforest"
    anomalous = anomalous.sort_values("anomaly_score", ascending=True)

    count = int(anomaly_mask.sum())
    return {
        "status": "completed",
        "method": "detrended_isolation_forest",
        "features": [str(c) for c in valid_cols],
        "contamination": contamination,
        "random_state": random_state,
        "count": count,
        "rate": round(count / len(work), 4),
        "rows": anomalous.head(100).to_dict(orient="records"),
    }


def correlation_analysis(df: pd.DataFrame, method: str = "pearson") -> dict[str, Any]:
    """Compute correlation matrix, p-values, and 95% Fisher-z CIs for numeric column pairs."""
    numeric = df.select_dtypes(include="number")
    cols = [str(c) for c in numeric.columns]
    if len(cols) < 2:
        return {"matrix": pd.DataFrame(), "pairs": [], "method": method}

    matrix = numeric.corr(method=method, numeric_only=True).round(3)
    pairs: list[dict[str, Any]] = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c1, c2 = cols[i], cols[j]
            paired = numeric[[c1, c2]].dropna()
            n = len(paired)
            if n < 3 or paired[c1].nunique() <= 1 or paired[c2].nunique() <= 1:
                continue
            if method == "spearman":
                r_val, p_val = stats.spearmanr(paired[c1], paired[c2])
            else:
                r_val, p_val = stats.pearsonr(paired[c1], paired[c2])

            ci_low, ci_high = None, None
            if n >= 4 and abs(float(r_val)) < 0.9999:
                z = np.arctanh(float(r_val))
                se = 1.0 / np.sqrt(n - 3)
                ci_low = round(float(np.tanh(z - 1.96 * se)), 4)
                ci_high = round(float(np.tanh(z + 1.96 * se)), 4)

            pairs.append({
                "col_a": c1,
                "col_b": c2,
                "correlation": round(float(r_val), 4),
                "p_value": round(float(p_val), 6),
                "ci_95_low": ci_low,
                "ci_95_high": ci_high,
                "sample_size": int(n),
                "significant": bool(p_val < 0.05 and abs(r_val) >= 0.3),
                "interpretation": (
                    f"{'Significant' if p_val < 0.05 else 'Non-significant'} {method} association "
                    f"(r={r_val:.3f}, p={p_val:.4f}, n={n}); correlation does not imply causation."
                ),
            })

    pairs.sort(key=lambda item: abs(item["correlation"]), reverse=True)
    return {"matrix": matrix, "pairs": pairs, "method": method}


def run_statistical_tests(df: pd.DataFrame) -> dict[str, Any]:
    """Run assumption-checked normality, group-comparison, and categorical independence tests."""
    numeric_cols = [str(c) for c in df.select_dtypes(include="number").columns]
    cat_cols = [
        str(c)
        for c in df.select_dtypes(include=["object", "string", "category"]).columns
        if 2 <= df[c].nunique(dropna=True) <= 8
    ]

    normality_results: list[dict[str, Any]] = []
    normal_map: dict[str, bool] = {}

    for col in numeric_cols[:10]:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        n = len(series)
        if n < 3 or series.nunique() <= 1:
            continue
        if n <= 5000:
            stat_val, p_val = stats.shapiro(series)
            test_name = "Shapiro-Wilk"
        else:
            stat_val, p_val = stats.normaltest(series)
            test_name = "D'Agostino-Pearson"
        is_normal = bool(p_val >= 0.05)
        normal_map[col] = is_normal
        normality_results.append({
            "column": col,
            "test": test_name,
            "statistic": round(float(stat_val), 4),
            "p_value": round(float(p_val), 6),
            "sample_size": int(n),
            "normal_at_0_05": is_normal,
            "interpretation": (
                f"Data is consistent with normality (p={p_val:.4f} >= 0.05)."
                if is_normal
                else f"Data departs from normality (p={p_val:.4f} < 0.05); prefer non-parametric tests."
            ),
        })

    group_tests: list[dict[str, Any]] = []
    if cat_cols and numeric_cols:
        for cat in cat_cols[:2]:
            for num in numeric_cols[:3]:
                sub = df[[cat, num]].dropna()
                groups = [
                    pd.to_numeric(grp[num], errors="coerce").dropna().values
                    for _, grp in sub.groupby(cat, observed=True)
                ]
                groups = [g for g in groups if len(g) >= 3]
                if len(groups) < 2:
                    continue

                ci_diff_low, ci_diff_high = None, None
                if len(groups) == 2 and normal_map.get(num, False):
                    stat_val, p_val = stats.ttest_ind(groups[0], groups[1], equal_var=False)
                    test_name = "Welch's t-test"
                    m1, m2 = float(np.mean(groups[0])), float(np.mean(groups[1]))
                    se_diff = float(np.sqrt(np.var(groups[0], ddof=1) / len(groups[0]) + np.var(groups[1], ddof=1) / len(groups[1])))
                    if se_diff > 0:
                        ci_diff_low = round((m1 - m2) - 1.96 * se_diff, 3)
                        ci_diff_high = round((m1 - m2) + 1.96 * se_diff, 3)
                elif len(groups) == 2:
                    stat_val, p_val = stats.mannwhitneyu(groups[0], groups[1], alternative="two-sided")
                    test_name = "Mann-Whitney U"
                elif normal_map.get(num, False):
                    stat_val, p_val = stats.f_oneway(*groups)
                    test_name = "One-way ANOVA"
                else:
                    stat_val, p_val = stats.kruskal(*groups)
                    test_name = "Kruskal-Wallis"

                if np.isnan(stat_val) or np.isnan(p_val):
                    continue
                sig = bool(p_val < 0.05)
                group_tests.append({
                    "categorical": cat,
                    "numeric": num,
                    "test": test_name,
                    "groups": len(groups),
                    "total_n": int(sum(len(g) for g in groups)),
                    "statistic": round(float(stat_val), 4),
                    "p_value": round(float(p_val), 6),
                    "diff_ci_95_low": ci_diff_low,
                    "diff_ci_95_high": ci_diff_high,
                    "significant_at_0_05": sig,
                    "interpretation": (
                        f"Statistically significant difference in '{num}' across '{cat}' groups (p={p_val:.4f})."
                        if sig
                        else f"No statistically significant difference in '{num}' across '{cat}' groups (p={p_val:.4f})."
                    ),
                })

    categorical_tests: list[dict[str, Any]] = []
    if len(cat_cols) >= 2:
        c1, c2 = cat_cols[0], cat_cols[1]
        contingency = pd.crosstab(df[c1], df[c2])
        if contingency.shape[0] >= 2 and contingency.shape[1] >= 2:
            chi2, p_val, dof, expected = stats.chi2_contingency(contingency)
            min_expected = float(expected.min())
            assumption_met = bool(min_expected >= 5.0)
            categorical_tests.append({
                "col_a": c1,
                "col_b": c2,
                "test": "Chi-square independence",
                "statistic": round(float(chi2), 4),
                "p_value": round(float(p_val), 6),
                "dof": int(dof),
                "min_expected_freq": round(min_expected, 2),
                "assumption_met": assumption_met,
                "significant_at_0_05": bool(p_val < 0.05),
                "warning": (
                    None
                    if assumption_met
                    else f"Expected cell count ({min_expected:.2f} < 5) violates Chi-square assumption; interpret cautiously."
                ),
            })

    return {
        "normality": normality_results,
        "group_comparisons": group_tests,
        "categorical_associations": categorical_tests,
    }


def _calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | None]:
    err = y_true - y_pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    nonzero = y_true != 0
    mape = round(float(np.mean(np.abs(err[nonzero] / y_true[nonzero])) * 100.0), 2) if np.any(nonzero) else None
    return {"mae": round(mae, 3), "rmse": round(rmse, 3), "mape": mape}


def forecast_series(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    horizon: int = 6,
) -> dict[str, Any]:
    """
    Forecast a numeric target over time using strict chronological train/test split
    (no data leakage) and baseline comparisons (Naive, Moving Average, Ridge Linear Trend).
    """
    if date_col not in df.columns or target_col not in df.columns:
        raise ValueError("Invalid date or target column for forecasting.")

    work = df[[date_col, target_col]].copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce", format="mixed")
    work[target_col] = pd.to_numeric(work[target_col], errors="coerce")
    work = work.dropna(subset=[date_col, target_col]).sort_values(date_col)

    if work.empty:
        return {
            "status": "skipped",
            "reason": "No valid datetime/numeric pairs after parsing.",
            "limitations": ["Provide a valid datetime column and numeric target column."],
        }

    ts = work.groupby(date_col, as_index=False)[target_col].sum().sort_values(date_col).reset_index(drop=True)
    n = len(ts)
    if n < 8:
        return {
            "status": "skipped",
            "reason": f"At least 8 distinct chronological periods are required for holdout evaluation (found {n}).",
            "limitations": [
                f"Insufficient history ({n} periods). Collect at least 8 periods before trusting trend forecasts."
            ],
        }

    horizon = max(1, min(int(horizon), 24))
    test_size = max(2, min(horizon, n // 4))
    train_df = ts.iloc[:-test_size].copy()
    test_df = ts.iloc[-test_size:].copy()

    # Strictly fit only on train_df to prevent data leakage
    y_train = train_df[target_col].to_numpy(dtype=float)
    y_test = test_df[target_col].to_numpy(dtype=float)

    naive_pred = np.full(test_size, y_train[-1], dtype=float)
    ma_window = min(3, len(y_train))
    ma_pred = np.full(test_size, float(np.mean(y_train[-ma_window:])), dtype=float)

    t_train = np.arange(len(y_train), dtype=float).reshape(-1, 1)
    t_test = np.arange(len(y_train), len(y_train) + test_size, dtype=float).reshape(-1, 1)
    ridge = Ridge(alpha=1.0).fit(t_train, y_train)
    ridge_pred = ridge.predict(t_test)

    evaluations = {
        "naive_baseline": _calc_metrics(y_test, naive_pred),
        "moving_average_baseline": _calc_metrics(y_test, ma_pred),
        "ridge_linear_trend": _calc_metrics(y_test, ridge_pred),
    }

    best_model = min(evaluations, key=lambda name: evaluations[name]["rmse"])
    best_baseline_rmse = min(
        evaluations["naive_baseline"]["rmse"],
        evaluations["moving_average_baseline"]["rmse"],
    )
    beats_baseline = bool(evaluations["ridge_linear_trend"]["rmse"] < best_baseline_rmse)

    # Refit selected model on full history for future horizon
    y_full = ts[target_col].to_numpy(dtype=float)
    diffs = ts[date_col].diff().dropna()
    step = diffs.median() if not diffs.empty else pd.Timedelta(days=1)
    if pd.isna(step) or step <= pd.Timedelta(0):
        step = pd.Timedelta(days=1)

    last_date = ts[date_col].iloc[-1]
    future_dates = [last_date + step * (i + 1) for i in range(horizon)]

    if best_model == "naive_baseline":
        future_preds = np.full(horizon, y_full[-1], dtype=float)
        resid_std = float(np.std(y_test - naive_pred, ddof=1)) if test_size > 1 else float(np.std(y_full))
    elif best_model == "moving_average_baseline":
        future_preds = np.full(horizon, float(np.mean(y_full[-min(3, n):])), dtype=float)
        resid_std = float(np.std(y_test - ma_pred, ddof=1)) if test_size > 1 else float(np.std(y_full))
    else:
        t_full = np.arange(n, dtype=float).reshape(-1, 1)
        t_future = np.arange(n, n + horizon, dtype=float).reshape(-1, 1)
        full_ridge = Ridge(alpha=1.0).fit(t_full, y_full)
        future_preds = full_ridge.predict(t_future)
        resid_std = float(np.std(y_test - ridge_pred, ddof=1)) if test_size > 1 else float(np.std(y_full - full_ridge.predict(t_full)))

    margin = 1.96 * resid_std if np.isfinite(resid_std) else 0.0

    forecast_rows = [
        {
            "date": d.strftime("%Y-%m-%d"),
            "forecast": round(float(val), 3),
            "lower_95": round(float(val - margin), 3),
            "upper_95": round(float(val + margin), 3),
        }
        for d, val in zip(future_dates, future_preds)
    ]

    history_rows = [
        {"date": d.strftime("%Y-%m-%d"), "actual": round(float(val), 3)}
        for d, val in zip(ts[date_col], y_full)
    ]

    test_comparison = [
        {
            "date": d.strftime("%Y-%m-%d"),
            "actual": round(float(act), 3),
            "naive": round(float(nv), 3),
            "moving_avg": round(float(ma), 3),
            "ridge_trend": round(float(rg), 3),
        }
        for d, act, nv, ma, rg in zip(test_df[date_col], y_test, naive_pred, ma_pred, ridge_pred)
    ]

    limitations = [
        f"Evaluated on {test_size} chronological holdout period(s) using {len(train_df)} training period(s) without leakage.",
        "Univariate trend/baseline models do not account for external promotions, macro shifts, or abrupt regime changes.",
    ]
    if n < 24:
        limitations.append(
            f"Short history ({n} periods): prediction intervals may underestimate long-horizon uncertainty."
        )

    logger.info(
        "Forecasted '%s' over '%s': n=%d, test=%d, best_model=%s (RMSE=%.3f)",
        target_col, date_col, n, test_size, best_model, evaluations[best_model]["rmse"],
    )

    return {
        "status": "completed",
        "date_col": date_col,
        "target_col": target_col,
        "total_periods": n,
        "train_periods": len(train_df),
        "test_periods": test_size,
        "horizon": horizon,
        "selected_model": best_model,
        "beats_baseline": beats_baseline,
        "evaluations": evaluations,
        "history": history_rows,
        "test_comparison": test_comparison,
        "forecast": forecast_rows,
        "limitations": limitations,
    }
