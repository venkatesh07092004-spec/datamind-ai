"""Reproducible Phase 9 benchmark suite for DataMind AI."""

import ctypes
import json
import os
from pathlib import Path
import platform
import sys
import time
import tracemalloc
from typing import Any

import pandas as pd
import plotly.graph_objects as go

from agents.profiler import cleaning_suggestions, profile_dataset
from agents.reviewer import review_findings
from agents.sql_agent import generate_and_validate_sql
from agents.statistician import (
    correlation_analysis,
    detrended_isolation_forest_outliers,
    forecast_series,
    iqr_outliers,
    isolation_forest_outliers,
    run_statistical_tests,
)
from core.cleaning import clean_dataframe
from core.ingestion import detect_datetime_columns, load_dataframe, validate_dataframe
from core.sql_engine import execute_readonly_query
from experiments.generate_datasets import generate_all_datasets
from orchestration.graph import run_analysis_workflow


def _get_hardware_specs() -> dict[str, Any]:
    ram_gb = None
    try:
        class _MEMSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        stat = _MEMSTATUSEX()
        stat.dwLength = ctypes.sizeof(stat)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        ram_gb = round(stat.ullTotalPhys / (1024 ** 3), 2)
    except Exception:
        ram_gb = None

    return {
        "os": platform.platform(),
        "python_version": sys.version.split()[0],
        "processor": platform.processor() or platform.machine(),
        "cpu_cores": os.cpu_count(),
        "total_ram_gb": ram_gb,
        "gpu_required": False,
    }


def _run_equivalent_baseline(df: pd.DataFrame, question: str = "how many rows") -> dict[str, Any]:
    """
    Fair single-process analytical baseline performing equivalent computations
    (profiling, DuckDB SQL, IQR, Isolation Forest, correlations, statistical tests, forecast)
    without LangGraph orchestration, chart building, Reviewer verification, or PDF generation.
    """
    tracemalloc.start()
    t0 = time.perf_counter()

    _ = profile_dataset(df)
    res = generate_and_validate_sql(question, df)
    _ = execute_readonly_query(df, res["sql"])
    num_cols = list(df.select_dtypes(include="number").columns)
    if num_cols:
        _ = iqr_outliers(df, num_cols[0])
        _ = isolation_forest_outliers(df)
        _ = correlation_analysis(df)
        _ = run_statistical_tests(df)
        dt_cols = detect_datetime_columns(df)
        if dt_cols:
            _ = forecast_series(df, dt_cols[0], num_cols[0], horizon=6)

    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "elapsed_ms": elapsed_ms,
        "peak_memory_mb": round(peak_bytes / (1024 ** 2), 3),
    }


def _evaluate_sql_accuracy(df: pd.DataFrame, queries: list[tuple[str, Any]]) -> dict[str, Any]:
    passed = 0
    details = []
    for question, check_fn in queries:
        try:
            res = generate_and_validate_sql(question, df)
            out = execute_readonly_query(df, res["sql"])
            ok = bool(check_fn(out))
            if ok:
                passed += 1
            details.append({"question": question, "sql": res["sql"], "passed": ok})
        except Exception as exc:
            details.append({"question": question, "sql": None, "passed": False, "error": str(exc)})
    return {
        "total": len(queries),
        "passed": passed,
        "accuracy_pct": round((passed / max(len(queries), 1)) * 100.0, 1),
        "details": details,
    }


def _score_anomaly_predictions(df: pd.DataFrame, feature_cols: list[str], res: dict[str, Any]) -> dict[str, Any]:
    gt = df["ground_truth_anomaly"].to_numpy(dtype=int)
    flagged_indices = set()
    flagged_rows = pd.DataFrame(res.get("rows", []))
    if not flagged_rows.empty:
        flagged_keys = set(tuple(row[c] for c in feature_cols) for _, row in flagged_rows.iterrows())
        for idx, row in df.iterrows():
            if tuple(row[c] for c in feature_cols) in flagged_keys:
                flagged_indices.add(idx)
    pred = [1 if i in flagged_indices else 0 for i in range(len(df))]
    tp = sum(1 for g, p in zip(gt, pred) if g == 1 and p == 1)
    fp = sum(1 for g, p in zip(gt, pred) if g == 0 and p == 1)
    fn = sum(1 for g, p in zip(gt, pred) if g == 1 and p == 0)
    precision = round(tp / max(tp + fp, 1), 3)
    recall = round(tp / max(tp + fn, 1), 3)
    f1 = round((2 * precision * recall) / max(precision + recall, 1e-9), 3)
    return {
        "ground_truth_count": int(gt.sum()),
        "flagged_count": int(sum(pred)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _evaluate_anomaly_comparison(
    df: pd.DataFrame, feature_cols: list[str], date_col: str
) -> dict[str, Any] | None:
    if "ground_truth_anomaly" not in df.columns:
        return None
    clean_sub = df.dropna(subset=feature_cols).reset_index(drop=True)
    raw_iso = isolation_forest_outliers(clean_sub, columns=feature_cols, contamination=0.05, random_state=42)
    detrended_iso = detrended_isolation_forest_outliers(
        clean_sub, columns=feature_cols, date_col=date_col, contamination=0.05, random_state=42
    )
    return {
        "standard_isolation_forest": _score_anomaly_predictions(clean_sub, feature_cols, raw_iso),
        "detrended_isolation_forest": _score_anomaly_predictions(clean_sub, feature_cols, detrended_iso),
    }


def _evaluate_insight_reviewer_30(df: pd.DataFrame, profile: dict[str, Any]) -> dict[str, Any]:
    """Evaluate Insight Reviewer across 32 diverse test claims."""
    rows_n = profile["rows"]
    miss_n = profile["missing_cells"]
    cols_n = profile["columns"]
    dup_n = profile["duplicate_rows"]

    test_cases: list[dict[str, Any]] = []
    # 10 verified facts (expected: approved)
    for idx in range(5):
        test_cases.append({
            "category": "verified_fact",
            "expected": "approved",
            "finding": {
                "claim": f"[Fact {idx}] Dataset has {rows_n} rows.",
                "evidence": "profiler.profile_dataset",
                "metric": "profile.rows",
                "expected_value": rows_n,
                "sample_size": rows_n,
            },
        })
    for idx in range(3):
        test_cases.append({
            "category": "verified_fact",
            "expected": "approved",
            "finding": {
                "claim": f"[Fact {idx}] Missing cells: {miss_n}.",
                "evidence": "profiler.profile_dataset",
                "metric": "profile.missing_cells",
                "expected_value": miss_n,
                "sample_size": rows_n,
            },
        })
    test_cases.extend([
        {
            "category": "verified_fact",
            "expected": "approved",
            "finding": {
                "claim": f"Dataset has {cols_n} columns.",
                "evidence": "profiler.profile_dataset",
                "metric": "profile.columns",
                "expected_value": cols_n,
                "sample_size": rows_n,
            },
        },
        {
            "category": "verified_fact",
            "expected": "approved",
            "finding": {
                "claim": f"Duplicate rows: {dup_n}.",
                "evidence": "profiler.profile_dataset",
                "metric": "profile.duplicate_rows",
                "expected_value": dup_n,
                "sample_size": rows_n,
            },
        },
    ])

    # 8 numerical mismatches (expected: rejected)
    for bad_val in [999, -1, rows_n + 50, rows_n * 2, 12345, 0, cols_n + 10, miss_n + 99]:
        metric = "profile.rows" if bad_val not in {cols_n + 10, miss_n + 99} else (
            "profile.columns" if bad_val == cols_n + 10 else "profile.missing_cells"
        )
        test_cases.append({
            "category": "numerical_mismatch",
            "expected": "rejected",
            "finding": {
                "claim": f"Mismatched claim {metric}={bad_val}.",
                "evidence": "profiler.profile_dataset",
                "metric": metric,
                "expected_value": bad_val,
                "sample_size": rows_n,
            },
        })

    # 6 unsupported causal claims (expected: rejected)
    for verb in ["causes", "caused", "proves", "proved", "guarantees", "causes direct"]:
        test_cases.append({
            "category": "causal_overreach",
            "expected": "rejected",
            "finding": {
                "claim": f"Variable A {verb} Variable B to increase.",
                "evidence": "correlation_analysis",
                "sample_size": rows_n,
            },
        })

    # 4 insufficient sample-size claims (expected: rejected)
    for n_small in [0, 1, 2, 4]:
        test_cases.append({
            "category": "insufficient_sample",
            "expected": "rejected",
            "finding": {
                "claim": f"Significant difference with n={n_small}.",
                "evidence": "statistical_test",
                "sample_size": n_small,
            },
        })

    # 4 missing evidence / malformed claims (expected: rejected)
    test_cases.extend([
        {"category": "missing_evidence", "expected": "rejected", "finding": {"claim": "No evidence 1."}},
        {"category": "missing_evidence", "expected": "rejected", "finding": {"claim": "No evidence 2.", "evidence": ""}},
        {"category": "missing_evidence", "expected": "rejected", "finding": {"claim": "", "evidence": "ok"}},
        {"category": "missing_evidence", "expected": "rejected", "finding": {"claim": "Unknown metric.", "evidence": "ok", "metric": "profile.nonexistent", "expected_value": 1}},
    ])

    findings = [tc["finding"] for tc in test_cases]
    res = review_findings(df, findings, computed_context={"profile": profile})
    approved_claims = {f["claim"] for f in res["approved"]}

    correct = 0
    by_category: dict[str, dict[str, int]] = {}
    for tc in test_cases:
        cat = tc["category"]
        by_category.setdefault(cat, {"total": 0, "correct": 0})
        by_category[cat]["total"] += 1
        actual = "approved" if tc["finding"]["claim"] in approved_claims else "rejected"
        if actual == tc["expected"]:
            correct += 1
            by_category[cat]["correct"] += 1

    return {
        "total_cases": len(test_cases),
        "correct_decisions": correct,
        "accuracy_pct": round((correct / len(test_cases)) * 100.0, 1),
        "by_category": by_category,
    }


def _evaluate_insight_reviewer_unseen(df: pd.DataFrame, profile: dict[str, Any]) -> dict[str, Any]:
    """Separate 16-case unseen adversarial & boundary suite for Insight Reviewer."""
    rows_n = profile["rows"]
    cases: list[tuple[str, Any, str]] = [
        # (label, finding, expected_status)
        ("within_tolerance", {"claim": "Row count within tolerance.", "evidence": "profiler", "metric": "profile.rows", "expected_value": rows_n + 0.005, "tolerance": 0.01, "sample_size": rows_n}, "approved"),
        ("outside_tolerance", {"claim": "Row count outside tolerance.", "evidence": "profiler", "metric": "profile.rows", "expected_value": rows_n + 0.05, "tolerance": 0.01, "sample_size": rows_n}, "rejected"),
        ("sample_boundary_pass", {"claim": "Boundary n=5.", "evidence": "stats", "sample_size": 5}, "approved"),
        ("sample_boundary_fail", {"claim": "Boundary n=4.", "evidence": "stats", "sample_size": 4}, "rejected"),
        ("experimental_causal_pass", {"claim": "Treatment causes lift (RCT).", "evidence": "rct", "experimental_evidence": True, "sample_size": 20}, "approved"),
        ("observational_causal_fail", {"claim": "Metric proves revenue growth.", "evidence": "corr", "experimental_evidence": False, "sample_size": 20}, "rejected"),
        ("non_dict_string", "Unstructured string claim", "rejected"),
        ("non_dict_int", 42, "rejected"),
        ("whitespace_only_claim", {"claim": "   ", "evidence": "profiler"}, "rejected"),
        ("none_evidence", {"claim": "Valid text.", "evidence": None}, "rejected"),
        ("empty_list_evidence", {"claim": "Valid text.", "evidence": []}, "rejected"),
        ("missing_nested_metric", {"claim": "Bad path.", "evidence": "profiler", "metric": "profile.unknown_key", "expected_value": 1}, "rejected"),
        ("exact_quality_score", {"claim": f"Quality {profile['quality_score']}.", "evidence": "profiler", "metric": "profile.quality_score", "expected_value": profile["quality_score"], "sample_size": rows_n}, "approved"),
        ("wrong_quality_score", {"claim": "Quality 1000.", "evidence": "profiler", "metric": "profile.quality_score", "expected_value": 1000.0, "sample_size": rows_n}, "rejected"),
        ("exact_missing_pct", {"claim": f"Missing pct {profile['missing_pct']}.", "evidence": "profiler", "metric": "profile.missing_pct", "expected_value": profile["missing_pct"], "sample_size": rows_n}, "approved"),
        ("negative_sample_size", {"claim": "Negative n.", "evidence": "stats", "sample_size": -5}, "rejected"),
    ]

    findings = [c[1] for c in cases]
    res = review_findings(df, findings, computed_context={"profile": profile})
    approved_claims = {f["claim"] for f in res["approved"] if isinstance(f, dict)}

    correct = 0
    for _, finding, expected in cases:
        if isinstance(finding, dict) and str(finding.get("claim", "")).strip():
            actual = "approved" if finding["claim"] in approved_claims else "rejected"
        else:
            actual = "rejected"
        if actual == expected:
            correct += 1

    return {
        "total_cases": len(cases),
        "correct_decisions": correct,
        "accuracy_pct": round((correct / len(cases)) * 100.0, 1),
        "approved_count": len(res["approved"]),
        "rejected_count": len(res["rejected"]),
    }


def _generate_benchmark_charts(results: dict[str, Any], out_dir: Path) -> list[str]:
    datasets = list(results["datasets"].keys())
    wf_times = [results["datasets"][d]["multi_agent_workflow_ms"] for d in datasets]
    base_times = [results["datasets"][d]["equivalent_baseline"]["elapsed_ms"] for d in datasets]

    fig_time = go.Figure(data=[
        go.Bar(name="LangGraph Multi-Agent Workflow", x=datasets, y=wf_times, marker_color="#38BDF8"),
        go.Bar(name="Equivalent Single-Process Baseline", x=datasets, y=base_times, marker_color="#A78BFA"),
    ])
    fig_time.update_layout(
        template="plotly_dark",
        barmode="group",
        title="Execution Time: Multi-Agent Workflow vs. Equivalent Baseline",
        xaxis_title="Dataset",
        yaxis_title="Execution Time (ms)",
    )
    p1 = out_dir / "execution_time_comparison.html"
    fig_time.write_html(str(p1))

    sql_unclean = [results["datasets"][d]["sql_accuracy_uncleaned"]["accuracy_pct"] for d in datasets]
    sql_clean = [results["datasets"][d]["sql_accuracy_cleaned"]["accuracy_pct"] for d in datasets]
    fig_sql = go.Figure(data=[
        go.Bar(name="Uncleaned Dataset SQL Acc (%)", x=datasets, y=sql_unclean, marker_color="#FBBF24"),
        go.Bar(name="Cleaned Dataset SQL Acc (%)", x=datasets, y=sql_clean, marker_color="#34D399"),
    ])
    fig_sql.update_layout(
        template="plotly_dark",
        barmode="group",
        title="SQL Execution Accuracy: Uncleaned vs. Cleaned Datasets",
        xaxis_title="Dataset",
        yaxis_title="Accuracy (%)",
        yaxis_range=[0, 110],
    )
    p2 = out_dir / "sql_and_reviewer_accuracy.html"
    fig_sql.write_html(str(p2))

    fc_datasets = [d for d in datasets if results["datasets"][d]["forecasting"] is not None]
    naive_rmse = [results["datasets"][d]["forecasting"]["evaluations"]["naive_baseline"]["rmse"] for d in fc_datasets]
    ma_rmse = [results["datasets"][d]["forecasting"]["evaluations"]["moving_average_baseline"]["rmse"] for d in fc_datasets]
    ridge_rmse = [results["datasets"][d]["forecasting"]["evaluations"]["ridge_linear_trend"]["rmse"] for d in fc_datasets]
    fig_fc = go.Figure(data=[
        go.Bar(name="Naive Baseline RMSE", x=fc_datasets, y=naive_rmse, marker_color="#94A3B8"),
        go.Bar(name="Moving Average RMSE", x=fc_datasets, y=ma_rmse, marker_color="#38BDF8"),
        go.Bar(name="Ridge Linear Trend RMSE", x=fc_datasets, y=ridge_rmse, marker_color="#A78BFA"),
    ])
    fig_fc.update_layout(
        template="plotly_dark",
        barmode="group",
        title="Chronological Holdout Forecasting RMSE Comparison",
        xaxis_title="Dataset",
        yaxis_title="Holdout RMSE (lower is better)",
    )
    p3 = out_dir / "forecast_rmse_comparison.html"
    fig_fc.write_html(str(p3))
    return [str(p1), str(p2), str(p3)]


def run_benchmarks() -> dict[str, Any]:
    paths = generate_all_datasets()
    results: dict[str, Any] = {
        "hardware": _get_hardware_specs(),
        "datasets": {},
    }

    # 22 predefined NL-to-SQL questions across the 5 datasets (filtering, grouping, aggregation, sorting, top-N)
    sql_suites_uncleaned: dict[str, list[tuple[str, Any]]] = {
        "retail_sales": [
            ("how many rows", lambda out: int(out.iloc[0, 0]) == 120),
            ("total revenue by region", lambda out: len(out) == 4 and "total_revenue" in out.columns),
            ("average revenue by product", lambda out: len(out) == 4),
            ("top 2 products by revenue", lambda out: len(out) == 2),
            ("bottom 2 regions by revenue", lambda out: len(out) == 2),
            ("show records where revenue > 1500", lambda out: (out["revenue"] > 1500).all()),
        ],
        "app_catalog": [
            ("how many rows", lambda out: int(out.iloc[0, 0]) == 82),
            ("average Rating by Category", lambda out: len(out) == 4),
            ("top 3 Category by Rating", lambda out: len(out) == 3),
            ("show records in Tools", lambda out: len(out) >= 1 and (out["Category"].str.strip() == "Tools").all()),
        ],
        "healthcare_vitals": [
            ("how many rows", lambda out: int(out.iloc[0, 0]) == 100),
            ("average systolic_bp by department", lambda out: len(out) == 3),
            ("maximum heart_rate by department", lambda out: len(out) == 3),
            ("show records where systolic_bp > 150", lambda out: (out["systolic_bp"] > 150).all()),
        ],
        "ecommerce_traffic": [
            ("how many rows", lambda out: int(out.iloc[0, 0]) == 90),
            ("total revenue by channel", lambda out: len(out) == 4),
            ("average sessions by channel", lambda out: len(out) == 4),
            ("top 2 channel by conversions", lambda out: len(out) == 2),
            ("monthly revenue trend", lambda out: len(out) >= 2),
        ],
        "hr_survey": [
            ("how many rows", lambda out: int(out.iloc[0, 0]) == 60),
            ("unique department", lambda out: len(out) == 4),
            ("unique satisfaction", lambda out: len(out) == 3),
            ("show records in Engineering", lambda out: (out["department"] == "Engineering").all()),
        ],
    }

    # Cleaned row count for app_catalog is 80 (after 2 duplicates removed)
    sql_suites_cleaned = dict(sql_suites_uncleaned)
    sql_suites_cleaned["app_catalog"] = [
        ("how many rows", lambda out: int(out.iloc[0, 0]) == 80),
        ("average Rating by Category", lambda out: len(out) == 4),
        ("top 3 Category by Rating", lambda out: len(out) == 3),
        ("show records in Tools", lambda out: len(out) >= 1 and (out["Category"].str.strip() == "Tools").all()),
    ]

    anom_configs: dict[str, tuple[list[str], str]] = {
        "retail_sales": (["quantity", "revenue"], "date"),
        "healthcare_vitals": (["systolic_bp", "heart_rate", "bmi"], "visit_date"),
    }

    total_sql_q = 0
    total_sql_pass_clean = 0
    total_sql_pass_unclean = 0

    for name, path in paths.items():
        t_ing = time.perf_counter()
        with open(path, "rb") as f:
            df = load_dataframe(f, path.name)
        validate_dataframe(df)
        ingestion_ms = round((time.perf_counter() - t_ing) * 1000.0, 2)

        analysis_df = df.drop(columns=["ground_truth_anomaly"], errors="ignore")

        t_prof = time.perf_counter()
        prof = profile_dataset(analysis_df)
        _ = cleaning_suggestions(analysis_df)
        profiling_ms = round((time.perf_counter() - t_prof) * 1000.0, 2)

        cleaned_df, _ = clean_dataframe(analysis_df, trim_text=True, fill_numeric_median=True, drop_duplicates=True)

        tracemalloc.start()
        t0 = time.perf_counter()
        wf = run_analysis_workflow(cleaned_df, question="how many rows", horizon=6)
        wf_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        baseline = _run_equivalent_baseline(cleaned_df, question="how many rows")
        sql_unclean = _evaluate_sql_accuracy(analysis_df, sql_suites_uncleaned[name])
        sql_clean = _evaluate_sql_accuracy(cleaned_df, sql_suites_cleaned[name])
        total_sql_q += sql_clean["total"]
        total_sql_pass_unclean += sql_unclean["passed"]
        total_sql_pass_clean += sql_clean["passed"]

        rev_eval = _evaluate_insight_reviewer_30(analysis_df, prof)
        rev_unseen = _evaluate_insight_reviewer_unseen(analysis_df, prof)
        anom_eval = (
            _evaluate_anomaly_comparison(df, anom_configs[name][0], anom_configs[name][1])
            if name in anom_configs
            else None
        )

        fcst = wf.get("forecast_result", {})
        fcst_summary = None
        if fcst.get("status") == "completed":
            best = fcst["selected_model"]
            fcst_summary = {
                "target_col": fcst["target_col"],
                "selected_model": best,
                "beats_baseline": fcst["beats_baseline"],
                "evaluations": fcst["evaluations"],
            }

        results["datasets"][name] = {
            "rows": int(df.shape[0]),
            "columns": int(analysis_df.shape[1]),
            "ingestion_ms": ingestion_ms,
            "profiling_ms": profiling_ms,
            "multi_agent_workflow_ms": wf_ms,
            "multi_agent_peak_memory_mb": round(peak_bytes / (1024 ** 2), 3),
            "equivalent_baseline": baseline,
            "sql_accuracy_uncleaned": sql_unclean,
            "sql_accuracy_cleaned": sql_clean,
            "reviewer_evaluation": rev_eval,
            "reviewer_unseen_evaluation": rev_unseen,
            "anomaly_comparison": anom_eval,
            "forecasting": fcst_summary,
        }

    results["summary"] = {
        "total_sql_questions": total_sql_q,
        "sql_uncleaned_accuracy_pct": round((total_sql_pass_unclean / max(total_sql_q, 1)) * 100.0, 1),
        "sql_cleaned_accuracy_pct": round((total_sql_pass_clean / max(total_sql_q, 1)) * 100.0, 1),
    }

    out_dir = Path("experiments/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    results["generated_charts"] = _generate_benchmark_charts(results, out_dir)
    (out_dir / "benchmark_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    lines = [
        "# DataMind AI — Phase 9 Reproducible Benchmark Results",
        "",
        f"- **OS**: {results['hardware']['os']} | **Python**: {results['hardware']['python_version']}",
        f"- **CPU Cores**: {results['hardware']['cpu_cores']} | **RAM (GB)**: {results['hardware']['total_ram_gb']}",
        f"- **Total SQL Questions Evaluated**: {total_sql_q} (Uncleaned: {results['summary']['sql_uncleaned_accuracy_pct']}%, Cleaned: {results['summary']['sql_cleaned_accuracy_pct']}%)",
        "",
        "| Dataset | Rows x Cols | Ingest (ms) | Profile (ms) | Multi-Agent (ms) | Peak Mem (MB) | Equiv Baseline (ms) | Base Mem (MB) | SQL Unclean | SQL Clean | Reviewer (32) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, d in results["datasets"].items():
        lines.append(
            f"| {name} | {d['rows']} x {d['columns']} | {d['ingestion_ms']} | {d['profiling_ms']} | "
            f"{d['multi_agent_workflow_ms']} | {d['multi_agent_peak_memory_mb']} | "
            f"{d['equivalent_baseline']['elapsed_ms']} | {d['equivalent_baseline']['peak_memory_mb']} | "
            f"{d['sql_accuracy_uncleaned']['accuracy_pct']}% | {d['sql_accuracy_cleaned']['accuracy_pct']}% | "
            f"{d['reviewer_evaluation']['accuracy_pct']}% |"
        )
    (out_dir / "benchmark_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return results


if __name__ == "__main__":
    run_benchmarks()
    print(Path("experiments/results/benchmark_summary.md").read_text(encoding="utf-8"))
