"""Insight Reviewer Agent: verifies analytical claims against computed evidence, sample size, and assumptions."""

import math
import re
from typing import Any

import pandas as pd

from core.logging_config import get_logger

logger = get_logger(__name__)

_CAUSAL_WORDS = re.compile(r"\b(causes|caused|proves|proved|guarantees)\b", re.IGNORECASE)


def _resolve_metric(metric_path: str, context: dict[str, Any]) -> Any:
    """Resolve a dotted metric path like 'profile.rows' from computed_context."""
    current: Any = context
    for part in metric_path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


def review_findings(
    df: pd.DataFrame,
    findings: list[dict[str, Any]],
    computed_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Verify analytical claims against computed evidence, numerical accuracy,
    sample-size adequacy, and statistical assumptions.
    """
    approved: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    context = computed_context or {}

    for finding in findings:
        if not isinstance(finding, dict):
            rejected.append({
                "finding": {"claim": str(finding)},
                "reason": "Finding must be a structured dictionary.",
            })
            continue

        claim = str(finding.get("claim", "")).strip()
        evidence = finding.get("evidence")
        if not claim or not evidence:
            rejected.append({
                "finding": finding,
                "reason": "Missing claim text or verifiable evidence reference.",
            })
            continue

        # 1. Sample-size check for statistical claims
        sample_size = finding.get("sample_size")
        if sample_size is not None and isinstance(sample_size, (int, float)) and sample_size < 5:
            rejected.append({
                "finding": finding,
                "reason": f"Insufficient sample size (n={sample_size} < 5) to support statistical claim.",
            })
            continue

        # 2. Causal overreach check on observational correlation/profiling claims
        if _CAUSAL_WORDS.search(claim) and not finding.get("experimental_evidence", False):
            rejected.append({
                "finding": finding,
                "reason": "Unsupported causal language ('causes'/'proves') on observational data.",
            })
            continue

        # 3. Numerical accuracy check against computed_context when metric + expected_value are given
        metric_path = finding.get("metric")
        expected_val = finding.get("expected_value")
        if metric_path and expected_val is not None and context:
            actual_val = _resolve_metric(str(metric_path), context)
            if actual_val is None:
                rejected.append({
                    "finding": finding,
                    "reason": f"Metric '{metric_path}' not found in computed evidence.",
                })
                continue
            if isinstance(expected_val, (int, float)) and isinstance(actual_val, (int, float)):
                tol = float(finding.get("tolerance", 1e-2))
                if not math.isclose(float(expected_val), float(actual_val), rel_tol=0.0, abs_tol=tol):
                    rejected.append({
                        "finding": finding,
                        "reason": (
                            f"Numerical mismatch for '{metric_path}': claimed {expected_val}, "
                            f"computed {actual_val} (tolerance {tol})."
                        ),
                    })
                    continue

        # 4. DataFrame row-count sanity check if df is empty
        if df.empty:
            rejected.append({
                "finding": finding,
                "reason": "Cannot approve findings on an empty DataFrame.",
            })
            continue

        approved.append(finding)

    status = "approved" if not rejected else ("needs_review" if approved else "rejected")
    logger.info("Reviewer evaluated %d findings: %d approved, %d rejected.", len(findings), len(approved), len(rejected))

    return {
        "approved": approved,
        "rejected": rejected,
        "status": status,
        "note": (
            "Verified evidence references, sample-size thresholds, causal-claim boundaries, "
            "and numerical consistency against computed outputs."
        ),
    }


def build_verified_findings(
    df: pd.DataFrame,
    profile: dict[str, Any],
    iqr_summary: dict[str, Any] | None = None,
    iso_summary: dict[str, Any] | None = None,
    corr_summary: dict[str, Any] | None = None,
    stat_tests: dict[str, Any] | None = None,
    forecast_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Construct structured findings from computed results and run them through review_findings."""
    candidates: list[dict[str, Any]] = [
        {
            "claim": (
                f"Dataset contains {profile['rows']:,} rows and {profile['columns']} columns "
                f"(data quality score: {profile.get('quality_score', 'N/A')}/100)."
            ),
            "evidence": "profiler.profile_dataset: shape & quality_score",
            "metric": "profile.rows",
            "expected_value": profile["rows"],
            "sample_size": profile["rows"],
        },
        {
            "claim": (
                f"Detected {profile['missing_cells']:,} missing cells ({profile.get('missing_pct', 0.0)}%) "
                f"and {profile['duplicate_rows']:,} duplicate rows."
            ),
            "evidence": "profiler.profile_dataset: missing_cells & duplicate_rows",
            "metric": "profile.missing_cells",
            "expected_value": profile["missing_cells"],
            "sample_size": profile["rows"],
        },
    ]

    if iqr_summary and iqr_summary.get("column"):
        candidates.append({
            "claim": (
                f"IQR analysis on '{iqr_summary['column']}' flagged {iqr_summary['count']} "
                "potential outlier row(s) for review."
            ),
            "evidence": f"statistician.iqr_outliers({iqr_summary['column']})",
            "metric": "iqr.count",
            "expected_value": iqr_summary["count"],
            "sample_size": len(df),
        })

    if iso_summary and iso_summary.get("status") == "completed":
        candidates.append({
            "claim": (
                f"Isolation Forest flagged {iso_summary['count']} multivariate anomaly row(s) "
                f"({round(iso_summary['rate'] * 100, 2)}% of rows) across {len(iso_summary['features'])} features."
            ),
            "evidence": f"statistician.isolation_forest_outliers({iso_summary['features']})",
            "metric": "isolation_forest.count",
            "expected_value": iso_summary["count"],
            "sample_size": len(df),
        })

    if corr_summary and corr_summary.get("pairs"):
        top_pair = corr_summary["pairs"][0]
        candidates.append({
            "claim": (
                f"Strongest linear association observed between '{top_pair['col_a']}' and "
                f"'{top_pair['col_b']}' (r={top_pair['correlation']}, p={top_pair['p_value']}, "
                f"n={top_pair['sample_size']}); correlation does not imply causation."
            ),
            "evidence": "statistician.correlation_analysis",
            "sample_size": top_pair["sample_size"],
        })

    if stat_tests and stat_tests.get("group_comparisons"):
        top_test = stat_tests["group_comparisons"][0]
        sig_text = "statistically significant" if top_test["significant_at_0_05"] else "not statistically significant"
        candidates.append({
            "claim": (
                f"{top_test['test']} comparing '{top_test['numeric']}' across '{top_test['categorical']}' "
                f"is {sig_text} (stat={top_test['statistic']}, p={top_test['p_value']}, n={top_test['total_n']})."
            ),
            "evidence": f"statistician.run_statistical_tests:{top_test['test']}",
            "sample_size": top_test["total_n"],
        })

    if forecast_summary and forecast_summary.get("status") == "completed":
        best = forecast_summary["selected_model"]
        rmse = forecast_summary["evaluations"][best]["rmse"]
        candidates.append({
            "claim": (
                f"Chronological holdout evaluation ({forecast_summary['test_periods']} test periods) "
                f"selected '{best}' for '{forecast_summary['target_col']}' with RMSE={rmse}."
            ),
            "evidence": "statistician.forecast_series",
            "sample_size": forecast_summary["total_periods"],
        })

    context = {
        "profile": profile,
        "iqr": iqr_summary or {},
        "isolation_forest": iso_summary or {},
        "correlation": corr_summary or {},
        "forecast": forecast_summary or {},
    }
    return review_findings(df, candidates, computed_context=context)
