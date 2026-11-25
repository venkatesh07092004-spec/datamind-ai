import pandas as pd

from agents.profiler import profile_dataset
from agents.reviewer import build_verified_findings, review_findings


def test_reviewer_rejects_fabricated_and_causal_claims():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6], "y": [2, 4, 6, 8, 10, 12]})
    profile = profile_dataset(df)
    findings = [
        {
            "claim": "Dataset has 6 rows.",
            "evidence": "profiler.profile_dataset",
            "metric": "profile.rows",
            "expected_value": 6,
            "sample_size": 6,
        },
        {
            "claim": "Dataset has 999 rows.",
            "evidence": "profiler.profile_dataset",
            "metric": "profile.rows",
            "expected_value": 999,
            "sample_size": 6,
        },
        {
            "claim": "x causes y to double.",
            "evidence": "correlation",
            "sample_size": 6,
        },
        {
            "claim": "Small sample claim.",
            "evidence": "test",
            "sample_size": 2,
        },
        {
            "claim": "Missing evidence claim.",
        },
    ]
    res = review_findings(df, findings, computed_context={"profile": profile})
    assert len(res["approved"]) == 1
    assert len(res["rejected"]) == 4
    assert res["status"] == "needs_review"


def test_build_verified_findings():
    df = pd.DataFrame({"x": [10, 20, 30, 40, 50], "cat": ["a", "b", "a", "b", "a"]})
    profile = profile_dataset(df)
    res = build_verified_findings(df, profile)
    assert res["status"] == "approved"
    assert len(res["approved"]) >= 2
