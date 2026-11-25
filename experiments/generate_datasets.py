"""Generate 5 deterministic, domain-diverse datasets for reproducible DataMind AI evaluation."""

from pathlib import Path
import numpy as np
import pandas as pd


def generate_all_datasets(out_dir: Path | str = "experiments/datasets") -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    paths: dict[str, Path] = {}

    # 1. Retail Sales (Time-series + injected anomalies + missing values)
    dates_1 = pd.date_range("2025-01-05", periods=120, freq="D")
    rev_1 = 1000.0 + np.arange(120) * 12.5 + rng.normal(0, 45, size=120)
    qty_1 = np.maximum(1, (rev_1 / 120.0 + rng.normal(0, 1.5, size=120)).round()).astype(float)
    is_anom_1 = np.zeros(120, dtype=int)
    for idx in [15, 58, 102]:
        rev_1[idx] *= 4.2
        qty_1[idx] *= 4.0
        is_anom_1[idx] = 1
    df_retail = pd.DataFrame({
        "date": dates_1.strftime("%Y-%m-%d"),
        "region": rng.choice(["North", "South", "East", "West"], size=120),
        "product": rng.choice(["Laptop", "Monitor", "Keyboard", "Mouse"], size=120),
        "quantity": qty_1,
        "revenue": np.round(rev_1, 2),
        "ground_truth_anomaly": is_anom_1,
    })
    df_retail.loc[[5, 22, 77], "quantity"] = np.nan
    p1 = out / "retail_sales.csv"
    df_retail.to_csv(p1, index=False)
    paths["retail_sales"] = p1

    # 2. Mobile App Catalog (Dirty text, semantic columns Size/Price/Rating/Last Updated, duplicates)
    n_app = 80
    app_dates = pd.date_range("2024-01-01", periods=n_app, freq="3D")
    df_app = pd.DataFrame({
        "App Name": [f"App_{i:03d}" for i in range(n_app)],
        "Category": rng.choice(["Tools", "Games", "Finance", "Productivity"], size=n_app),
        "Size": [f"{rng.integers(5, 95)}M" for _ in range(n_app)],
        "Price": [rng.choice(["$0", "$1.99", "$4.99", "$9.99"]) for _ in range(n_app)],
        "Rating": np.round(rng.uniform(3.2, 4.9, size=n_app), 2),
        "Last Updated": app_dates.strftime("%B %d, %Y"),
    })
    df_app.loc[[2, 10, 25], "Rating"] = np.nan
    df_app.loc[[1, 4], "Category"] = " Tools "
    df_app = pd.concat([df_app, df_app.iloc[[0, 3]]], ignore_index=True)  # 2 duplicates
    p2 = out / "app_catalog.csv"
    df_app.to_csv(p2, index=False)
    paths["app_catalog"] = p2

    # 3. Healthcare Patient Vitals (Multivariate anomalies, group differences)
    n_hc = 100
    visit_dates = pd.date_range("2025-02-01", periods=n_hc, freq="D")
    dept = rng.choice(["Cardiology", "General", "Pediatrics"], size=n_hc)
    bp = rng.normal(120, 10, size=n_hc) + np.where(dept == "Cardiology", 18.0, 0.0)
    hr = rng.normal(74, 8, size=n_hc)
    bmi = rng.normal(24.5, 3.2, size=n_hc)
    is_anom_3 = np.zeros(n_hc, dtype=int)
    for idx in [12, 44, 79, 91]:
        bp[idx] = 215.0
        hr[idx] = 148.0
        is_anom_3[idx] = 1
    df_hc = pd.DataFrame({
        "visit_date": visit_dates.strftime("%Y-%m-%d"),
        "department": dept,
        "age": rng.integers(18, 82, size=n_hc),
        "bmi": np.round(bmi, 1),
        "systolic_bp": np.round(bp, 1),
        "heart_rate": np.round(hr, 1),
        "ground_truth_anomaly": is_anom_3,
    })
    p3 = out / "healthcare_vitals.csv"
    df_hc.to_csv(p3, index=False)
    paths["healthcare_vitals"] = p3

    # 4. E-Commerce Traffic & Conversions (Chronological trend)
    n_ec = 90
    ec_dates = pd.date_range("2025-03-01", periods=n_ec, freq="D")
    sessions = 500 + np.arange(n_ec) * 6 + rng.normal(0, 25, size=n_ec)
    conversions = np.maximum(5, (sessions * 0.045 + rng.normal(0, 3, size=n_ec)).round())
    revenue_ec = conversions * 52.0 + rng.normal(0, 40, size=n_ec)
    df_ec = pd.DataFrame({
        "date": ec_dates.strftime("%Y-%m-%d"),
        "channel": rng.choice(["Organic", "Paid", "Email", "Social"], size=n_ec),
        "sessions": np.round(sessions).astype(int),
        "conversions": conversions.astype(int),
        "revenue": np.round(revenue_ec, 2),
    })
    p4 = out / "ecommerce_traffic.csv"
    df_ec.to_csv(p4, index=False)
    paths["ecommerce_traffic"] = p4

    # 5. HR Employee Survey (Purely categorical / ordinal)
    n_hr = 60
    df_hr = pd.DataFrame({
        "employee_id": [f"EMP-{i:04d}" for i in range(1, n_hr + 1)],
        "department": rng.choice(["Engineering", "Sales", "Support", "Finance"], size=n_hr),
        "role_level": rng.choice(["Junior", "Mid", "Senior", "Lead"], size=n_hr),
        "satisfaction": rng.choice(["High", "Medium", "Low"], size=n_hr, p=[0.5, 0.35, 0.15]),
        "remote_preference": rng.choice(["Remote", "Hybrid", "Onsite"], size=n_hr),
    })
    p5 = out / "hr_survey.csv"
    df_hr.to_csv(p5, index=False)
    paths["hr_survey"] = p5

    return paths


if __name__ == "__main__":
    generated = generate_all_datasets()
    for name, p in generated.items():
        print(f"Generated {name}: {p}")
