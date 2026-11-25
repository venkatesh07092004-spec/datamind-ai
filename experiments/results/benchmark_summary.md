# DataMind AI — Phase 9 Reproducible Benchmark Results

- **OS**: Windows-11-10.0.26200-SP0 | **Python**: 3.14.10
- **CPU Cores**: 12 | **RAM (GB)**: 15.7
- **Total SQL Questions Evaluated**: 23 (Uncleaned: 100.0%, Cleaned: 100.0%)

| Dataset | Rows x Cols | Ingest (ms) | Profile (ms) | Multi-Agent (ms) | Peak Mem (MB) | Equiv Baseline (ms) | Base Mem (MB) | SQL Unclean | SQL Clean | Reviewer (32) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| retail_sales | 120 x 5 | 6.81 | 120.08 | 4244.78 | 21.352 | 1906.06 | 0.427 | 100.0% | 100.0% | 100.0% |
| app_catalog | 82 x 6 | 4.4 | 100.33 | 3330.44 | 1.301 | 1785.37 | 0.455 | 100.0% | 100.0% | 100.0% |
| healthcare_vitals | 100 x 6 | 6.91 | 98.16 | 2894.29 | 1.471 | 1863.96 | 0.436 | 100.0% | 100.0% | 100.0% |
| ecommerce_traffic | 90 x 5 | 5.25 | 43.29 | 3374.97 | 1.786 | 2143.84 | 0.441 | 100.0% | 100.0% | 100.0% |
| hr_survey | 60 x 5 | 6.91 | 67.68 | 781.15 | 0.925 | 230.48 | 0.09 | 100.0% | 100.0% | 100.0% |
