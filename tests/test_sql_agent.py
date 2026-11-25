import pandas as pd
import pytest
from agents.sql_agent import generate_and_validate_sql, generate_sql
from core.sql_engine import execute_readonly_query, validate_readonly_sql


def test_row_count_sql():
    df = pd.DataFrame({"revenue": [10, 20]})
    sql = generate_sql("how many rows", df)
    result = execute_readonly_query(df, sql)
    assert int(result.iloc[0, 0]) == 2


def test_region_revenue_sql():
    df = pd.DataFrame({"region": ["East", "West", "East"], "revenue": [10, 20, 30]})
    sql = generate_sql("total revenue by region", df)
    result = execute_readonly_query(df, sql)
    values = dict(zip(result["region"], result["total_revenue"]))
    assert values["East"] == 40
    assert values["West"] == 20


def test_filtering_grouping_and_top_n_sql():
    df = pd.DataFrame({
        "region": ["North", "North", "North", "South"],
        "product": ["Laptop", "Mouse", "Monitor", "Laptop"],
        "revenue": [1300, 180, 720, 650],
    })
    sql = generate_sql("top 2 products by revenue in North", df)
    result = execute_readonly_query(df, sql)
    assert len(result) == 2
    assert result.iloc[0]["product"] == "Laptop"
    assert result.iloc[0]["total_revenue"] == 1300

    sql_num = generate_sql("show records where revenue > 700", df)
    res_num = execute_readonly_query(df, sql_num)
    assert len(res_num) == 2


def test_whitespace_resilient_sql_without_mutating_df():
    df = pd.DataFrame({
        "Category": ["Tools", " Tools ", "Games", "Games"],
        "Rating": [4.0, 5.0, 3.0, 4.0],
    })
    sql = generate_sql("average Rating by Category", df)
    out = execute_readonly_query(df, sql)
    assert len(out) == 2
    assert df.loc[1, "Category"] == " Tools "  # Original df remains unmodified


def test_trend_and_structured_sql():
    df = pd.DataFrame({
        "date": ["2026-01-01", "2026-01-15", "2026-02-01"],
        "revenue": [100, 150, 250],
    })
    res = generate_and_validate_sql("monthly revenue trend", df)
    assert res["valid"] is True
    assert res["source"] == "deterministic"
    out = execute_readonly_query(df, res["sql"])
    assert len(out) == 2


def test_reject_write_and_injection_queries():
    df = pd.DataFrame({"x": [1]})
    for bad in [
        "DELETE FROM data",
        "SELECT * FROM data; DROP TABLE data",
        "/* comment */ DROP TABLE data",
        "SELECT * FROM read_csv('secret.csv')",
        "SELECT * FROM data WHERE x = 1; COPY data TO 'out.csv'",
    ]:
        with pytest.raises(ValueError):
            validate_readonly_sql(bad)
        with pytest.raises(ValueError):
            execute_readonly_query(df, bad)
