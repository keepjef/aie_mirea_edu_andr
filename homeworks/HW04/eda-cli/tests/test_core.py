from __future__ import annotations

import pandas as pd

from eda_cli.core import (
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [10, 20, 30, None],
            "height": [140, 150, 160, 170],
            "city": ["A", "B", "A", None],
        }
    )

def _sample_df2() -> pd.DataFrame:
    """Расширенный датафрейм для тестирования."""
    return pd.DataFrame(
        {
            "age": [10, 20, 30, None, 40, 50, 60, 70, 80, 90],
            "height": [140, 150, 160, 170, 180, 190, 200, 210, 220, 230],
            "city": ["A", "B", "A", None, "C", "D", "E", "F", "G", "H"],
            "constant_col": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Константная колонка
            "mostly_nulls": [None] * 9 + [1],  # Колонка с 90% пропусков
            "some_nulls": [1, 2, None, None, 5, 6, 7, 8, 9, 10],
        }
    )


def test_summarize_dataset_basic():
    df = _sample_df()
    summary = summarize_dataset(df)

    assert summary.n_rows == 4
    assert summary.n_cols == 3
    assert any(c.name == "age" for c in summary.columns)
    assert any(c.name == "city" for c in summary.columns)

    summary_df = flatten_summary_for_print(summary)
    assert "name" in summary_df.columns
    assert "missing_share" in summary_df.columns


def test_missing_table_and_quality_flags():
    df = _sample_df()
    missing_df = missing_table(df)

    assert "missing_count" in missing_df.columns
    assert missing_df.loc["age", "missing_count"] == 1

    summary = summarize_dataset(df)
    flags = compute_quality_flags(summary, missing_df)
    assert 0.0 <= flags["quality_score"] <= 1.0

def test_my_quality_flag():
    # Домашнее задание, все тесты новых флагов подключены
    df2 = _sample_df2()
    missing_df2 = missing_table(df2)

    summary2 = summarize_dataset(df2)
    flags2 = compute_quality_flags(summary2, missing_df2)

    assert 0.0 <= flags2["quality_score"] <= 1.0
    assert flags2["max_missing_share"] == 0.9
    assert flags2["too_few_rows"] == True
    assert flags2["too_many_columns"] == False
    assert flags2["too_many_missing"] == True
    assert flags2["has_constant_columns"] == True
    assert flags2["has_high_cardinality_categoricals"] == False

def test_correlation_and_top_categories():
    df = _sample_df()
    corr = correlation_matrix(df)
    # корреляция между age и height существует
    assert "age" in corr.columns or corr.empty is False

    top_cats = top_categories(df, max_columns=5, top_k=2)
    assert "city" in top_cats
    city_table = top_cats["city"]
    assert "value" in city_table.columns
    assert len(city_table) <= 2
