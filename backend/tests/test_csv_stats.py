import numpy as np
import pandas as pd
from core.analysis.csv_stats import (
    clean_for_json, detect_problem_type, compute_numerical_stats,
    compute_categorical_stats, compute_correlation_matrix, compute_imbalance,
    compute_dataset_overview,
)


def test_clean_for_json_converts_numpy_types_recursively():
    messy = {"a": np.int64(3), "b": [np.float64(1.5), {"c": np.bool_(True)}], "d": np.array([1, 2])}
    clean = clean_for_json(messy)
    assert clean == {"a": 3, "b": [1.5, {"c": True}], "d": [1, 2]}
    assert type(clean["a"]) is int and type(clean["b"][0]) is float


def test_problem_type_detection():
    df = pd.DataFrame({
        "label": ["a", "b", "a", "b"] * 5,
        "few": [0, 1, 0, 1] * 5,
        "many": list(range(20)),
    })
    assert detect_problem_type(df, "label") == "classification"
    assert detect_problem_type(df, "few") == "classification"
    assert detect_problem_type(df, "many") == "regression"


def test_numerical_stats_on_data_with_one_obvious_outlier():
    s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 100])
    stats = compute_numerical_stats(s)
    assert stats["mean"] == 14.5
    assert stats["median"] == 5.5
    assert (stats["min"], stats["max"], stats["range"]) == (1, 100, 99)
    assert stats["outliers_iqr_count"] == 1
    assert stats["missing"] == 0


def test_categorical_stats():
    s = pd.Series(["x", "x", "x", "y", None])
    stats = compute_categorical_stats(s)
    assert stats["most_common"] == "x"
    assert stats["top_5_frequent"] == {"x": 3, "y": 1}
    assert stats["missing"] == 1 and stats["missing_pct"] == 20.0
    assert not stats["high_cardinality"]


def test_correlation_flags_perfectly_correlated_columns():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [2, 4, 6, 8, 10], "c": [5, 1, 4, 2, 3]})
    result = compute_correlation_matrix(df)
    pairs = [(p["col1"], p["col2"]) for p in result["high_correlation_pairs"]]
    assert ("a", "b") in pairs
    assert result["multicollinearity_warning"]


def test_imbalance_ratio():
    result = compute_imbalance(pd.Series(["a", "a", "a", "b"]))
    assert result["imbalance_ratio"] == 3.0
    assert not result["imbalance_warning"]


def test_dataset_overview_counts():
    df = pd.DataFrame({"n": [1, 1, 2], "t": ["a", "a", None]})
    o = compute_dataset_overview(df)
    assert (o["rows"], o["columns"]) == (3, 2)
    assert o["duplicate_rows"] == 1
    assert o["total_missing"] == 1
    assert o["numerical_columns"] == ["n"]