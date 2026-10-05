import pandas as pd
from core.utils.data_cleaning import strip_stray_quotes


def test_quoted_numbers_become_numeric():
    df = pd.DataFrame({"cls": ["'1'", "'2'", "'3'"]})
    out = strip_stray_quotes(df)
    assert pd.api.types.is_numeric_dtype(out["cls"])
    assert out["cls"].tolist() == [1, 2, 3]


def test_quoted_text_is_stripped_but_stays_text():
    df = pd.DataFrame({"name": ["'alice'", '"bob"']})
    out = strip_stray_quotes(df)
    assert out["name"].tolist() == ["alice", "bob"]


def test_mixed_numbers_and_text_stay_text():
    df = pd.DataFrame({"code": ["'1'", "x"]})
    out = strip_stray_quotes(df)
    assert out["code"].tolist() == ["1", "x"]


def test_real_numeric_columns_are_left_alone():
    df = pd.DataFrame({"age": [10, 20, 30]})
    out = strip_stray_quotes(df)
    assert out["age"].tolist() == [10, 20, 30]