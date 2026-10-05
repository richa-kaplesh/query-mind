import pandas as pd
import pytest
from core.tools.safe_namespace import build_safe_df, build_safe_pd


def test_safe_pd_allows_listed_functions():
    safe_pd = build_safe_pd()
    assert safe_pd.to_numeric(pd.Series(["1", "2"])).tolist() == [1, 2]


def test_safe_pd_blocks_unlisted_functions():
    with pytest.raises(AttributeError):
        build_safe_pd().read_csv("anything.csv")


def test_safe_df_allows_reading():
    df = build_safe_df(pd.DataFrame({"age": [10, 20, 30]}))
    assert df.shape == (3, 1)
    assert len(df) == 3
    assert df["age"].sum() == 60


def test_safe_df_blocks_unlisted_methods():
    df = build_safe_df(pd.DataFrame({"age": [10]}))
    with pytest.raises(AttributeError):
        df.to_csv("out.csv")


def test_safe_df_blocks_attribute_style_column_access():
    df = build_safe_df(pd.DataFrame({"age": [10]}))
    with pytest.raises(AttributeError):
        df.age


def test_safe_df_can_add_a_column():
    df = build_safe_df(pd.DataFrame({"age": [10, 20]}))
    df["double"] = df["age"] * 2
    assert df["double"].tolist() == [20, 40]