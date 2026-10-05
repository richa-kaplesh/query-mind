from core.models import ColumnSchema, CSVSchema


def col(name, dtype="int64", **extra):
    return ColumnSchema(name=name, dtype=dtype, null_count=0, samples=["1", "2"], **extra)


def test_small_schema_lists_every_column_in_detail():
    schema = CSVSchema(source="a.csv", row_count=5,
                       columns=[col("age", min=1.0, max=9.0, mean=5.0), col("city", "object", unique_count=2)])
    text = schema.to_prompt_string()

    assert text.splitlines()[0] == "File: a.csv | Rows: 5 | Cols: 2"
    assert "Col: age" in text and "Min: 1.0" in text and "Mean: 5.0" in text
    assert "Col: city" in text and "Unique: 2" in text
    assert "Warnings" not in text


def test_warnings_are_appended():
    schema = CSVSchema(source="a.csv", row_count=0, columns=[col("age")], warnings=["File has 0 rows"])
    assert schema.to_prompt_string().endswith("Warnings: File has 0 rows")


def test_wide_schema_details_only_the_first_20_columns():
    columns = [col(f"c{i}") for i in range(25)]
    text = CSVSchema(source="wide.csv", row_count=1, columns=columns).to_prompt_string()

    assert "Col: c19 " in text            # 20th column: full detail
    assert "Col: c20 " not in text        # 21st: only listed by name
    assert "+5 more int64 columns" in text
    assert "c24" in text
    assert "pandas_sandbox" in text       # tells the LLM how to inspect the rest