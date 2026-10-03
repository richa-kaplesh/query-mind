import pytest
from core.extractors.csv_extractor import CSVExtractor


def write_csv(tmp_path, text, name="data.csv"):
    path = tmp_path / name
    path.write_text(text)
    return str(path)


def test_normal_csv_builds_correct_schema(tmp_path):
    path = write_csv(tmp_path, "age,city\n10,Delhi\n20,Mumbai\n30,Delhi\n")

    schema = CSVExtractor().extract_schema(path)

    assert schema.source == "data.csv"
    assert schema.row_count == 3
    assert [c.name for c in schema.columns] == ["age", "city"]

    age, city = schema.columns
    assert (age.min, age.max, age.mean) == (10.0, 30.0, 20.0)
    assert age.null_count == 0
    assert city.unique_count == 2
    assert sorted(city.unique_values) == ["Delhi", "Mumbai"]
    assert schema.warnings == []