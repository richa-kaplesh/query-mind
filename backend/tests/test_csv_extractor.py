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


def test_header_only_csv_warns_about_zero_rows(tmp_path):
    path = write_csv(tmp_path, "age,city\n")
    schema = CSVExtractor().extract_schema(path)
    assert schema.row_count == 0
    assert any("0 rows" in w for w in schema.warnings)


def test_entirely_null_column_warns(tmp_path):
    path = write_csv(tmp_path, "age,notes\n10,\n20,\n")
    schema = CSVExtractor().extract_schema(path)
    assert any("notes" in w and "entirely null" in w for w in schema.warnings)


def test_semicolon_separated_csv_is_detected(tmp_path):
    path = write_csv(tmp_path, "age;city\n10;Delhi\n20;Mumbai\n")
    schema = CSVExtractor().extract_schema(path)
    assert [c.name for c in schema.columns] == ["age", "city"]


def test_empty_file_raises_value_error(tmp_path):
    path = write_csv(tmp_path, "")
    with pytest.raises(ValueError):
        CSVExtractor().extract_schema(path)


def test_missing_csv_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        CSVExtractor().extract_schema(str(tmp_path / "nope.csv"))


def test_stray_quotes_are_cleaned_so_numbers_become_numeric(tmp_path):
    path = write_csv(tmp_path, "cls,name\n'1',a\n'2',b\n'3',c\n")
    cls = CSVExtractor().extract_schema(path).columns[0]
    assert (cls.min, cls.max) == (1.0, 3.0)


def test_latin1_encoded_file_is_loaded(tmp_path):
    path = tmp_path / "latin.csv"
    path.write_bytes("name,city\nJosé,Zürich\n".encode("latin-1"))
    schema = CSVExtractor().extract_schema(str(path))
    assert schema.columns[0].samples == ["José"]
    assert schema.columns[1].samples == ["Zürich"]


def test_extract_wraps_the_schema(tmp_path):
    path = write_csv(tmp_path, "age\n1\n2\n")
    result = CSVExtractor().extract(path)
    assert result.schema.row_count == 2