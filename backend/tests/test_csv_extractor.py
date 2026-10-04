# import pytest
# from core.extractors.csv_extractor import CSVExtractor


# def write_csv(tmp_path, text, name="data.csv"):
#     path = tmp_path / name
#     path.write_text(text)
#     return str(path)


# def test_normal_csv_builds_correct_schema(tmp_path):
#     path = write_csv(tmp_path, "age,city\n10,Delhi\n20,Mumbai\n30,Delhi\n")

#     schema = CSVExtractor().extract_schema(path)

#     assert schema.source == "data.csv"
#     assert schema.row_count == 3
#     assert [c.name for c in schema.columns] == ["age", "city"]

#     age, city = schema.columns
#     assert (age.min, age.max, age.mean) == (10.0, 30.0, 20.0)
#     assert age.null_count == 0
#     assert city.unique_count == 2
#     assert sorted(city.unique_values) == ["Delhi", "Mumbai"]
#     assert schema.warnings == []


# def test_header_only_csv_warns_about_zero_rows(tmp_path):
#     path = write_csv(tmp_path, "age,city\n")
#     schema = CSVExtractor().extract_schema(path)
#     assert schema.row_count == 0
#     assert len(schema.columns) == 2
#     assert any("0 rows" in w for w in schema.warnings)


# def test_entirely_null_column_warns(tmp_path):
#     path = write_csv(tmp_path, "age,notes\n10,\n20,\n")
#     schema = CSVExtractor().extract_schema(path)
#     assert any("notes" in w and "entirely null" in w for w in schema.warnings)


# def test_semicolon_separated_csv_is_detected(tmp_path):
#     path = write_csv(tmp_path, "age;city\n10;Delhi\n20;Mumbai\n")
#     schema = CSVExtractor().extract_schema(path)
#     assert [c.name for c in schema.columns] == ["age", "city"]


# def test_empty_file_raises_value_error(tmp_path):
#     path = write_csv(tmp_path, "")
#     with pytest.raises(ValueError):
#         CSVExtractor().extract_schema(path)


# def test_missing_csv_raises(tmp_path):
#     with pytest.raises(FileNotFoundError):
#         CSVExtractor().extract_schema(str(tmp_path / "nope.csv"))