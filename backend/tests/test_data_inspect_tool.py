from core.tools.data_inspect_tool import DataInspectTool


def make_tool(tmp_path, text="age,city\n10,Delhi\n20,Mumbai\n30,Delhi\n", name="d.csv"):
    path = tmp_path / name
    path.write_text(text)
    return DataInspectTool(str(path))


def test_shape_and_columns(tmp_path):
    tool = make_tool(tmp_path)
    assert tool.run("shape") == "rows=3, columns=2"
    assert tool.run("columns") == "2 columns: age, city"


def test_column_actions(tmp_path):
    tool = make_tool(tmp_path)
    assert tool.run("unique_count", "city") == "2 unique values"
    assert tool.run("null_count", "age") == "0 missing of 3"
    assert "Delhi" in tool.run("value_counts", "city")


def test_null_count_without_column_summarises_the_whole_file(tmp_path):
    assert make_tool(tmp_path).run("null_count") == "no missing values"
    with_gaps = make_tool(tmp_path, "a,b\n1,\n2,\n", name="gaps.csv")
    assert "b" in with_gaps.run("null_count")


def test_unknown_action_is_an_error_message_not_a_crash(tmp_path):
    assert make_tool(tmp_path).run("explode").startswith("Error: unknown action")


def test_missing_column_error_suggests_the_right_case(tmp_path):
    out = make_tool(tmp_path).run("describe", "AGE")
    assert "does not exist" in out and "Did you mean 'age'" in out


def test_action_that_needs_a_column_says_so(tmp_path):
    assert "needs a 'column'" in make_tool(tmp_path).run("describe")


def test_unreadable_file_is_reported_as_text(tmp_path):
    tool = DataInspectTool(str(tmp_path / "nope.csv"))
    assert tool.run("shape").startswith("Error: could not load the dataset")


def test_json_args_must_be_an_object(tmp_path):
    tool = make_tool(tmp_path)
    assert tool.run_json_args({"action": "shape"}) == "rows=3, columns=2"
    assert tool.run_json_args("shape").startswith("Error")