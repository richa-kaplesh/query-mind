from core.extractors.csv_extractor import CSVExtractor
from core.tools.pandas_sandbox_tool import PandasSandboxTool
from core.generator import Generator
import json

def test_schema():
    print("=== SCHEMA EXTRACTION ===")
    extractor = CSVExtractor()
    result = extractor.extract("test_data.csv")
    schema = result.schema
    print(schema.to_prompt_string())
    print()

def test_sandbox():
    print("=== PANDAS SANDBOX ===")
    sandbox = PandasSandboxTool(file_path="test_data.csv")
    result = sandbox.run("result = df['Salary'].mean()")
    print("Mean salary:", result)
    print()

def test_tool_schema():
    print("=== GENERATOR TOOL SCHEMA ===")
    g = Generator(tools=[PandasSandboxTool(file_path="test_data.csv")])
    schema_defs = g._build_tool_schema(g.tools)
    print(json.dumps(schema_defs, indent=2))
    print()

if __name__ == "__main__":
    test_schema()
    test_sandbox()
    test_tool_schema()
    print("All checks passed!")
