from core.tools.pandas_sandbox_tool import PandasSandboxTool

if __name__ == "__main__":
    file_path = "uploads/Titanic-Dataset.csv"
    sandbox = PandasSandboxTool(file_path)

    llm_code_1 = """```python
# Calculate the average age of passengers
result = df['Age'].mean()
```"""
    print("--- TEST 1: Average Age ---")
    output_1 = sandbox.execute_query(file_path, llm_code_1)
    print(f"Result: {output_1}\n")