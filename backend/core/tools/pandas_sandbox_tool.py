from core.tools.base_tool import BaseTool
import pandas as pd
from typing import Any
import ast
import multiprocessing 
import textwrap

def _worker(file_path: str, executable_code: str, safe_builtins: dict, queue: multiprocessing.Queue):
    try:
        df = pd.read_csv(file_path, low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding="latin-1", low_memory=False)

        local_vars = {"pd":pd, "df":df, "result":None}
        restricted_globals = {"__builtins__": safe_builtins}
        exec(executable_code, restricted_globals, local_vars)
        queue.put(("ok",str(local_vars["result"])))
    except Exception as e:
        queue.put(("error", str(e)))


class PandasSandboxTool(BaseTool):

    DANGEROUS_CALLS = {"eval", "exec", "open", "compile", "__import__","globals", "locals", "vars", "getattr", "setattr", "delattr"}

    SAFE_BUILTINS = {
        "len":len, "str":str, "int":int, "float": float, "bool":bool,
        "round":round, "sum": sum, "min": min, "max" : max, "abs":abs,
        "sorted": sorted, "list":list, "dict": dict, "tuple":tuple,"set":set,
        "range": range, "enumerate": enumerate, "zip":zip, "print": print,
    }

    def __init__(self, file_path: str, timeout_seconds: int=5):
        self.file_path = file_path
        self.timeout_seconds=timeout_seconds

    @property
    def name(self) -> str:
        return "pandas_sandbox"

    @property
    def description(self) -> str:
        return (
            "Execute dynamic Python/Pandas code on DataFrame 'df' to answer exact mathematical, "
            "filtering, aggregation, or analytical queries. Assign your final answer to variable 'result'."
        )

    def clean_code(self, raw_code: str) -> str:
        clean_code = raw_code.strip()
        if clean_code.startswith("```python"):
            clean_code = clean_code.removeprefix("```python")
        elif clean_code.startswith("```Python"):
            clean_code = clean_code.removeprefix("```Python")
        elif clean_code.startswith("```"):
            clean_code = clean_code.removeprefix("```")
            
        if clean_code.endswith("```"):
            clean_code = clean_code.removesuffix("```")

        clean_code = textwrap.dedent(clean_code)
        return clean_code.strip()

    

    def _validate_code(self, code: str) -> tuple[bool, str]:
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"Syntax error in generated code: {e}"

        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                return False, "Import statements are not allowed"

            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id in self.DANGEROUS_CALLS:
                    return False, f"Call to '{func.id}' is not allowed"
                if isinstance(func, ast.Attribute) and func.attr in self.DANGEROUS_CALLS:
                    return False, f"Call to '{func.attr}' is not allowed"

            if isinstance(node, ast.Attribute):
                if node.attr.startswith("__") and node.attr.endswith("__"):
                    return False, f"Access to '{node.attr}' is not allowed"

            if isinstance(node, ast.Name):
                if node.id.startswith("__") and node.id.endswith("__"):
                    return False, f"Access to '{node.id}' is not allowed"

        return True, "" 
    def execute_query(self, file_path: str, llm_code: str) -> str:
            executable_code = self.clean_code(llm_code)
            is_valid , error_message = self._validate_code(executable_code)
            if not is_valid:
                return f"Rejected:{error_message}"

            

            queue = multiprocessing.Queue()
            process = multiprocessing.Process(
                target=_worker,
                args=(file_path, executable_code, self.SAFE_BUILTINS, queue)

            )
            process.start()
            process.join(self.timeout_seconds)

            if process.is_alive():
                process.terminate()
                process.join()
                return f"Error: Code execution exceeded {self.timeout_seconds} seconds"

            if queue.empty():
                return "Error: Process ended unexpectedly with no result"

            status, payload = queue.get()
            if status == "error":
                return f"Error executing Pandas code: {payload}"
            return payload

    def run(self, code: str) -> dict:
        result = self.execute_query(self.file_path, code)
        return {"result": result}
        

