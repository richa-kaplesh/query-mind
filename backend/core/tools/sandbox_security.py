import ast

# Flat allowlist — name-based, not type-aware. A lightweight AST walk can't
# distinguish "df.sum()" from "pd.sum" from "series.sum()", so one set covers
# all of them. Documented limitation, not an oversight.
ALLOWED_ATTRIBUTES = {
    "columns", "shape", "dtypes", "index", "values", "empty", "size", "ndim", "T",
    "loc", "iloc", "at", "iat", "isin", "between",
    "sum", "mean", "median", "mode", "std", "var", "min", "max", "count",
    "nunique", "unique", "value_counts", "quantile", "corr", "cov",
    "cumsum", "cumprod", "cummax", "cummin",
    "isna", "isnull", "notna", "notnull", "dropna", "fillna", "ffill", "bfill",
    "apply", "applymap", "map", "astype", "rename", "replace", "round", "abs", "clip",
    "sort_values", "sort_index", "reset_index", "set_index", "drop",
    "drop_duplicates", "duplicated", "groupby", "merge", "join",
    "pivot_table", "melt", "stack", "unstack", "transpose",
    "head", "tail", "sample", "copy", "describe",
    "str", "dt", "cat",
    "contains", "upper", "lower", "strip", "lstrip", "rstrip", "split",
    "startswith", "endswith", "extract", "len",
    "year", "month", "day", "hour", "minute", "second", "weekday", "date",
    "DataFrame", "Series", "concat", "to_datetime", "to_numeric",
    "cut", "qcut", "get_dummies", "date_range", "crosstab",      "agg", "aggregate", "nsmallest", "nlargest", "kind",
}

# query() excluded deliberately — supports @variable injection into arbitrary namespaces
DANGEROUS_CALLS = {
    "eval", "exec", "open", "compile", "__import__",
    "globals", "locals", "vars", "getattr", "setattr", "delattr", "query",
}

MAX_AST_NODES = 400
MAX_CODE_LENGTH = 3000


class CodeValidationError(Exception):
    pass


class CodeValidator:
    """Single responsibility: judge whether LLM-generated code is safe.
    Knows nothing about processes or pandas execution — pure static analysis."""

    def validate(self, code: str) -> None:
        if len(code) > MAX_CODE_LENGTH:
            raise CodeValidationError(f"Code exceeds {MAX_CODE_LENGTH} character limit")

        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            raise CodeValidationError(f"Syntax error in generated code: {e}")

        nodes = list(ast.walk(tree))
        if len(nodes) > MAX_AST_NODES:
            raise CodeValidationError(f"Code is too complex ({len(nodes)} AST nodes, max {MAX_AST_NODES})")

        for node in nodes:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                raise CodeValidationError("Import statements are not allowed")

            if isinstance(node, (ast.Lambda, ast.FunctionDef, ast.AsyncFunctionDef)):
                raise CodeValidationError("Defining functions/lambdas is not allowed")

            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id in DANGEROUS_CALLS:
                    raise CodeValidationError(f"Call to '{func.id}' is not allowed")
                if isinstance(func, ast.Attribute) and func.attr in DANGEROUS_CALLS:
                    raise CodeValidationError(f"Call to '{func.attr}' is not allowed")

            if isinstance(node, ast.Attribute):
                if node.attr.startswith("__") and node.attr.endswith("__"):
                    raise CodeValidationError(f"Access to '{node.attr}' is not allowed")
                if node.attr not in ALLOWED_ATTRIBUTES:
                    raise CodeValidationError(f"Attribute/method '{node.attr}' is not on the allowed list")

            if isinstance(node, ast.Name):
                if node.id.startswith("__") and node.id.endswith("__"):
                    raise CodeValidationError(f"Access to '{node.id}' is not allowed")