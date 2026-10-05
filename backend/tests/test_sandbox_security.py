import pytest
from core.tools.sandbox_security import (
    CodeValidator, CodeValidationError, MAX_CODE_LENGTH, MAX_AST_NODES,
)

validator = CodeValidator()


@pytest.mark.parametrize("code", [
    "result = df['age'].mean()",
    "result = df.groupby('city')['age'].sum()",
    "result = df[df['age'] > 5].shape[0]",
    "result = df.sort_values('age').head(3)",
    "result = pd.to_numeric(df['age']).max()",
])
def test_normal_analysis_code_is_allowed(code):
    validator.validate(code)          # must not raise


@pytest.mark.parametrize("code", [
    "import os",
    "from os import path",
    "__import__('os')",
    "eval('1+1')",
    "exec('x = 1')",
    "open('secrets.txt')",
    "getattr(df, 'shape')",
    "df.query('age > 5')",
    "df.__class__",
    "lambda x: x",
    "def f():\n    pass",
    "pd.read_csv('other.csv')",
    "df.to_csv('out.csv')",
    "os.system('ls')",
])
def test_dangerous_code_is_rejected(code):
    with pytest.raises(CodeValidationError):
        validator.validate(code)


def test_syntax_error_is_rejected():
    with pytest.raises(CodeValidationError, match="Syntax error"):
        validator.validate("result = (")


def test_too_long_code_is_rejected():
    with pytest.raises(CodeValidationError, match="character limit"):
        validator.validate("x = 1\n" * (MAX_CODE_LENGTH // 5))


def test_too_complex_code_is_rejected():
    code = "result = " + " + ".join(["1"] * (MAX_AST_NODES // 2))
    assert len(code) < MAX_CODE_LENGTH
    with pytest.raises(CodeValidationError, match="too complex"):
        validator.validate(code)