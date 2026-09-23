import pandas as pd
from core.tools.sandbox_security import ALLOWED_ATTRIBUTES


class SafeModuleProxy:
    """Second gate for `pd`, independent of the AST check. Even if a future
    change to CodeValidator ever let something slip past, this refuses
    anything not on the allowlist at actual access time."""

    def __init__(self, module, allowed: set[str]):
        self._module = module
        self._allowed = allowed

    def __getattr__(self, name: str):
        if name not in self._allowed:
            raise AttributeError(f"'{name}' is not an allowed pandas function")
        return getattr(self._module, name)


class SafeDataFrameProxy:
    """Same idea for `df`. Objects RETURNED by allowed calls (a Series from
    df['col'], a GroupBy from df.groupby(...)) are NOT re-wrapped —
    enforcement for those relies on the AST allowlist, not recursive
    taint-tracking, which is out of scope for a lightweight sandbox."""

    def __init__(self, dataframe, allowed: set[str]):
        object.__setattr__(self, "_df", dataframe)
        object.__setattr__(self, "_allowed", allowed)

    def __getattr__(self, name: str):
        if name not in self._allowed:
            raise AttributeError(f"'{name}' is not an allowed DataFrame attribute/method")
        return getattr(self._df, name)

    def __getitem__(self, key):
        return self._df[key]

    def __setitem__(self, key, value):
        self._df[key] = value

    def __len__(self):
        return len(self._df)

    def __repr__(self):
        return repr(self._df)

    def __str__(self):
        return str(self._df)


def build_safe_pd() -> SafeModuleProxy:
    return SafeModuleProxy(pd, ALLOWED_ATTRIBUTES)


def build_safe_df(dataframe) -> SafeDataFrameProxy:
    return SafeDataFrameProxy(dataframe, ALLOWED_ATTRIBUTES)