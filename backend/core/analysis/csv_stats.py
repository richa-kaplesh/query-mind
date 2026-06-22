import pandas as pd
import numpy as np
from scipy import stats as scipy_stats
import json


def clean_for_json(obj):
    if isinstance(obj, dict):
        return {key: clean_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def detect_problem_type(df: pd.DataFrame, target_col: str) -> str:
    target = df[target_col]
    unique_ratio = target.nunique() / len(target)

    if target.dtype == 'object' or target.nunique() <= 10:
        return "classification"
    elif unique_ratio > 0.05:
        return "regression"
    else:
        return "classification"


def compute_numerical_stats(series: pd.Series) -> dict:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers_iqr = series[(series < lower_bound) | (series > upper_bound)]
    zscores = np.abs(scipy_stats.zscore(series.dropna()))

    clean_series = series.dropna()
    sample_size = min(len(clean_series), 5000)
    _, normality_p_value = scipy_stats.shapiro(clean_series.sample(sample_size))

    return {
        "mean": round(series.mean(), 4),
        "median": round(series.median(), 4),
        "std": round(series.std(), 4),
        "variance": round(series.var(), 4),
        "min": round(series.min(), 4),
        "max": round(series.max(), 4),
        "range": round(series.max() - series.min(), 4),
        "skewness": round(series.skew(), 4),
        "kurtosis": round(series.kurtosis(), 4),
        "q1": round(q1, 4),
        "q3": round(q3, 4),
        "iqr": round(iqr, 4),
        "outliers_iqr_count": len(outliers_iqr),
        "outliers_iqr_pct": round(len(outliers_iqr) / len(series) * 100, 2),
        "outliers_zscore_count": int((zscores > 3).sum()),
        "is_normal": normality_p_value > 0.05,
        "normality_p_value": round(normality_p_value, 4),
        "missing": int(series.isnull().sum()),
        "missing_pct": round(series.isnull().sum() / len(series) * 100, 2)
    }


def compute_categorical_stats(series: pd.Series) -> dict:
    value_counts = series.value_counts()
    return {
        "unique_values": series.nunique(),
        "top_5_frequent": value_counts.head(5).to_dict(),
        "most_common": str(value_counts.index[0]) if not value_counts.empty else None,
        "missing": int(series.isnull().sum()),
        "missing_pct": round(series.isnull().sum() / len(series) * 100, 2),
        "high_cardinality": series.nunique() > 50
    }


def compute_correlation_matrix(df: pd.DataFrame) -> dict:
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numerical_cols].corr()

    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.85:
                high_corr_pairs.append({
                    "col1": corr_matrix.columns[i],
                    "col2": corr_matrix.columns[j],
                    "correlation": round(corr_val, 3)
                })

    return {
        "matrix": corr_matrix.to_dict(),
        "high_correlation_pairs": high_corr_pairs,
        "multicollinearity_warning": len(high_corr_pairs) > 0
    }


def compute_imbalance(target: pd.Series) -> dict:
    value_counts = target.value_counts()
    majority = value_counts.iloc[0]
    minority = value_counts.iloc[-1]
    imbalance_ratio = round(majority / minority, 2)

    return {
        "class_distribution": value_counts.to_dict(),
        "imbalance_ratio": imbalance_ratio,
        "imbalance_warning": imbalance_ratio > 10
    }


def compute_categorical_vs_target(df: pd.DataFrame, target_col: str, target: pd.Series) -> dict:
    results = {}
    for col in df.select_dtypes(include=['object']).columns:
        if col != target_col:
            contingency_table = pd.crosstab(df[col], target)
            chi2, p_value, _, _ = scipy_stats.chi2_contingency(contingency_table)
            results[col] = {
                "test": "chi-square",
                "statistic": round(chi2, 4),
                "p_value": round(p_value, 4),
                "significant": p_value < 0.05
            }
    return results


def compute_numerical_vs_target(df: pd.DataFrame, target_col: str, target: pd.Series) -> dict:
    results = {}
    classes = target.unique()
    num_classes = len(classes)

    for col in df.select_dtypes(include=[np.number]).columns:
        if col != target_col:
            if num_classes == 2:
                group1 = df[df[target_col] == classes[0]][col].dropna()
                group2 = df[df[target_col] == classes[1]][col].dropna()
                stat, p_value = scipy_stats.ttest_ind(group1, group2)
                test_name = "t-test"
            else:
                groups = [df[df[target_col] == c][col].dropna() for c in classes]
                stat, p_value = scipy_stats.f_oneway(*groups)
                test_name = "anova"

            results[col] = {
                "test": test_name,
                "statistic": round(stat, 4),
                "p_value": round(p_value, 4),
                "significant": p_value < 0.05
            }
    return results


def compute_target_analysis(df: pd.DataFrame, target_col: str, problem_type: str) -> dict:
    target = df[target_col]

    if problem_type == "classification":
        return {
            **compute_imbalance(target),
            "categorical_feature_tests": compute_categorical_vs_target(df, target_col, target),
            "numerical_feature_tests": compute_numerical_vs_target(df, target_col, target)
        }
    else:
        return {
            "distribution": compute_numerical_stats(target),
            "skew_warning": abs(target.skew()) > 1,
            "log_transform_recommended": abs(target.skew()) > 1
        }


def compute_dataset_overview(df: pd.DataFrame) -> dict:
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "duplicate_rows": int(df.duplicated().sum()),
        "total_missing": int(df.isnull().sum().sum()),
        "missing_pct": round(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100, 2),
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 3),
        "numerical_columns": list(df.select_dtypes(include=[np.number]).columns),
        "categorical_columns": list(df.select_dtypes(exclude=[np.number]).columns)
    }


def run_full_statistics(df: pd.DataFrame, problem_statement: str, target_col: str = None) -> dict:
    if target_col is None:
        target_col = df.columns[-1]

    problem_type = detect_problem_type(df, target_col)
    overview = compute_dataset_overview(df)

    column_stats = {}
    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64]:
            column_stats[col] = compute_numerical_stats(df[col])
        else:
            column_stats[col] = compute_categorical_stats(df[col])

    correlation = compute_correlation_matrix(df)
    target_analysis = compute_target_analysis(df, target_col, problem_type)

    result = {
        "problem_type": problem_type,
        "target_column": target_col,
        "overview": overview,
        "column_stats": column_stats,
        "correlation": correlation,
        "target_analysis": target_analysis
    }
    return clean_for_json(result)