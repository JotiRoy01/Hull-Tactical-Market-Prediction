from __future__ import annotations

from typing import Any

import pandas as pd


TARGET_COLUMNS = (
    "forward_returns",
    "risk_free_rate",
    "market_forward_excess_returns",
)


def analyze_targets(
    dataframe: pd.DataFrame,
    target_columns: tuple[str, ...] = TARGET_COLUMNS,
) -> dict[str, Any]:
    """Analyze target/reference variables."""

    result: dict[str, Any] = {}

    for target in target_columns:

        series = dataframe[target].dropna()

        result[target] = {
            "count": int(series.count()),
            "missing": int(
                dataframe[target].isna().sum()
            ),
            "mean": float(series.mean()),
            "std": float(series.std()),
            "min": float(series.min()),
            "q01": float(series.quantile(0.01)),
            "median": float(series.median()),
            "q99": float(series.quantile(0.99)),
            "max": float(series.max()),
            "skewness": float(series.skew()),
            "kurtosis": float(series.kurtosis()),
        }

    return result


def analyze_target_relationships(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Analyze relationships between the three
    training-only market variables.
    """

    return dataframe[
        list(TARGET_COLUMNS)
    ].corr(method="pearson")