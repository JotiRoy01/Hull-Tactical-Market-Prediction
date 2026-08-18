from __future__ import annotations

import numpy as np
import pandas as pd


def analyze_distributions(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Generate distribution statistics for all features."""

    rows = []

    for feature in feature_columns:

        series = dataframe[feature].dropna()

        if series.empty:
            rows.append(
                {
                    "feature": feature,
                    "count": 0,
                    "mean": np.nan,
                    "std": np.nan,
                    "min": np.nan,
                    "q01": np.nan,
                    "q05": np.nan,
                    "median": np.nan,
                    "q95": np.nan,
                    "q99": np.nan,
                    "max": np.nan,
                    "skewness": np.nan,
                    "kurtosis": np.nan,
                }
            )
            continue

        rows.append(
            {
                "feature": feature,
                "count": int(series.count()),
                "mean": float(series.mean()),
                "std": float(series.std()),
                "min": float(series.min()),
                "q01": float(series.quantile(0.01)),
                "q05": float(series.quantile(0.05)),
                "median": float(series.median()),
                "q95": float(series.quantile(0.95)),
                "q99": float(series.quantile(0.99)),
                "max": float(series.max()),
                "skewness": float(series.skew()),
                "kurtosis": float(series.kurtosis()),
            }
        )

    return pd.DataFrame(rows)