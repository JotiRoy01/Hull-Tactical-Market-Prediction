from __future__ import annotations

import numpy as np
import pandas as pd


def analyze_feature_stability(
    dataframe: pd.DataFrame,
    date_column: str,
    feature_columns: list[str],
    n_periods: int = 5,
) -> pd.DataFrame:
    """
    Compare feature behavior across chronological periods.

    For every feature and time period we calculate:
    - mean
    - standard deviation
    - missing fraction
    """

    ordered = dataframe.sort_values(
        date_column
    ).reset_index(drop=True)

    periods = pd.qcut(
        ordered.index,
        q=min(n_periods, len(ordered)),
        labels=False,
        duplicates="drop",
    )

    rows = []

    for period in sorted(
        periods.dropna().unique()
    ):

        period_data = ordered.loc[
            periods == period
        ]

        for feature in feature_columns:

            series = period_data[
                feature
            ]

            rows.append(
                {
                    "period": int(period),
                    "feature": feature,
                    "mean": float(
                        series.mean()
                    ),
                    "std": float(
                        series.std()
                    ),
                    "missing_fraction": float(
                        series.isna().mean()
                    ),
                }
            )

    return pd.DataFrame(rows)


def analyze_target_autocorrelation(
    dataframe: pd.DataFrame,
    target_column: str,
    max_lag: int = 20,
) -> pd.DataFrame:
    """
    Calculate autocorrelation of the target over time.
    """

    target = dataframe[
        target_column
    ].dropna()

    rows = []

    for lag in range(
        1,
        max_lag + 1,
    ):

        correlation = target.autocorr(
            lag=lag
        )

        rows.append(
            {
                "lag": lag,
                "autocorrelation": correlation,
            }
        )

    return pd.DataFrame(rows)


def analyze_target_regimes(
    dataframe: pd.DataFrame,
    target_column: str,
    n_periods: int = 10,
) -> pd.DataFrame:
    """
    Examine whether target behavior changes over time.
    """

    periods = pd.qcut(
        dataframe.index,
        q=min(
            n_periods,
            len(dataframe),
        ),
        labels=False,
        duplicates="drop",
    )

    target = dataframe[
        target_column
    ]

    result = (
        pd.DataFrame(
            {
                "period": periods,
                "target": target,
            }
        )
        .groupby("period")
        .agg(
            mean_return=(
                "target",
                "mean",
            ),
            volatility=(
                "target",
                "std",
            ),
            observations=(
                "target",
                "count",
            ),
        )
        .reset_index()
    )

    return result