from __future__ import annotations

import pandas as pd


def analyze_future_relationship(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    max_lag: int = 5,
) -> pd.DataFrame:
    """
    Look for suspicious relationships between current features
    and future target values.

    This is a screening tool, not a formal proof of leakage.
    """

    rows = []

    for feature in feature_columns:

        feature_series = dataframe[
            feature
        ]

        for lag in range(
            0,
            max_lag + 1,
        ):

            future_target = dataframe[
                target_column
            ].shift(-lag)

            pair = pd.concat(
                [
                    feature_series,
                    future_target,
                ],
                axis=1,
            ).dropna()

            if len(pair) < 3:
                correlation = None
            else:
                correlation = pair.iloc[
                    :, 0
                ].corr(
                    pair.iloc[:, 1]
                )

            rows.append(
                {
                    "feature": feature,
                    "target": target_column,
                    "future_lag": lag,
                    "correlation": correlation,
                    "absolute_correlation": (
                        abs(correlation)
                        if correlation is not None
                        else None
                    ),
                    "observations": len(pair),
                }
            )

    return (
        pd.DataFrame(rows)
        .sort_values(
            "absolute_correlation",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def compare_feature_values_across_train_test(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """
    Compare train and test feature distributions.

    Large distribution differences may indicate
    dataset shift. They are not automatically leakage.
    """

    rows = []

    for feature in feature_columns:

        train_series = train[
            feature
        ].dropna()

        test_series = test[
            feature
        ].dropna()

        rows.append(
            {
                "feature": feature,
                "train_mean": float(
                    train_series.mean()
                ),
                "test_mean": float(
                    test_series.mean()
                ),
                "train_std": float(
                    train_series.std()
                ),
                "test_std": float(
                    test_series.std()
                ),
                "train_missing_fraction": float(
                    train[feature].isna().mean()
                ),
                "test_missing_fraction": float(
                    test[feature].isna().mean()
                ),
            }
        )

    return pd.DataFrame(rows)