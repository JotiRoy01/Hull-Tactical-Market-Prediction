from __future__ import annotations

import pandas as pd


def analyze_target_correlations(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
) -> pd.DataFrame:
    """
    Calculate feature-target Pearson and Spearman correlations.

    This is exploratory evidence, not proof of predictive power.
    """

    rows = []

    target = dataframe[target_column]

    for feature in feature_columns:

        pair = dataframe[
            [feature, target_column]
        ].dropna()

        if len(pair) < 3:
            pearson = None
            spearman = None
        else:
            pearson = pair[feature].corr(
                pair[target_column],
                method="pearson",
            )

            spearman = pair[feature].corr(
                pair[target_column],
                method="spearman",
            )

        rows.append(
            {
                "feature": feature,
                "pearson": pearson,
                "spearman": spearman,
                "absolute_pearson": (
                    abs(pearson)
                    if pearson is not None
                    else None
                ),
                "absolute_spearman": (
                    abs(spearman)
                    if spearman is not None
                    else None
                ),
                "observations": len(pair),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            "absolute_spearman",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def analyze_feature_correlation_matrix(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
    method: str = "spearman",
) -> pd.DataFrame:
    """Create a feature-to-feature correlation matrix."""

    return dataframe[
        feature_columns
    ].corr(method=method)