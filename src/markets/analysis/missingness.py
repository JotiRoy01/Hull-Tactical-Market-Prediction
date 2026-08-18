from __future__ import annotations

import pandas as pd


def analyze_missingness(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """
    Analyze missing values for every feature.

    Returns one row per feature.
    """

    rows = len(dataframe)

    result = pd.DataFrame(
        {
            "feature": feature_columns,
            "missing_count": [
                int(
                    dataframe[column].isna().sum()
                )
                for column in feature_columns
            ],
        }
    )

    result["missing_fraction"] = (
        result["missing_count"] / rows
    )

    result["observed_count"] = (
        rows - result["missing_count"]
    )

    result["completely_missing"] = (
        result["missing_count"] == rows
    )

    return result.sort_values(
        "missing_fraction",
        ascending=False,
    ).reset_index(drop=True)


def analyze_missingness_by_group(
    dataframe: pd.DataFrame,
    feature_groups: dict[str, list[str]],
) -> pd.DataFrame:
    """Aggregate missingness by feature family."""

    rows = []

    for group, columns in feature_groups.items():

        missing_count = int(
            dataframe[columns]
            .isna()
            .sum()
            .sum()
        )

        total_cells = (
            dataframe[columns]
            .shape[0]
            * dataframe[columns]
            .shape[1]
        )

        rows.append(
            {
                "group": group,
                "feature_count": len(columns),
                "missing_count": missing_count,
                "missing_fraction": (
                    missing_count / total_cells
                    if total_cells
                    else 0.0
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            "missing_fraction",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def analyze_missingness_over_time(
    dataframe: pd.DataFrame,
    date_column: str,
    feature_columns: list[str],
    n_periods: int = 10,
) -> pd.DataFrame:
    """
    Measure missingness across chronological periods.

    This helps identify whether missingness itself changes
    with market regime or time.
    """

    ordered = dataframe.sort_values(
        date_column
    ).reset_index(drop=True)

    period_ids = pd.qcut(
        ordered.index,
        q=min(n_periods, len(ordered)),
        labels=False,
        duplicates="drop",
    )

    missing_fraction = (
        ordered[feature_columns]
        .isna()
        .mean(axis=1)
    )

    result = pd.DataFrame(
        {
            "period": period_ids,
            "missing_fraction": missing_fraction,
        }
    )

    return (
        result.groupby("period")
        .agg(
            mean_missing_fraction=(
                "missing_fraction",
                "mean",
            ),
            max_missing_fraction=(
                "missing_fraction",
                "max",
            ),
        )
        .reset_index()
    )