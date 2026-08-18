from __future__ import annotations

from typing import Any

import pandas as pd


def analyze_dataset(
    dataframe: pd.DataFrame,
    date_column: str,
    feature_columns: list[str],
) -> dict[str, Any]:
    """Generate a structural summary of a dataset."""

    numeric_features = dataframe[
        feature_columns
    ].select_dtypes(include="number")

    return {
        "rows": int(dataframe.shape[0]),
        "columns": int(dataframe.shape[1]),
        "feature_count": len(feature_columns),
        "date_column": date_column,
        "date_min": _to_python(
            dataframe[date_column].min()
        ),
        "date_max": _to_python(
            dataframe[date_column].max()
        ),
        "unique_dates": int(
            dataframe[date_column].nunique()
        ),
        "duplicate_rows": int(
            dataframe.duplicated().sum()
        ),
        "numeric_features": int(
            len(numeric_features.columns)
        ),
        "total_missing_values": int(
            dataframe.isna().sum().sum()
        ),
    }


def _to_python(value: Any) -> Any:
    """Convert NumPy/Pandas scalar to Python scalar."""

    if hasattr(value, "item"):
        return value.item()

    return value