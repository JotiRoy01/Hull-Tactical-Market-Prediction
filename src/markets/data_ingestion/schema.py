import re
from dataclasses import dataclass

import pandas as pd


_FEATURE_PATTERN = re.compile(
    r"^(?P<group>[A-Z])(?P<number>\d+)$"
)


@dataclass(frozen=True)
class MarketDataSchema:
    """Expected structural schema of the competition dataset.
    feature_columns :
        The dataset contain same column name in multiple time. so this function count the group and collect them
    n_feature :
        it return the total number of column from feature_columns return columns name
    train_columns :
        it return all the feature whose should be trained in model traing
    test_columns :
        it return all the feature whose should be trained in model testing
    get_feature_groups :
        it define the group name and the index number in this order
        group D index 1
        
    """

    date_column: str
    feature_groups: dict[str, int]
    train_only_columns: tuple[str, ...]
    test_only_columns: tuple[str, ...]

    @property
    def feature_columns(self) -> tuple[str, ...]:
        """
        Return all expected feature names in deterministic order.
        """

        columns: list[str] = []

        for group, count in self.feature_groups.items():
            for index in range(1, count + 1):
                columns.append(
                    f"{group}{index}"
                )
                print(f"group {group} index {index}")
        print(f"columns {columns}")
        return tuple(columns)

    @property
    def n_features(self) -> int:
        """Return the total number of expected features."""
        return len(self.feature_columns)

    @property
    def train_columns(self) -> tuple[str, ...]:
        """Return all expected training columns."""

        return (
            self.date_column,
            *self.feature_columns,
            *self.train_only_columns,
        )

    @property
    def test_columns(self) -> tuple[str, ...]:
        """Return all expected test columns."""

        return (
            self.date_column,
            *self.feature_columns,
            *self.test_only_columns,
        )

    def get_feature_groups(
        self,
        columns: list[str] | pd.Index,
    ) -> dict[str, list[str]]:
        """
        Group actual feature columns by family.

        Example:
        {
            "D": ["D1", "D2", ...],
            "E": ["E1", "E2", ...],
        }
        """

        groups = {
            group: []
            for group in self.feature_groups
        }

        for column in columns:
            column = str(column)

            match = _FEATURE_PATTERN.fullmatch(column)

            if not match:
                continue

            group = match.group("group")

            if group in groups:
                groups[group].append(column)

        for group in groups:
            groups[group].sort(
                key=lambda value: int(
                    _FEATURE_PATTERN.fullmatch(
                        value
                    ).group("number")
                )
            )
        #print(f"goupe {group}")
        return groups

    def build_from_config(
        self,
    ) -> None:
        """
        Reserved for future schema-generation logic.

        Keeping the schema definition centralized makes it easier
        to extend the dataset contract later.
        """
        return None