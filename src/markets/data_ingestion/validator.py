from dataclasses import dataclass

import pandas as pd

from .schema import MarketDataSchema


class DataValidationError(ValueError):
    """Raised when the dataset violates the expected schema.
    """


@dataclass(frozen=True)
class ValidationReport:
    """Summary of a successful dataset validation."""

    dataset_name: str
    rows: int
    columns: int
    features: int
    missing_values: int


class MarketDataValidator:
    """Validate raw train and test datasets.
    _validate_expected_columns :
        take the pandas DataFrame, expected_column: set[str], dataset_name: str
        first collect the total numeber of column
        calcualte the missing columns
        calculate the unexpected column
    _validate_duplicate_columns :
        take the pandas DataFrame
        dataset_name: str
        check the duplicate columns
    _validate_date_columns :
        take the pandas DataFrame, dataset_name
        check whether exist any missing columns
        Is any null value
        Is numeric dtype
        check dataset is unique or not
        sort the datetime value
    _validate_features :
        take the pandas DataFrame, dataset_name
        check missing feature
        check non numeric
        If required fill the missing value
    _validate_train_specific_columns :
        take the DataFrame
        avoid the test_only_column from the schema
        check whether exist any unexpected_test_columns columns
        check the train the value numeric or not
    _validate_test_specific_columns :
        take the pandas DataFrame
        avoid the train dataset
    _build_report :
        take pandas DataFrame
        take dataset_name
        return the validationReport(dataset_name, row, columns, feature, missing_values)

        
    """

    def __init__(
        self,
        schema: MarketDataSchema,
        *,
        require_non_empty: bool = True,
        require_unique_dates: bool = True,
        require_sorted_dates: bool = True,
        require_numeric_features: bool = True,
        allow_missing_feature_values: bool = True,
    ) -> None:

        self.schema = schema

        self.require_non_empty = (
            require_non_empty
        )

        self.require_unique_dates = (
            require_unique_dates
        )

        self.require_sorted_dates = (
            require_sorted_dates
        )

        self.require_numeric_features = (
            require_numeric_features
        )

        self.allow_missing_feature_values = (
            allow_missing_feature_values
        )

    def validate_train(
        self,
        dataframe: pd.DataFrame,
    ) -> ValidationReport:
        """Validate train.csv."""

        self._validate_common(
            dataframe=dataframe,
            dataset_name="train",
        )

        self._validate_expected_columns(
            dataframe=dataframe,
            expected_columns=set(
                self.schema.train_columns
            ),
            dataset_name="train",
        )

        self._validate_train_specific_columns(
            dataframe
        )

        return self._build_report(
            dataframe,
            "train",
        )

    def validate_test(
        self,
        dataframe: pd.DataFrame,
    ) -> ValidationReport:
        """Validate test.csv."""

        self._validate_common(
            dataframe=dataframe,
            dataset_name="test",
        )

        self._validate_expected_columns(
            dataframe=dataframe,
            expected_columns=set(
                self.schema.test_columns
            ),
            dataset_name="test",
        )

        self._validate_test_specific_columns(
            dataframe
        )

        return self._build_report(
            dataframe,
            "test",
        )

    def _validate_common(
        self,
        dataframe: pd.DataFrame,
        dataset_name: str,
    ) -> None:

        if (
            self.require_non_empty
            and dataframe.empty
        ):
            raise DataValidationError(
                f"{dataset_name}: dataset is empty."
            )

        self._validate_duplicate_columns(
            dataframe,
            dataset_name,
        )

        self._validate_date_column(
            dataframe,
            dataset_name,
        )

        self._validate_features(
            dataframe,
            dataset_name,
        )

    def _validate_expected_columns(
        self,
        dataframe: pd.DataFrame,
        expected_columns: set[str],
        dataset_name: str,
    ) -> None:

        actual_columns = set(
            dataframe.columns
        )

        missing_columns = sorted(
            expected_columns - actual_columns
        )

        unexpected_columns = sorted(
            actual_columns - expected_columns
        )

        if missing_columns:
            raise DataValidationError(
                f"{dataset_name}: missing expected "
                f"columns: {missing_columns}"
            )

        if unexpected_columns:
            raise DataValidationError(
                f"{dataset_name}: unexpected "
                f"columns: {unexpected_columns}"
            )

    @staticmethod
    def _validate_duplicate_columns(
        dataframe: pd.DataFrame,
        dataset_name: str,
    ) -> None:

        duplicated = (
            dataframe.columns[
                dataframe.columns.duplicated()
            ]
            .tolist()
        )

        if duplicated:
            raise DataValidationError(
                f"{dataset_name}: duplicate "
                f"column names: {duplicated}"
            )

    def _validate_date_column(
        self,
        dataframe: pd.DataFrame,
        dataset_name: str,
    ) -> None:

        date_column = self.schema.date_column

        if date_column not in dataframe.columns:
            raise DataValidationError(
                f"{dataset_name}: missing date "
                f"column '{date_column}'."
            )

        if dataframe[date_column].isna().any():
            raise DataValidationError(
                f"{dataset_name}: '{date_column}' "
                f"contains missing values."
            )

        if not pd.api.types.is_numeric_dtype(
            dataframe[date_column]
        ):
            raise DataValidationError(
                f"{dataset_name}: '{date_column}' "
                f"must be numeric."
            )

        if (
            self.require_unique_dates
            and not dataframe[
                date_column
            ].is_unique
        ):
            raise DataValidationError(
                f"{dataset_name}: '{date_column}' "
                f"contains duplicate values."
            )

        if (
            self.require_sorted_dates
            and not dataframe[
                date_column
            ].is_monotonic_increasing
        ):
            raise DataValidationError(
                f"{dataset_name}: '{date_column}' "
                f"is not sorted in ascending order."
            )

    def _validate_features(
        self,
        dataframe: pd.DataFrame,
        dataset_name: str,
    ) -> None:

        features = self.schema.feature_columns

        missing_features = [
            feature
            for feature in features
            if feature not in dataframe.columns
        ]

        if missing_features:
            raise DataValidationError(
                f"{dataset_name}: missing feature "
                f"columns: {missing_features}"
            )

        if self.require_numeric_features:

            non_numeric = [
                feature
                for feature in features
                if not pd.api.types.is_numeric_dtype(
                    dataframe[feature]
                )
            ]

            if non_numeric:
                raise DataValidationError(
                    f"{dataset_name}: non-numeric "
                    f"feature columns: {non_numeric}"
                )

        if (
            not self.allow_missing_feature_values
        ):

            missing_values = (
                dataframe[list(features)]
                .isna()
                .sum()
            )

            columns_with_missing = (
                missing_values[
                    missing_values > 0
                ]
            )

            if not columns_with_missing.empty:
                raise DataValidationError(
                    f"{dataset_name}: missing feature "
                    f"values detected: "
                    f"{columns_with_missing.to_dict()}"
                )

    def _validate_train_specific_columns(
        self,
        dataframe: pd.DataFrame,
    ) -> None:

        unexpected_test_columns = [
            column
            for column in self.schema.test_only_columns
            if column in dataframe.columns
        ]

        if unexpected_test_columns:
            raise DataValidationError(
                "train: test-only columns found: "
                f"{unexpected_test_columns}"
            )

        for column in self.schema.train_only_columns:

            if not pd.api.types.is_numeric_dtype(
                dataframe[column]
            ):
                raise DataValidationError(
                    f"train: '{column}' must "
                    f"be numeric."
                )

    def _validate_test_specific_columns(
        self,
        dataframe: pd.DataFrame,
    ) -> None:

        unexpected_train_columns = [
            column
            for column in self.schema.train_only_columns
            if column in dataframe.columns
        ]

        if unexpected_train_columns:
            raise DataValidationError(
                "test: training-only columns found: "
                f"{unexpected_train_columns}"
            )

    def _build_report(
        self,
        dataframe: pd.DataFrame,
        dataset_name: str,
    ) -> ValidationReport:

        return ValidationReport(
            dataset_name=dataset_name,
            rows=len(dataframe),
            columns=len(dataframe.columns),
            features=self.schema.n_features,
            missing_values=int(
                dataframe.isna()
                .sum()
                .sum()
            ),
        )