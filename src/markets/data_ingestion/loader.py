from pathlib import Path

import pandas as pd


class MarketDataLoader:
    """Load raw Hull Tactical Market Prediction datasets."""

    def __init__(
        self,
        train_path: str | Path,
        test_path: str | Path,
    ) -> None:
        self.train_path = Path(train_path)
        self.test_path = Path(test_path)

    @staticmethod
    def _validate_file(path: Path) -> None:
        """Validate that a configured data file exists."""

        if not path.exists():
            raise FileNotFoundError(
                f"Data file does not exist: {path}"
            )

        if not path.is_file():
            raise ValueError(
                f"Data path is not a file: {path}"
            )

        if path.suffix.lower() != ".csv":
            raise ValueError(
                f"Expected a CSV file, got: {path}"
            )

    @classmethod
    def _load_csv(cls, path: Path) -> pd.DataFrame:
        """Load one CSV file without modifying its contents."""

        cls._validate_file(path)

        dataframe = pd.read_csv(path)

        if dataframe.empty:
            raise ValueError(
                f"Dataset is empty: {path}"
            )

        return dataframe

    def load_train(self) -> pd.DataFrame:
        """Load the training dataset."""
        return self._load_csv(self.train_path)

    def load_test(self) -> pd.DataFrame:
        """Load the test dataset."""
        return self._load_csv(self.test_path)

    def load(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load train and test datasets."""
        train = self.load_train()
        test = self.load_test()

        return train, test