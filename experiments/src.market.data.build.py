"""Simple smoke test for the project data files.

This script uses:
- configs/data.yaml
- data/raw/train.csv
- data/raw/test.csv

It loads the real project data and validates it with the data package.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from src.markets.data.loader import MarketDataLoader
from src.markets.data.schema import MarketDataSchema
from src.markets.data.validator import MarketDataValidator


def load_config() -> dict:
    config_path = PROJECT_ROOT / "configs" / "data.yaml"
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main() -> None:
    config = load_config()

    data_config = config["data"]
    schema_config = config["schema"]
    validation_config = config["validation"]

    schema = MarketDataSchema(
        date_column=schema_config["date_column"],
        feature_groups=schema_config["feature_groups"],
        train_only_columns=tuple(schema_config["train_only_columns"]),
        test_only_columns=tuple(schema_config["test_only_columns"]),
    )

    loader = MarketDataLoader(
        train_path=PROJECT_ROOT / data_config["train_path"],
        test_path=PROJECT_ROOT / data_config["test_path"],
    )

    validator = MarketDataValidator(schema=schema, **validation_config)

    train, test = loader.load()
    train_report = validator.validate_train(train)
    test_report = validator.validate_test(test)

    print("Project data smoke test")
    print(f"train shape: {train.shape}")
    print(f"test shape:  {test.shape}")
    print(f"expected features: {schema.n_features}")
    print(f"train report: {train_report}")
    print(f"test report:  {test_report}")
    print("Validation passed")


if __name__ == "__main__":
    main()