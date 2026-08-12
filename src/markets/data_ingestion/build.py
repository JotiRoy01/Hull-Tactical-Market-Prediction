from pathlib import Path

import yaml

from .loader import MarketDataLoader
from .schema import MarketDataSchema
from .validator import MarketDataValidator


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config() -> dict:
    """Load the data configuration."""

    config_path = (
        PROJECT_ROOT
        / "configs"
        / "data.yaml"
    )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(file)


def main() -> None:
    """Load and validate competition data.
    frist collect the train and test path.
    build the schema of the dataset(data_column, feature_group, train_only_columns, test_only_columns)
    load and validate the using the MarketDataLoader , MarketDataValidator
    also load the train and test datda from loader.load() function
    generated the test report
    """

    config = load_config()

    data_config = config["data"]
    schema_config = config["schema"]
    validation_config = config["validation"]

    train_path = (
        PROJECT_ROOT
        / data_config["train_path"]
    )

    test_path = (
        PROJECT_ROOT
        / data_config["test_path"]
    )

    schema = MarketDataSchema(
        date_column=schema_config[
            "date_column"
        ],
        feature_groups=schema_config[
            "feature_groups"
        ],
        train_only_columns=tuple(
            schema_config[
                "train_only_columns"
            ]
        ),
        test_only_columns=tuple(
            schema_config[
                "test_only_columns"
            ]
        ),
    )

    loader = MarketDataLoader(
        train_path=train_path,
        test_path=test_path,
    )

    validator = MarketDataValidator(
        schema=schema,
        **validation_config,
    )

    train, test = loader.load()

    train_report = (
        validator.validate_train(
            train
        )
    )

    test_report = (
        validator.validate_test(
            test
        )
    )

    print("=" * 60)
    print("HULL TACTICAL MARKET PREDICTION")
    print("DATA VALIDATION")
    print("=" * 60)

    print(
        f"Train shape: "
        f"{train.shape}"
    )

    print(
        f"Test shape:  "
        f"{test.shape}"
    )

    print(
        f"Expected features: "
        f"{schema.n_features}"
    )

    print(
        f"Train missing values: "
        f"{train_report.missing_values:,}"
    )

    print(
        f"Test missing values: "
        f"{test_report.missing_values:,}"
    )

    print("\nFeature groups:")

    for (
        group,
        count,
    ) in schema.feature_groups.items():

        print(
            f"  {group}: {count}"
        )

    print("\nValidation successful.")


if __name__ == "__main__":
    main()