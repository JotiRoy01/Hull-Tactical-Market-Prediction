from __future__ import annotations

import json
from pathlib import Path

from markets.data_ingestion.loader import MarketDataLoader
from markets.data_ingestion.schema import MarketDataSchema
from markets.data_ingestion.validator import MarketDataValidator

from .correlation import (
    analyze_feature_correlation_matrix,
    analyze_target_correlations,
)
from .descriptive import analyze_dataset
from .distribution import analyze_distributions
from .leakage import (
    analyze_future_relationship,
    compare_feature_values_across_train_test,
)
from .missingness import (
    analyze_missingness,
    analyze_missingness_by_group,
    analyze_missingness_over_time,
)
from .target import (
    analyze_target_relationships,
    analyze_targets,
)
from .temporal import (
    analyze_feature_stability,
    analyze_target_autocorrelation,
    analyze_target_regimes,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_config() -> dict:
    """Load data configuration."""

    import yaml

    path = (
        PROJECT_ROOT
        / "configs"
        / "data.yaml"
    )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(file)


def save_json(
    data: dict,
    path: Path,
) -> None:
    """Save dictionary as JSON."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=2,
        )


def main() -> None:
    """Run the complete first-pass dataset analysis."""

    config = load_config()

    data_config = config["data"]
    schema_config = config["schema"]

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

    train, test = loader.load()

    validator = MarketDataValidator(
        schema=schema,
        **config["validation"],
    )

    validator.validate_train(train)
    validator.validate_test(test)

    feature_columns = list(
        schema.feature_columns
    )

    feature_groups = (
        schema.get_feature_groups(
            feature_columns
        )
    )

    output_dir = (
        PROJECT_ROOT
        / "reports"
        / "analysis"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------
    # 1. Descriptive analysis
    # -------------------------------------------------

    train_summary = analyze_dataset(
        train,
        schema.date_column,
        feature_columns,
    )

    test_summary = analyze_dataset(
        test,
        schema.date_column,
        feature_columns,
    )

    save_json(
        {
            "train": train_summary,
            "test": test_summary,
        },
        output_dir
        / "dataset_summary.json",
    )

    # -------------------------------------------------
    # 2. Missingness
    # -------------------------------------------------

    missingness = analyze_missingness(
        train,
        feature_columns,
    )

    missingness.to_csv(
        output_dir
        / "missingness.csv",
        index=False,
    )

    missingness_by_group = (
        analyze_missingness_by_group(
            train,
            feature_groups,
        )
    )

    missingness_by_group.to_csv(
        output_dir
        / "missingness_by_group.csv",
        index=False,
    )

    missingness_over_time = (
        analyze_missingness_over_time(
            train,
            schema.date_column,
            feature_columns,
        )
    )

    missingness_over_time.to_csv(
        output_dir
        / "missingness_over_time.csv",
        index=False,
    )

    # -------------------------------------------------
    # 3. Distribution
    # -------------------------------------------------

    distribution = analyze_distributions(
        train,
        feature_columns,
    )

    distribution.to_csv(
        output_dir
        / "distribution.csv",
        index=False,
    )

    # -------------------------------------------------
    # 4. Correlations
    # -------------------------------------------------

    target_correlations = (
        analyze_target_correlations(
            train,
            feature_columns,
            "market_forward_excess_returns",
        )
    )

    target_correlations.to_csv(
        output_dir
        / "correlations.csv",
        index=False,
    )

    feature_correlation = (
        analyze_feature_correlation_matrix(
            train,
            feature_columns,
            method="spearman",
        )
    )

    feature_correlation.to_csv(
        output_dir
        / "feature_correlation_matrix.csv"
    )

    # -------------------------------------------------
    # 5. Target analysis
    # -------------------------------------------------

    target_summary = analyze_targets(
        train
    )

    target_relationships = (
        analyze_target_relationships(
            train
        )
    )

    target_relationships.to_csv(
        output_dir
        / "target_relationships.csv"
    )

    save_json(
        target_summary,
        output_dir
        / "target_analysis.json",
    )

    # -------------------------------------------------
    # 6. Temporal analysis
    # -------------------------------------------------

    stability = analyze_feature_stability(
        train,
        schema.date_column,
        feature_columns,
    )

    stability.to_csv(
        output_dir
        / "temporal_stability.csv",
        index=False,
    )

    target_autocorrelation = (
        analyze_target_autocorrelation(
            train,
            "market_forward_excess_returns",
        )
    )

    target_autocorrelation.to_csv(
        output_dir
        / "target_autocorrelation.csv",
        index=False,
    )

    target_regimes = analyze_target_regimes(
        train,
        "market_forward_excess_returns",
    )

    target_regimes.to_csv(
        output_dir
        / "target_regimes.csv",
        index=False,
    )

    # -------------------------------------------------
    # 7. Leakage / shift screening
    # -------------------------------------------------

    leakage_flags = (
        analyze_future_relationship(
            train,
            feature_columns,
            "market_forward_excess_returns",
        )
    )

    leakage_flags.to_csv(
        output_dir
        / "leakage_screen.csv",
        index=False,
    )

    train_test_shift = (
        compare_feature_values_across_train_test(
            train,
            test,
            feature_columns,
        )
    )

    train_test_shift.to_csv(
        output_dir
        / "train_test_shift.csv",
        index=False,
    )

    print("=" * 60)
    print("HULL TACTICAL MARKET PREDICTION")
    print("FIRST-PASS ANALYSIS")
    print("=" * 60)

    print(f"Train shape : {train.shape}")
    print(f"Test shape  : {test.shape}")
    print(f"Features    : {len(feature_columns)}")

    print(
        "\nAnalysis reports saved to:"
    )

    print(output_dir)

    print("\nCompleted:")
    print("  ✓ Dataset summary")
    print("  ✓ Missingness")
    print("  ✓ Feature distributions")
    print("  ✓ Target correlations")
    print("  ✓ Feature correlation matrix")
    print("  ✓ Target analysis")
    print("  ✓ Temporal stability")
    print("  ✓ Target autocorrelation")
    print("  ✓ Target regimes")
    print("  ✓ Leakage screening")
    print("  ✓ Train/test distribution shift")

    print("=" * 60)


if __name__ == "__main__":
    main()