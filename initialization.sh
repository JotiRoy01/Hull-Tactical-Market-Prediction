#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="$(basename "$BASE_DIR")"

echo "Initializing layout for: $PROJECT_NAME"

# Array of standard project top-level directories
ROOT_DIRS=(
    "configs"
    "notebooks"
    "experiments"
    "reports"
    "submissions"
    "tests"
    "docs"
    "logs"
)

# Array of storage layers for data partitioning
DATA_SUBDIRS=(
    "raw"
    "interim"
    "processed"
    "external"
    "validation"
)

# Array of quantitative core components for source code
SRC_SUBDIRS=(
    "data_ingestion"
    "preprocessing"
    "features"
    "signals"
    "models"
    "portfolio"
    "risk"
    "backtest"
    "evaluation"
    "visualization"
    "utils"
)

# 1. Loop through and create general project folders
for dir in "${ROOT_DIRS[@]}"; do
    mkdir -p "$BASE_DIR/$dir"
done

# 2. Loop through and create data engineering structures
for sub_dir in "${DATA_SUBDIRS[@]}"; do
    mkdir -p "$BASE_DIR/data/$sub_dir"
done

# 3. Loop through and create Python package sub-modules
for src_dir in "${SRC_SUBDIRS[@]}"; do
    mkdir -p "$BASE_DIR/src/markets/$src_dir"
done

# 4. Generate root structural files without overwriting existing files
for file in requirements.txt README.md; do
    if [ ! -e "$BASE_DIR/$file" ]; then
        touch "$BASE_DIR/$file"
    fi
done

echo "Success! Project structure is ready."