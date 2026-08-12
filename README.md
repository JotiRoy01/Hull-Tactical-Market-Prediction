# Hull-Tactical-Market-Prediction
Hull Tactical Market Prediction is a Kaggle competition project.

## Setup
### Prerequisites
* Python 3.10+
### Installation
1. Install the dependencies:
```
pip install -r requirements.txt
```
2. Install the virtual environment and package
```
conda create --prefix ./market python=3.10 -y

pip install -e .
```

## Suggested Project Structure
This competition is notebook-only on Kaggle, so the repository should stay small,
reproducible, and easy to export into a submission notebook.

```text
data/
	raw/            # original Kaggle files
	interim/        # cleaned intermediate tables
	processed/      # feature-ready artifacts
docs/             # problem statement, metric notes, data notes
experiments/      # quick local experiments and debugging scripts
notebooks/        # EDA, baseline, training, and submission notebooks
reports/          # figures, validation summaries, and findings
src/markets/
	data/           # loading, validation, schema, dataset wrappers
	features/       # feature engineering and transforms
	models/         # model training and inference wrappers
	evaluation/     # custom metric, CV, and backtest helpers
	portfolio/      # allocation, betting, and position sizing logic
	risk/           # volatility and drawdown constraints
	visualization/  # plots and reporting helpers
	utils/          # shared helpers
tests/            # unit tests for core pure-python logic
submissions/      # exported Kaggle submission artifacts
```

## Kaggle Build Order
1. Read the competition metric and submission rules first.
2. Build a local baseline notebook that can reproduce the Kaggle submission flow.
3. Move reusable code into `src/markets/` only after it stabilizes.
4. Keep the final notebook self-contained, since Kaggle requires notebook submission.
