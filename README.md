# Hank's Tank MLB Prediction Pipeline

Senior-level ML/analytics project that builds an end-to-end MLB prediction system: data collection, validation, feature engineering, model training, and daily inference readiness. The pipeline uses historical game data (2015–2025) with a strict temporal split to avoid leakage and benchmarks models against a 2025 holdout season.

## Highlights

- **Reproducible ML workflow** with clear train/validation splits (2015–2024 train, 2025 validation)
- **Feature engineering at scale** (57 engineered features, rolling form, rest, park effects, matchup deltas)
- **Model evaluation and documentation** with traceable performance comparisons
- **Production-oriented tooling** (data validation reports, automation scripts, model artifacts)

## Results (2025 Holdout)

| Version | Algorithm | Features | Accuracy | AUC |
|---|---|---:|---:|---:|
| V1 | Logistic Regression | 5 | 54.0% | 0.543 |
| V2 | Logistic Regression | 44 | 54.4% | 0.534 |
| V3 | XGBoost | 57 | **54.6%** | **0.546** |

Detailed write-ups:
- [docs/V3_TRAINING_RESULTS.md](docs/V3_TRAINING_RESULTS.md)
- [docs/MODEL_EVOLUTION_COMPLETE.md](docs/MODEL_EVOLUTION_COMPLETE.md)

## Tech Stack

- Python, pandas, NumPy, scikit-learn
- XGBoost, LightGBM, Optuna (experimentation)
- BigQuery ingestion and validation
- Parquet-based training datasets

## Repo Structure

- [src](src) — feature engineering and model training scripts
- [data](data) — training/validation datasets and logs
- [docs](docs) — model results and validation summaries
- [scripts](scripts) — automation and validation helpers
- [notebooks](notebooks) — exploration and prototyping

## Key Scripts

- [src/build_training_data.py](src/build_training_data.py) — base feature pipeline (V1)
- [src/build_v2_features.py](src/build_v2_features.py) — enhanced features (V2)
- [src/build_v3_features.py](src/build_v3_features.py) — advanced features (V3)
- [src/train_game_models.py](src/train_game_models.py) — baseline training
- [src/train_v2_models.py](src/train_v2_models.py) — V2 training
- [src/train_v3_models.py](src/train_v3_models.py) — V3 training (best)
- [src/train_v4_models.py](src/train_v4_models.py) — experimental stacked ensemble
- [src/train_v4_tuned.py](src/train_v4_tuned.py) — experimental hyperparameter tuning
- [src/predict_2026_games.py](src/predict_2026_games.py) — 2026 inference pipeline

## Data Quality & Validation

Daily validation reports are generated during season monitoring.

- [docs/BIGQUERY_DATA_SCHEMA.md](docs/BIGQUERY_DATA_SCHEMA.md)
- [logs/validation](logs/validation)

## Notes

This project focuses on deterministic, leakage-safe MLB prediction modeling. Future iterations aim to add external signals (injuries, roster changes, market lines) to push beyond the historical-data ceiling.
