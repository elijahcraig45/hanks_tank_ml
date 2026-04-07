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
- [src/train_v6_models.py](src/train_v6_models.py) — pitcher arsenal + venue splits
- [src/train_v7_models.py](src/train_v7_models.py) — **V7 (bullpen + moon + venue)** ⭐
- [src/predict_today_games.py](src/predict_today_games.py) — 2026 daily inference
- [src/predict_2026_games.py](src/predict_2026_games.py) — batch prediction pipeline
- [src/backfill_v7_predictions.py](src/backfill_v7_predictions.py) — backfill missed predictions

## V7 Model (2026 Production)

The V7 model is the current production version deployed for daily autonomous predictions.

**Setup & Deployment**:
- [docs/V7_AUTONOMOUS_SETUP.md](docs/V7_AUTONOMOUS_SETUP.md) — Complete setup guide for training, cloud scheduler config, and troubleshooting

**Quick Start**:
```bash
# Train V7 model locally
python src/train_v7_models.py

# Make predictions for today
python src/predict_today_games.py

# Backfill missed dates
python src/backfill_v7_predictions.py --start 2026-03-27 --end 2026-04-06
```

**Performance (2025 Validation)**:
- Accuracy: 56.31%
- AUC: 0.5665
- Features: 113 (V6 base + bullpen health + moon phase + pitcher venue splits)

## Data Quality & Validation

Daily validation reports are generated during season monitoring.

- [docs/BIGQUERY_DATA_SCHEMA.md](docs/BIGQUERY_DATA_SCHEMA.md)
- [logs/validation](logs/validation)

## Notes

This project focuses on deterministic, leakage-safe MLB prediction modeling with production automation. The V7 model adds bullpen health, moon phase, and pitcher venue dynamics for better 2026 predictions.
