# Hank's Tank ML — MLB Game Prediction Pipeline

> **Production endpoint:** Cloud Function `mlb-2026-daily-pipeline` (us-central1)  
> **Part of:** [hanks_tank](../hanks_tank) · [hanks_tank_backend](../hanks_tank_backend) · **hanks_tank_ml** ←

End-to-end MLB game outcome prediction system. Trains an ensemble of gradient boosting and neural-network models on 11 seasons of historical data (2015–2024), then runs daily inference for every scheduled game and writes predictions to BigQuery for the frontend to consume.

**Best result — V8 Ensemble (live in production, 2026 season):**

| Metric | Value |
|---|---|
| Accuracy (2025 holdout, 1,243 games) | **57.65%** |
| AUC-ROC | 0.613 |
| Models | CatBoost (home) + CatBoost (away) + LightGBM + MLP |
| Features | 89 engineered features |

---

## 📈 Model Evolution

| Version | Algorithm | Features | Accuracy | Notes |
|---|---|---:|---:|---|
| V1 | Logistic Regression | 5 | 54.0% | Baseline |
| V2 | Logistic Regression | 44 | 54.4% | Expanded features |
| V3 | XGBoost | 57 | 54.6% | First tree model |
| V4 | Stacked Ensemble | 57 | 54.9% | Experimental |
| V5 | CatBoost | 57 | 55.1% | Better categoricals |
| V6 | CatBoost + arsenal | 68 | 55.8% | Pitcher arsenal splits |
| V7 | CatBoost + bullpen/moon/venue | 78 | 56.4% | ⭐ Previous production |
| **V8** | **CatBoost×2 + LGB + MLP** | **89** | **57.65%** | **✅ Current production** |

Detailed experiment write-ups: [`docs/`](docs/)

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────┐
│           Daily Pipeline (Cloud Function Gen2)           │
│                                                          │
│  1. Fetch today's schedule  ← MLB Stats API              │
│  2. Load Elo ratings        ← BigQuery elo_ratings       │
│  3. Load team features      ← BigQuery team_features     │
│  4. Build 89 features per game                           │
│  5. Run V8 ensemble inference                            │
│     ├ CatBoost (home-focused)                            │
│     ├ CatBoost (away-focused)                            │
│     ├ LightGBM                                           │
│     └ MLP (PyTorch)                                      │
│  6. Soft-vote → win probability + confidence tier        │
│  7. Write predictions → BigQuery game_predictions        │
└──────────────────────────────────────────────────────────┘
         │                                  ↑
Cloud Scheduler (6 AM ET daily)        V8 model artifacts (GCS)
```

---

## 🤖 Feature Groups (V8, 89 features)

| Group | Count | Examples |
|---|---:|---|
| Elo ratings | 4 | elo_home, elo_away, elo_differential, elo_home_win_prob |
| Pythagorean W% | 4 | home/away_pythag_season, pythag_differential |
| Recent form (L10) | 8 | home/away_run_diff_10g, run_diff_7g, run_diff_3g |
| Win/loss streaks | 2 | home/away_current_streak |
| Head-to-head | 3 | h2h_win_pct_3yr, h2h_game_count_3yr |
| Pitcher arsenal | 20 | avg_fastball_velo, strikeout_rate, woba_allowed, xFIP … |
| Bullpen | 6 | home/away_bullpen_era, fatigue, saves_pct … |
| Venue / park | 8 | venue_era, altitude, park_factor_hr … |
| Lineup wOBA | 6 | home/away_lineup_woba_vs_hand … |
| Schedule | 4 | days_rest, is_divisional, is_home … |
| Moon phase | 1 | moon_phase |
| Misc / flags | 23 | game_time, temp, wind, pitcher ERA/WHIP … |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.12 |
| Gradient boosting | CatBoost 1.2, LightGBM 4.x |
| Neural network | PyTorch 2.x (MLP) |
| Hyperparameter tuning | Optuna |
| Data processing | pandas, NumPy, pyarrow |
| Data warehouse | Google BigQuery (`mlb_2026_season` dataset) |
| Model storage | Google Cloud Storage (`gs://hankstank-models/`) |
| Serving | Cloud Functions Gen2 (2 GB RAM, 2 vCPU, 540s timeout) |
| Scheduling | Cloud Scheduler (6 AM ET) |

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install -r requirements.txt

# Authenticate to GCP
gcloud auth application-default login
export GCP_PROJECT=hankstank
export BIGQUERY_DATASET=mlb_2026_season
```

### Run Pipeline Locally

```bash
python src/daily_pipeline.py --date $(date +%Y-%m-%d)
```

### Train a New Model Version

```bash
# Build training data
python src/build_v8_features.py

# Train ensemble
python src/train_v8_models.py

# Evaluate on 2025 holdout
python scripts/evaluate_model.py --model v8 --holdout 2025
```

### Deploy Cloud Function

```bash
gcloud functions deploy mlb-2026-daily-pipeline \
  --gen2 --region=us-central1 --runtime=python312 \
  --source=src/ --entry-point=daily_pipeline \
  --memory=2048MB --cpu=2 --timeout=540s \
  --service-account=hankstank@hankstank.iam.gserviceaccount.com \
  --set-env-vars="GCP_PROJECT=hankstank,BIGQUERY_DATASET=mlb_2026_season" \
  --trigger-http --allow-unauthenticated
```

---

## 📁 Repository Structure

```
hanks_tank_ml/
├── src/
│   ├── daily_pipeline.py       # Cloud Function entry point
│   ├── build_v8_features.py    # Feature engineering (V8, 89 features)
│   ├── train_v8_models.py      # Ensemble training (CB×2 + LGB + MLP)
│   ├── elo_system.py           # Elo rating engine (seeded from 2015)
│   ├── inference.py            # Batch inference + BQ write
│   └── ...
├── scripts/
│   ├── seed_elo.py             # Bootstrap Elo from 2015 historical data
│   ├── backfill_features.py    # Backfill team_features_2026 table
│   └── evaluate_model.py       # Holdout accuracy / AUC report
├── docs/
│   ├── V8_FEATURES.md          # Full V8 feature documentation
│   ├── V8_EXPERIMENT_COMPLETE.md
│   ├── MODEL_EVOLUTION_COMPLETE.md
│   ├── BIGQUERY_DATA_SCHEMA.md
│   └── CONFIDENCE_QUICK_CARD.md
├── notebooks/                  # EDA and prototyping
├── data/training/              # Parquet training datasets
├── cloud_functions/            # Legacy CF scaffolding
└── requirements.txt
```

---

## 📊 BigQuery Schema (key table)

```sql
-- mlb_2026_season.game_predictions
SELECT
  game_pk, game_date, home_team_name, away_team_name,
  home_win_probability, away_win_probability,
  predicted_winner, confidence_tier,
  model_version, model_accuracy, lineup_confirmed,
  -- V8 signals
  elo_home, elo_away, elo_differential, elo_home_win_prob,
  home_pythag_season, away_pythag_season,
  home_run_diff_10g, away_run_diff_10g,
  home_current_streak, away_current_streak,
  h2h_win_pct_3yr, is_divisional,
  -- V7 features (retained)
  home_bullpen_era, away_bullpen_era, moon_phase,
  home_starter_arsenal_score, away_starter_arsenal_score
FROM `hankstank.mlb_2026_season.game_predictions`
WHERE game_date = CURRENT_DATE()
ORDER BY game_date DESC;
```

---

## 📄 License

MIT — see [LICENSE](../hanks_tank/LICENSE)
