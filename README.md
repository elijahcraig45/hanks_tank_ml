# Hank's Tank ML

> **Primary production function:** `mlb-2026-daily-pipeline` (`us-central1`)  
> **Companion repos:** [hanks_tank](../hanks_tank) · [hanks_tank_backend](../hanks_tank_backend)

This repo owns the machine-learning and scheduled data-pipeline side of Hank's Tank. It trains MLB game-outcome models, builds daily feature sets, generates prediction rows and scouting reports, and writes the resulting artifacts into BigQuery for the backend and frontend to consume.

## Current project state

- **Best published benchmark:** V8 ensemble at **57.65% accuracy** on a 1,243-game 2025 holdout
- **Current live pipeline:** production daily pipeline plus newer V10 feature/scouting enrichment work
- **Primary warehouse outputs:** `game_predictions`, `game_scouting_reports`, `game_v10_features`, `games`, `statcast_pitches`

## GCP footprint

### BigQuery

The project currently writes to two datasets in the `hankstank` GCP project:

- `mlb_2026_season` for current-season operational tables
- `mlb_historical_data` for historical warehouse tables

Current-season tables are partitioned and clustered where it matters most, including:

- `games`
- `game_predictions`
- `game_scouting_reports`
- `game_v10_features`
- `statcast_pitches`

### Cloud Functions and jobs

The active production entry point is a **Gen2 Cloud Function**:

| Resource | Current deployed details |
|---|---|
| Function | `mlb-2026-daily-pipeline` |
| Region | `us-central1` |
| Runtime | Python 3.12 |
| Memory | 1024 MB |
| Timeout | 540 seconds |
| Max instances | 3 |

Older helper functions were removed from the live GCP project during cleanup. The deprecated `cloud_functions/` directory remains only as historical reference.

### Cloud Scheduler

Key scheduler jobs currently driving the project include:

| Job | Purpose |
|---|---|
| `mlb-2026-daily` | Daily collection/validation/feature pipeline |
| `mlb-2026-validate` | Daily validation pass |
| `mlb-2026-weekly-predict` | Weekly forward predictions |
| `mlb-2026-v7-features-daily` | Daily V7 feature refresh |
| `mlb-schedule-pregame-tasks` | Pregame task enqueueing via backend endpoint |

An older paused Cloud Run scheduler path was removed during cleanup, and the active schedulers now consistently use `America/New_York` for business-facing run times.

## Model evolution

| Version | Algorithm | Features | Accuracy | Notes |
|---|---|---:|---:|---|
| V1 | Logistic Regression | 5 | 54.0% | Baseline |
| V2 | Logistic Regression | 44 | 54.4% | Expanded features |
| V3 | XGBoost | 57 | 54.6% | First tree model |
| V4 | Stacked Ensemble | 57 | 54.9% | Experimental |
| V5 | CatBoost | 57 | 55.1% | Better categoricals |
| V6 | CatBoost + arsenal | 68 | 55.8% | Pitcher arsenal splits |
| V7 | CatBoost + bullpen/moon/venue | 78 | 56.4% | Previous production |
| V8 | CatBoost x2 + LightGBM + MLP | 89 | 57.65% | Best published holdout |
| V10 | XGBoost SP-quality enrichment | ongoing | ongoing | Starter-quality and scouting enrichment work |

Detailed experiment notes live in [`docs/`](docs/) and [`research/`](research/).

## Live confidence calibration snapshot

As of **2026-04-20**, the latest-settled BigQuery read on `mlb_2026_season.game_predictions`
joined to final scores in `mlb_2026_season.games` showed the following for the current live
pipeline:

| Scope | Games | Accuracy | Avg max win prob |
|---|---:|---:|---:|
| All latest production predictions | 189 | 49.21% | 56.49% |
| Latest `v10` predictions only | 45 | 46.67% | 57.87% |

The current `v10` confidence tiers in `src/predict_today_games.py` are:

- `high`: `>= 0.64`
- `medium`: `>= 0.57`
- `low`: `< 0.57`

Current settled `v10` bucket performance:

| Tier | Rule | Games | Accuracy |
|---|---|---:|---:|
| High | `>= 64%` | 7 | 71.43% |
| Medium | `57% - 63.99%` | 15 | 60.00% |
| Low | `< 57%` | 23 | 30.43% |

### Provisional `very_high` recommendation

For later calibration work, the best current candidate is a **`very_high`** label at
`>= 0.67` max win probability **for `v10` only**.

| Proposed cutoff | Games | Accuracy |
|---|---:|---:|
| `>= 64%` | 7 | 71.43% |
| `>= 65%` | 6 | 66.67% |
| `>= 66%` | 6 | 66.67% |
| `>= 67%` | 3 | 100.00% |
| `>= 68%` | 2 | 100.00% |
| `>= 70%` | 1 | 100.00% |

This is intentionally documented as **provisional**, not production-ready calibration:
the hit rate jumps sharply at `>= 67%`, but the settled sample is still too small to treat
as stable. If the live `v10` sample continues to grow without regression, `>= 0.67` is the
best first cutoff to revisit for a true `very_high` bucket.

## Pipeline outline

```text
1. Fetch schedule and live context
2. Refresh or read current-season warehouse tables
3. Build matchup and model features
4. Run ensemble inference
5. Build per-game scouting reports
6. Write predictions and report payloads to BigQuery
7. Let the backend/frontend consume those tables
```

## Local development

1. Install dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```
2. Authenticate to GCP:
   ```bash
   gcloud auth application-default login
   ```
3. Set environment variables:
   ```bash
   set GCP_PROJECT=hankstank
   set BIGQUERY_DATASET=mlb_2026_season
   ```
4. Run the pipeline locally:
   ```bash
   python src/daily_pipeline.py --date 2026-04-20
   ```

## Validation

This repo now includes a GitHub Actions workflow that installs dependencies and compiles the Python source tree to catch import/syntax regressions early. For local validation:

```bash
python -m compileall src scripts cloud_functions
```

## Repository structure

```text
hanks_tank_ml/
├── src/
├── scripts/
├── docs/
├── research/
├── notebooks/
├── cloud_functions/
└── requirements.txt
```

## License

MIT
