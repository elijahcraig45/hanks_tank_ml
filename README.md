# Hank's Tank ML

This repo is the ML and scheduled-pipeline side of Hank's Tank. It builds features, predicts
MLB, NFL and college football games, runs shadow models and power rankings, and writes
everything to BigQuery in GCP project `hankstank`. The companion repos only read that data:
[hanks_tank_backend](../hanks_tank_backend) serves it over HTTP, and
[hanks_tank](../hanks_tank) is the React site.

## Start here

| Doc | What it covers |
|---|---|
| [docs/ML_SYSTEM.md](docs/ML_SYSTEM.md) | The main reference: live deploys, every Cloud Function mode, Scheduler jobs and Cloud Tasks, tables written, fallback chains, safe run/test/deploy, and monitoring queries |
| [docs/EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md) | Every model generation and experiment, V1 to the 2026-09 football work, with the measured results and the verdicts |
| [docs/MODEL_CARDS.md](docs/MODEL_CARDS.md) | One card per live or shadow model |
| [docs/SHADOW_MODELS.md](docs/SHADOW_MODELS.md) | MLB shadow writers (logit3, sim_blend) |
| [research/backtest_2026/FINDINGS.md](research/backtest_2026/FINDINGS.md) | The 2026 MLB backtest programme |
| [research/football_2026_09/README.md](research/football_2026_09/README.md) | Football experiments |

## What is running (verified 2026-09-25)

| Sport | Production | Shadow | Live measure |
|---|---|---|---|
| MLB | V10 XGBoost (73 features, trained 2026-04-29) | 3-feature logistic; PA-sim v2 + strength blend | **53.98%** on 1,984 pregame predictions vs 52.82% home rate |
| NFL | XGBoost `nfl_v1_pure_epa` | margin ridge | small 2026 sample |
| CFB | XGBoost `cfb_v1` | margin ridge | ridge beats XGB by ~0.05 log loss in backtests |
| Rankings | Bradley-Terry (MLB W/L; NFL/CFB margin) with bootstrap rank bands | — | — |

Four gen2 Cloud Functions run in us-central1: `mlb-2026-daily-pipeline`,
`mlb-2026-sim-blend`, `nfl-weekly-pipeline` and `cfb-weekly-pipeline`. Twelve Scheduler jobs
and per-game Cloud Tasks drive them. The details are in ML_SYSTEM.md.

These older figures do not describe the live system:

- **61.48% V10 accuracy:** measured on 283 early games with rebuilt features.
- **57.65% V8:** a single 2025 holdout.
- **"158 features":** the canonical list. The deployed artifact uses 73.

## Real entry points

| File | Role |
|---|---|
| `src/cloud_function_main.py` | MLB function; `daily_pipeline` dispatches on `mode` (`src/main.py` re-exports it) |
| `src/season_2026_pipeline.py` | MLB data collection (`--dry-run`, `--date`) |
| `src/predict_today_games.py` | MLB prediction and the V10 → V4 model fallback chain |
| `src/build_v8_features_live.py`, `src/build_v10_features_live.py` | Live feature builders (`--seed-elo`, `--backfill` on V8) |
| `src/train_v10_models.py` | Builds the V10 artifact |
| `src/logit3_shadow.py`, `src/pa_sim/` | MLB shadow models |
| `src/nfl/main.py`, `src/cfb/main.py` | Football functions |
| `src/rankings/` | Power rankings (`build.py`, `evaluate.py`, `fpi_games.py`) |
| `src/stats/` | Player stats, CFBD, pick'em sheet |
| `scripts/gcp/2026_season/deploy_v10.sh`, `deploy_sim_blend.sh`, `scripts/gcp/nfl/deploy_nfl.sh`, `scripts/gcp/cfb/deploy_cfb.sh` | Deploys (all support `--dry-run`) |

`cloud_functions/` is a deprecated audit trail; see its `DEPRECATED.md`.

## Safety rules (short form)

- **Deploys are manual and local.** GitHub Actions only runs tests and has no GCP credentials. Prefix gcloud with `CLOUDSDK_CORE_ACCOUNT=elijahcraig45@gmail.com` and `unset GOOGLE_APPLICATION_CREDENTIALS`.
- **For a code-only MLB deploy, use `deploy_v10.sh --skip-model-upload`.** The script stages `git ls-files src/` and also recreates the 6 MLB Scheduler jobs.
- **Pass `--shadow` to the NFL and CFB deploys** to keep the shadow writers on.
- **Never call the football functions to test them.** On MLB, `daily` still DELETEs duplicate `games` rows even with `dry_run`.
- **Score models only on pregame rows**, i.e. `predicted_at < game_time_utc` inside the dedupe.

## Local development and tests

```bash
python -m pip install -r requirements.txt pytest "polars==1.*"
python -m pytest tests -v
python -m compileall src scripts cloud_functions
python3 src/season_2026_pipeline.py --dry-run --date 2026-09-24
```

`requirements.txt` is pinned and mirrored byte-for-byte to `src/requirements.txt`. Change both
together.

## License

MIT
