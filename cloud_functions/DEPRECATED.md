# DEPRECATED — Legacy Cloud Functions Directory

> **This directory is superseded.** Do not modify or deploy these files.
> 
> **Live status:** the old standalone Gen1 helper functions were removed from GCP during cleanup. This directory remains only as historical reference.

## What happened

This directory was the original Cloud Functions deployment structure (circa V6/early-V7).
It contained four separate HTTP-triggered Cloud Functions:

| Function | Status | Issue |
|---|---|---|
| `update_statcast_2026` | ❌ Dead code | `fetch_statcast_from_baseball_savant()` returns `[]` always — never implemented |
| `update_pitcher_stats_2026` | ❌ Dead code | References the unimplemented statcast fetch |
| `rebuild_v7_features` | ✅ Was functional | Correct but now subsumed by `src/cloud_function_main.py` |
| `predict_today_games` | ❌ Dead code | Returns hardcoded placeholder `{"predictions_generated": 0}` — never wired up |

`setup_scheduler.sh` in this directory has been replaced with a stub because the old
top-level function names no longer match the deployed entry point.

## What replaced it

**`src/cloud_function_main.py`** — single `daily_pipeline()` HTTP handler with
mode-based routing (`mode: daily | backfill | v8_features | pregame_v8 | ...`).

Deployed via: `scripts/gcp/2026_season/deploy_v8.sh`  
Scheduler setup: `scripts/gcp/2026_season/deploy_v8.sh --only-scheduler`

## Why kept (not deleted)

- Audit trail: leaving the files shows exactly what the old architecture looked like
- Reference: the `rebuild_v7_features` import pattern in `daily_updates.py` was
  the model for how `src/cloud_function_main.py` imports V7/V8 builders
- Safety: the old scheduler script was replaced with a hard stop so this directory
  cannot accidentally recreate obsolete jobs

## Migration notes

| Old scheduler job name | New mode | New request body |
|---|---|---|
| `fetch-statcast-2026-daily` | `daily` | `{"mode": "daily"}` (includes data collection) |
| `fetch-pitcher-stats-2026-daily` | `daily` | subsumed by daily pipeline |
| `rebuild-v7-features-daily` | `daily` | `{"mode": "daily"}` (includes V7 features) |
| `predict-today-games-daily` | `predict_today` | `{"mode": "predict_today"}` |

The new unified daily pipeline job covers all of these in a single invocation.
