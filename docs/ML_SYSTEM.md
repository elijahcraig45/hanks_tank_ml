# Hank's Tank ML — System Reference

Last verified: **2026-09-25**, against `main` at `3fd0b98` (PR #6), the live GCP project
`hankstank`, and stored research outputs. This is the main reference for what runs, how it
runs, and how to change it safely. Model history is in [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).
Per-model details are in [MODEL_CARDS.md](MODEL_CARDS.md).

**Evidence labels.** Each claim carries one of these labels:

- **[M]** Measured on 2026-09-25 from code, deployed config (`gcloud describe`/`list`), a BigQuery SELECT, a GCS listing, or a named stored result file.
- **[I]** Inferred from code or config, but not observed running.
- **[H]** Historical: a claim from an older doc or memory note that was not re-run today.

---

## 1. System map

```
                 Cloud Scheduler (12 jobs, America/New_York)
                          │ HTTP POST + OIDC (hankstank@appspot SA)
          ┌───────────────┼──────────────────────┬─────────────────────┐
          ▼               ▼                      ▼                     ▼
 mlb-2026-daily-pipeline  nfl-weekly-pipeline   cfb-weekly-pipeline   mlb-2026-sim-blend
  (src/, staged)          (src/nfl + rankings    (src/cfb + 3 nfl      (same src as daily,
   modes: daily, pregame_  + stats, staged)       modules + rankings    4 GiB, shadow only)
   v10, train_weekly ...)                         + stats, staged)             ▲
          │   ▲                    │                       │                   │
          │   └── Cloud Tasks (lineup-pregame queue) ◄── hanks_tank_backend ───┘
          │        pregame_v10 per game, T-6h/3h/90m/45m      (App Engine; enqueues
          ▼                                                    tasks after the 10am
   BigQuery: mlb_2026_season / mlb_historical_data /           schedule_pregame_tasks
             nfl_season / nfl_historical / cfb_season /        call)
             cfb_historical         GCS: gs://hanks_tank_data/models/vertex/...
          │
          ▼  (read-only SELECTs)
   hanks_tank_backend (Express, App Engine `default`) ──HTTP /api──► hanks_tank (React, App Engine `frontend`)
```

The contract between the repos:

- **The ML repo is the only writer** of the prediction, feature, shadow and ranking tables. [I] This comes from code search; the backend reads these tables but does not write them.
- **The backend only reads them.** One exception: it enqueues the Cloud Tasks that call back into the ML function. [M] See `hanks_tank_backend/src/services/lineup-scheduler.service.ts` on `origin/main`.
- **The frontend only calls the backend.**
- **There is no request-time inference.** Every prediction is batch-written by a Scheduler job or a Cloud Task. [M]
- **Column names in `game_predictions` are a contract.** The NFL and CFB tables deliberately mirror the MLB names so the frontend can reuse its components. Add new columns; never rename existing ones (see `src/nfl/predict_nfl.py`, top of file). [M]
- **BigQuery DATE/TIMESTAMP values** come back as `{value: ...}` objects. The backend flattens them in `normalizePredictionRow`, so a new temporal column needs to be added there. [H] From mlb/CLAUDE.md; not re-read today.

### Live deploys (2026-09-25) [M]

| Resource | Revision / version | Memory / CPU | Timeout | Max inst. | Notes |
|---|---|---|---|---|---|
| `mlb-2026-daily-pipeline` | `00035-xet` (20:41 UTC) | 1024M / 0.5833 | 540s | 3 | concurrency 1; entry `daily_pipeline` |
| `mlb-2026-sim-blend` (new) | `00002-fow` (21:41 UTC) | 4Gi / 2 | 540s | 1 | env `SIM_BLEND_MEMORY_MB=4096` |
| `nfl-weekly-pipeline` | `00013-kob` (21:25 UTC) | 2048M / 1 | 540s | 20 | env `NFL_RIDGE_SHADOW=1`, `FPI_SNAPSHOT=1` |
| `cfb-weekly-pipeline` | `00014-wof` (21:27 UTC) | 2048M / 1 | 540s | 20 | env `CFB_RIDGE_SHADOW=1`, `FPI_SNAPSHOT=1`; secret `CFBD_API_KEY` |
| Backend (App Engine `default`) | `20260925t180718` | | | | env `ML_FUNCTION_URL` and `SIM_BLEND_FUNCTION_URL` set; `LINEUP_TASK_QUEUE=lineup-pregame` |
| Frontend (App Engine `frontend`) | `20260925t181259` | | | | |

All four functions are gen2, python312, ingress `ALLOW_ALL`, deployed with
`--no-allow-unauthenticated`, and run as service account `hankstank@appspot.gserviceaccount.com`. [M]

**GitHub Actions never deploys.** There are no GCP credentials in the repo secrets, so any
deploy step logs a skip and still reports success. [M] The CI in this repo
(`.github/workflows/`) runs `pytest tests` and `compileall` only. Every deploy is run by hand
from a laptop with the personal account (see §6).

---

## 2. Cloud Functions and modes

### 2.1 `mlb-2026-daily-pipeline` (`src/cloud_function_main.py`)

The function has one HTTP entry point, `daily_pipeline`, and dispatches on `mode` in the POST
body. [M]

- **The body is parsed with `request.get_json(silent=True) or {}`.** A malformed body is silently treated as `{}`, which means `mode=daily` against yesterday.
- **`date` defaults to yesterday.** The one exception is `schedule_pregame_tasks`, which defaults to today.

| Mode | Steps (in order) | Writes | Supports `dry_run`? |
|---|---|---|---|
| `daily` | collection → validation (`fix_duplicates=True`) → V3/V4 features → Elo update → V8 features → V10 features → power rankings → (Mondays) rosters | `games`, `standings`, `team_stats`, `player_stats`, `transactions`, `statcast_pitches`, `game_features`, `team_elo_ratings`, `game_v8_features`, `game_v10_features`, `power_rankings`, `rosters` | **Partly.** Validation ignores `dry_run` and DELETEs duplicate `games` rows (see §6) |
| `backfill` | collection for `start`..`end` | as collection | yes (collection honours it) [I] |
| `validate` | validation with `fix_duplicates=True` | `games` (dedupe DELETE) | **no** |
| `features` | V3/V4 `game_features` | `game_features` | yes |
| `update_elo` | Elo from `target`'s Final games | `team_elo_ratings` | yes |
| `v8_features` / `v10_features` | builders for `target` or `game_pks` | `game_v8_features` / `game_v10_features` | yes |
| `power_rankings` | Bradley-Terry board, `n_boot=200` | `mlb_2026_season.power_rankings` | yes (skipped) |
| `predict` | weekly batch prediction (`predict_2026_weekly.py`) | `weekly_predictions` | yes |
| `lineups`, `matchup_features`, `matchup_v7_features` | single steps | `lineups`, `matchup_features`, `matchup_v7_features` | yes |
| `pregame`, `pregame_v7`, `pregame_v8`, **`pregame_v10`** | lineups → matchup → (V7) → (V8) → (V10) → predict → scouting report; plus opt-in shadows on `pregame_v10` | `lineups`, `matchup_features`, `matchup_v7_features`, `game_v8_features`, `game_v10_features`, **`game_predictions`**, `game_scouting_reports`, plus shadow tables | yes |
| `predict_today` | prediction only | `game_predictions` | yes |
| `logit3` | 3-feature shadow | `game_predictions_logit3` | yes |
| `sim_blend` | v2 PA sim + strength blend (refuses below 3072 MB) | `game_predictions_sim_blend`, `game_props_sim` | yes |
| `pa_sim` / `pregame_sim` | v1 PA sim (deprecated shadow) | `game_predictions_sim`. That table does not exist, so a real run fails at its DELETE. [I] | yes |
| `train_weekly` | `model_version` `v10`/`v8` → subprocess `train_v8_models.py --weekly-update`; then refresh the SP percentile parquet in GCS | `gs://hanks_tank_data/sp_quality/statcast_sp_{year}.parquet` | yes |
| `backfill_v7` / `backfill_v8` / `backfill_v10` | feature rebuild for a range | feature tables | yes |
| `rosters` | roster snapshot | `rosters` | yes |
| `scouting_reports` | reports for `date` | `game_scouting_reports` | yes |
| `schedule_pregame_tasks` | GET `BACKEND_URL/api/lineup/schedule-today?date=` | none directly; the backend enqueues Cloud Tasks | yes (no call made) |

Opt-in shadow flags on `pregame_v10` are `run_logit3`, `run_sim_blend` and `run_pa_sim`. Shadow
failures are caught by `_shadow()` and reported as a step with `status:error`. They never fail
the production run. [M]

**`train_weekly` is broken.** [M] It has failed every Sunday. Cloud Logging shows the same
error on 2026-09-14, 09-20 and 09-21:
`FileNotFoundError: '/logs/v8_experiments.log'`, raised at import in `train_v8_models.py:59`.
It would fail even without that error, for two reasons:

- `train_v8_models.py` does not accept `--weekly-update`. Its argparse has only `--iterations` and `--quick`.
- It has no GCS upload.

So **V10 has never been retrained** since its artifact was written on 2026-04-29. The V8
artifact was last written on 2026-04-09. [M]

### 2.2 `mlb-2026-sim-blend`

Same staged source and entry point as the daily function. It is deployed separately because the
v2 simulator needs more than the daily function's 1 GB (measured peak 1.3–2.5 GB,
`data/backtest_2026/rich/logs/runtime.txt`). Tasks call it with
`{"mode":"sim_blend","game_pks":[pk],"date":...}`. It has no Scheduler job. [M]

### 2.3 `nfl-weekly-pipeline` (`src/nfl/main.py`)

| Mode | What it does | Writes | `dry_run` |
|---|---|---|---|
| `ingest` | full schedule + teams from nflverse; EPA for the current season only; then rankings, player stats, pick'em sheet, FPI snapshot (if enabled) | `nfl_historical.games`, `teams`, `team_week_epa`; `nfl_season.power_rankings`, `player_season_stats`, `stat_leaders`; `pickem.games`; `fpi_game_predictions` | **refused (400)** |
| `predict_week` | XGBoost (`nfl_v1_pure_epa`) for the next unplayed week; ridge shadow and FPI if enabled | `nfl_season.game_predictions` (game_id-scoped DELETE + append); `game_predictions_ridge_shadow`; `fpi_game_predictions` | **yes**: computes and returns, writes nothing |
| `score` | UPDATE result columns from `nfl_historical.games` | `game_predictions` | refused |
| `rankings`, `stats`, `fpi_snapshot` | single steps | as above | refused |
| `backfill` | walk-forward season rebuild; game_id-scoped DELETE + append | `game_predictions` | refused |

### 2.4 `cfb-weekly-pipeline` (`src/cfb/main.py`)

The modes are `ingest`, `score`, `rankings`, `stats`, `cfbd`, `predict_week`/`predict_next`,
`backfill` and `fpi_snapshot`.

- **`dry_run` is honoured only on `predict_week`/`predict_next`.** Every other mode returns 400 when given `dry_run`. [M]
- **Predictions are written with `replace_game_ids`**, never season-scoped.
- **The ingest uses `replace_seasons` with a shrink guard.** See memory note `cfb-games-truncate-wiped-history`. [H]

### 2.5 The 2026-09-25 incident, and what changed

Before commit `0f7b709`, both football functions ignored `dry_run`. [H]
- A `{"dry_run":true}` call to `predict_week`/`predict_next` ran for real. It replaced NFL week 3 (16 games) and CFB week 4 (122 games), and wrote FPI snapshots.
- Three already-kicked-off games lost their true pregame rows: ATL@GB, Liberty@Coastal Carolina and Army@Temple.
- The user restored those three rows with BigQuery time travel (`mlb/ml_writeups/restore_3_pregame_rows.sql`).

The fixes are now live in NFL `00013` and CFB `00014`. [M, code]

- `dry_run` is honoured on the predict modes and refused everywhere else.
- `_upcoming()` (NFL) and `predict_week` (CFB) drop games that have kicked off. A rerun can no longer overwrite a real pregame row.
- `upsert_week` / `replace_game_ids` delete only the game_ids being written. They no longer clear the whole week.

**Standing rule, unchanged:** do not send any request to the football functions to "test" them,
even with `dry_run`. Test locally against a staged tree instead.

---

## 3. Triggers

### 3.1 Cloud Scheduler [M]

All 12 jobs share these settings:

- `America/New_York` timezone
- OIDC auth
- `attemptDeadline` of 540s
- `ENABLED` state
- a body that parses as valid JSON

| Job | Cron | Target | Body |
|---|---|---|---|
| `mlb-2026-daily` | `0 4 * 3-11 *` | mlb daily | `{"mode":"daily"}` |
| `mlb-2026-pregame-schedule` | `0 10 * 3-11 *` | mlb daily | `{"mode":"schedule_pregame_tasks"}` |
| `mlb-2026-weekly-predict` | `0 5 * 3-11 5` (Fri) | mlb daily | `{"mode":"predict"}` |
| `mlb-2026-weekly-train-v10` | `0 2 * 3-11 0` (Sun) | mlb daily | `{"mode":"train_weekly","model_version":"v10"}` |
| `mlb-2026-roster-refresh` | `0 3 * 3-11 1` (Mon) | mlb daily | `{"mode":"rosters"}` |
| `mlb-2026-power-rankings` | `30 6 * 3-11 1` (Mon) | mlb daily | `{"mode":"power_rankings"}` |
| `nfl-weekly-ingest` | `0 6 * 9-12,1,2 2` (Tue) | nfl | `{"mode":"ingest"}` |
| `nfl-weekly-score` | `0 7 * 9-12,1,2 2` (Tue) | nfl | `{"mode":"score"}` |
| `nfl-weekly-predict` | `0 6 * 9-12,1,2 3` (Wed) | nfl | `{"mode":"predict_week"}` |
| `cfb-weekly-ingest` | `0 6 * 8-12,1 0` (Sun) | cfb | `{"mode":"ingest"}` |
| `cfb-weekly-cfbd` | `0 7 * 8-12,1 0` (Sun) | cfb | `{"mode":"cfbd"}` |
| `cfb-weekly-predict` | `0 6 * 8-12,1 2` (Tue) | cfb | `{"mode":"predict_next"}` |

To decode the bodies yourself:

```bash
CLOUDSDK_CORE_ACCOUNT=elijahcraig45@gmail.com gcloud scheduler jobs list --location us-central1 \
  --project hankstank --format=json | python3 -c "import json,sys,base64
for j in json.load(sys.stdin):
    b=base64.b64decode(j['httpTarget'].get('body','')).decode(); print(j['name'].split('/')[-1], j['schedule'], b)"
```

### 3.2 Cloud Tasks (per MLB game) [M]

The flow starts at 10:00 ET, when `schedule_pregame_tasks` calls the backend's
`/api/lineup/schedule-today`. For each game that has not started, the backend enqueues tasks on
queue `lineup-pregame` (max 5 concurrent, 2/s, 3 attempts):

- **`pregame_v10` tasks:** one immediately (`baseline`), then one each at T−360, T−180, T−90 and T−45 minutes. Each body is `{"mode":"pregame_v10","game_pks":[pk],"date":...,"run_logit3":true}` and targets the daily function.
- **`sim_blend` task:** one at T−90, targeting `mlb-2026-sim-blend`. It is only enqueued when `SIM_BLEND_FUNCTION_URL` is set, and it is set in backend `20260925t180718`.

Consequences:

- **A game can be re-predicted up to 5 times.** Each `pregame_v10` run DELETEs the game's `game_predictions` row and inserts a fresh one. The DELETE is skipped inside the streaming-buffer window, which can leave duplicate rows. [I]
- **logit3 appends one row per task run.** [I]
- **`run_logit3` only applies to tasks enqueued after the backend deploy.** That deploy was 2026-09-25 22:08 UTC, so the first logit3 and sim_blend rows are expected from 2026-09-26's slate. [I]
- **429 errors:** the queue allows 5 concurrent tasks against `maxInstances=3` × concurrency 1. Logs show 14–19 HTTP 429 per day before today. [M]
- **500 errors, now fixed:** until revision 00034, there were 8–68 HTTP 500 per day from `UnboundLocalError: game_pitcher_splits`. Revision 00035 served 29/29 tasks with HTTP 200 on 2026-09-25. [M]

---

## 4. Production vs shadow vs research, per sport

| Sport | Model | Status | Writes | Evidence |
|---|---|---|---|---|
| MLB | **V10 XGBoost** (`V10_LineupMatchupXGB`, 73 features) | **Production** | `game_predictions` (`model_version='v10'`) | artifact metadata [M] |
| MLB | V8 / V8_nocat / V7 / V6 / V5 / V4 | Fallbacks in the load chain only | — | GCS listing [M] |
| MLB | V4 (`game_outcome_2026/model.pkl`) | **Still live** in `predict` mode (Friday job) | `weekly_predictions` (no backend reader found) [I] | `predict_2026_weekly.py:39` [M] |
| MLB | logit3 (3-feature L1 logistic) | **Shadow**, enabled via task flag; 0 rows so far | `game_predictions_logit3` | table exists, 0 rows [M] |
| MLB | sim_blend (PA sim v2 + strength) | **Shadow**, separate function; 0 rows so far | `game_predictions_sim_blend`, `game_props_sim` | 0 rows [M] |
| MLB | PA sim v1 | Deprecated shadow, never enabled | (`game_predictions_sim`, absent) | [M] |
| MLB | Power rankings, Bradley-Terry W/L | **Production** | `power_rankings` (`model='bt'`) | [M] |
| NFL | XGBoost `nfl_v1_pure_epa` (EPA fixed) | **Production** since 2026-09-25 | `nfl_season.game_predictions` | 15 rows, week 3 [M] |
| NFL | Margin ridge `nfl_v2_margin_ridge` | **Shadow** | `game_predictions_ridge_shadow` (15 rows) | [M] |
| NFL | ESPN FPI | Comparison snapshot (not ours) | `fpi_game_predictions` (32 rows) | [M] |
| NFL | Power rankings, margin ridge | **Production** | `nfl_season.power_rankings` (`model='margin'`) | [M] |
| CFB | XGBoost `cfb_v1` | **Production** | `cfb_season.game_predictions` | [M] |
| CFB | Margin ridge `cfb_v2_margin_ridge` | **Shadow** | `game_predictions_ridge_shadow` (121 rows) | [M] |
| CFB | ESPN FPI | Comparison snapshot | `fpi_game_predictions` (138 rows) | [M] |
| CFB | Power rankings, margin ridge | **Production** | `cfb_season.power_rankings` (`model='margin'`, 266 teams) | [M] |
| All | Drive sim (NFL), matchup features, injuries, weather, state-space, series unit, autosearch | **Research only** | — | see EXPERIMENT_LOG |

The comments in `src/rankings/sources.py` still say the margin boards are "EXPERIMENT … not
deployed". That is stale: the live boards were computed with `model='margin'` at
2026-09-25 22:10 UTC. [M]

---

## 5. Model resolution and fallback chains

### MLB game predictions (`src/predict_today_games.py` → `DailyPredictor.load_model`) [M]

The loader tries each version in order, taking the first one it can load. For each version it
checks for a local file under `models/` first, then GCS `gs://hanks_tank_data/<path>`.

1. `v10` → `models/vertex/game_outcome_2026_v10/model.pkl` (**exists**, 2026-04-29)
2. `v8_nocat_calibrated` → `game_outcome_2026_v8_nocat_calibrated` (**absent in GCS**)
3. `v8_nocat` → exists (2026-04-09)
4. `v8_calibrated` → **absent**
5. `v8` → exists (2026-04-09)
6. `v7` → exists (2026-04-26); `v6` → exists; `v5` → exists
7. Final fallback `v4` → `models/vertex/game_outcome_2026/model.pkl` (2026-03-27)

The fallback is silent. Every step logs only a warning, and the written `model_version` column
is the only trace. In a Cloud Function, `_REPO_ROOT` resolves to `/`, so the local files never
exist and the loader always goes to GCS. [I]

`gs://hankstank-models/` **does not exist** (404). Model artifacts live only under
`gs://hanks_tank_data/models/vertex/`. [M]

Every `model_version` value in `game_predictions` for 2026, with rows and date range: [M]

| `model_version` | Rows | Dates |
|---|---|---|
| `v10` | 3,294 | 03-27 → 09-25 |
| `V7_BullpenMoonVenueSplit` | 370 | 04-06/07 |
| `v8_nocat` | 26 | 04-09 → 04-15 |
| `V10_LineupMatchupXGB` | 2 | 04-29 |

Confidence tiers are absolute thresholds on max(p, 1−p): high ≥ 0.64, medium ≥ 0.57, else low.
The tiers are applied by `CONFIDENCE_TIERS` in `predict_today_games.py` for V10, and by
`tier()` in `logit3_shadow.py`. Football uses high ≥ 0.72 and medium ≥ 0.60
(`src/nfl/predict_nfl.py`). [M]

### Football [M]

Football has no artifact chain. Both production XGBoost models are **refit on every run**:

- **NFL** fits on every completed game in `predict_nfl.predict_week`. It refuses to run without EPA rows (`_require_epa`).
- **CFB** fits per division, on all prior games in that division, and needs at least 300 rows.
- **The ridge shadows** refit a decaying 2-season window.

---

## 6. How to run, test and deploy safely

### Account and credentials

The personal account prefix and the credentials unset are both required on this laptop:

```bash
unset GOOGLE_APPLICATION_CREDENTIALS            # .zshrc points it at a revoked SA key
export CLOUDSDK_CORE_ACCOUNT=elijahcraig45@gmail.com
```

Python clients need ADC. See memory note `adc-broken-by-stale-sa-key-env`. [H]

### Tests

- **CI** runs `python -m pytest tests -v` and `compileall src scripts cloud_functions`. It was green on the PR #6 merge (run 36195628615). [M]
- **Locally**, use `pip install -r requirements.txt pytest "polars==1.*"`. The existing `.venv` lacks `functions_framework` and `polars`, so without those installs 3 tests fail or error. [M]
- **Safety-critical suites:**
  - `tests/test_football_write_safety.py`: dry_run refusal and kickoff skipping.
  - `tests/test_cfb_games_write.py`: shrink guard.
  - `tests/test_logit3_shadow.py`, `tests/test_sim_blend.py`: shadow isolation.
  - `tests/test_build_v10_features_live.py`: the train/serve fixes.

### `dry_run`, per function

| Function / mode | `dry_run` behaviour |
|---|---|
| MLB `pregame_*`, `predict_today`, feature modes, `logit3`, `sim_blend`, `power_rankings`, `train_weekly`, `schedule_pregame_tasks` | Honoured; each step is passed `dry_run` [M] |
| MLB `daily`, `validate` | **Not fully honoured.** `_run_validation()` builds `DataValidator(fix_duplicates=True)` with no dry_run, so duplicate `games` rows are DELETEd even on a dry run [M] |
| NFL `predict_week`; CFB `predict_week`/`predict_next` | Honoured: compute, return the game list, write nothing [M] |
| Every other NFL/CFB mode | Rejected with HTTP 400 [M] |
| Standing rule | **Never invoke the football functions to test**, dry_run or not |

The local scripts have their own flags:
- `python3 src/season_2026_pipeline.py --dry-run`
- `python3 src/predict_today_games.py --dry-run`
- `python src/nfl/predict_nfl.py --season S --week W --no-write`

**Beware `predict_nfl.py --backfill S` without `--no-write`.** For XGBoost it loads with
`WRITE_TRUNCATE` and replaces the whole `nfl_season.game_predictions` table. `bq_io.load_table`
also defaults to `WRITE_TRUNCATE`. [M]

### Deploy scripts

| Script | Stages | Flags | Notes |
|---|---|---|---|
| `scripts/gcp/2026_season/deploy_v10.sh` | `git ls-files src/` copied into a temp dir. Working-tree content of tracked files is copied, with a warning if src/ has uncommitted edits | `--dry-run`, `--skip-model-upload` (**use for code-only deploys**), `--only-scheduler`, `--only-upload-model`, `--skip-function` | **Always deletes and recreates all 6 MLB Scheduler jobs**; there is no flag to skip that. Does not set `--max-instances` (live value 3 is retained) [M] |
| `scripts/gcp/2026_season/deploy_sim_blend.sh` | `git ls-files src/`; fails if `pa_sim` is untracked | `--dry-run` | Prints the URL to put in backend `app.yaml` |
| `scripts/gcp/nfl/deploy_nfl.sh` | `src/nfl/*.py` + `cp -R src/rankings src/stats`. **Not** `git ls-files`, so untracked files in those dirs ship | `--dry-run`, `--only-scheduler`, **`--shadow`** (sets `NFL_RIDGE_SHADOW=1,FPI_SNAPSHOT=1`) | Omitting `--shadow` on a redeploy turns the shadows off |
| `scripts/gcp/cfb/deploy_cfb.sh` | `src/cfb/*.py` + `nfl/{features,models,margin_ridge}.py` + rankings/stats | `--dry-run`, `--only-scheduler`, **`--shadow`** | Mounts `CFBD_API_KEY` from Secret Manager |
| `deploy_v8.sh`, `deploy_v7.sh`, `deploy_2026_pipeline.sh`, `deploy_pa_sim.sh` | older paths | — | Do not use; `deploy_v8.sh` deploys `src/` directly and ships untracked code |

Other deploy facts:

- **Backend and frontend deploy locally** with `npm run deploy` in their repos, not via CI.
- **`requirements.txt` changes** must be mirrored byte-for-byte into `src/requirements.txt`.
- **Invoke a function once after deploying it** and read `gcloud functions logs read <name> --gen2`. Import-time failures only show up in the deployed flat tree (memory `cf-flat-tree-breaks-repo-root-paths`). For football, a read-only check of the logs and tables is the limit.

---

## 7. Monitoring and verification queries (read-only)

The honest MLB live score filters to pregame rows **inside** the dedupe:

```sql
WITH g AS (SELECT game_pk, ANY_VALUE(home_score) hs, ANY_VALUE(away_score) aws
           FROM `hankstank.mlb_2026_season.games`
           WHERE status IN ('Final','Completed Early') AND home_score IS NOT NULL GROUP BY 1),
p AS (SELECT * EXCEPT(rn) FROM (
        SELECT game_pk, home_win_probability hp,
               ROW_NUMBER() OVER (PARTITION BY game_pk ORDER BY predicted_at DESC) rn
        FROM `hankstank.mlb_2026_season.game_predictions`
        WHERE predicted_at < game_time_utc) WHERE rn = 1)
SELECT COUNT(*) n, AVG(IF((hp>=0.5)=(hs>aws),1,0)) acc,
       AVG(-(IF(hs>aws,1,0)*LN(hp)+IF(hs>aws,0,1)*LN(1-hp))) log_loss,
       AVG(IF(hs>aws,1,0)) home_rate
FROM p JOIN g USING (game_pk) WHERE hs != aws;
```

Result on 2026-09-25 [M]:

| Scope | n | Accuracy | 95% CI | Log loss |
|---|---|---|---|---|
| Pregame rows only | 1,984 | **53.98%** | 51.8–56.2 | 0.6907 |
| Home-win base rate, same games | | 52.82% | | |
| Latest row regardless of time | 2,372 | 54.17% | | 0.6881 |

829 rows have `predicted_at >= game_time_utc`. The earlier count was 418 on 2026-09-08. [M]

Shadow scoreboard: for each shadow table, take the latest row per game with
`predicted_at < game_time_utc`, join to `games`, and score as above. Readers must use the
latest pre-first-pitch row, because the shadow tables are append-only.

```sql
-- row counts and freshness of every prediction/shadow table
SELECT 'v10' t, COUNT(*) n, MAX(predicted_at) last FROM `hankstank.mlb_2026_season.game_predictions`
UNION ALL SELECT 'logit3', COUNT(*), MAX(predicted_at) FROM `hankstank.mlb_2026_season.game_predictions_logit3`
UNION ALL SELECT 'sim_blend', COUNT(*), MAX(predicted_at) FROM `hankstank.mlb_2026_season.game_predictions_sim_blend`
UNION ALL SELECT 'nfl_ridge', COUNT(*), MAX(predicted_at) FROM `hankstank.nfl_season.game_predictions_ridge_shadow`
UNION ALL SELECT 'cfb_ridge', COUNT(*), MAX(predicted_at) FROM `hankstank.cfb_season.game_predictions_ridge_shadow`;
```

**Football:** `SELECT season, model_version, COUNT(*), AVG(prediction_correct) FROM
hankstank.{nfl,cfb}_season.game_predictions GROUP BY 1,2`. The 2024/2025 rows are walk-forward
backfills, not live predictions. [M]

**Function health:** check HTTP status per revision in Cloud Logging:

```bash
gcloud logging read 'resource.type="cloud_run_revision" AND logName:"run.googleapis.com%2Frequests" AND resource.labels.service_name="mlb-2026-daily-pipeline"' \
  --project hankstank --freshness=2d --format="value(timestamp,resource.labels.revision_name,httpRequest.status)"
```

**Step timings:** each MLB step prints a `NOTICE` JSON line, `[mode] step X took Ns`. That is
the way to see which step a 504 died in. [M, code]

**Weekly training:** search the logs for `V8 training error`. It currently fires every Sunday (§2.1).

---

## 8. Known open issues (2026-09-25)

1. **`train_weekly` fails every week** (§2.1). V10 and V8 are frozen at their April artifacts. [M]
2. **`mlb-2026-weekly-predict` still serves the V4 model** into `weekly_predictions`. The backend has no reader for that table. [M code / I reader]
3. **`daily`/`validate` DELETE under `dry_run`** (§6). [M]
4. **The confidence tiers are absolute.** The 2026-09-08 backtest recommends quantile tiers and display-only recalibration. That change is not implemented. [H]
5. **Streaming-buffer DELETE skips** can leave duplicate `game_predictions` rows on re-runs within about 90 minutes. [H]
6. **`daily` builds V8/V10 feature rows for yesterday's (already final) games.** This is a source of post-first-pitch feature rows. Readers must filter `computed_at < game_time_utc`, as `logit3_shadow.py` does. [I]
7. **`src/pa_sim/v2.py`'s docstring is out of date.** It says "not wired into any pipeline", but `blend.py` wires it. [M]
8. **Duplicated or untracked research code.** `src/edge/`, `data/odds/` and `data/backtest_2026/` are untracked and exist only in the `mlb/hanks_tank_ml` checkout. Several research results cite them. [M]
