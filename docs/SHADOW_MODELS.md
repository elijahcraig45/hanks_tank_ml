> **Status update (2026-09-25):** the "Enabling" steps below have been carried out. The three tables exist but hold 0 rows so far.
> `mlb-2026-sim-blend` is deployed (rev 00002, 4Gi).
> Backend `20260925t180718` enqueues `run_logit3` on every `pregame_v10` task, plus one `sim_blend` task per game at T-90m.
> PA sim v1 (`game_predictions_sim`) was never enabled.
> See [ML_SYSTEM.md](ML_SYSTEM.md) and [MODEL_CARDS.md](MODEL_CARDS.md).
>
> **2026-09-28:** sim_blend missed 8 of 27 games on 9/26-27 and wrote duplicate rows; the
> fixes (6 GiB / 3 instances, its own queue, T-80 + T-35 tasks, warm engine, idempotent
> writes) are under [Operations](#operations-2026-09-28) below. Not deployed until the
> deploy steps there are run.

# MLB shadow models

Shadow models write their own tables and never `game_predictions`, so turning one on
cannot change what hankstank.com serves. All are off by default. Their only reader is the
backend's model comparison (`/api/models/mlb/compare`), which scores a row only if it was
written strictly before first pitch.

| Model | Code | Table | Trigger | Memory |
|---|---|---|---|---|
| PA sim v1 (existing) | `src/pa_sim/pipeline.py` | `game_predictions_sim` | `mode=pa_sim`, or `run_pa_sim` on `pregame_v10` | ~1 GB |
| 3-feature logistic | `src/logit3_shadow.py` | `game_predictions_logit3` | `mode=logit3`, or `run_logit3` on `pregame_v10` | small; fits the live 1 GB function |
| v2 sim + strength blend | `src/pa_sim/blend.py` | `game_predictions_sim_blend`, `game_props_sim`, `game_sim_distributions`, `player_sim_projections` | `mode=sim_blend` (backend tasks at T-80 and T-35) | peak 2.8 GB cold, 2.4 GB warm, 3.5 GB on a date change (local RSS, 2026-09-28); deployed at 6 GiB |

Both new writers are append-only (one row per game per run; readers take the latest row
before first pitch), skip games that have already started, accept `game_pks` to limit the
slate, and load with `CREATE_NEVER`: until the table exists a run reports
`table_missing` and writes nothing. A shadow failure is caught in `cloud_function_main`
and reported as a step with `status: error`; it never fails the production run.

## What each one is

**logit3** — `elo_differential`, `pythag_differential`, `sp_quality_composite_diff` from
`game_v10_features` (latest row with `computed_at < game_time_utc`), train-median
imputed, standardised, L1 logistic with C = 0.557 (the autosearch winner,
`data/backtest_2026/autosearch_best.json`). Refit every run on this season's Final
regular-season games before the target date; skips below 200 games. Measured on the
2026 holdout (400 games): log loss 0.6821 vs V10 0.6859 (difference not significant).

**sim_blend** — the frozen v2 simulator config (`full_x50`, `FROZEN_CONFIG.json`) plays
each game 3,000 times from the pregame lineups. Its raw home-win share is stacked with a
team-strength logistic (Elo K=4 with log margin, 24-point home edge, 1/3 regression per
season; season pythag with a 10-game prior; fit on the 3 prior seasons):

    blend = sigmoid(-0.0490 + 0.5141*logit(strength_p) + 0.7581*logit(sim_p))

Coefficients are in `src/pa_sim/blend_coefs.json`, fit on 2016-2025 (22,746 games) from
the research test frame. This is the one simulator output with a measured winner gain:
+0.0021 nats over strength alone on 15,000 test games 2020-26, CI [+0.0010, +0.0032]. The
sim alone only ties strength.

`game_props_sim` holds what the simulator is actually good at: the total-runs pmf (tilted
down by the 0.47-run over-prediction the raw sim showed over the fit window) and each
starter's strikeout pmf (backtest CRPS 1.269 vs 1.298 for a naive rate, 30k starts).
There is no live market total, so `market_total_line` / `p_over_market` are NULL; the
research gain for totals (+0.010 log score) comes from the sim *shape* tilted to the
market mean, which a reader can apply wherever a line exists. Batter props are written
only with `experimental_props: true` because P(>=1 hit) is over-predicted (64.5% vs 60.8%).

Differences from the research harness: venue ids come from `games_historical` /
`mlb_2026_season.games` instead of statsapi schedule metadata; scheduled innings are
always 9; team strength is keyed on MLB team id rather than statcast abbreviation; the
strength model sees every Final regular-season game, not only games with a stored
lineup. A local run of the ported inputs on the research parquet files reproduced the
research PA table row for row and the week-of-2026-09-14 win probabilities at r = 0.985.

## Enabling (needs approval; nothing here has been run)

1. Create the tables:

       bq query --use_legacy_sql=false --project_id=hankstank < scripts/gcp/2026_season/create_game_predictions_logit3.sql
       bq query --use_legacy_sql=false --project_id=hankstank < scripts/gcp/2026_season/create_game_predictions_sim_blend.sql
       bq query --use_legacy_sql=false --project_id=hankstank < scripts/gcp/2026_season/create_game_props_sim.sql

2. logit3: redeploy `mlb-2026-daily-pipeline` from tracked code only (stage
   `git ls-files src/` into a temp dir, as mlb/CLAUDE.md says), then add
   `"run_logit3": true` to the pregame task body built by the backend's
   `cloud-tasks.service.ts`, or run it once a day before first pitch:

       gcloud scheduler jobs create http mlb-2026-logit3-shadow --location=us-central1 \
         --schedule="0 11 * 3-11 *" --time-zone=America/New_York --uri=<function-uri> \
         --http-method=POST --message-body='{"mode":"logit3","date":"<today>"}' ...

   Note `target` defaults to yesterday: a Scheduler body cannot say "today", so the
   per-game pregame task (`run_logit3`) is the reliable path.

3. sim_blend: do NOT add it to the 1 GB function — it will report
   `insufficient_memory`. Deploy the same source as a separate gen2 function, e.g.
   `mlb-2026-sim-blend`, with `scripts/gcp/2026_season/deploy_sim_blend.sh` (6Gi, 2 vCPU,
   540s, 3 instances, and its `sim-blend` queue; see Operations), and call it with `{"mode":"sim_blend","date":"<today>"}` once
   lineups post (or per game with `game_pks`). A full load pulls ~2M PAs from
   BigQuery on each cold start (~1 GB scanned); the cheaper alternative is precomputed
   rate tables in GCS, not built yet. `SIM_BLEND_MIN_MEMORY_MB` (default 3072) sets the
   guard; `SIM_BLEND_MEMORY_MB` overrides the detected limit.


## Operations (2026-09-28)

What went wrong on 2026-09-26/27 and what changed. All measurements are read-only
(`gcloud logging read`, BigQuery SELECTs, local dry runs); scripts are in the weekend
review's scratchpad.

**1. Missed games (8 of 27).** Three causes, each fixed:

| Cause | Evidence | Fix |
|---|---|---|
| Concurrency: per-game sim tasks shared `lineup-pregame` (5 concurrent dispatches) with a 1-instance function | 429 "no available instance" / 500s in the function log; 10 of 15 games on 9/27 start 19:05-19:10 UTC, so their T-90 tasks all fire at once | own queue `sim-blend`, 2 concurrent dispatches, function max 3 instances, so a dispatched task always finds an idle or startable instance. A discrete-event replay of both days' real arrival times reproduces the old misses (9/26 mean 2.1 of 13, 9/27 8.7 of 15, lineup misses included in the observed count) and gives 0 with the new settings, also with every task doubled and with all 15 games in one burst |
| Memory: OOM at 4,271 MiB > 4,096 | every request rebuilt the ~2M-PA inputs; freed heap was not returned, so the 2nd request on a warm instance landed on top of the 1st. Local dry runs: 2.93 GB cold, 3.86 GB second run | warm engine per instance and date (`blend.warm_state`): later tasks that day only simulate, 2.36 GB and 7 s vs 25-50 s. `release_memory()` (malloc_trim) on rebuild; `load_inputs` drops each raw frame once consumed (cold 2.80 GB). Date change on a warm instance 3.53 GB. Memory 6 GiB: production ran ~10% above local (4,271 vs 3,863 MiB), so the worst case is ~3.9 GB against 6 GiB. Warm and cold runs give bit-identical rows (all four tables, checked on real games) |
| Missing lineups: the sim task fired at T-90, the same moment as the pregame task that fetches the lineup; pregame_v10 runs take 146 s mean, 213 s max | 7 of the 8 missed games had their lineup complete at T-85..89 (fetched by that T-90 task) or at T-44..50 | tasks at T-80 (after the T-90 fetch) and T-35 (after the T-45 fetch). The T-35 one sends `lineup_fallback`: a side still incomplete uses the team's previous game's lineup (research 51_test_report: +0.0014 [-0.0009, +0.0037] 2025, -0.0012 [-0.0038, +0.0014] 2026 log loss), recorded in `lineup_source`. A missing starter is never filled in |

Failures still never cost a pregame task: the sim tasks are on their own queue and are
enqueued in their own try/catch after the pregame tasks.

**2. Duplicate rows.** From 2026-09-11 (when the Scheduler body fix made
`mlb-2026-pregame-schedule` work) both it and App Engine cron call `schedule-today` at
10:00 ET, so every task ran twice. sim_blend's twins ran back to back on its one instance
(same seed, identical rows 1-6 min apart); V10's ran concurrently, both DELETEd and both
INSERTed (identical rows < 1 s apart). Fixes:

- backend task names are a hash of queue, url, body, checkpoint and first pitch, so the
  second `schedule-today` is rejected by Cloud Tasks (ALREADY_EXISTS) and counted `deduped`;
- sim_blend skips a game already written today (per table, so a run that died between
  tables finishes the rest; `force` overrides), and every load job id is derived from the
  rows' content minus `predicted_at`, so a recomputation that slips through gets 409;
- V10 loads first under a content + 10-minute-bucket job id, then deletes only rows older
  than itself, so twins can neither double-insert nor delete each other's row.

Readers already pick one row per key (`pickLatestPregameRow`); on the weekend rows it
returns 19/19/57 rows for sim_blend/distributions/V10 and 2,280 player rows from 3,600.
Existing duplicates: `scripts/gcp/2026_season/cleanup_duplicate_predictions_2026_09_28.sql`
(reviewed by hand, not run by the pipeline).

**Deploy order:** `deploy_sim_blend.sh` (creates `sim-blend` and redeploys the function),
optionally `alter_sim_blend_lineup_source.sql`, then the backend (`SIM_BLEND_TASK_QUEUE` in
app.yaml). Tasks already enqueued for the day keep their old T-90 schedule.

## Operator note: pausing and pinning from the control plane

The lab (`mllab control ...`) is the only writer of `control.model_control_current`; the pipeline only
reads it, once per invocation (`src/model_control.py`). Any read failure, an empty view, or
`MODEL_CONTROL_DISABLED=1` means "no control": everything behaves as if this feature did not exist.

Pause (`run_state = paused`, keyed by model key):
- A paused model returns HTTP 200 with `status: "paused"`, writes nothing, and logs one JSON line
  (`event: model_paused`). It is a success on purpose: an error would make Cloud Tasks retry forever.
- MLB keys: `v10` (production, `predict_today`/`pregame_*`), `pa_sim`, `logit3`, `sim_blend`.
  Football keys, per sport: `xgb` (production `predict_week`/`predict_next`), `ridge`, `drive_sim`, `fpi`.
- A paused shadow is skipped exactly like a shadow that was never enabled. Pausing production `v10`
  skips only the prediction step of a `pregame_v10` chain; lineups/features/scouting still run.
- Not covered: manual `backfill` modes and the weekly `predict` batch (`predict_2026_weekly`).
- A pause takes effect on the next invocation; in-flight runs finish.

Pin (MLB `v10` only: `lifecycle = live` + `artifact_uri` + `artifact_sha256`, all three required):
- `load_model` loads the pinned `gs://` artifact first and checks the sha256 of the raw bytes before
  unpickling. A mismatch RAISES (HTTP 500, nothing written); it never falls back to another model.
- If the object is missing or unreadable, an ERROR is logged and the normal chain is used, so a
  typo in the URI degrades to the old behaviour rather than an outage. Check logs after pinning.
- Rows carry `model_version` = the artifact payload's `version`. Give every pinned artifact a
  `feature_set` (`"v10"` or `"v8"`); without it the feature path is guessed from the label and a
  WARNING is logged, and a label other than `v10`/`v8` would be treated as a legacy model.
- `model_sha256` is stored per row only once `scripts/gcp/control/03_add_model_sha256.sql` has been
  run; until then the pipeline silently omits the column. Un-pinning (clear the fields or set the
  lifecycle to anything but `live`) returns to the normal chain on the next run. (`lifecycle` was called `role` before 2026-09-29; the reader falls back to a `role` column during the transition.)
- `--fallback-v4` requests ignore the pin. CLI runs and the backfill scripts never read the control
  plane (only the Cloud Function passes it in).

Tiers (`tier_high`, `tier_medium`, need `0.5 < medium < high < 1`, both set) replace the confidence
tier cut-offs of the production key (MLB `v10`, football `xgb`) for new predictions. Invalid values
are ignored. Football uses the same "winning-side probability" test as MLB, so the override maps
directly. Shadow models keep their constants.

Deploy note: the NFL and CFB deploy scripts now stage `src/model_control.py`; the football
functions fail open if it is absent, so redeploy them for the control plane to take effect.
