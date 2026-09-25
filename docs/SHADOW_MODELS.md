# MLB shadow models

Shadow models write their own tables and never `game_predictions`, so turning one on
cannot change what hankstank.com serves. All are off by default. Their only reader is the
backend's model comparison (`/api/models/mlb/compare`), which scores a row only if it was
written strictly before first pitch.

| Model | Code | Table | Trigger | Memory |
|---|---|---|---|---|
| PA sim v1 (existing) | `src/pa_sim/pipeline.py` | `game_predictions_sim` | `mode=pa_sim`, or `run_pa_sim` on `pregame_v10` | ~1 GB |
| 3-feature logistic | `src/logit3_shadow.py` | `game_predictions_logit3` | `mode=logit3`, or `run_logit3` on `pregame_v10` | small; fits the live 1 GB function |
| v2 sim + strength blend | `src/pa_sim/blend.py` | `game_predictions_sim_blend`, `game_props_sim` | `mode=sim_blend`, or `run_sim_blend` on `pregame_v10` | peak 1.3-2.2 GB measured in research, ~3.4 GB in a local smoke run; needs >= 4 GiB |

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
   `mlb-2026-sim-blend`, with `--memory=4Gi --cpu=2 --timeout=540s
   --max-instances=1`, and call it with `{"mode":"sim_blend","date":"<today>"}` once
   lineups post (or per game with `game_pks`). A full load pulls ~2M PAs from
   BigQuery on each cold start (~1 GB scanned); the cheaper alternative is precomputed
   rate tables in GCS, not built yet. `SIM_BLEND_MIN_MEMORY_MB` (default 3072) sets the
   guard; `SIM_BLEND_MEMORY_MB` overrides the detected limit.
