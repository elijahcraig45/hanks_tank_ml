# Model Cards

This file has one card per live or shadow model. It was verified 2026-09-25; the evidence labels
match [ML_SYSTEM.md](ML_SYSTEM.md): **[M]** measured, **[I]** inferred, **[H]** historical.
The experiments behind each model are in [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).

**Owner: Henry (elijahcraig45) for every model below.** Only the ML repo writes these tables.

| Model | Sport | Status | Table | `model_version` |
|---|---|---|---|---|
| V10 XGBoost | MLB | production | `mlb_2026_season.game_predictions` | `v10` |
| logit3 | MLB | shadow | `mlb_2026_season.game_predictions_logit3` | `logit3_l1_v1` |
| sim_blend (PA sim v2 + strength) | MLB | shadow | `game_predictions_sim_blend`, `game_props_sim` | `sim_blend_v2` |
| V4 weekly | MLB | legacy, live | `weekly_predictions` | V4 |
| NFL XGBoost | NFL | production | `nfl_season.game_predictions` | `nfl_v1_pure_epa` |
| NFL margin ridge | NFL | shadow | `nfl_season.game_predictions_ridge_shadow` | `nfl_v2_margin_ridge` |
| CFB XGBoost | CFB | production | `cfb_season.game_predictions` | `cfb_v1` |
| CFB margin ridge | CFB | shadow | `cfb_season.game_predictions_ridge_shadow` | `cfb_v2_margin_ridge` |
| Power rankings | all | production | `{mlb_2026,nfl,cfb}_season.power_rankings` | `model` = `bt` / `margin` |
| ESPN FPI snapshot | NFL/CFB | comparison only | `fpi_game_predictions` | ESPN's model, not ours |
| Season sim (rest-of-season Monte Carlo) | NFL/CFB | shadow | `{nfl,cfb}_season.season_sim_team`, `season_sim_bracket`, `season_sim_games` | `season_sim_v1` |

---

## MLB — V10 XGBoost (production)

- **Artifact** [M]
  - Location: `gs://hanks_tank_data/models/vertex/game_outcome_2026_v10/model.pkl`, written 2026-04-29.
  - Format: a dict with keys `model`, `version='v10'`, `model_name='V10_LineupMatchupXGB'`, `features`, `fill_values`, `metrics` and `trained_at`.
- **Inputs:** 73 features. [M] They cover:
  - Elo and Elo win probability;
  - pythag, season and last 30 games;
  - luck factor;
  - win% over 7 games and the season;
  - run differential, runs scored and allowed, and ERA proxy, each over 10 and 30 games;
  - scoring momentum and streaks;
  - lineup wOBA and K% vs starter hand, by lineup slot group;
  - head-to-head wOBA and `matchup_advantage_home`;
  - `day_of_week`, `month` and month dummies, and `season_pct_complete`.

  There are **no SP-quality, park or rest features**, whatever the V10 docs say. Missing values
  are filled with training medians.
- **Training window:** `data/training/train_v8_2015_2024.parquet`, plus lineup joins from `matchup_features_historical`. The 2025 parquet is used as an eval set, without early stopping. XGBoost settings: depth 4, 450 trees, learning rate 0.035. [M]
- **Output:** home-win probability. It is written per game by `pregame_v10` Cloud Tasks at up to 5 checkpoints, and the latest write replaces the row. [M]
- **Metrics**

| Measure | Accuracy | AUC | Log loss | Brier | Label |
|---|---|---|---|---|---|
| Artifact metadata, 2025 | 56.39% | 0.569 | 0.6816 | 0.2443 | [M] |
| Honest 2026 live, pregame rows (n=1,984) | **53.98%** (CI 51.8–56.2) | | 0.6907 | | [M] |
| Home rate, same games | 52.82% | | | | [M] |

  The 2025 parquet has frozen team features, so the 2025 figure is optimistic. [H]
- **Calibration:** poor. The 0.60–0.65 bucket is too confident, and the whole signal is about 0.005 nats over the base rate. On the other hand, the high tier (≥0.64) scored 60.10% on 203 games, and V10 wins at the sharp end against every simpler candidate. [H] Tiers are absolute: 0.64 and 0.57.
- **Known issues:**
  - It has never been retrained, because `train_weekly` is broken.
  - Three train/serve skews were fixed in the features on 2026-09-25 (rev 00035). The model itself is unchanged.
  - The fallback chain is silent.
  - Backfill rows contaminate naive evaluation, so filter `predicted_at < game_time_utc`.
- **Status:** production. Do not swap it without a both-windows win.

## MLB — logit3 (shadow)

- **Code:** `src/logit3_shadow.py`. It runs as `mode=logit3`, or with `run_logit3:true` on every `pregame_v10` task (backend `20260925t180718`). [M]
- **Inputs:** `elo_differential`, `pythag_differential` and `sp_quality_composite_diff`. They are taken from the latest `game_v10_features` row with `computed_at < game_time_utc`. [M]
- **Model:** train-median imputation, standardisation, then L1 logistic regression with C=0.557 (the autosearch winner). [M]
- **Training window:** refit every run on this season's Final regular-season games before the target date. It skips the run below 200 games. [M]
- **Output:** append-only rows with `coef_json`, `n_train` and `features_computed_at`. Games that have already started are skipped. [M]
- **Evidence:** holdout log loss 0.6821 vs V10 0.6859, not significant. Accuracy −0.18pp in the head-to-head. [H]
- **Calibration:** better calibrated than V10, but it almost never reaches the 0.64 tier. [H]
- **Status:** shadow. The table exists with 0 rows as of 2026-09-25. [M]

## MLB — sim_blend (shadow)

- **Code:** `src/pa_sim/blend.py` and `v2.py`, with coefficients in `src/pa_sim/blend_coefs.json`. It runs on function `mlb-2026-sim-blend` (4 GiB), one task per game at T−90. [M]
- **Inputs:**
  - about 2M historical plate appearances, from Statcast, loaded on each cold start;
  - pregame lineups and starters;
  - park and venue data.
- **Method:** each game is simulated 3,000 times with the frozen `full_x50` config. The result is combined with a strength logistic:
  - **Strength inputs:** Elo with K=4 and a 24-point home edge, plus season pythag with a 10-game prior.
  - **Strength fit:** the 3 prior seasons.
  - **Blend:** `blend = σ(−0.0490 + 0.5141·logit(strength) + 0.7581·logit(sim))`, with coefficients fit on 2016–2025 (22,746 games). [M]
- **Outputs:**
  - `game_predictions_sim_blend` holds P(home), the raw and calibrated sim probabilities, `strength_p` and mean runs.
  - `game_props_sim` holds the total-runs pmf, tilted down by 0.47 runs, and each starter's K pmf. Batter props are written only with `experimental_props`. [M]
- **Evidence** (2020–26, 15,000 games) [M]:
  - The blend beats strength alone by +0.0021 nats [+0.0010, +0.0032].
  - The sim alone ties strength.
  - The market beats both.
  - Totals and starter-K props are where the real gains are.
- **Known issues:**
  - Raw totals run hot, by about 0.68 runs per game overall, cause unknown.
  - P(≥1 hit) is over-predicted.
  - The coefficients were fit through 2025, so only the 2026 scores are out of sample.
  - The cold start scans about 1 GB in BigQuery.
- **Status:** shadow, 0 rows as of 2026-09-25. [M]

## MLB — V4 weekly (legacy, still live)

`mode=predict` (Friday 5 AM ET) loads `models/vertex/game_outcome_2026/model.pkl` (V4,
2026-03-27) and writes `weekly_predictions`. [M] No reader of this table was found in the
backend. [I]

- **Status:** legacy. It is a candidate for retirement, or for pointing at V10.

## NFL — XGBoost `nfl_v1_pure_epa` (production)

- **Inputs:** about 90 team features from nflverse: Elo, rolling form, rest, and EPA from `nfl_historical.team_week_epa`. Market columns are excluded (the "pure" variant). Of the 90, 38 are exact linear duplicates. [H]
- **Training:** refit each run on every completed game (`predict_nfl.predict_week`). XGBoost settings: depth 3, `min_child_weight` 20, 400 trees. It refuses to run if EPA is missing. [M]
- **Output:** `nfl_season.game_predictions`, written with a game_id-scoped DELETE and append. Tiers are high 0.72 and medium 0.60. [M]
- **Evidence:** 2025 walk-forward log loss 0.6416 [H]. Stored 2025 rows score 0.645 [M]. After the EPA fix it is within noise of the ridge (−0.008, CI crosses 0) [H]. Vegas scores 0.609.
- **Known issues:**
  - The EPA feed was NULL for all live rows until 2026-09-25.
  - `win_pct_season` is actually a last-8-games rate.
  - `--backfill` without `--no-write` truncates the table.
- **Status:** production.

## NFL — margin ridge (shadow)

- **Model:** `home_margin = r_home − r_away + HFA`, a weighted ridge with a decaying 2-season window, then P(win) = Φ(margin/σ). Settings: α=3, τ=16 weeks, σ=12.48. The config lives in `src/nfl/margin_ridge.py`. [M]
- **Evidence:** walk-forward 2017–24 log loss 0.6342 vs XGB 0.6455; 2025 0.6339 vs 0.6416. [H] Against the market it is 50.5% ATS. [H]
- **Output:** `game_predictions_ridge_shadow` (15 rows), which includes the predicted margin and the team power ratings. [M]
- **Status:** shadow; env `NFL_RIDGE_SHADOW=1`.

## CFB — XGBoost `cfb_v1` (production)

- **Inputs:** Elo, pythag, 3-game point differential, streaks, divisional and cross-division flags. Trained per division (FBS, FCS) on every prior game in that division, with a minimum of 300 games. The games cache starts in 2021. [M]
- **Evidence:** 2025 FBS walk-forward log loss 0.5313. Stored 2026 FBS rows score 85.9% on 256 games, but that sample includes FBS-vs-FCS blowouts and is early-season. [M]
- **Known issues:**
  - Cross-division tagging for transitioning teams (memory `cfb-2026-cross-division-mistagging`). [H]
  - The margin ridge beats this model by about 0.05 in log loss.
- **Status:** production. It is the first candidate for replacement, by the ridge.

## CFB — margin ridge (shadow)

- **Config:** α=1, τ=16, σ=15.55, with a ±45-point margin cap and an FBS/FCS covariate. One fit covers both divisions. [M]
- **Evidence** [H]:
  - Log loss: 2023–24 0.4948 vs XGB 0.5406; 2025 0.4801 vs 0.5313.
  - Against FPI: FPI is ahead by only −0.011 [−0.023, +0.001], all of it in weeks 1–4.
- **Output:** `game_predictions_ridge_shadow` (121 rows), written with `replace_game_ids`. [M]
- **Status:** shadow; env `CFB_RIDGE_SHADOW=1`.

## Season sim — rest-of-season Monte Carlo (shadow, EXPERIMENT)

- **Code:** `src/season_sim/` (`engine.py`, `nfl.py`, `cfb.py`, `run.py`, `store.py`). Mode `season_sim` in both football functions. [M]
- **Game model:** the margin ridge (`NFL_RIDGE` / `CFB_RIDGE`), margin ~ Normal(r_home − r_away + HFA [+ division term], sigma). Each simulated season first draws every rating from the ridge's Gaussian posterior (s²(X'WX + αI)⁻¹), and per-game noise is shrunk so a single game's variance stays at the fitted sigma (NFL 12.48 → 11.2; CFB 15.55 → 13.8). Variant `draw`. [M]
- **NFL:** 17-game standings; division, common-games, conference, SOV, SOS, net-points tiebreakers and the coin (points-ranking and net-touchdown steps skipped); 14-team playoff with reseeding; Super Bowl neutral. Reproduces 48/48 real conference fields 2002-25. [M]
- **CFB:** conference standings from conference games; top two (or Sun Belt division winners; SEC/B1G/MAC 2022-23, ACC/MWC 2022) meet in the title game; approximate tiebreakers (h2h, common opponents, rating, coin). CFP by season: four teams to 2023; 2024 five best champions + 7, champions seeded 1-4; 2025 same field, straight seeding; 2026 Power-4 champions + best Group of Six team + 7, straight seeding. Committee proxy = end rating − 8·losses + 4·champion − 12·G6. [M]
- **Evidence:** EXPERIMENT_LOG §C (2026-09-28 rows). Better than record extrapolation and coin flips with CIs clear of 0; worse than a closing-line sim (which is leaky). [M]
- **Runtime at 10,000 sims** (local, CF pins): NFL 0.6 s / 367 MB peak; CFB 2.9 s simulation, 29 s end to end with the BigQuery read and ESPN schedule calls, 970 MB peak. Limits are 2 GB / 540 s. [M]
- **Known limits:** no injuries or QB changes; ratings drift only through the draw; the committee is a proxy; conference tiebreakers are approximate; bowls are not simulated; the sim stops once the postseason starts (it would re-simulate played playoff games). As of 2026-09-28 NFL week 3 had MNF pending, which is simulated.
- **Status:** shadow. Tables need `scripts/gcp/football/create_season_sim_tables.sql`; jobs `nfl-weekly-season-sim` (Tue 7:30 ET) and `cfb-weekly-season-sim` (Sun 8:00 ET) are created only by `deploy_* --shadow`.

## Power rankings (all sports)

- **Code:** `src/rankings/` (`core.py`, `build.py`, `sources.py`, `evaluate.py`). [M]
- **Method** [M]:
  - Ridge Bradley-Terry on W/L (MLB), or a margin ridge (NFL, and CFB with a division term).
  - A decaying prior on last season: `w0·exp(−week/τ)`.
  - Rank bands (`rank_p05/p50/p95`) come from a bootstrap over games that uses the same model and weights as the point estimate.

| Sport | Model | C | w0 | τ (weeks) | Other |
|---|---|---|---|---|---|
| MLB | `bt` | 0.06 | 1.0 | 20 | |
| NFL | `margin` | 0.5 | 0.5 | 8 | cap 21, scale 4.69 |
| CFB | `margin` | 2.0 | 0.25 | 16 | α 0.3, scale 10.62 |

- **Cadence:**
  - MLB: every `daily` run, plus the Monday 6:30 job.
  - NFL: inside the Tuesday ingest.
  - CFB: inside the Sunday ingest.
- **Evidence:** see EXPERIMENT_LOG §D. MLB ratings barely separate teams (0.012 nats), so the boards must show rank ranges, and the ratings are not a game predictor. [H]
- **Status:** production. All three boards were refreshed on 2026-09-25. [M]

## ESPN FPI snapshot (comparison only)

`src/rankings/fpi_games.py` appends ESPN's pregame win probability and margin before kickoff.
Backfilled rows are marked `source='backfill_after_kickoff'`. It is not our model and nothing is
tuned toward it. [M]
