# Scope — plate-appearance simulation for MLB game prediction

Written 2026-09-08. Data claims below were verified against BigQuery, not assumed.

## Why this is the one remaining path

Every approach tried so far predicts a **binary** label from **team-level aggregates**:
~1,300 usable games per season, which is ~1,300 bits of training signal. Five distinct
mechanisms and 450 search trials all failed to beat a 3-feature logistic (see FINDINGS.md).

A PA simulator trains on a different substrate entirely:

| Substrate | Training rows | Source |
|---|---|---|
| Game outcomes, 2026 | ~1,300 | current approach |
| Game outcomes, 2015–2026 | ~28,600 | measured as not transferring well |
| **Plate appearances, 2015–2025** | **1,906,551** | `mlb_historical_data.statcast_pitches` |
| Plate appearances, 2026 | 163,987 | `mlb_2026_season.statcast_pitches` |
| Pitches, 2015–2026 | ~8.85M | 750k/season, 2020 short at 289k |

**~1,600× more training rows**, and a PA outcome is far less noisy than a game outcome.
That is the entire argument. Everything else is engineering.

## The honest counter-evidence

**A crude version of this already exists and shows nothing.** `matchup_v7_features` /
`game_v10_features` already carry `lineup_woba_differential`, `home_top3_woba_vs_hand`,
`matchup_advantage_home` and friends — lineup-vs-handedness aggregates. Measured on 2026,
`matchup_advantage_home` scores **AUC 0.5292**, inside the noise floor, and adding the
lineup group to the 3-feature model does not help.

So the bet is specific: **that log5 odds-ratio matchups plus a base-out run model extract
signal that naive summation of lineup wOBA destroys.** That is a real bet, not a certainty.
It is plausible — summing nine wOBAs discards the run-scoring nonlinearity, the sequencing,
and the pitcher-specific interaction — but it must be gated, not assumed.

## Architecture — four components

**A. PA outcome model.** Multinomial over 8 classes: `K, BB+HBP, 1B, 2B, 3B, HR,
out_in_play, other`. The 2026 event taxonomy maps cleanly — `field_out` 66k, `strikeout`
36k, `single` 23k, `walk` 14k, `double` 6.7k, `home_run` 5k, plus `force_out`,
`grounded_into_double_play`, `hit_by_pitch`, `sac_fly`, `triple`. Per class, combine batter
rate × pitcher rate × league baseline via the **odds-ratio (log5) method**, with
**empirical-Bayes shrinkage** toward the league mean by PA count — this is what makes a
40-PA rookie behave sanely. Handedness splits come straight from statcast (`stand`,
`p_throws` are present); `player_season_splits` has **no** handedness breakdown, so do not
rely on it.

**B. Lineup and usage model.** Nine batters and the starter from `mlb_2026_season.lineups`
(2,066 of ~2,193 games = 94% coverage, 99.7% of rows pregame). **Note: the table holds
~3 snapshots per game — 27 batting-order rows per game, not 9 — so dedupe on `fetched_at`
before use.** Model starter innings as a distribution, not a point estimate, then a bullpen
sequence. Relief is ~35% of innings, so getting this wrong is expensive.

**C. Run generator.** A **24-state base-out Markov chain** per half-inning, giving an
analytic run distribution. Prefer this over Monte Carlo: it is exact, vectorises in numpy,
and avoids sampling noise in a setting where the whole edge is ~0.005 nats wide.

**D. Game aggregator.** Convolve the two teams' run distributions into P(home win) and
handle extra innings explicitly — 2026 has **zero** tied final scores, so the tie mass must
be resolved rather than split.

## Phases, with a kill gate

| Phase | Work | Effort | Gate |
|---|---|---|---|
| **0. Premise test** | log5 lineup-vs-starter projection as a single feature; test against the 3-feature model with the both-windows gate | **1–2 days** | **If it shows nothing beyond the existing crude aggregate, stop here.** |
| 1. PA outcome model | multinomial + empirical-Bayes shrinkage on 1.9M PAs | 3–5 days | held-out PA log-loss beats league-baseline and batter-only models |
| 2. Run generator | base-out Markov chain, convolution, extra innings | 3–5 days | **predicted vs actual team runs** on ~4,800 team-games |
| 3. Bullpen & usage | starter hook distribution, reliever sequencing | 3–4 days | run-distribution fit improves vs a fixed-innings assumption |
| 4. Backtest | game-level walk-forward, both-windows gate, calibration | 2–3 days | **beats the 3-feature logistic in BOTH windows, or it does not ship** |
| 5. Productionise | Cloud Function mode, GCS artifacts, Scheduler, `game_predictions` writer | 3–4 days | — |

**Realistic total: 3–4 weeks of focused work**, with a genuine stop-decision at day 2.

## The methodological reason this is worth doing even if it fails

Each phase validates on a target with **orders of magnitude more power than the final one**:

- Phase 1 validates on **1.9M PAs**
- Phase 2 validates on **~4,800 team-game run totals**
- Phase 4 validates on **~1,300 binary wins**

Components can therefore be proven or killed long before reaching the underpowered
game-win target where everything so far has died in the noise. It is the opposite of the
current situation, where a whole model stands or falls on 400 coin flips.

## Risks, in order

1. **The premise fails.** PA-level detail may not survive aggregation to a game result.
   Phase 0 finds this out for two days of work instead of four weeks.
2. **Bullpen modelling dominates the error budget.** ~35% of innings, and bullpen *fatigue*
   is already measured at AUC 0.5021 (worthless), so quality and sequencing must carry it.
3. **Lineups arrive late.** Confirmed lineups land ~2h before first pitch; the existing
   Cloud Tasks scheduler fires ~90 min out, so this fits — but ~6% of games need a
   projected-lineup fallback.
4. **Compute.** The Cloud Function is 2GB / 540s. An analytic Markov chain fits easily; a
   10k-iteration Monte Carlo across a full slate would not. Reason to build C analytically
   from the start.
5. **Defense, baserunning and park are not in v1.** Fold park in via the existing
   `home_park_factor`; treat defense as a team-level run adjustment.

## What it would plug into

Nothing about the write path changes: a new `mode` in
`hanks_tank_ml/src/cloud_function_main.py` (alongside `pregame_v10`), artifacts in
`gs://hanks_tank_data/models/`, and the same `game_predictions` writer. Scheduler and
backend need no changes. Keep the V10 row alongside the simulator's during any trial so both
are scored on the same games.

## Recommendation

Do **Phase 0 only**, then decide. Two days, it reuses the harness in
`research/backtest_2026/`, and it answers the actual question — does PA-level matchup detail
carry game-level signal — before committing a month. Run it against the both-windows
consistency gate from script `22_`, because on this dataset a single-window improvement has
been wrong every single time.

**In parallel and independent of all of this: get market odds.** They remain the cheaper,
more certain path to the ~58% benchmark, and they are what tells you whether a simulator
that reaches 57% is good or mediocre.
