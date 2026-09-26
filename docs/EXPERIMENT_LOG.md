# Experiment Log

This log lists every model generation and experiment we can evidence, in chronological order.
The evidence labels match [ML_SYSTEM.md](ML_SYSTEM.md):

- **[M]** Measured: re-verified 2026-09-25, or read from a named stored result file.
- **[H]** Historical: taken from an older doc or a memory note, not re-run.
- **[I]** Inferred.

Memory notes live in `~/.claude/projects/-Users-VTNX82W-Documents-personalDev-mlb/memory/`.
They are cited by file name.

## Read this first: how much to trust each row

- **V1–V10 accuracy numbers are single-holdout figures** from the version docs. Most were chosen after trying many variants on the same holdout, so they carry selection bias.
- **The old docs contradict each other.** `README.md` and `MODEL_EVOLUTION_COMPLETE.md` list different algorithms and feature counts for V4–V10. Both are shown where they disagree.
- **None of the V-doc numbers is a live pregame score.** The only honest live MLB number is the pregame-filtered one: **53.98%** (n=1,984, 95% CI 51.8–56.2) against a 52.82% home rate, measured 2026-09-25. [M]
- **Log-loss differences in MLB are tiny.** The whole modelled signal is about 0.005–0.01 nats. A single split with P≈0.95 has been wrong before (see "Elo edge" below). Trust a result only if it holds in both windows, or under walk-forward and season holdouts.
- **Where 95% CIs are given, they come from paired bootstraps** over games or weeks, unless stated otherwise.

---

## A. MLB game models, V1 → V10

| Date | Version | Question / change | Method | Result as documented | Verdict | Source |
|---|---|---|---|---|---|---|
| ≤2026-01 | V1 | Baseline | LR, 5 features, 2025 holdout | 54.0% acc, AUC 0.543 [H] | superseded | `docs/V3_TRAINING_RESULTS.md` |
| ≤2026-01 | V2 | More features | LR, 44 features | 54.4%, AUC 0.534 [H] | superseded | same |
| 2026-01-16 | V3 | Trees + interactions | XGBoost, 57 features | 54.6%, AUC 0.546; LR 54.0%, RF 53.4% [H] | superseded | `src/train_v3_models.py`, `docs/V3_TRAINING_RESULTS.md` |
| 2026-01-21 | V4 | Stacked ensemble (README) / XGB + Elo (EVOLUTION doc) | — | 54.8–54.9% [H]; the two docs disagree | last-resort fallback; **still used by `predict` mode** | `src/train_v4_models.py` |
| 2026-04-02 | V5 | CatBoost / Statcast basics | — | 55.1% [H] | fallback | `src/train_v5_models.py` |
| 2026-04-03 | V6 | Pitcher arsenal + venue | stacked | 55.3–55.8% [H]; production in early April | fallback | `docs/V6_FEATURES.md` |
| 2026-04-06 | V7 | Bullpen fatigue, moon phase, venue splits | CatBoost stack | 55.5–56.4% [H]; WF-CV 53.8% [H]. Served 370 rows on 04-06/07 [M] | fallback | `docs/V7_FEATURES.md` |
| 2026-04-08 | V8 | Elo + pythag + run diff; CatBoost ×2 + LGB + MLP | 8 iterations on a 2025 holdout; walk-forward 54.81% ±0.96 | **57.65%** best (EXT3 CatBoost + team IDs) [H]. Selection bias: best of ~20 variants on one holdout | superseded | `docs/V8_EXPERIMENT_COMPLETE.md`, `src/train_v8_models.py` |
| 2026-04-09 | V8 team-ID failure | Team IDs as CatBoost categoricals | 2026 live | AUC 0.455 in 2026, worse than random [H] | → `v8_nocat` retrain (served 26 rows 04-09..15 [M]) | `docs/MODEL_LESSONS_LEARNED.md` §1, `src/retrain_v8_no_team_ids.py` |
| 2026-04-17 | V9 | Statcast quality (xERA, xFIP, xwOBA) | research, 2024 dev / 2025 holdout | 55.80% full, 57.33% XGB-Optuna, WF-CV 55.79% [H] | **research only, never deployed** (no V9 artifact in GCS [M]) | `research/v9_experiment/`, `docs/V9_EXPERIMENT_COMPLETE.md` |
| 2026-04-17 | V10 (research build) | SP quality, park factors, rest/travel, series context; 139 features | research, 2025 holdout + 283 early 2026 games | 57.70% on 2025; **61.48% "live"** [H] | **Not what shipped.** The 61.48% used 283 early games with retrospectively rebuilt Statcast features (FINDINGS.md) | `research/v10_experiment/`, `docs/V10_EXPERIMENT_COMPLETE.md` |
| 2026-04-29 | **V10 (production artifact)** | V8 training parquet + lineup matchup + calendar; XGBoost depth 4, 450 trees | train 2015–2024, eval 2025; no early stopping | Artifact metadata: 2025 holdout 56.39% / AUC 0.569 / LL 0.6816; 2026 games to date 53.6% / LL 0.6918 [M] | **production** | `src/train_v10_models.py`, GCS artifact `metrics` |
| 2026-09-08 | V10 live audit | Real live accuracy | 1,763 pregame rows, `predicted_at < game_time_utc` | 53.72% vs home 52.81%; AUC 0.5454; Brier 0.2500 (the same as predicting 0.5 every game) [H] | model kept; measurement fixed | `research/backtest_2026/FINDINGS.md` |
| 2026-09-25 | V10 re-measured | Same query, more games | 1,984 pregame rows | **53.98%** [51.8, 56.2], LL 0.6907; home 52.82% [M] | — | ML_SYSTEM.md §7 |

**The production V10 artifact uses 73 features, not the 158 (now 145) in
`V10_MODEL_FEATURES`.** [M] Training keeps only the `V10_MODEL_FEATURES` columns that exist in
the V8 parquet. So the artifact has:

- no SP xERA or percentile features;
- no park factors;
- no rest or travel features.

Its inputs are Elo, pythag, run-differential, streak, lineup-vs-hand, head-to-head wOBA and
calendar columns. It was never retrained after 2026-04-29 (ML_SYSTEM.md §2.1).

---

## B. MLB experiments after V10 (the 2026-09 research programme)

| Date | Question | Method / holdout | Measured result (95% CI) | Verdict | Source |
|---|---|---|---|---|---|
| 2026-09-08 | Contamination of the live score | Counted rows with `predicted_at >= game_time_utc` | 418 of 2,167 rows post-first-pitch; they score 55.02% vs 53.72% pregame [H]. 829 such rows on 09-25 [M] | **Adopted:** always filter pregame inside the dedupe | FINDINGS.md; memory `game-predictions-backfill-contamination` |
| 2026-09-08 | Single-feature signal | AUC with bootstrap CI on 44 stored features | 6 of 44 clear the CI; `pythag_differential` alone (AUC 0.5457) ≈ the whole model (0.5456) [H] | informs logit3 | FINDINGS.md |
| 2026-09-08 | Simpler models, walk-forward | Expanding window, 1,163 games 06-09..09-07 | logit3 LL 0.6869 vs V10 0.6917; gain +0.0037..0.0049, P(better) 0.87–0.90, CI crosses 0 [H] | not conclusive | FINDINGS.md, `02_`–`04_` |
| 2026-09-08 | More history vs this season | Same 2-feature LR, 2015–25 (26.9k games) vs 2026 only | 2026-only LL 0.6888 vs history 0.6902 [H]. Addendum 3 correction: with 54 features and within-season z-scoring, history *helps* (0.68592 vs 0.69319) but still loses to the in-season 3-feature model [H] | constraint is features × rows, jointly | FINDINGS.md, `05_`, `21_` |
| 2026-09-08 | Head-to-head before a swap | Weekly refit operational sim, 1,083 games, 18 sensitivity configs | No accuracy gain significant (McNemar p 0.37–1.00). V10 is best at the sharp end: top-100 62.0%; 130 picks ≥0.64 at 62.3%. V10 has the worst LL in all 18 configs, but by only ~0.005 nats [H] | **Do not swap**; ship pregame filter + display recalibration (not yet done) | FINDINGS.md, `06_`, `07_` |
| 2026-09-08 | Shrinkage toward base rate | λ≈0.38 | +1.11pp accuracy, but all 164 changed picks go away→home; the rule "pick home unless hp<0.48" is identical [H] | **Rejected** as a home-bias rule in disguise | FINDINGS.md |
| 2026-09-08 | Hunt for a 60% game-level slice | 67 rules, discovery/validation split | Only conf≥0.64: 60.10% on n=203 [53.2, 66.6], underpowered (n≥310 needed). Correctness gate AUC 0.503 [H] | **Rejected**: 60% is not reachable at game level on this data | FINDINGS.md, `08_`–`10_` |
| 2026-09-08 | Series unit | 511 decided series, game-1 pregame features only | V10 game-1 sign 57.53% [53.2, 61.7] vs home 54.40%; Elo+V10+pythag agree 60.14% on 286 [54.4, 65.6]; stable across halves [H]. A cross-series Elo average gave 62.43%, which was a **leak** | **Research**, candidate product | FINDINGS.md, `11_`, `12_`; memory `mlb-series-unit-beats-game-unit` |
| 2026-09-08 | Autonomous search | Optuna/TPE, 300 + 150 trials, 7 families, 161 + 13 features; nested; last 400 games untouched | Both runs chose `logit_l1` on 3 features (C=0.557). Holdout LL 0.6821 vs hand-picked 0.6820 vs V10 0.6859; winner vs V10 P=0.687 [H] | **Search exhausted**; logit3 kept as challenger | FINDINGS.md Add. 2, `15_`, `16_`; memory `mlb-autosearch-exhausted-and-sample-cap` |
| 2026-09-08 | Pitch-level features | 13 features from 772k pitches | `sp_form_k_diff` AUC 0.5457 (joint best, r≤0.33 with existing features); bullpen fatigue AUC 0.5021; adding any of them is worse (+0.0005 to +0.0056 LL) [H] | **Rejected** as additions; delete bullpen fatigue from the roadmap | FINDINGS.md, `17_`, `18_` |
| 2026-09-08 | Other mechanisms | Margin regression, Poisson/Skellam, learned map, multi-season GBM, ensembles; gate = better in both windows | Nothing passes; every gain in one window flips sign in the other [H] | **Rejected** | FINDINGS.md Add. 3, `19_`–`22_` |
| 2026-09-08 | Market benchmark | 14,859 games 2015–21 with real closing lines | Market 58.50%, LL 0.67152; its edge over always-home is +0.01902 nats vs ours +0.00717, a **2.65×** ratio [H]. `docs/MODEL_HISTORY.html` quotes **1.75×** on a 3,121-game subset (ours 0.67882, market 0.67067, home 0.68967). Both are measured, on different samples | "beat Vegas" dropped as a goal | memory `mlb-no-edge-over-market-measured`; `31_`–`33_` |
| 2026-09-08 | Orthogonal edge | `logit(market)` + our features, train 2015–19 / test 2020–21 | −0.00065 nats [−0.00145, +0.00012], P(help)=0.05; betting disagreements with edge >2% gave ROI −10.0%, >3% gave −27.3% [H] | **Rejected**: no edge over the market | same |
| 2026-09-08 | Weather | Pre-registered null for winners | weather-only AUC 0.4985; adding it −0.00054 nats. Temperature moves totals 8.56→10.04 runs across quintiles; wind r=−0.005 [H] | **Rejected** for moneyline; totals-only signal | memory `mlb-weather-is-a-totals-signal-not-moneyline`; `34_`, `35_` |
| 2026-09-08 | Injuries/IL | arena probe | +0.00012 nats; IL-only AUC 0.511 [H] | **Rejected** | memory `mlb-three-experiments-all-null-and-a-false-positive`; `36_` |
| 2026-09-08 | Kalman state-space strength | vs Elo | AUC 0.5880 vs 0.5877; −0.00105 nats in the full model [H] | **Rejected** (code kept in untracked `src/edge/`) | same; `37_` |
| 2026-09-08 | "Elo edge over market" | One split gave P=0.970 | Killed by 4 stress tests: full test +0.00003; walk-forward 20/42 folds, p=0.878; 2018 significantly hurts; all-season ROI −2.44% [H] | **False positive**; stress-test every P>0.95 | same; `38_` |
| 2026-09-25 | PA simulator v1 | Log5 multinomial, 24-state Markov; 415-game holdout | 1-season + calibrated LL 0.68134 vs logit3 0.68281 vs V10 0.68711; gain CI [−0.003, +0.008], 7 variants tried [H] | **Shadow code only, never enabled** | memory `mlb-pa-sim-and-v10-collinearity`; `23_`–`30_`, `src/pa_sim/pipeline.py` |
| 2026-09-25 | PA simulator v2 | 23 variants on dev 2016–19, frozen (`FROZEN_CONFIG.json`, variant `full_x50`), scored once on 2020–26 (15,000 games) | Winners: sim_cal vs strength +0.0005 [−0.0014, +0.0023] (a **tie**). sim + strength stacked +0.0021 [+0.0010, +0.0032]. Market beats the sim on dev 2017–19 (0.66965 vs 0.67555) [M]. Totals: sim shape tilted to the market mean +0.010 LS [+0.001, +0.020]. Starter K CRPS 1.269 vs 1.298. Batter P(≥1 H) 64.5% predicted vs 60.8% actual. Raw totals run +0.68 runs/game hot [M] | **Shadow** (`sim_blend`) | `data/backtest_2026/rich/test_report.json` and `logs/dev_table.txt` (untracked, `mlb/hanks_tank_ml` only); `42_`–`53_`; `src/pa_sim/v2.py`, `blend.py` |
| 2026-09-25 | V10 collinearity and builder bugs | rank / constant-feature audit | 142 non-constant features, rank 103; `fg_xfip`=ERA, `fg_woba`=OBP, constant-50 Statcast columns, `season_pct_complete` pinned at 1 [H] | fixed in builder (commit `02b8ec5`); placeholder columns dropped | memory `mlb-pa-sim-and-v10-collinearity` |
| 2026-09-25 | V10 train/serve skew | walk-forward, 1,743 games | 3 skewed inputs: `season_pct_complete` (1.0 live vs 0.5±0.3 in training), `day_of_week` (Mon=0 live vs Sun=1 in training), `luck_differential` sign. Fixing them changes 163 of 1,963 picks, LL ~0.001 (noise). V10 top 185 picks +4.7pp vs logit3 [−3.1, +12.8], P=0.86 [H] | **Adopted** (deployed in rev 00035 [M]); model not swapped | `research/v10_fixes/eval_v10_fixes.py`; memory `mlb-v10-train-serve-skew-and-daily-timeout` |
| 2026-09-25 | logit3 as a live shadow | — | table created, 0 rows yet [M] | **Shadow** | `src/logit3_shadow.py` |

---

## C. Football

| Date | Question | Method | Measured result (95% CI) | Verdict | Source |
|---|---|---|---|---|---|
| 2026-08-28 | NFL/CFB v1 | XGBoost (depth 3) + Elo/pythag/EPA features; walk-forward; 2025 held out | Stored walk-forward rows: NFL 2024 LL 0.607 / 2025 0.645; CFB-FBS 2024 0.562 / 2025 0.532 [M] | **production** | `src/nfl/train_nfl_models.py`, `src/cfb/pipeline.py` |
| 2026-09-25 | NFL EPA production bug | Checked stored rows | All 2025/26 live rows had `net_epa_8g` NULL because EPA was only in a cold /tmp cache [H] | **Fixed**; `nfl_v1_pure_epa` writes from 09-25 [M] | commit `74a7dae` |
| 2026-09-25 | CFB cold-cache bug | — | 382 rows for 2026 weeks 2–4 had `home_point_diff_3g=0` [H] | **Fixed** (reads `cfb_historical.games`) | commit `a3e8815` |
| 2026-09-25 | Margin ridge vs XGBoost | Walk-forward, decaying 2-season window, Φ(margin/σ) | NFL 2017–24: ridge 0.6342 vs XGB 0.6455 vs Elo 0.6462 vs Vegas 0.6086. NFL 2025: 0.6339 vs 0.6416. CFB-FBS 2023–24: 0.4948 vs 0.5406. CFB 2025: 0.4801 vs 0.5313. ATS 50.5% [H]. After the EPA fix, the NFL ridge vs fixed XGB gap is −0.008, CI crosses 0 [H] | **Shadow** (CFB ridge is the lead candidate) | `research/football_2026_09/football_eval.py`; memory `football-margin-ridge-beats-xgb` |
| 2026-09-25 | Run/pass unit matchups | Same folds, vs ridge | CFB split +0.033..+0.037 LL (worse); NFL 2025 +0.019. Best add-on is ridge + scalar EPA: CFB 2025 −0.0034 [−0.0089, +0.0020]. Style coefficient NFL −3.3±1.8, CFB −0.4±0.7 [M] | **Rejected** | `research/football_2026_09/out_{nfl,cfb}.txt`, `matchup_eval.py` |
| 2026-09-25 | Injury value (NFL 2008–25) | Opening-day starters | RB out −0.4 pts [−1.6, +0.8]; QB out −2.0 pts [−3.9, −0.2]. Using the "carries leader" as RB1 fakes −1.9 (selection on the outcome) [H] | QB ≈−2 is the only supportable term; not built | `injury_value.py`; memory `football-matchups-and-injuries-null` |
| 2026-09-25 | NFL drive simulator | 111,627 drives; config frozen on 2010–16 | Winners 2017–24: 0.6335 vs ridge 0.6342 (a tie). 2025: sim variants tie or lose (e.g. stk_c_g 0.6337 vs ridge 0.6339). Margin shape at the spread beats a normal curve (−0.061) and the league pmf (−0.029, Bonferroni-surviving on the full span). Totals lose to the market (MAE 10.85 vs 10.53) [M for 2025 holdout, H for 2017–24] | **Research only**; CFB not run (needs CFBD key) | `research/football_2026_09/drive_sim/eval_holdout.txt`, `eval_wf.txt` |
| 2026-09-25 | ESPN FPI backtest | Core-API predictor values for past games | NFL 2024–25 (n=569): XGB .6257, ridge .6313, FPI .6279, market .5981; all three models tie. CFB-FBS 2025 (n=933): XGB .5318, ridge .4823, FPI .4718, market .4663. FPI−ridge −0.011 [−0.023, +0.001], all of it in weeks 1–4 [H] | FPI **snapshotted for comparison only**, never a target | `scripts/football/eval_fpi_vs_models.py`, `src/rankings/fpi_games.py`; memory `espn-fpi-game-predictions-historical` |

---

## D. Power rankings

| Date | Question | Method | Measured result | Verdict | Source |
|---|---|---|---|---|---|
| 2026-08-31 | BT tuning | `rankings/tune.py` walk-forward | NFL 2021–25: C=0.5, LL 0.6451 (coin flip 0.6931). MLB 2023–25: C=0.06, LL 0.6808; whole surface 0.6808–0.6909 [H] | adopted (MLB still uses it) | memory `mlb-rankings-barely-separate` |
| 2026-09-25 | Audit | — | Prior too sticky (NFL wk-2 weight 0.779); MLB suspended game counted twice; bootstrap bands refit without weights; D2/NAIA teams on the FCS board [H] | **fixed** (`fa510fb`) | memory `power-rankings-audit-2026-09-25` |
| 2026-09-25 | Margin likelihood | `rankings/evaluate.py`; tune seasons then frozen eval seasons; CI by resampling weeks | CFB 2022–25: 0.5447→0.5090, −0.036 [−0.043, −0.028]. NFL, pooled over 15 unseen seasons: −0.0085 [−0.0137, −0.0034], better in 12/15. MLB: +0.0007 [−0.0007, +0.0023], a null. "No prior" board is worse in every sport (MLB +0.0035) [M from `sources.py` comments; H for the runs] | **Adopted, live** for CFB/NFL (`model='margin'` in `power_rankings` [M]); MLB stays W/L | `src/rankings/evaluate.py`, `src/rankings/sources.py` |

## E. What would change the conclusions

- **MLB game level.** New information, not new models: market odds as a labelled feature, and starter recent form if the sample limit is ever lifted. The series unit is the only 60% product with stable evidence.
- **Football.** Live shadow scoring of the ridge over the rest of 2026. Adopt only if the live gap matches the backtest. That gap is clear of 0 for CFB and within noise for NFL.
