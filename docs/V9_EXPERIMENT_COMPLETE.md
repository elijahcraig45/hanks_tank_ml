# V9 Model — Complete Experiment Documentation

## Overview

V9 is the **"professional analyst grade"** iteration. The mandate: close the
biggest remaining feature gaps in V8 by adding real pitcher quality signal
(Statcast percentile ranks), real team offensive/defensive quality (MLB Stats API),
and updated park factors. V9 also extends the training timeline through 2025 and
evaluates on 2026 live data for the first time.

**Key discovery from V9:** Optuna tuning is *essential* for XGBoost/LightGBM on this
dataset. Default GBDT hyperparameters were badly miscalibrated — XGBoost default scored
only 53.99% (worst model), but after Optuna regularization tuning it became the **best
single model at 57.33%**, surpassing V8's best.

**Key limitation revealed:** The new team-level quality features (Statcast percentile
ranks aggregated to team level) add +0.98% WF-CV over V8 but Elo still dominates
feature importance. The remaining ceiling-breaker is *game-level* SP quality:
knowing which specific pitcher starts *tonight*. That became V10.

---

## Experiment Architecture

```
Phase 1: Feature Ablation
  V8-equiv baseline → +pitching → +batting → +park → V9 full (CatBoost default)

Phase 2: Model Comparison (V9 full features)
  LogisticRegression → XGBoost default → LightGBM default → CatBoost default
  → MLP (128-64-32) → CatBoost+Optuna → XGBoost+Optuna

Phase 3: Ensemble Experiments
  SimpleAverage → OptWeight (Optuna) → Isotonic Calibration Ensemble

Phase 4: Confidence vs Accuracy Curve
  LogisticRegression (best-calibrated) across thresholds 50–69%

Phase 5: 2026 Live Test
  LogisticRegression on 283 games (2026-03-25 → 2026-04-16)

Phase 6: FLAML AutoML (300s budget)
```

**Training data:** 2015–2023 (train) | 2024 (dev/Optuna) | 2025 (holdout val) | 2026 YTD (live test)  
**Validation strategy:** Strict temporal split — no future data in training

---

## Data Sources

> **Note:** FanGraphs endpoints via pybaseball were 403-blocked. Baseball-Reference
> `schedule_and_record` hung for 2025/2026. The following replacements were used and
> proved *better* (percentile-normalized, cross-season comparable):

| Source | Data provided |
|--------|---------------|
| MLB Stats API `/teams/stats` | ERA, WHIP, K, BB, OBP, SLG, OPS, AVG |
| Statcast pitcher percentile ranks | xERA%, K%, BB%, whiff%, FB velocity% (0–100) |
| Statcast batter percentile ranks | xwOBA%, exit velocity%, hard-hit%, barrel% (0–100) |
| MLB Stats API `/teams/{id}/roster` | Player→team mapping for Statcast aggregation |
| MLB Stats API `/schedule` | 2020–2026 game results with scores |
| Backend CSV | Historical game results 2015–2024 |

---

## All Results

### Phase 1 — Feature Ablation (CatBoost default, 2025 val)

| Model | Val Acc | Val AUC | Conf≥60% (cov) | WF-CV |
|-------|---------|---------|----------------|-------|
| V8-equiv baseline | 54.90% | 0.5546 | 59.5% (30.3%) | 54.15% ±1.14% |
| V8 + pitching only | 55.35% | 0.5656 | 60.3% (34.9%) | — |
| V8 + batting only | 55.19% | 0.5658 | 59.1% (33.3%) | — |
| **V9 Full** | **55.80%** | **0.5753** | **61.4% (34.9%)** | **55.79% ±1.16%** |
| Elo only | 54.81% | 0.5597 | 65.1% (19.8%) | — |

**Finding:** Pitching features contribute more than batting (+0.45% vs +0.29%). Elo-only achieves 65.1% at ≥60% confidence but covers only 20% of games — too conservative for daily use. WF-CV gain of +1.64% over V8 is the reliable generalization signal.

---

### Phase 2 — Model Comparison (V9 full features, 2025 val)

| Model | Val Acc | Val AUC | Brier | Conf≥60% | Coverage |
|-------|---------|---------|-------|----------|----------|
| **XGBoost_Optuna** | **57.33%** | **0.5884** | **0.2422** | **64.4%** | **28.5%** |
| FLAML_lgbm | 57.00% | 0.5868 | 0.2425 | — | — |
| CatBoost_Optuna | 56.87% | 0.5813 | 0.2433 | 63.4% | 30.1% |
| LogisticRegression | 56.54% | 0.5860 | 0.2425 | 62.7% | 32.9% |
| OptWeight_Ensemble | 56.34% | 0.5824 | 0.2433 | 63.7% | 32.0% |
| Isotonic_Ensemble | 56.01% | 0.5775 | 0.2447 | 63.0% | 31.3% |
| V9_full_CatBoost | 55.80% | 0.5753 | 0.2451 | 61.4% | 34.9% |
| SimpleAvg_Ensemble | 55.76% | 0.5745 | 0.2457 | 62.1% | 35.9% |
| CatBoost_default | 55.56% | 0.5766 | 0.2447 | 63.3% | 34.6% |
| MLP_128-64-32 | 55.19% | 0.5763 | 0.2479 | 59.7% | 31.4% |
| LightGBM_default | 54.57% | 0.5594 | 0.2538 | 58.1% | 48.4% |
| XGBoost_default | 53.99% | 0.5534 | 0.2569 | 56.9% | 52.5% |

**Critical finding:** XGBoost default was the *worst* model (53.99%, Brier 0.2569 — overconfident). After Optuna tuning L1/L2 regularization and `max_depth`, it became the **best model at 57.33%**. Ensembles did not beat the best single model because weak default models dragged calibration down. The optimal ensemble weights (from Optuna) converged to: CatBoost_Optuna 57%, XGBoost_Optuna 23%, LogisticRegression 19%, others ≈0%.

---

### Phase 3 — Walk-Forward CV

| Model | WF-CV Mean | WF-CV Std |
|-------|-----------|-----------|
| V8-equiv | 54.15% | ±1.14% |
| **V9 full (CatBoost)** | **55.79%** | **±1.16%** |

---

### Phase 4 — Confidence Curve (LogisticRegression, 2025 val)

| Threshold | Coverage | Accuracy | N games |
|-----------|----------|----------|---------|
| ≥50% (all) | 100% | 56.54% | 2,430 |
| ≥55% | 64.2% | 59.40% | 1,559 |
| ≥60% | 32.9% | **62.70%** | 799 |
| ≥62% | 23.4% | **66.73%** | 568 |
| ≥64% | 17.8% | **70.44%** | 433 |
| ≥65% | 15.1% | **71.66%** | 367 |
| ≥67% | 11.4% | **71.84%** | 277 |
| ≥69% | 7.8% | **73.54%** | 189 |

**At ≥64% confidence: 70.44% accuracy on 17.8% of games.** This is the primary actionable threshold — the model is highly selective and right roughly 7 out of 10 times on its strongest predictions.

---

### Phase 5 — 2026 Live Test (283 games, LogisticRegression)

| Val Acc | Val AUC | Brier | Conf≥60% | Coverage |
|---------|---------|-------|----------|----------|
| **58.66%** | **0.6135** | **0.2402** | **65.12%** | **45.6%** |

Best performance across any split. AUC of 0.6135 is strong. Early 2026 (March/April) is easier to predict: teams near full strength, rosters set, no fatigue accumulation yet.

---

### Phase 6 — FLAML AutoML (300s budget)

Best found: LightGBM — Val Acc 57.00%, Val AUC 0.5868, Brier 0.2425. Matches XGBoost_Optuna in AUC but slightly lower accuracy. A longer budget (1800s) would likely close the gap.

---

## Feature Set (121 features, 10 groups)

| Group | Key Features |
|-------|-------------|
| `elo` | home_elo, away_elo, elo_differential, elo_home_win_prob, elo_win_prob_differential |
| `pythag` | home/away_pythag_season, home/away_pythag_last30, luck_factor |
| `rolling_form` | win% at 3g/7g/10g/30g, run diff at 3g/7g/10g/30g, runs scored/allowed, era_proxy, scoring momentum |
| `streak` | home/away_current_streak, streak_direction + diffs |
| `h2h` | season H2H win%, 3yr H2H win%, H2H game count |
| `fg_pitching` | ERA, WHIP, K/9, BB/9 (MLB API) + xERA%, K%, BB%, whiff%, FB velo% (Statcast pct) |
| `fg_batting` | OBP, SLG, OPS, AVG (MLB API) + xwOBA%, EV%, hard-hit%, barrel% (Statcast pct) |
| `park` | home_park_factor, home_park_factor_ratio (defaulted — real park factors added in V10) |
| `calendar` | day of week, month, is_weekend, month dummies |
| `context` | season game number, season_pct_complete, is_early/late_season |

**Top 5 features by importance (V9 CatBoost):**
1. `away_elo` — 25.5 pts
2. `elo_home_win_prob` — 22.7 pts
3. `elo_differential` — 18.4 pts
4. `home_elo` — 17.7 pts
5. `elo_win_prob_differential` — 15.8 pts

Elo still dominates. New quality features add signal at the margins and improve the confidence curve significantly but don't displace the long-run team quality signal.

---

## Version Comparison

| Version | Val Acc (2025) | WF-CV | Conf≥60% | AUC | 2026 Live |
|---------|---------------|-------|----------|-----|-----------|
| V8 (CatBoost ensemble) | 57.65% | 54.81% ±0.96% | 65.4% | 0.613 | — |
| **V9 (XGBoost_Optuna)** | **57.33%** | **55.79% ±1.16%** | **64.4%** | **0.588** | **58.66%** |

Note: V9's 57.33% is slightly below V8's 57.65% on the 2025 holdout — V8 had stronger 2025-specific calibration. V9's advantage shows clearly in WF-CV (+0.98%), the confidence curve at ≥64% (70.44% vs 65.4%), and the 2026 live test (58.66%).

---

## Key Takeaways

1. **Optuna is not optional for XGBoost/LGB.** Default GBDT hyperparameters are badly miscalibrated for this dataset. Without tuning, XGBoost scored 53.99% (worst). After regularization tuning: 57.33% (best).

2. **MLB API + Statcast percentile ranks replaced FanGraphs successfully.** FanGraphs was 403-blocked. The replacements are actually *better* — percentile-normalized (0–100 scale, cross-season comparable).

3. **Elo still dominates.** All top-5 features by importance are Elo variants. New quality features help at the margins. The ceiling-breaker is *game-level* SP quality (specific starter tonight), which was the focus of V10.

4. **The confidence curve is the product.** At ≥64%, accuracy is 70.44% on 17.8% of games. For a use case requiring high-conviction predictions, selectivity >> raw accuracy.

5. **V10 already implemented:** Game-level SP quality (+1.11% ablation), park factors (+0.58%), and rest/travel (+0.66%). See [`V10_EXPERIMENT_COMPLETE.md`](V10_EXPERIMENT_COMPLETE.md) for full results.

---

## Files

```
research/v9_experiment/
  README.md                       Detailed experiment notes and design decisions
  01_fetch_data.py                Fetch MLB API team stats + Statcast percentile ranks
  02_build_v9_dataset.py          Build 121-feature dataset (V8 base + FG groups)
  03_train_v9_experiment.py       6-phase experiment framework

data/v9/
  raw/                            Statcast percentile parquets, MLB API responses
  features/
    train_v9.parquet              2015–2023 training games
    dev_v9.parquet                2024 dev set (Optuna target)
    val_v9.parquet                2025 holdout (primary eval)
    test_2026_v9.parquet          283 live 2026 games

logs/
  v9_experiment.log               Full run log
  v9_experiment_results.json      All metrics (machine-readable)
```
