# MLB Game Prediction — Model Evolution V1 → V10

Complete history of model development, experiments, and results for the Hank's Tank MLB prediction system.

---

## Quick Summary — All Versions

| Version | Algorithm | Features | Val Acc | WF-CV Acc | 2026 Live | Status |
|---------|-----------|----------|---------|-----------|-----------|--------|
| V1 | Logistic Regression | 5 | 54.0% | — | — | ✅ Baseline |
| V2 | Logistic Regression | 44 | 54.4% | — | — | ✅ Feature expansion |
| V3 | XGBoost | 57 | 54.6% | — | — | ✅ First XGB win |
| V4 | XGBoost | 68 | 54.8% | — | — | ✅ Elo integration |
| V5 | XGBoost | 78 | 55.1% | — | — | ✅ Statcast basic |
| V6 | XGBoost + ensemble | 94 | 55.3% | — | — | ✅ Ensemble attempt |
| V7 | XGBoost | 102 | 55.5% | 53.8% | ~53% | ✅ H2H + bullpen |
| V8 | XGBoost (Optuna) | 112 | 55.9% | 54.1% | 53.9% | ✅ Optuna tuning |
| V9 | XGBoost (Optuna) | 121 | 56.2% | 54.3% | 54.1% | ✅ Statcast quality |
| **V10** | **XGBoost (Optuna)** | **139** | **56.8%** | **54.6%** | **54.7%** | ✅ **PRODUCTION** |

**Val Acc** = 2025 holdout season. **WF-CV Acc** = walk-forward cross-validation (2019–2024). **2026 Live** = out-of-sample live season results.

---

## Evolution by Generation

### Generation 1: Baselines (V1–V3)
**Goal**: Establish working prediction pipeline.

- **V1** (LR, 5 features): Rolling win %, home field. Proved 54% is achievable with just team form.
- **V2** (LR, 44 features): Added pitcher quality, rest days, park factors, EMA weighting. +0.4% for +39 features — diminishing returns clear.
- **V3** (XGB, 57 features): Derivative interaction features (fatigue_index, momentum). XGBoost first outperforms LR when interactions present. +0.2%.

**Key Lesson**: Feature quality > quantity. Interaction effects need tree models.

---

### Generation 2: Domain Knowledge (V4–V7)
**Goal**: Add domain-specific baseball signals.

- **V4** (XGB, 68): Team Elo ratings integrated. Ratings capture long-run quality better than short-window win%.
- **V5** (XGB, 78): Statcast basics (barrel%, exit velocity). Modern metrics outperform traditional box score stats.
- **V6** (XGB+ensemble, 94): Tried soft voting ensemble; minimal gain, complexity not worth it.
- **V7** (XGB, 102): Head-to-head records, bullpen fatigue (inherited runners, high-leverage appearances), deep rest signals. Walk-forward CV revealed 53.8% true generalization — overfit warning.

**Key Lesson**: Walk-forward CV is critical. Holdout accuracy overstates true generalization by ~1.5–2%.

---

### Generation 3: Statistical Rigor (V8)
**Goal**: Reduce overfitting, establish honest WF-CV baseline.

- **V8** (XGB Optuna, 112): Systematic Optuna hyperparameter search (500 trials). Added offensive/defensive splits. WF-CV became primary eval metric. Established `54.1%` as honest generalization baseline.

**Key Lesson**: Optuna tuning gave +0.3% WF-CV over manual grid search.

---

### Generation 4: Statcast Quality (V9)
**Goal**: True pitcher quality signals (not just outcomes).

- **V9** (XGB Optuna, 121): Added xERA, xFIP, xwOBA (quality-independent), movement profiles, Stuff+ for SP. Six-phase experiment. Live 2026 accuracy: **54.1%** — first model to hold WF-CV accuracy in true live test.

**Key Lesson**: Process metrics (xERA, Stuff+) generalize better than results (ERA, WHIP). Live test is king.

---

### Generation 5: Game-Level Context (V10)
**Goal**: Integrate game-day specific context: who is pitching today, where, under what conditions.

- **V10** (XGB Optuna, 139): SP quality mapped per-game (not season-average), ballpark factors (run environment), rest/travel fatigue (consecutive days, miles traveled), series position effects. Five-phase experiment. Live 2026 accuracy: **54.7%** — best ever, first to beat WF-CV in live test.

**Key Lesson**: Game-level context (who's pitching *today*) adds more than season-average quality metrics.

---

## Accuracy Trajectory

```
Live 2026 Accuracy (honest out-of-sample):

V7:  ~53.0%  ████████████████████████████░░░
V8:   53.9%  █████████████████████████████░░
V9:   54.1%  █████████████████████████████░░
V10:  54.7%  ██████████████████████████████░
             50%                           56%

WF-CV Accuracy (2019-2024 walk-forward):

V8:  54.1%  ████████████████████████████░░
V9:  54.3%  ████████████████████████████░░
V10: 54.6%  █████████████████████████████░
```

---

## Current Production System (V10)

### Model
- **File**: `models/game_outcome_2026_v10.pkl`
- **Algorithm**: XGBoost (Optuna-tuned, 500 trials)
- **Features**: 139 total (team quality, Elo, Statcast quality, SP game-level, park factors, rest/travel)
- **Training data**: 2015–2025 (26,900+ games)

### SP Quality Display (Frontend)
- Reads `home_sp_fbv_pct`, `home_sp_k_pct`, `home_sp_bb_pct`, `home_sp_whiff_pct` (Baseball Savant percentile ranks 0–100)
- Shows: *"Glasnow has the SP quality edge — xERA 84th · K% 85th · BB% 27th · Whiff 70th · FBV 70th"*
- Falls back to league-average display for unknown SPs

### Daily Pipeline
```
Cloud Function (runs daily 10:45 AM ET):
1. Fetch today's games from MLB API
2. build_v10_features_live.py → 139 features per game
3. predict_today_games.py → win probabilities
4. Write to BigQuery game_predictions table
5. Frontend reads from BigQuery via REST API
```

### Key BQ Tables
- `mlb_2026_season.game_predictions` — predictions with SP percentile columns
- `mlb_2026_season.v10_features_2026` — daily feature matrix
- `mlb_2026_season.sp_statcast_2026` — Baseball Savant SP quality
- `mlb_2026_season.team_elo_ratings` — Elo ratings updated daily

---

## 6 Key Lessons Learned

1. **Walk-forward CV is the only honest validation** — holdout accuracy overstates true generalization by 1.5–2%. Always report WF-CV alongside holdout.

2. **Process metrics generalize; results don't** — xERA beats ERA, Stuff+ beats K/9. Outcome stats are noisy; process stats predict future performance.

3. **Game-level context beats season averages** — knowing *who is pitching today* adds more signal than knowing a team's average pitcher quality. Specificity wins.

4. **Optuna tuning is worth it** — 500-trial Bayesian search gave consistent +0.3% WF-CV vs manual grid search, across V8, V9, and V10.

5. **Live testing is king** — V9 was the first model that held its WF-CV accuracy in a true live test. V10 exceeded it. Never trust only held-out 2025 data.

6. **Data quality gates everything** — FanGraphs-sourced features introduced 40% NaN rates that couldn't be recovered. Baseball Savant direct API is the right source.

---

## Documentation Index

| Document | Versions | Contents |
|----------|----------|----------|
| [V3_TRAINING_RESULTS.md](V3_TRAINING_RESULTS.md) | V1–V3 | Full V1-V3 comparison, deployment architecture |
| [V6_FEATURES.md](V6_FEATURES.md) | V4–V6 | Feature descriptions for V4–V6 additions |
| [V7_FEATURES.md](V7_FEATURES.md) | V7 | H2H, bullpen fatigue, deep ensemble |
| [V8_EXPERIMENT_COMPLETE.md](V8_EXPERIMENT_COMPLETE.md) | V8 | Full experiment — all 8 iterations + EXT |
| [V9_EXPERIMENT_COMPLETE.md](V9_EXPERIMENT_COMPLETE.md) | V9 | Statcast quality features, Optuna tuning |
| [V10_EXPERIMENT_COMPLETE.md](V10_EXPERIMENT_COMPLETE.md) | V10 | Game-level SP quality, park factors, rest/travel |
| [ARCHITECTURE.md](ARCHITECTURE.md) | All | System architecture, BigQuery schema |
| [BIGQUERY_DATA_SCHEMA.md](BIGQUERY_DATA_SCHEMA.md) | All | BQ table schemas |
| [MODEL_LESSONS_LEARNED.md](MODEL_LESSONS_LEARNED.md) | All | Cross-version lessons |
| [CONFIDENCE_QUICK_CARD.md](CONFIDENCE_QUICK_CARD.md) | V9+ | Confidence threshold reference card |
