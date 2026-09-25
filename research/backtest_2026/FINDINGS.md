# 2026 Season Backtest — Findings

Run 2026-09-08 against the completed 2026 regular season. Scripts in this directory;
data cached to `data/backtest_2026/`.

## Headline

**V10 in production is at 53.72% over the full season, not the 61.48% the V10 doc
reports.** Always picking the home team scores 52.81% on the same games. The real edge
is **+0.91pp**, and the Brier score is 0.2500 — identical to predicting 0.500 on every
game.

| | Games | Accuracy | AUC | Brier |
|---|---|---|---|---|
| Always pick home | 1,763 | 52.81% | 0.500 | 0.2492 |
| **V10, as actually served** | **1,763** | **53.72%** | **0.5454** | **0.2500** |
| Raw Elo formula, no ML at all | 1,761 | 53.78% | 0.5360 | 0.2613 |
| V10 doc's claim (283 early games) | 283 | 61.48% | 0.6482 | 0.2322 |

## Two measurement bugs that inflated the old number

1. **The 283-game sample was early season only** (2026-03-25 → 04-16) and was scored with
   *retrospectively rebuilt* features — end-of-season Statcast percentiles that were not
   knowable at first pitch. The production pipeline uses as-of-gameday values.
2. **`game_predictions` is also written by post-hoc backfills.** 418 of 2,167 rows have
   `predicted_at >= game_time_utc`, some 17 days after the game. Deduping on
   `ORDER BY predicted_at DESC` without a pregame guard silently selects those rows, and
   they score higher (55.02%) than genuine pregame rows. **Any query measuring model
   performance must filter `predicted_at < game_time_utc`.**

## The signal ceiling is real

Of 44 stored as-of-gameday features, only **6 clear a 95% CI on single-feature AUC**:

| Feature | AUC |
|---|---|
| `pythag_differential` | 0.5457 |
| `away_starter_k_bb_pct` | 0.4603 |
| `sp_quality_composite_diff` | 0.5369 |
| `sp_xera_diff` | 0.5364 |
| `elo_differential` / `elo_home_win_prob` | 0.5360 |

`pythag_differential` **alone** (AUC 0.5457) matches the entire 139-feature model (0.5456).

## Walk-forward results

Expanding window, train on everything before the block, predict the block, refit. 1,163
test games (2026-06-09 → 09-07). All candidates scored on identical games; imputation,
scaling, blend weights and shrinkage fit on the train fold only.

| Model | Acc | AUC | Brier | LogLoss |
|---|---|---|---|---|
| logit, 2 probability-logit stack | 54.69% | 0.5598 | 0.2468 | 0.6868 |
| **logistic, 3 features** | **54.51%** | **0.5594** | **0.2469** | **0.6869** |
| V10 shrunk toward base rate | **55.98%** | 0.5475 | 0.2474 | 0.6880 |
| always home | 52.97% | 0.500 | 0.2491 | 0.6914 |
| **V10 (incumbent)** | 54.17% | 0.5487 | 0.2491 | 0.6917 |
| XGBoost, all 44 features, V10-like depth | 53.65% | 0.5425 | 0.2976 | 0.8497 |
| logistic, all 44 features | 52.11% | 0.5090 | 0.2573 | 0.7098 |

**More features is strictly worse.** A 3-feature logistic beats the 139-feature model on
every metric; using all 44 features drops AUC below the always-home baseline.

Paired bootstrap vs V10, 4,000 resamples: log-loss gains of +0.0037 to +0.0049,
P(better) = 0.87–0.90, CIs crossing zero. **Directionally consistent, not conclusive.**
The accuracy gain from shrinkage (+1.81pp) is the largest single effect.

When the blend weight is fit inside each training fold it converges to **0.94 on the
3-feature logistic and 0.06 on V10** — the training data wants to discard V10.

## The one unambiguous defect: confidence tiers are miscalibrated

V10's shrink factor tunes to λ≈0.38 in every fold: probabilities should be compressed to
~38% of their distance from the base rate. The calibration table shows why:

| V10 predicted | n | Mean pred | Actual |
|---|---|---|---|
| 0.50–0.55 | 271 | 0.526 | 0.539 |
| 0.55–0.60 | 200 | 0.573 | 0.595 |
| **0.60–0.65** | **145** | **0.622** | **0.510** |
| 0.65–1.00 | 85 | 0.693 | 0.624 |

**The 0.60–0.65 bucket is a coin flip.** 145 games are served to users as 62% confident
and land at 51.0%. Since the V9/V10 docs conclude "the confidence curve is the product,"
this is the highest-value fix in the stack and needs no retraining.

## More history does not help

Same 2-feature logistic (the two features present in both the historical V8 parquet and
the 2026 stored rows), scored on the same 1,163 games. Feature distributions checked
comparable first, so the transfer is valid.

| Training set | Acc | AUC | LogLoss |
|---|---|---|---|
| **2026 season only (600–1,713 games)** | **54.69%** | 0.5458 | **0.6888** |
| 2015–2025 history (~26,900 games) + in-season recalibration | 53.65% | 0.5384 | 0.6894 |
| 2015–2025 history (~26,900 games) | 53.40% | 0.5427 | 0.6902 |

45× more training data makes it *worse*. The Elo/pythag→outcome relationship is not
stable across seasons — this is the same instability that
`MODEL_LESSONS_LEARNED.md` §1 found with team IDs, and it explains why V10 validated on
history and then failed live.

## Head-to-head vs the incumbent (operational simulation)

Added 2026-09-08 in response to "how does it compare before we push". Weekly refit
(matching a Scheduler retrain), train only on completed games, serve frozen parameters
for the following week. 1,083 test games, 2026-06-16 → 09-07. Scripts `06_`/`07_`.

| Model | Acc | AUC | Brier | LogLoss | ECE | high-tier n | high-tier acc |
|---|---|---|---|---|---|---|---|
| blend50 | **55.68%** | 0.5595 | 0.2471 | 0.6873 | 0.0229 | 0 | — |
| v10_shrunk | 55.49% | 0.5482 | 0.2477 | 0.6885 | 0.0339 | 0 | — |
| stack | 54.48% | **0.5603** | **0.2470** | **0.6871** | **0.0133** | 2 | — |
| logit3 | 54.20% | 0.5598 | 0.2470 | 0.6871 | 0.0185 | 2 | — |
| **prod_v10 (incumbent)** | 54.39% | 0.5502 | 0.2491 | 0.6916 | 0.0299 | **130** | **62.31%** |
| always_home | 52.45% | 0.500 | 0.2494 | 0.6920 | 0.0055 | 0 | — |

### Three findings that argue *against* pushing a model swap

**1. No accuracy gain is statistically significant.** Paired McNemar on the same games:

| Candidate | Δacc | Picks flipped | McNemar p | LogLoss gain | P(better) |
|---|---|---|---|---|---|
| blend50 | +1.29pp | 113 W / 99 L | 0.372 | +0.00437 | 0.872 |
| v10_shrunk | +1.11pp | 88 W / 76 L | 0.390 | +0.00313 | 0.815 |
| stack | +0.09pp | 126 W / 125 L | 1.000 | +0.00451 | 0.865 |
| logit3 | −0.18pp | 147 W / 149 L | 0.954 | +0.00452 | 0.845 |

Every log-loss CI crosses zero. The candidates win coin flips, not games.

**2. The incumbent is better at the sharp end — which is the product.** Top-k accuracy
on the games each model is most confident about:

| Model | top50 | top100 | top150 | top200 | top300 | top500 |
|---|---|---|---|---|---|---|
| **prod_v10** | 56.00 | **62.00** | **60.67** | 57.50 | 56.00 | 57.20 |
| logit3 | 54.00 | 55.00 | 59.33 | 57.50 | **58.67** | 58.20 |
| stack | 54.00 | 57.00 | 58.67 | **58.00** | **58.67** | **58.40** |
| v10_shrunk | **62.00** | 61.00 | 56.00 | 55.00 | 55.67 | 55.40 |

V10 ranks its top 100–150 games better than any candidate. The candidates win only in the
broad middle (top300–500). Swapping wholesale trades the high-conviction tier — 130 picks
at 62.31% — for a marginally better middle. **The candidates almost never clear a fixed
0.64 confidence threshold at all** (n = 0–2), because being well calibrated on a
near-random process means never being confident.

**3. The shrinkage "win" is a home-bias rule in disguise.** All 164 picks that shrinkage
changes are away→home, all in the narrow band hp ∈ [0.420, 0.499]. And the trivial rule
*"pick home unless hp < 0.48"* scores **55.49% — identical to the tuned shrinkage.** The
gain is not recalibration; it is "stop making marginal away picks," which works because
V10's away picks run 52.52% vs 55.56% on home picks. That is one season of home-field
environment (2026 home rate 52.8%) and should not be shipped as a rule on this evidence.

Note also that shrinkage does **not** preserve the pick set as a single monotone transform
would (Spearman ρ=0.708, only 105/130 of the top picks shared) precisely because λ is
refit weekly and re-crosses the 0.5 boundary differently each week.

### What *is* robust

Sensitivity across 18 harness configurations (min_train ∈ {400,600,800} × cadence ∈
{7,14} × C ∈ {0.03,0.1,0.3}): **the incumbent has the worst log-loss in all 18**, and one
of the candidates is best in every row. The calibration improvement is real and
setting-independent. Monthly, the candidates beat the incumbent in 3 of 4 months
(June: 49.26% incumbent vs 53.69% blend50).

But keep the magnitude in perspective: 0.6916 → 0.6871 is a **0.65% relative** log-loss
reduction, against 0.6920 for always predicting the base rate. The entire modelled signal
in this stack is worth **~0.005 nats** — the same "barely separates" regime already
measured for MLB power rankings.

## Recommendations

Ordered by evidence strength, not by appeal.

1. **Ship now — the pregame filter** (`predicted_at < game_time_utc` inside the dedupe).
   This is a measurement correctness bug, not a modelling choice. Every accuracy number
   the project reports is contaminated without it.
2. **Ship now — display-only recalibration.** Keep V10's picks and ordering exactly as
   served; recalibrate only the *displayed* probability, and re-derive the tier cut-offs
   as **quantiles** (e.g. top 12% = high) rather than the absolute 0.64. This buys honest
   confidence numbers with zero risk to pick quality, and avoids the tier emptying out.
   Do **not** let recalibration change which side is picked.
3. **Do not swap the model.** No candidate shows a significant accuracy gain, and the
   incumbent is better on top-100/150 pick quality, which is the high-conviction product.
   The 3-feature logistic remains the more honest *baseline* and is worth keeping in the
   repo as a permanent challenger, but it has not earned production.
4. **Treat the away-pick weakness as a hypothesis, not a fix.** 52.52% vs 55.56% is worth
   instrumenting and re-checking on 2027 data. Shipping "favour home when unsure" now
   would be fitting a single season's home-field environment.
5. **Real headroom needs new information, not new models.** 38 of 44 features are noise
   and 45× more history hurts. Bullpen quality, pitcher recent form, and above all
   **market odds** — as both a feature and the benchmark. Vegas runs ~58–60%; until the
   stack is measured against that line, "better" has no scale.

---

# Addendum — hunting for a 60%+ category (2026)

Added 2026-09-08. Scripts `08_`–`12_`. Question asked: can any category clear 60%?

**Answer: yes, one — but only by changing the unit of prediction from the game to the
series, and it sits right at the 60% line.**

## The power constraint that governs everything

To show 60% is genuinely above the ~53% base rate (one-sided α=.05, power .80) needs
**n ≥ 310 games**. A 60% result on n=100 has CI [50.4, 69.6] — it cannot even rule out the
base rate. Every claim below is reported with a Wilson CI and a powered/underpowered flag.

## Game-level: only one slice reaches 60%, and it is underpowered

67 rules searched (confidence thresholds, model consensus, feature-magnitude slices,
context splits), discovered on the first 60% of games and tested on the rest. Full-season
re-scoring of the survivors:

| Rule | n | Acc | 95% CI | Verdict |
|---|---|---|---|---|
| **v10 conf ≥ 0.64** | 203 | **60.10%** | [53.2, 66.6] | 60%+ but underpowered |
| v10 conf ≥ 0.62 | 326 | 57.98% | [52.6, 63.2] | not distinguishable |
| v10 conf ≥ 0.60 | 464 | 56.68% | [52.1, 61.1] | not distinguishable |
| \|sp_quality_diff\| ≥ q70 | 530 | 56.42% | [52.2, 60.6] | not distinguishable |
| v10 conf ≥ 0.60 & home-fav | 315 | 57.14% | [51.6, 62.5] | not distinguishable |

The existing high tier is the *only* game-level category at 60%, and one season cannot
establish it as truly above 60%.

**Two traps this search had to survive:**

- **The window confound.** In the raw discovery/validation search, several rules "cleared
  60% on held-out games" — but the late-season window is simply easier (V10's own baseline
  jumps there). Re-scored as *lift over the same-window baseline*, most of those winners
  had been at or below baseline during discovery. A slice that only works in the second
  window is a coin that landed heads twice.
- **Multiplicity.** 67 rules at α=.05 yields ~3.4 false hits by chance before any real
  effect exists.

## A clean negative: no learnable "when is V10 right" signal

A selective-prediction gate (logistic and GBM) was trained on the discovery window to
predict whether V10 would be *correct*, using the features plus V10's own confidence and
its agreement with the challenger model. Held-out **gate AUC = 0.5032 (logistic) and
0.5121 (GBM)** — no ability whatsoever. Beyond the confidence number V10 already emits,
there is no exploitable structure in this feature set telling you which games it will get
right. Selectivity is the only lever, and it is already exposed.

## Series instead of games: the one real 60%

Reconstructing series as consecutive-date blocks of the same (home, away) pair gives 511
decided series (28 even splits dropped as undecided) over 1,579 games. **Features are the
game-1 pregame row only** — see the leak note below.

| Predictor | n | Acc | 95% CI | vs base |
|---|---|---|---|---|
| always home takes series | 511 | 54.40% | [50.1, 58.7] | — |
| Elo differential sign | 511 | 55.97% | [51.6, 60.2] | +1.57 |
| **V10 game-1 prob sign** | **511** | **57.53%** | [53.2, 61.7] | **+3.13** |
| Elo+V10 agree | 355 | 59.72% | [54.5, 64.7] | +5.32 |
| **Elo+V10+pythag all agree** | **286** | **60.14%** | [54.4, 65.6] | **+5.74** |
| Elo+V10 agree & \|elo\| ≥ q50 | 195 | 61.54% | [54.5, 68.1] | +7.14 |
| Elo sign, \|elo\| ≥ q75 | 128 | 61.72% | [53.1, 69.7] | +7.32 |

The series unit is worth **+3.8pp over the same model's game-level accuracy** (57.53% vs
53.72%) at full power, and consensus filtering pushes it to ~60%. It is also **stable
across the season** — Elo sign scores 56.08% in H1 and 55.86% in H2; with |elo| ≥ q50,
57.39% and 58.16%. That split-half stability is the strongest evidence in this whole
document, and it is what the game-level rules lacked.

### Leak warning, recorded because it was nearly missed

A first pass averaged `elo_differential` *across* the series and produced 62.43% for Elo
sign, 71.22% at |elo| ≥ q60. **Those numbers are artifacts.** Elo is updated after every
game, so games 2–3 of a series carry game-1's result — the "prediction" already knew part
of the outcome. Enforcing game-1-only features dropped Elo sign from **62.43% → 55.97%**.
Any future series or multi-game work must take features from the first game's pregame row
only.

## What this means for a 60% target

- **Game level: 60% is not reachable on this feature set.** The signal is ~0.005 nats;
  the only 60% slice is the existing conf ≥ 0.64 tier at n=203, and the gate experiment
  shows there is no further selectivity to extract.
- **Series level: 60% is reachable now**, via consensus (Elo+V10+pythag agreeing) at
  60.14% on 286 series, with a genuinely stable underlying edge.
- **If a 60%+ game-level product is the goal, it needs new information, not new models**
  — market odds first (Vegas ~58–60% is the benchmark and the honest scale), then bullpen
  quality and pitcher recent form.

---

# Addendum 2 — autonomous search, and new information from pitch-level data

Added 2026-09-08. Scripts `14_`–`18_`. Question asked: can autonomous local search find
a better model?

**It was run twice, 450 trials total, and both runs independently rediscovered the same
3-feature logistic.** The search space is now exhausted, which is a more useful result
than the hand-picked model merely happening to work.

## What was searched

Two gaps in the earlier work were closed first:

1. **The full feature table.** Everything before this used the 44 feature columns stored
   in `game_predictions`. `game_v10_features` carries **166 columns** — lineup matchups
   vs handedness, rolling 3/7/10/30-game windows, era proxies, luck factors, FanGraphs-style
   team rates. Extracted with the same pregame guard (`computed_at < game_time_utc`):
   1,743 games × 161 usable numeric features, no column above 5% null.
2. **New information from pitch level.** 772,573 pitches across 2,612 games in
   `statcast_pitches`, used to build V10's own untested "Tier 1 / Tier 2" features —
   starter recent form and bullpen fatigue (script `17_`). 95–99% coverage.

### Protocol

`FINAL HOLDOUT` = the last 400 games (2026-08-08 → 09-07), untouched until one single
evaluation. `SEARCH SET` = everything before it. Optuna/TPE sampled model family ×
hyperparameters × feature-selection strategy, each trial scored by expanding-window
walk-forward log-loss inside the search set. **Feature selection ran inside each training
fold**, never globally.

## Search result: 7 families, 161 features, 300 trials → a 3-feature logistic

| Family | Best WF log-loss | Trials |
|---|---|---|
| **logit_l1** | **0.68871** | 219 |
| logit | 0.68886 | 20 |
| mlp | 0.68951 | 11 |
| xgb | 0.69009 | 14 |
| rf | 0.69044 | 14 |
| extratrees | 0.69086 | 10 |
| lgb | 0.69217 | 12 |

| Feature strategy | Best WF log-loss |
|---|---|
| **the 3 hand-picked features** | **0.68871** |
| top-k by in-fold AUC | 0.68985 |
| all 161 | 0.69315 |

TPE spent **265 of 300 trials** on the 3-feature set because it kept winning. Winner:
`logit_l1, manual3, C=0.557`.

### On the untouched holdout

| Model | n | Acc | 95% CI | AUC | Brier | LogLoss |
|---|---|---|---|---|---|---|
| Search winner | 400 | 56.75% | [51.9, 61.5] | **0.5816** | **0.2445** | **0.6821** |
| 3-feature logistic (hand-picked, C=0.1) | 400 | 57.00% | [52.1, 61.8] | 0.5816 | 0.2444 | 0.6820 |
| V10 (incumbent) | 400 | **57.50%** | [52.6, 62.3] | 0.5668 | 0.2463 | 0.6859 |
| always home | 400 | 54.25% | [49.4, 59.1] | 0.500 | 0.2487 | 0.6905 |

**300 trials bought nothing over the hand-picked trio** (0.6821 vs 0.6820), and the
incumbent still has the best raw accuracy. Search winner vs V10: log-loss +0.00375,
CI [−0.01112, +0.01828], P(better) = 0.687 — not significant.

**Overfit gap: −0.00661** (holdout *better* than the search score). The search did not
overfit, because it converged on a three-parameter model with almost no capacity to. That
is the search telling you the signal is small, not the harness failing.

## Pitch-level features: one real find, one dead roadmap item

Single-feature AUC of the 13 new features:

| Feature | AUC | Clears noise floor |
|---|---|---|
| **`sp_form_k_diff`** (starter K-rate, last 3 starts) | **0.5457** | **yes** |
| `sp_form_velo_diff` | 0.5420 | yes |
| `sp_form_xwoba_diff` | 0.5401 | yes |
| `home_sp_form_xwoba` | 0.4606 | yes |
| `bp_xwoba_7d_diff` (bullpen quality) | 0.5133 | no |
| **`bp_pitches_3d_diff`** (bullpen fatigue) | **0.5021** | **no** |

**Starter recent form is the joint-best feature in the entire stack** — `sp_form_k_diff`
at 0.5457 exactly ties `pythag_differential`. That is a genuine find, and it is new
information: correlation with the closest existing feature is only r = +0.328
(`sp_k_pct_diff`), and `sp_form_xwoba_diff` tops out at r = +0.216 against anything
already present. These are *not* redundant. (`sp_form_velo_diff` is the exception —
r = +0.762 with `sp_fbv_diff`, so it is already covered.)

**Bullpen fatigue is worthless.** `bp_pitches_3d_diff` scores AUC 0.5021 — indistinguishable
from a coin. This kills the **top item on V10's own "Tier 1 — highest impact" roadmap**.
Worth knowing before anyone builds it.

### And yet adding them does not help

| Feature set | WF log-loss | vs 3 features |
|---|---|---|
| **3 features** | **0.68938** | — |
| 3 + SP recent form | 0.68990 | +0.00052 |
| 3 + bullpen | 0.69322 | +0.00384 |
| 3 + all 13 new | 0.69495 | +0.00557 |
| the 13 new features alone | 0.69801 | +0.00864 |

A second 150-trial search with the new features available again picked
`logit_l1, manual3` (0.68912), ahead of `manual3+new` (0.69315).

**This is the real ceiling, and it is not a feature-quality problem.** `sp_form_k_diff`
has as much standalone signal as the best feature in the model and is largely independent
of it — and adding it is still a wash. With ~1,300 training games, **each additional
parameter costs more in estimation variance than the signal it contributes.** The binding
constraint is the number of games in a season, not what can be measured about them. And
the obvious escape — training on more history — was already measured as *worse*
(Addendum: 26.9k games loses to ~1.2k).

## Revised recommendation

Nothing here changes the earlier conclusion, and two searches now support it from an
independent direction:

- **Do not swap the model, and do not add features.** The search had 161 features, 7
  model families and 450 trials and declined all of it.
- **Delete bullpen fatigue from the roadmap.** Measured at AUC 0.5021.
- **`sp_form_k_diff` is the one feature worth keeping on the list** — not because it
  improves this model, but because it is real, independent signal that would pay off *if*
  the sample-size constraint were ever lifted.
- **Market odds remain the only lever that changes the picture** — they are a different
  information source, not another transformation of the same box score.

---

# Addendum 3 — is this the peak? Four different mechanisms, and a correction

Added 2026-09-08. Scripts `19_`–`22_`. Question asked: with all this data, is ~54% the peak?

**Short answer: no, this is not the peak of MLB prediction — but it is the peak of this
substrate, and getting past it needs different information or a different granularity,
not another model.**

## A correction to Addendum 2

Addendum 2 concluded "the binding constraint is how many games a season has." **That was
overstated in two ways**, and the tests below show why:

1. It was a claim about a **binary** label. A season's scores carry far more information
   than its win/loss bits, so switching target was an untested escape route.
2. The supporting "more history hurts" result used **2 raw features and one model** — far
   too weak to support a ceiling claim.

On (2), the corrected result: with **54 features**, history genuinely *helps*. Historical
GBM (11 z-scored, recency-weighted seasons) scores **0.68592** log-loss on the holdout
versus **0.69319** for the same 54 features trained on 2026 alone. The real constraint is
**joint** — features × rows — not games per season. One season supports ~3 features; 11
seasons support ~54. Neither beats the 3-feature in-season model.

## The four mechanisms tested

Each is a different *mechanism*, not another feature set. All use the same nested protocol
and the same untouched holdout (2026-08-08 →), and all are judged by one gate: **better in
both the search window and the holdout.**

| Mechanism | Script | Result |
|---|---|---|
| Margin regression → P(margin ≥ 1) via fitted residual distribution | `19_` | best holdout AUC (0.5889) but worst log-loss — a scaling defect |
| Two Poisson arms → P(home runs > away runs) via Skellam | `19_` | worse than binary on both windows |
| Margin regression → **learned** probability map | `20_` | holdout 0.68017 (best) but search-window 0.69010 (worse) — **inconsistent** |
| Multi-season transfer: within-season z-scoring + recency weights + GBM | `21_` | 0.68777 search / 0.68592 holdout — beats always-home in both, loses to the 3-feature model on holdout |
| Ensembles of in-season + historical (average / tuned weight / logistic stack) | `22_` | **all inconsistent** |

### The consistency gate, log-loss gain over the in-season 3-feature model

| Mechanism | Search window | Holdout | Verdict |
|---|---|---|---|
| historical GBM (11 seasons) | +0.00237 | −0.00408 | inconsistent |
| average of the two | +0.00253 | −0.00127 | inconsistent |
| weight tuned on train window | +0.00076 | −0.00103 | inconsistent |
| logistic stack of the two | +0.00098 | −0.00169 | inconsistent |

**Nothing passes.** Every candidate that looks better in one window is worse in the other.
Worth noting what would have shipped without this gate: on the holdout alone, the stack and
the average both reach **57.38% accuracy** against the 3-feature model's 56.90%, and the
calibrated-margin model posts the best log-loss of anything tested. Both evaporate on the
larger window.

## So what would actually break 58%?

Two things, neither of which is a model:

1. **Market odds.** Not another transformation of the same box score — a different
   information source. Odds price injury news, weather, lineup scratches and sharp money,
   none of which exists in any aggregate computed here. They are simultaneously the
   benchmark (~58–60%) and the single most valuable feature available. This is a data
   acquisition problem, not a modelling one.
2. **Bottom-up plate-appearance simulation.** V10's own untested "Tier 3". The `lineups`
   table, player splits and 772k pitch-level rows are all present; the mechanism is to
   project each batter's outcome against the actual starter and simulate the game rather
   than regress a team-level aggregate. This is how Vegas-grade models are built, it is
   the only remaining path with a plausible route past 58%, and it is a real project —
   weeks, not an afternoon.

And the thing that already clears 60% today remains the **series unit** (Addendum 1):
60.14% on 286 series via consensus, with split-half stability.

## Standing recommendation, unchanged

Ship the pregame filter and display-only recalibration with quantile tiers. Do not swap
the game-level model — after 450 search trials and five distinct mechanisms, nothing has
beaten a 3-feature logistic in both windows. Build the series product. Get odds.
