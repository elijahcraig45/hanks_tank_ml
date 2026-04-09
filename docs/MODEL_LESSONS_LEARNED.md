# Model Development Lessons Learned

Accumulated lessons from production issues. Read this before training a new model version.

---

## 1. Never use team IDs as categorical features in tree models

**What happened:** V8 used `home_team_id` and `away_team_id` as CatBoost categorical features (54% of
ensemble weight). The model trained fine on 2015–2024 data and validated well on 2025. But on 2026
holdout it achieved **AUC = 0.455** — worse than random — because the team embeddings encoded
2015–2025-specific home/away tendencies that no longer held in 2026.

With perfectly neutral quantitative features, the model predicted only **41.6% home win probability**
(should be ~53%). A +200 Elo differential moved it only to 45%.

**The fix (v8_nocat):** Removed `home_team_id`/`away_team_id` from all sub-models. Retrained on
2015–2024 + 2026 YTD data. Result: **57.6% accuracy, AUC 0.624** on 2026 holdout, avg home pred
of 0.538 (well-calibrated).

**Rule for future models:**
- Do not use team IDs, franchise IDs, or any static team identifier as a model feature.
- If you want team-level signals, encode them as rolling statistics (win rate, run diff, Elo) that
  update continuously. These are already in the feature set as `home_elo`, `home_pythagorean_win_pct`,
  `home_avg_run_diff`, etc.
- When validating a new model, always check `avg_home_pred` on a holdout set. It should be 0.52–0.56.
  If it is significantly below 0.50, the model has an away-team bias and team ID leakage is the
  most likely culprit.

---

## 2. Always evaluate on a same-season holdout before deploying

**What happened:** V8 was evaluated on 2025 validation data (57.5% acc) but not on any 2026 games
before going live. The 2026 holdout AUC of 0.455 would have caught the team embedding problem
immediately.

**Rule for future models:**
- Before deploying any model mid-season, run evaluation on completed games from the current season
  (even if only 20–30 games are available).
- Specifically check: accuracy, AUC, and `avg_home_pred`. All three should look reasonable.
- `calibrate_v8_2026.py` and `retrain_v8_no_team_ids.py` both include 2026 holdout evaluation — use
  them as templates for any future calibration or retraining work.

---

## 3. Calibration cannot fix a fundamentally broken signal

**What happened:** After discovering V8's AUC = 0.455, we tried Platt scaling and Isotonic
calibration on the 2026 games. The AUC remained 0.455 — calibration maps probabilities to a
different range but cannot invert the direction of a broken signal.

**Rule for future models:**
- Calibration is a post-processing step. It improves probability reliability (Brier score,
  calibration curve) but does not fix a model whose predictions are inversely correlated with
  outcomes.
- If AUC < 0.50 on a holdout set, the model has a fundamental problem. Fix the model, not the
  calibration.
- Calibration is useful when AUC > 0.52 but probabilities are skewed (e.g. model always predicts
  55–70%, rarely predicts near 50%).

---

## 4. Cloud Run filesystem is read-only — never assume you can cache to disk

**What happened:** `predict_today_games.py` downloaded the V8 model from GCS successfully, then
tried to write a local cache file. On Cloud Run the `/models` directory is read-only, so the write
raised `[Errno 13] Permission denied`. This was caught inside the same `try/except` block as the
download, causing the entire V8 load attempt to fail silently and fall through to V5.

**The fix:** Split into two separate `try/except` blocks — one for the download (must succeed), one
for the local cache write (best-effort, catch and log only).

```python
# WRONG — one block means a cache write failure kills the download too
try:
    data = pickle.loads(blob.download_as_bytes())
    with open(local_path, "wb") as f:   # fails on Cloud Run
        pickle.dump(data, f)
except Exception as e:
    logger.warning("model unavailable: %s", e)  # silently falls through!

# RIGHT — separate concerns
try:
    data = pickle.loads(blob.download_as_bytes())   # download must succeed
except Exception as e:
    logger.warning("model unavailable: %s", e)
    continue  # try next version

try:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    with open(local_path, "wb") as f:
        pickle.dump(data, f)
except Exception as cache_err:
    logger.debug("Could not cache locally (non-fatal): %s", cache_err)
```

**Rule for future models:** When deploying to Cloud Run/Cloud Functions, any local file write must
be wrapped in its own `try/except` and treated as optional. Downloads from GCS must be in a
separate block.

---

## 5. One-hot encoded columns may be stored as `object` dtype in Parquet

**What happened:** The training parquet (`train_v8_2015_2024.parquet`) stores one-hot encoded
month and day-of-week columns (`month_3`, `month_4`, ..., `dow_1`, ..., `dow_7`) as `object` dtype
rather than `int` or `bool`. CatBoost and LightGBM reject `object` dtype columns with a hard error.

The retrain script had a cast for `bool` columns but not `object`:

```python
# Misses object-typed boolean columns
for col in X.select_dtypes(include=["bool"]).columns:
    X[col] = X[col].astype(int)
```

**The fix:** Include `"object"` in the dtype selector and use `pd.to_numeric(errors="coerce")` as a
safe conversion path:

```python
for col in X.select_dtypes(include=["bool"]).columns:
    X[col] = X[col].astype(int)
for col in X.select_dtypes(include=["object", "category"]).columns:
    X[col] = pd.to_numeric(X[col], errors="coerce")
X = X.fillna(X.median(numeric_only=True))
```

**Rule for future models:** After loading any parquet training file, always inspect column dtypes
with `df.dtypes.value_counts()` before passing to a model. Add a dtype assertion or coercion step
in every `prepare_xy`-style function.

---

## 6. Cloud Task pipeline mode must match the model version

**What happened:** `lineup-scheduler.service.ts` was hardcoded to send `mode: 'pregame'` for every
pre-game Cloud Task. This ran the pre-V8 pipeline (no V8 feature build step), so V8 features were
either stale or missing for some games, even after V8 was deployed.

**The fix:** Changed to `mode: 'pregame_v8'`, which runs: lineups → matchup → V7 features →
**V8 features** → predict.

**Rule:** When introducing a new model version that depends on new features, update the Cloud Task
payload `mode` field in `lineup-scheduler.service.ts` at the same time as the model deployment.
These two changes must be deployed together.

---

## 7. Always run a same-season evaluation before making a model the primary

A checklist for promoting a model to production:

| Check | Pass condition |
|---|---|
| 2025 val accuracy | > 55% |
| 2025 val AUC | > 0.55 |
| 2026 holdout accuracy | > 52% |
| 2026 holdout AUC | > 0.52 |
| avg home_win_prob (2026 holdout) | 0.50 – 0.58 |
| No team ID / franchise ID features | Confirmed |
| Cloud Task mode updated | Confirmed |
| GCS model path uploaded | Confirmed |
| predict_today_games resolution chain updated | Confirmed |
