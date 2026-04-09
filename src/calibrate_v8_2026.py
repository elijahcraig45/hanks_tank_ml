#!/usr/bin/env python3
"""
V8 Quick Calibration — 2026 Season Isotonic + Platt Re-calibration

The V8 ensemble (CatBoost + team IDs) has a systematic away-team bias:
  - Neutral features → ~41.6% home win probability (should be ~53%)
  - Elo differentials barely move the needle (+200 Elo → only 45%)
  - Root cause: CatBoost team embeddings (54% of ensemble weight) dominate
    quantitative signals, encoding team-specific 2015-2025 patterns

This script:
  1. Loads the existing V8 ensemble (game_outcome_2026_v8_final.pkl)
  2. Gathers all completed 2026 games that also have V8 features in BQ
  3. Gets raw ensemble probabilities for those games
  4. Fits two calibrators on a training split:
       a) Platt scaling  (logistic regression on raw proba → true prob)
       b) Isotonic regression (non-parametric, monotone)
  5. Evaluates both calibrators vs uncalibrated on a 2026 holdout split
  6. Saves the best calibrator as game_outcome_2026_v8_calibrated.pkl
     (same bundle format + calibrator key, used transparently by predictor)

Usage:
    python calibrate_v8_2026.py                    # calibrate and save
    python calibrate_v8_2026.py --dry-run          # evaluate only, no save
    python calibrate_v8_2026.py --min-games 50     # require at least N games
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from google.cloud import bigquery
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / "models"
SOURCE_PKL  = MODEL_DIR / "game_outcome_2026_v8_final.pkl"
OUTPUT_PKL  = MODEL_DIR / "game_outcome_2026_v8_calibrated.pkl"

PROJECT = "hankstank"
DATASET = "mlb_2026_season"

sys.path.insert(0, str(Path(__file__).parent))
from predict_today_games import V8EnsemblePredictor  # noqa: E402


# ---------------------------------------------------------------------------
# Platt/isotonic wrappers compatible with the ensemble output
# ---------------------------------------------------------------------------
class PlattCalibrator:
    """Logistic regression on raw ensemble probability → calibrated probability."""
    def __init__(self):
        self.lr = LogisticRegression(C=1.0, solver="lbfgs")

    def fit(self, raw_proba: np.ndarray, y: np.ndarray):
        X = raw_proba.reshape(-1, 1)
        self.lr.fit(X, y)
        return self

    def predict(self, raw_proba: np.ndarray) -> np.ndarray:
        return self.lr.predict_proba(raw_proba.reshape(-1, 1))[:, 1]


class IsotonicCalibrator:
    """Isotonic regression calibrator — non-parametric, monotone mapping."""
    def __init__(self):
        self.ir = IsotonicRegression(out_of_bounds="clip")

    def fit(self, raw_proba: np.ndarray, y: np.ndarray):
        self.ir.fit(raw_proba, y.astype(float))
        return self

    def predict(self, raw_proba: np.ndarray) -> np.ndarray:
        return self.ir.predict(raw_proba)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_2026_games_with_v8_features(bq: bigquery.Client) -> pd.DataFrame:
    """
    Returns completed 2026 games joined with V8 features and outcomes.
    Excludes today (no outcome yet). Uses latest computed V8 features per game.
    """
    sql = """
    WITH v8_latest AS (
        SELECT *
        FROM `hankstank.mlb_2026_season.game_v8_features`
        QUALIFY ROW_NUMBER() OVER (PARTITION BY game_pk ORDER BY computed_at DESC) = 1
    )
    SELECT
        g.game_pk,
        g.game_date,
        g.home_team_id,
        g.away_team_id,
        CAST(g.home_score > g.away_score AS INT64) AS home_won,
        v8.* EXCEPT (game_pk, game_date, home_team_id, away_team_id, computed_at, data_completeness)
    FROM `hankstank.mlb_2026_season.games` g
    JOIN v8_latest v8 ON g.game_pk = v8.game_pk
    WHERE g.status = 'Final'
      AND g.home_score IS NOT NULL
      AND g.game_date < CURRENT_DATE()
    ORDER BY g.game_date
    """
    logger.info("Loading 2026 completed games with V8 features from BigQuery...")
    df = bq.query(sql).to_dataframe()
    logger.info("  Loaded %d games (%s – %s)", len(df),
                df["game_date"].min(), df["game_date"].max())
    return df


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def evaluate(proba: np.ndarray, y: np.ndarray, label: str) -> dict:
    pred = (proba > 0.5).astype(int)
    acc  = accuracy_score(y, pred)
    auc  = roc_auc_score(y, proba)
    brier = brier_score_loss(y, proba)
    ll   = log_loss(y, proba)
    home_win_rate = proba.mean()

    # Confidence-tier accuracy
    conf_results = {}
    for offset in [0.03, 0.05, 0.08, 0.10, 0.12]:
        mask = np.abs(proba - 0.5) > offset
        n = mask.sum()
        if n >= 10:
            conf_acc = accuracy_score(y[mask], pred[mask])
            conf_results[f">={50 + offset*100:.0f}%"] = {
                "acc": round(conf_acc, 4), "n": int(n),
                "pct": round(n / len(y), 3)
            }

    logger.info(
        "  %-35s  acc=%.3f  AUC=%.3f  brier=%.4f  avg_home_pred=%.3f",
        label, acc, auc, brier, home_win_rate,
    )
    for thresh, v in conf_results.items():
        logger.info("    conf %s: %.3f (%d games, %.1f%%)",
                    thresh, v["acc"], v["n"], v["pct"] * 100)

    return {"label": label, "accuracy": acc, "auc": auc, "brier": brier,
            "log_loss": ll, "avg_home_pred": home_win_rate, "conf": conf_results}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Evaluate only — do not save calibrated model")
    parser.add_argument("--min-games", type=int, default=30,
                        help="Minimum completed games required (default: 30)")
    parser.add_argument("--train-frac", type=float, default=0.6,
                        help="Fraction of 2026 games used to fit calibrators (default: 0.6, time-ordered split)")
    parser.add_argument("--source-pkl", type=str, default=None,
                        help="Path to source model pkl (default: game_outcome_2026_v8_final.pkl)")
    parser.add_argument("--output-pkl", type=str, default=None,
                        help="Path to save calibrated model pkl (default: auto-derived from source)")
    args = parser.parse_args()

    src_pkl = Path(args.source_pkl) if args.source_pkl else SOURCE_PKL
    if args.output_pkl:
        out_pkl = Path(args.output_pkl)
    else:
        out_pkl = src_pkl.parent / (src_pkl.stem + "_calibrated.pkl")
        if src_pkl == SOURCE_PKL:
            out_pkl = OUTPUT_PKL

    # --- Load V8 ensemble ---
    logger.info("Loading V8 ensemble from %s", src_pkl)
    with open(src_pkl, "rb") as f:
        bundle = pickle.load(f)
    ensemble = V8EnsemblePredictor(bundle)
    logger.info("  Version: %s | %d features", bundle.get("version"), len(ensemble.all_features))

    # --- Load 2026 data ---
    import os
    os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)
    bq = bigquery.Client(project=PROJECT)
    df = load_2026_games_with_v8_features(bq)

    if len(df) < args.min_games:
        logger.error("Only %d completed games available (need %d). "
                     "Re-run once more games are played.", len(df), args.min_games)
        sys.exit(1)

    # --- Get raw V8 probabilities for all 2026 games ---
    feature_rows = []
    for _, row in df.iterrows():
        feat = {f: bundle["fill_values"].get(f, 0.0) for f in ensemble.all_features}
        for col in ensemble.all_features:
            if col in row.index and pd.notna(row[col]):
                feat[col] = row[col]
        feature_rows.append(feat)

    X_all = pd.DataFrame(feature_rows)
    y_all = df["home_won"].values.astype(int)

    logger.info("Getting raw V8 probabilities for %d games...", len(X_all))
    raw_proba = ensemble.predict_proba(X_all)[:, 1]

    # --- Time-ordered train/holdout split (no leakage) ---
    n_train = int(len(df) * args.train_frac)
    n_holdout = len(df) - n_train

    raw_train = raw_proba[:n_train]
    y_train_c = y_all[:n_train]
    raw_hold  = raw_proba[n_train:]
    y_hold    = y_all[n_train:]

    train_dates = df["game_date"].iloc[:n_train]
    hold_dates  = df["game_date"].iloc[n_train:]
    logger.info(
        "Train split: %d games (%s – %s) | Holdout: %d games (%s – %s)",
        n_train, train_dates.min(), train_dates.max(),
        n_holdout, hold_dates.min(), hold_dates.max(),
    )

    # --- Fit calibrators on training split ---
    platt = PlattCalibrator().fit(raw_train, y_train_c)
    iso   = IsotonicCalibrator().fit(raw_train, y_train_c)

    # --- Evaluate all three on holdout ---
    logger.info("\n--- HOLDOUT RESULTS (2026 games not used for fitting) ---")
    r_raw   = evaluate(raw_hold, y_hold, "V8 uncalibrated")
    r_platt = evaluate(platt.predict(raw_hold), y_hold, "V8 + Platt scaling")
    r_iso   = evaluate(iso.predict(raw_hold), y_hold, "V8 + Isotonic regression")

    # Pick the best calibrator on the holdout set
    best_name = max(
        [("platt", r_platt), ("iso", r_iso)],
        key=lambda x: x[1]["accuracy"]
    )
    logger.info("\n  Best calibrator on holdout: %s (acc=%.3f)",
                best_name[0], best_name[1]["accuracy"])
    best_calibrator = platt if best_name[0] == "platt" else iso

    # Also show full-dataset calibration (all 2026 games) for the winner
    # (re-fit on all available data before saving)
    if best_name[0] == "platt":
        final_calibrator = PlattCalibrator().fit(raw_proba, y_all)
    else:
        final_calibrator = IsotonicCalibrator().fit(raw_proba, y_all)

    # Sanity check: avg predicted home win probability after calibration
    final_cal_all = final_calibrator.predict(raw_proba)
    logger.info("\nPost-calibration sanity check (all 2026 games):")
    logger.info("  Avg predicted home win prob: %.3f (should be ~0.53–0.55)", final_cal_all.mean())
    logger.info("  Actual home win rate:        %.3f", y_all.mean())

    if args.dry_run:
        logger.info("\n[DRY RUN] Calibrated model not saved.")
        return

    # --- Save calibrated bundle ---
    new_bundle = dict(bundle)
    new_bundle["calibrator"]      = final_calibrator
    new_bundle["calibrator_type"] = best_name[0]
    new_bundle["calibrator_games"] = len(df)
    new_bundle["calibrator_dates"] = {
        "first": str(df["game_date"].min()),
        "last":  str(df["game_date"].max()),
    }
    new_bundle["version"] = bundle.get("version", "v8_final") + "_calibrated"

    with open(out_pkl, "wb") as f:
        pickle.dump(new_bundle, f)

    logger.info("\nSaved calibrated model → %s", out_pkl)
    logger.info("  Version: %s", new_bundle["version"])
    logger.info("  Fitted on: %d 2026 games (%s – %s)",
                len(df), df["game_date"].min(), df["game_date"].max())

    # Summary table
    logger.info("\n" + "="*65)
    logger.info("  APPROACH COMPARISON (holdout: %d games)", n_holdout)
    logger.info("  %-35s  %s  %s  %s", "Model", "Acc", "AUC", "Brier")
    logger.info("  " + "-"*60)
    for r in [r_raw, r_platt, r_iso]:
        logger.info("  %-35s  %.3f  %.3f  %.4f",
                    r["label"], r["accuracy"], r["auc"], r["brier"])
    logger.info("="*65)


if __name__ == "__main__":
    main()
