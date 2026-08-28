"""NFL v1 training + walk-forward evaluation.

The deliverable of this script is the comparison table, not the model. Raw accuracy is
close to meaningless in the NFL: always-home gets 56.4% and Elo alone gets ~62-64%
for free. What matters is whether the feature set beats Elo-only on log loss across
many seasons, and how close it gets to the closing line.

Two variants, per the plan:
  pure   — no market columns. THE HEADLINE.
  market — includes spread/total. Scores better, is ~90% Vegas, reported separately.

Usage:
  python train_nfl_models.py --walk-forward
  python train_nfl_models.py --variant pure --variant market
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import HOLDOUT_SEASON  # noqa: E402
from data import completed_games  # noqa: E402
from features import build_features, elo_expected, feature_columns  # noqa: E402
from models import build_lr, build_xgb, evaluate  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------
def baseline_metrics(df: pd.DataFrame) -> dict:
    """The three bars every model is judged against."""
    y = df["home_won"].values

    always_home = float(np.mean(y))

    elo_pred = (df["elo_home_win_prob"] > 0.5).astype(int).values
    elo_acc = float(np.mean(elo_pred == y))
    elo_ll = log_loss(y, df["elo_home_win_prob"].clip(1e-6, 1 - 1e-6).values)
    elo_brier = brier_score_loss(y, df["elo_home_win_prob"].values)

    lined = df[df["spread_line"].notna() & (df["spread_line"] != 0)]
    if len(lined):
        fav = (lined["spread_line"] > 0).astype(int).values
        vegas_acc = float(np.mean(fav == lined["home_won"].values))
    else:
        vegas_acc = float("nan")

    return {
        "always_home_acc": always_home,
        "elo_acc": elo_acc,
        "elo_log_loss": elo_ll,
        "elo_brier": elo_brier,
        "vegas_acc": vegas_acc,
        "n": len(df),
    }


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# Walk-forward CV — the only trustworthy estimator here
# --------------------------------------------------------------------------
def walk_forward(df: pd.DataFrame, cols: list[str], model_fn, folds: list[int]) -> pd.DataFrame:
    out = []
    for season in folds:
        train = df[df["season"] < season]
        test = df[df["season"] == season]
        if len(train) < 500 or test.empty:
            continue

        X_tr = train[cols].fillna(0.0).values
        X_te = test[cols].fillna(0.0).values
        y_tr = train["home_won"].values
        y_te = test["home_won"].values

        model = model_fn()
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)[:, 1]

        m = evaluate(y_te, proba)
        b = baseline_metrics(test)
        m.update(season=season, n=len(test),
                 always_home=b["always_home_acc"], elo_acc=b["elo_acc"],
                 elo_log_loss=b["elo_log_loss"], vegas_acc=b["vegas_acc"])
        out.append(m)
    return pd.DataFrame(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", action="append", choices=["pure", "market"], default=None)
    ap.add_argument("--model", choices=["lr", "xgb", "both"], default="both")
    ap.add_argument("--folds", type=int, default=8)
    ap.add_argument("--walk-forward", action="store_true")
    ap.add_argument("--no-epa", action="store_true",
                    help="ablation: build without EPA to measure what it adds")
    args = ap.parse_args()
    variants = args.variant or ["pure", "market"]

    games = completed_games()
    logger.info("loaded %d decided games (%d-%d)",
                len(games), games["season"].min(), games["season"].max())

    epa_df = None
    if not args.no_epa:
        from epa import EPA_CACHE
        import polars as pl

        if EPA_CACHE.exists():
            epa_pl = pl.read_parquet(EPA_CACHE)
            epa_df = epa_pl.to_pandas()
            epa_seasons = sorted(epa_pl["season"].unique().to_list())
            logger.info("EPA available for %d seasons (%d-%d)",
                        len(epa_seasons), min(epa_seasons), max(epa_seasons))
            # EPA only exists for some seasons; restrict modelling to those so the
            # walk-forward folds aren't silently comparing different feature sets.
            games = games[games["season"].isin(epa_seasons)]
        else:
            logger.warning("no EPA cache found — run epa.build_team_week_epa() first")

    feats = build_features(games, epa=epa_df)

    # 2025 is the touch-once holdout; exclude it from all walk-forward folds.
    modelling = feats[feats["season"] < HOLDOUT_SEASON]
    seasons = sorted(modelling["season"].unique())
    folds = seasons[-args.folds:]

    print("\n" + "=" * 78)
    print("OVERALL BASELINES  (all seasons, 1999-%d)" % modelling["season"].max())
    print("=" * 78)
    b = baseline_metrics(modelling)
    print(f"  games                    {b['n']}")
    print(f"  always-home accuracy     {b['always_home_acc']:.2%}")
    print(f"  Elo-only accuracy        {b['elo_acc']:.2%}   <-- the real floor")
    print(f"  Elo-only log loss        {b['elo_log_loss']:.4f}")
    print(f"  Vegas-favorite accuracy  {b['vegas_acc']:.2%}   <-- the bar")

    model_fns = {"lr": build_lr, "xgb": build_xgb}
    chosen = ["lr", "xgb"] if args.model == "both" else [args.model]

    for variant in variants:
        cols = feature_columns(feats, include_market=(variant == "market"))
        print("\n" + "=" * 78)
        print(f"VARIANT: {variant.upper()}  ({len(cols)} features)"
              + ("   <-- HEADLINE" if variant == "pure" else "   (~90% Vegas)"))
        print("=" * 78)

        for name in chosen:
            res = walk_forward(modelling, cols, model_fns[name], folds)
            if res.empty:
                continue
            print(f"\n  {name.upper()} — walk-forward, {len(res)} seasons "
                  f"({res['season'].min()}-{res['season'].max()})")
            print(f"  {'season':>6} {'n':>4} {'acc':>7} {'logloss':>8} {'brier':>7} "
                  f"{'vs home':>8} {'vs elo':>7} {'vegas':>7}")
            for r in res.itertuples(index=False):
                print(f"  {r.season:>6} {r.n:>4} {r.acc:>6.1%} {r.log_loss:>8.4f} "
                      f"{r.brier:>7.4f} {r.acc - r.always_home:>+7.1%} "
                      f"{r.acc - r.elo_acc:>+6.1%} {r.vegas_acc:>6.1%}")

            n = res["n"].sum()
            wacc = float(np.average(res["acc"], weights=res["n"]))
            wll = float(np.average(res["log_loss"], weights=res["n"]))
            well = float(np.average(res["elo_log_loss"], weights=res["n"]))
            se = float(np.sqrt(wacc * (1 - wacc) / n))
            print(f"  {'MEAN':>6} {n:>4} {wacc:>6.1%} {wll:>8.4f} "
                  f"{res['brier'].mean():>7.4f} "
                  f"{wacc - np.average(res['always_home'], weights=res['n']):>+7.1%} "
                  f"{wacc - np.average(res['elo_acc'], weights=res['n']):>+6.1%} "
                  f"{np.average(res['vegas_acc'], weights=res['n']):>6.1%}")
            print(f"         95% CI on pooled accuracy: +/-{1.96 * se:.1%}")
            verdict = "BEATS Elo-only" if wll < well else "DOES NOT beat Elo-only"
            print(f"         log loss vs Elo-only: {wll:.4f} vs {well:.4f}  -> {verdict}")

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
