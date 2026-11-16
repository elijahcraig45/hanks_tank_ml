"""Reproduce the NFL model comparison from docs/EXPERIMENT_LOG.md on a laptop.

    python demo/nfl_backtest.py                  # about 5 seconds
    python demo/nfl_backtest.py --weekly         # retrain XGBoost before every week instead of every season
    python demo/nfl_backtest.py --out preds.csv  # also write each game's predictions

It reads one file, demo/data/nfl_games_1999_2025.csv.gz (nflverse schedules, CC BY 4.0), and runs it through the
production code in src/nfl: the same feature builder, XGBoost settings and margin ridge. No network, no cloud
account, and nothing is written unless you pass --out.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SNAPSHOT = REPO / "demo" / "data" / "nfl_games_1999_2025.csv.gz"
sys.path.insert(0, str(REPO / "src" / "nfl"))

import margin_ridge as mr  # noqa: E402
from data import completed_games  # noqa: E402
from features import build_features, feature_columns  # noqa: E402
from models import build_xgb  # noqa: E402

TUNE_SEASONS = (2010, 2016)   # choose the ridge settings; never scored
SCORED = range(2017, 2026)    # 2017-2024 walk-forward, then 2025 as a holdout
VEGAS_SIGMA = 13.45           # points; turns the closing spread into a win probability
MODELS = {"elo": "Elo", "xgb": "XGBoost", "ridge": "Margin ridge", "vegas": "Vegas closing spread"}
PERIODS = {"2017-2024 walk-forward": lambda s: s < 2025, "2025 holdout": lambda s: s == 2025}


@dataclass
class Result:
    preds: pd.DataFrame        # one row per scored game, a home-win probability per model
    ridge: mr.RidgeConfig      # the settings tuning chose
    tune_log_loss: float
    n_games: int
    n_features: int


def ridge_frame(games: pd.DataFrame) -> pd.DataFrame:
    """What margin_ridge needs. predict_nfl.ridge_frame builds the same columns, but importing it loads BigQuery."""
    rg = games[["game_id", "season", "week", "home_team", "away_team"]].reset_index(drop=True)
    rg["margin"] = games["result"].to_numpy(float)
    rg["neutral"] = (games["location"].fillna("Home") != "Home").astype(int).to_numpy()
    return rg


def tune_ridge(rg: pd.DataFrame) -> tuple[mr.RidgeConfig, float]:
    won = (rg["margin"] > 0).astype(int).to_numpy()
    mask = rg.season.between(*TUNE_SEASONS).to_numpy()
    return mr.tune(rg, mask, won, mr.NFL_RIDGE.with_(covariates=()), alphas=(1, 3, 10, 30), taus=(6, 10, 16, 25))


def _blocks(F: pd.DataFrame, season: int, weekly: bool):
    """(train, test) pairs for one season. Training rows are always strictly earlier than the test rows."""
    if not weekly:
        yield F[F.season < season], F[F.season == season]
        return
    for week in sorted(F.loc[F.season == season, "week"].unique()):
        earlier = (F.season < season) | ((F.season == season) & (F.week < week))
        yield F[earlier], F[(F.season == season) & (F.week == week)]


def xgb_walk_forward(F: pd.DataFrame, weekly: bool) -> pd.Series:
    cols = feature_columns(F)
    out = []
    for season in SCORED:
        for train, test in _blocks(F, season, weekly):
            model = build_xgb()
            model.fit(train[cols].fillna(0).to_numpy(), train.home_won.to_numpy())
            out.append(pd.Series(model.predict_proba(test[cols].fillna(0).to_numpy())[:, 1], index=test.game_id))
    return pd.concat(out)


def run(weekly: bool = False, path: Path = SNAPSHOT) -> Result:
    games = completed_games(pd.read_csv(path))   # ties dropped, home_won added, as in production
    F = build_features(games, epa=None)          # production's feature set: scores and schedule only
    rg = ridge_frame(games)
    cfg, tune_ll = tune_ridge(rg)
    rg["ridge_margin"] = mr.walk_forward(rg, rg.season.isin(SCORED).to_numpy(), cfg)

    keep = ["game_id", "season", "week", "home_team", "away_team", "home_won", "elo_home_win_prob", "spread_line"]
    preds = F.loc[F.season.isin(SCORED), keep].rename(columns={"elo_home_win_prob": "elo"})
    preds["xgb"] = preds.game_id.map(xgb_walk_forward(F, weekly))
    preds = preds.merge(rg[["game_id", "ridge_margin"]], on="game_id")
    preds["ridge"] = mr.win_prob(preds.ridge_margin, cfg.sigma)
    preds["vegas"] = mr.win_prob(preds.spread_line, VEGAS_SIGMA)
    return Result(preds.reset_index(drop=True), cfg, tune_ll, len(games), len(feature_columns(F)))


def game_log_loss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def scores(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {"accuracy": float(np.mean((p > 0.5) == y)), "log_loss": float(game_log_loss(y, p).mean()),
            "brier": float(np.mean((p - y) ** 2))}


def paired_diff(y: np.ndarray, p: np.ndarray, ref: np.ndarray, n: int = 2000, seed: int = 0):
    """Mean per-game log-loss difference (p minus ref) with a 95% paired-bootstrap interval over games."""
    d = game_log_loss(y, p) - game_log_loss(y, ref)
    rng = np.random.default_rng(seed)
    boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return float(d.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def summary(preds: pd.DataFrame) -> dict[str, dict]:
    """Per period: each model's scores, and the ridge-minus-XGBoost log-loss difference."""
    out = {}
    for label, in_period in PERIODS.items():
        b = preds[in_period(preds.season)]
        y = b.home_won.to_numpy(float)
        out[label] = {"n": len(b), "models": {m: scores(y, b[m].to_numpy(float)) for m in MODELS},
                      "ridge_vs_xgb": paired_diff(y, b.ridge.to_numpy(float), b.xgb.to_numpy(float))}
    return out


def ships(cfg: mr.RidgeConfig) -> bool:
    shipped = mr.NFL_RIDGE
    return (cfg.alpha, cfg.tau, round(cfg.sigma, 2)) == (shipped.alpha, shipped.tau, shipped.sigma)


def print_report(res: Result, weekly: bool) -> None:
    cfg = res.ridge
    print(f"NFL model comparison: {res.n_games:,} decided games, 1999-2025 (nflverse schedules)\n")
    print(f"Margin ridge settings, chosen on 2010-2016 (never scored): alpha={cfg.alpha:g}, tau={cfg.tau:g}, "
          f"sigma={cfg.sigma:.2f}" + (", the settings production ships" if ships(cfg) else
                                      ". Production ships different settings"))
    print(f"XGBoost: {res.n_features} features, retrained before every {'week' if weekly else 'season'}")
    print("Every prediction uses only games played before it.")
    for label, s in summary(res.preds).items():
        print(f"\n{label}, {s['n']:,} games")
        print(f"  {'model':22s} {'accuracy':>8s} {'log loss':>9s} {'Brier':>7s}")
        for m, name in MODELS.items():
            r = s["models"][m]
            print(f"  {name:22s} {r['accuracy']:8.1%} {r['log_loss']:9.4f} {r['brier']:7.4f}")
        d, lo, hi = s["ridge_vs_xgb"]
        verdict = "" if hi < 0 or lo > 0 else ", not significant on its own"
        print(f"  Ridge minus XGBoost, log loss: {d:+.4f} (95% CI {lo:+.4f} to {hi:+.4f}{verdict})")
    print("\nLower is better for log loss and Brier. Always guessing 50% scores 0.6931 and 0.2500.")
    print("The intervals are a paired bootstrap over games (2,000 resamples). No model beats the betting market.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weekly", action="store_true", help="retrain XGBoost before every week, as the backfill does")
    ap.add_argument("--out", type=Path, help="write each scored game's predictions to this CSV")
    args = ap.parse_args()
    res = run(weekly=args.weekly)
    print_report(res, args.weekly)
    if args.out:
        cols = ["game_id", "season", "week", "home_team", "away_team", "home_won", *MODELS, "ridge_margin"]
        res.preds[cols].round(4).to_csv(args.out, index=False)
        print(f"\nwrote {len(res.preds):,} games to {args.out}")


if __name__ == "__main__":
    main()
