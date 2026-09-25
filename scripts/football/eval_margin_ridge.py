"""Walk-forward comparison of the football models. Read-only: local caches, no BQ.

    python scripts/football/eval_margin_ridge.py cfb
    python scripts/football/eval_margin_ridge.py nfl [--fast]

Protocol (same as the 2026-09-25 review, now through the production code path):
  * Ridge hyper-parameters (alpha, tau, sigma) are tuned on pre-holdout seasons that
    are NOT scored: NFL 2010-2016, CFB 2022 FBS. The CFB cache starts at 2021.
  * Scored periods: NFL walk-forward 2017-2024 + 2025 holdout; CFB 2023-24 + 2025.
  * Every prediction for a (season, week) is made from games strictly before it.
    XGBoost is retrained weekly, exactly like production's backfill; `--fast` retrains
    it once per season instead (slightly pessimistic for XGB, much quicker).
  * Paired bootstrap (2000 resamples of games) on the per-game log-loss difference.
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.disable(logging.INFO)

SRC = Path(__file__).resolve().parents[2] / "src"


def per_game_ll(y, p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def boot_ci(d, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def report(base: pd.DataFrame, models: list[str], ref: str, periods, label: str):
    for name, mask in periods:
        b = base[mask].dropna(subset=models)
        y = b["home_won"].to_numpy(float)
        print(f"\n{label} {name} (n={len(b)})")
        print(f"  {'model':28s} {'acc':>6} {'logloss':>8} {'brier':>7}")
        for m in models:
            p = b[m].to_numpy(float)
            print(f"  {m:28s} {np.mean((p > .5) == y):6.3f} {per_game_ll(y, p).mean():8.4f}"
                  f" {np.mean((p - y) ** 2):7.4f}")
        for m in models:
            if m == ref:
                continue
            d = per_game_ll(y, b[m]) - per_game_ll(y, b[ref])
            lo, hi = boot_ci(d)
            print(f"  {m} - {ref}: {d.mean():+.4f} (95% CI {lo:+.4f}, {hi:+.4f})")


# --------------------------------------------------------------------------- CFB
def run_cfb():
    sys.path.insert(0, str(SRC / "nfl"))
    sys.path.insert(0, str(SRC / "cfb"))
    import margin_ridge as mr
    import pipeline as cp

    raw = pd.read_parquet(SRC.parent / "data/cfb/raw/cfb_games_v2.parquet")
    raw = raw[raw.home_won.notna()]
    feats = cp.build(raw)

    rg = cp.ridge_frame(raw)
    won = (rg["margin"] > 0).astype(int).to_numpy()
    tune_mask = ((rg.season == 2022) & (rg.division == "fbs")).to_numpy()
    cfg, ll = mr.tune(rg, tune_mask, won, mr.CFB_RIDGE)
    print(f"CFB ridge tuned on 2022 FBS: alpha={cfg.alpha} tau={cfg.tau} "
          f"sigma={cfg.sigma:.2f} ll={ll:.4f}")

    test = rg.season.isin([2023, 2024, 2025]).to_numpy()
    rg["ridge_margin"] = mr.walk_forward(rg, test, cfg)
    rg["ridge"] = mr.win_prob(rg["ridge_margin"], cfg.sigma)
    # The config actually shipped in margin_ridge.CFB_RIDGE, to confirm it reproduces.
    rg["ridge_shipped"] = mr.win_prob(mr.walk_forward(rg, test, mr.CFB_RIDGE),
                                      mr.CFB_RIDGE.sigma)

    for div in ("fbs", "fcs"):
        outs = []
        for s in (2023, 2024, 2025):
            r = cp.backfill_division(feats, div, s, model="xgb")
            outs.append(r[["game_id", "season", "home_won", "home_win_probability",
                           "elo_home_win_prob"]])
        base = pd.concat(outs).rename(columns={"home_win_probability": "xgb (production)"})
        base = base.merge(rg[["game_id", "ridge", "ridge_shipped", "ridge_margin"]],
                          on="game_id", how="left")
        print(f"\n{div}: ridge covers {base.ridge.notna().mean():.3f} of XGB-scored games")
        report(base, ["elo_home_win_prob", "xgb (production)", "ridge", "ridge_shipped"],
               "xgb (production)",
               [("2023-24", base.season < 2025), ("holdout 2025", base.season == 2025)],
               f"CFB-{div.upper()}")


# --------------------------------------------------------------------------- NFL
def run_nfl(fast: bool):
    sys.path.insert(0, str(SRC / "nfl"))
    import margin_ridge as mr
    import predict_nfl as pn
    from data import completed_games
    from features import build_features, feature_columns
    from models import build_xgb

    games = completed_games()
    epa = pd.read_parquet(SRC.parent / "data/nfl/raw/team_week_epa.parquet")
    g06 = games[games.season >= int(epa.season.min())]
    F_epa = build_features(g06, epa=epa)
    F_no = build_features(games, epa=None)  # what production actually ran: no EPA, 1999+

    folds = list(range(2017, 2026))

    def xgb_wf(F, name):
        cols = feature_columns(F)
        out = []
        for s in folds:
            blocks = [None] if fast else sorted(F[F.season == s].week.unique())
            for wk in blocks:
                if wk is None:
                    tr, te = F[F.season < s], F[F.season == s]
                else:
                    tr = F[(F.season < s) | ((F.season == s) & (F.week < wk))]
                    te = F[(F.season == s) & (F.week == wk)]
                m = build_xgb()
                m.fit(tr[cols].fillna(0).values, tr.home_won.values)
                out.append(pd.DataFrame({"game_id": te.game_id.values,
                                         name: m.predict_proba(te[cols].fillna(0).values)[:, 1]}))
        return pd.concat(out)

    base = F_epa[F_epa.season.isin(folds)][["game_id", "season", "home_won",
                                            "elo_home_win_prob", "spread_line"]].copy()
    base = base.merge(xgb_wf(F_no, "xgb no-EPA (production)"), on="game_id")
    base = base.merge(xgb_wf(F_epa, "xgb + EPA (fixed)"), on="game_id")

    rg = pn.ridge_frame(games, feats=F_epa, epa=epa)
    won = (rg["margin"] > 0).astype(int).to_numpy()
    tune_mask = rg.season.between(2010, 2016).to_numpy()
    test = rg.season.isin(folds).to_numpy()

    variants = {
        "ridge": mr.NFL_RIDGE.with_(covariates=()),
        "ridge + rest": mr.NFL_RIDGE.with_(covariates=(("rest_diff", 1.0),)),
        "ridge + rest + epa form": mr.NFL_RIDGE.with_(
            covariates=(("rest_diff", 1.0), ("net_epa_8g", 65.0))),
        "ridge + rest + epa rating": mr.NFL_RIDGE.with_(
            covariates=(("rest_diff", 1.0), ("epa_rating_diff", 1.0))),
    }
    for name, base_cfg in variants.items():
        cfg, ll = mr.tune(rg, tune_mask, won, base_cfg,
                          alphas=(1, 3, 10, 30), taus=(6, 10, 16, 25))
        print(f"{name:28s} tuned 2010-16: alpha={cfg.alpha} tau={cfg.tau} "
              f"sigma={cfg.sigma:.2f} ll={ll:.4f}")
        mp = mr.walk_forward(rg, test, cfg)
        rg[name] = mr.win_prob(mp, cfg.sigma)
        if cfg.covariates:
            fitted = mr.fit(rg, mr.time_index(2025, 1), cfg)
            print(f"{'':28s} betas at 2025 wk1 (points per unit): "
                  + ", ".join(f"{c}={fitted.betas[c] * s:.3f}" for c, s in cfg.covariates)
                  + f"; hfa={fitted.hfa:.2f}")
    base = base.merge(rg[["game_id", *variants]], on="game_id", how="left")
    base["vegas"] = mr.win_prob(base.spread_line, 13.45)

    models = ["elo_home_win_prob", "xgb no-EPA (production)", "xgb + EPA (fixed)",
              *variants, "vegas"]
    periods = [("walk-forward 2017-24", base.season < 2025),
               ("holdout 2025", base.season == 2025)]
    report(base, models, "xgb no-EPA (production)", periods, "NFL")
    report(base, models, "xgb + EPA (fixed)", periods, "NFL")
    report(base, models, "ridge", periods, "NFL")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("sport", choices=["nfl", "cfb"])
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_args()
    run_cfb() if a.sport == "cfb" else run_nfl(a.fast)
