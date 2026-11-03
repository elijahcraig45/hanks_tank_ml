"""Does a preseason prior help the CFB margin ridge?  Research only; reads local CSVs.

    python3 research/cfb_preseason_prior/eval_prior.py

Baseline: EXACTLY production's CFB ridge (margin_ridge.CFB_RIDGE on pipeline.ridge_frame),
walk-forward, refit every (season, week) on games strictly before it.

Variant: the same solve, but each team's rating is shrunk toward a preseason mean mu
instead of toward 0:

    argmin_b  sum w (y - X b)^2 + alpha_p |r - lam * mu|^2 + alpha |rest|^2

With lam = 0 and alpha_p = alpha it IS the baseline (checked below to 1e-9).

mu (CFBD talent / returning production / recruiting were planned, but fetching them needs
the CFBD key, which this research run was not allowed to read): a program-strength prior
from our own data. R_u = season-only ridge rating of season u (same design, no decay, one
season). mu_s = OLS of R_u on (R_{u-1}, R_{u-2}, R_{u-3}) + intercept, per team division,
fit only on target seasons u < s, applied to (R_{s-1}, R_{s-2}, R_{s-3}). R_{s-3} lies
outside the production ridge's two-season window, R_{s-2} is nearly decayed out of it, so
mu carries multi-year program strength the baseline cannot see. Teams with no R_{s-1}
get mu = 0 (the baseline's own prior).

Protocol: lam / alpha_p chosen on the repo's CFB tune seasons (rankings/evaluate.py
SPLITS: 2017, 2018, 2019, 2021), frozen, scored on 2022-2024. Season 2025 is the
touch-once holdout and is never scored here (its games are only training data for the
2026 check). sigma stays production's 15.55 for both models. CIs resample whole weeks.
"""

from __future__ import annotations

import itertools
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1] / "src"
sys.path.insert(0, str(SRC / "nfl"))
sys.path.insert(0, str(SRC / "cfb"))

import margin_ridge as mr  # noqa: E402
import pipeline as cp  # noqa: E402

CFG = mr.CFB_RIDGE
W = mr.WEEKS_PER_SEASON
TUNE = [2017, 2018, 2019, 2021]
EVAL = [2022, 2023, 2024]
HOLDOUT = 2025          # never scored
CHECK = 2026
GRID_ALPHA_P = [1.0, 2.0, 4.0, 8.0, 16.0]
GRID_LAM = [0.25, 0.5, 0.75, 1.0]


def ll_vec(y, p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def block_ci(frame, a, b, n=2000, seed=0):
    """Mean per-game LL difference a-b, 95% CI resampling whole (season, week) blocks."""
    d = frame[a] - frame[b]
    blk = frame.assign(d=d).groupby(["season", "week"])["d"].agg(["sum", "count"])
    s, c = blk["sum"].to_numpy(), blk["count"].to_numpy()
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(s), (n, len(s)))
    boot = s[draws].sum(1) / c[draws].sum(1)
    return float(d.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


# ----------------------------------------------------------------------------- data
def load():
    g = pd.read_csv(HERE / "data/games_raw.csv", parse_dates=["game_date"])
    g = g[g["home_won"].notna()].drop_duplicates("game_id").reset_index(drop=True)
    rg = cp.ridge_frame(g)
    rg["home_won"] = (rg["margin"] > 0).astype(int)
    return g, rg


# ------------------------------------------------------------------- season ratings
def season_ratings(rg) -> dict[int, dict[str, float]]:
    """R_u: one-season ridge rating, same design and alpha, no decay."""
    cfg = CFG.with_(window_seasons=1, tau=1e12)
    out = {}
    for u in sorted(rg["season"].unique()):
        m = mr.fit(rg[rg["season"] == u], (u + 1) * W, cfg)
        if m is not None:
            out[int(u)] = m.ratings
    return out


def lag_frame(R, teams_div, u):
    """Rows (team, lag1, lag2, lag3[, target]) for season u, lags filled backwards."""
    prev = R.get(u - 1, {})
    rows = []
    for team, r1 in prev.items():
        r2 = R.get(u - 2, {}).get(team, r1)
        r3 = R.get(u - 3, {}).get(team, r2)
        rows.append((team, teams_div.get(team, "fcs"), r1, r2, r3, R.get(u, {}).get(team, np.nan)))
    return pd.DataFrame(rows, columns=["team", "div", "l1", "l2", "l3", "target"])


def priors(R, teams_div, seasons) -> dict[int, dict[str, float]]:
    """mu_s per team, regression fit causally on target seasons u < s."""
    first = min(R)
    out = {}
    for s in seasons:
        train = pd.concat([lag_frame(R, teams_div, u) for u in range(first + 1, s)
                           if u in R], ignore_index=True).dropna(subset=["target"])
        cur = lag_frame(R, teams_div, s)
        mu = {}
        for div in ("fbs", "fcs"):
            tr, te = train[train["div"] == div], cur[cur["div"] == div]
            if len(tr) < 30 or te.empty:
                continue
            Xtr = np.column_stack([np.ones(len(tr)), tr[["l1", "l2", "l3"]].to_numpy()])
            beta, *_ = np.linalg.lstsq(Xtr, tr["target"].to_numpy(), rcond=None)
            Xte = np.column_stack([np.ones(len(te)), te[["l1", "l2", "l3"]].to_numpy()])
            mu.update(dict(zip(te["team"], Xte @ beta)))
        out[s] = mu
    return out


# ---------------------------------------------------------------------- walk-forward
def walk_forward(rg, target, mu_by_season=None, alpha_p=None, lam=0.0):
    """mr.walk_forward with an optional prior center for the team ratings."""
    g = rg.reset_index(drop=True)
    target = np.asarray(target, bool)
    t = mr.time_index(g["season"], g["week"])
    margin = g["margin"].to_numpy(float)
    has = ~np.isnan(margin)
    teams = pd.Index(sorted(set(g["home_team"]) | set(g["away_team"])))
    T = len(teams)
    X = mr._design(g, teams, CFG)
    y = np.clip(margin, -CFG.margin_cap, CFG.margin_cap)
    pen = np.full(X.shape[1], CFG.alpha)
    if alpha_p is not None:
        pen[:T] = alpha_p
    pred = np.full(len(g), np.nan)
    for t_now in np.unique(t[target]):
        tr = mr._train_mask(t, t_now, has, CFG)
        if tr.sum() < CFG.min_train:
            continue
        w = np.exp(-(t_now - t[tr]) / CFG.tau)
        Xw = X[tr] * w[:, None]
        A = X[tr].T @ Xw + np.diag(pen)
        rhs = Xw.T @ y[tr]
        if mu_by_season is not None and lam:
            mu = mu_by_season.get(int(t_now // W), {})
            center = np.zeros(X.shape[1])
            center[:T] = lam * teams.map(lambda k: mu.get(k, 0.0)).to_numpy(float)
            rhs = rhs + pen * center
        b = np.linalg.solve(A, rhs)
        te = target & (t == t_now)
        pred[te] = X[te] @ b
    return pred


def scored(rg, mask, margin_pred, name):
    f = rg.loc[mask, ["game_id", "season", "week", "division", "home_won"]].copy()
    f[name] = ll_vec(f["home_won"], mr.win_prob(margin_pred[mask], CFG.sigma))
    f[name + "_p"] = mr.win_prob(margin_pred[mask], CFG.sigma)
    return f


def bucket(w):
    return np.where(w <= 3, "1-3", np.where(w <= 6, "4-6", "7+"))


def table(ev, label):
    print(f"\n### {label}")
    print(f"  {'slice':16s} {'n':>5} {'baseline':>9} {'prior':>9} {'diff':>8}  95% CI (weeks)")
    for div in ("fbs", "fcs", "all"):
        d0 = ev if div == "all" else ev[ev["division"] == div]
        for bk in ("1-3", "4-6", "7+", "all"):
            d = d0 if bk == "all" else d0[d0["bucket"] == bk]
            if d.empty:
                continue
            m, lo, hi = block_ci(d, "prior", "base")
            print(f"  {div + ' wk ' + bk:16s} {len(d):5d} {d['base'].mean():9.4f} "
                  f"{d['prior'].mean():9.4f} {m:+8.4f}  [{lo:+.4f}, {hi:+.4f}]")


def main():
    g, rg = load()
    div_of = cp.resolve_team_divisions(g)
    R = season_ratings(rg)
    seasons = TUNE + EVAL + [CHECK]
    MU = priors(R, div_of, sorted(set(seasons)))
    print("seasons with ratings:", min(R), "-", max(R), "| teams with a prior:",
          {s: len(MU[s]) for s in sorted(MU)})

    # 0. the variant with lam=0, alpha_p=alpha IS production's ridge
    m26 = (rg["season"] == 2024).to_numpy()
    a = mr.walk_forward(rg, m26, CFG)
    b = walk_forward(rg, m26, MU, alpha_p=CFG.alpha, lam=0.0)
    ok = ~np.isnan(a)
    print(f"parity vs margin_ridge.walk_forward (2024): max |diff| = "
          f"{np.nanmax(np.abs(a[ok] - b[ok])):.2e}, n={ok.sum()}")

    # 1. tune on TUNE seasons, all games, fixed sigma
    tmask = rg["season"].isin(TUNE).to_numpy()
    base_t = scored(rg, tmask, walk_forward(rg, tmask), "base")
    print(f"\ntune seasons {TUNE}: baseline LL {base_t['base'].mean():.4f} (n={len(base_t)})")
    best = None
    for ap, lam in itertools.product(GRID_ALPHA_P, GRID_LAM):
        f = scored(rg, tmask, walk_forward(rg, tmask, MU, ap, lam), "x")
        ll = f["x"].mean()
        early = f[f["week"] <= 6]["x"].mean()
        print(f"  alpha_p={ap:5.1f} lam={lam:.2f}  LL {ll:.4f}  (wk1-6 {early:.4f})")
        if best is None or ll < best[0]:
            best = (ll, ap, lam)
    _, AP, LAM = best
    print(f"FROZEN: alpha_p={AP} lam={LAM}")

    # 2. frozen eval 2022-2024 (2025 never scored)
    emask = rg["season"].isin(EVAL).to_numpy()
    ev = scored(rg, emask, walk_forward(rg, emask), "base").merge(
        scored(rg, emask, walk_forward(rg, emask, MU, AP, LAM), "prior")[["game_id", "prior", "prior_p"]],
        on="game_id")
    ev = ev.dropna(subset=["base", "prior"])
    ev["bucket"] = bucket(ev["week"])
    assert HOLDOUT not in set(ev["season"])
    table(ev, f"EVAL {EVAL} (frozen config, prior - baseline; negative = prior better)")

    # 3. one shot on 2026 games played so far
    cmask = (rg["season"] == CHECK).to_numpy()
    ck = scored(rg, cmask, walk_forward(rg, cmask), "base").merge(
        scored(rg, cmask, walk_forward(rg, cmask, MU, AP, LAM), "prior")[["game_id", "prior", "prior_p"]],
        on="game_id").dropna(subset=["base", "prior"])
    ck["bucket"] = bucket(ck["week"])
    table(ck, f"CHECK {CHECK} weeks {ck['week'].min()}-{ck['week'].max()} (one shot)")

    bench = pd.read_csv(HERE / "data/bench_2026.csv", dtype={"game_id": str})
    ck["game_id"] = ck["game_id"].astype(str)
    ck = ck.merge(bench, on="game_id", how="left")
    ck["market"] = ll_vec(ck["home_won"], mr.win_prob(ck["spread"], 15.55))
    ck["fpi_ll"] = ll_vec(ck["home_won"], ck["fpi"])
    print("\n  2026 vs market / FPI (same games):")
    for div in ("fbs", "fcs"):
        for bk in ("1-3", "4-6", "7+", "all"):
            d = ck[(ck["division"] == div) & ((ck["bucket"] == bk) if bk != "all" else True)]
            dm = d[d["spread"].notna()]
            line = f"    {div} wk {bk:4s} market n={len(dm):4d}: base {dm['base'].mean():.4f} prior {dm['prior'].mean():.4f} market {dm['market'].mean():.4f}"
            df = d[d["fpi"].notna()]
            if len(df):
                line += f" | FPI n={len(df):3d}: base {df['base'].mean():.4f} prior {df['prior'].mean():.4f} fpi {df['fpi_ll'].mean():.4f}"
            print(line)


if __name__ == "__main__":
    main()
