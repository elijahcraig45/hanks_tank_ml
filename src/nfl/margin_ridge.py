"""Margin ridge: a few-parameter football rating model, shared by NFL and CFB.

    home_margin = rating[home] - rating[away] + hfa * (not neutral) + sum_k beta_k * x_k

fit by weighted ridge regression on recent games, then turned into a win probability
with Phi(margin / sigma). One rating per team plus a handful of global terms — around
35 parameters in the NFL and ~260 in college, against the XGBoost classifier's ~90
features — and it models the margin, which carries more information per game than the
won/lost bit the classifier sees.

Walk-forward numbers that justified building this (football_eval.py, 2026-09-25):
CFB FBS 2023-24 log loss 0.4948 vs XGBoost 0.5406; NFL 2017-24 0.6342 vs 0.6455.

Design notes:
  * Weights decay exponentially in weeks (tau) and games older than `window_seasons`
    are dropped, so last season is a fading prior rather than a hard reset.
  * `covariates` are the extension point for game-level adjustments (rest, EPA form,
    unit matchups). Each is a column on the games frame, oriented home-minus-away,
    multiplied by its scale so the fitted coefficient is in points per unit. They are
    fit jointly with the team ratings, so an adjustment only gets credit for what the
    ratings do not already explain.
  * Pure numpy: the closed-form weighted ridge solve needs no sklearn or scipy, so it
    stages into either Cloud Function without touching requirements.
  * Every refit sees only games strictly before the block being predicted. A team with
    no games in the window gets rating 0 — the ridge prior, i.e. "average".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

WEEKS_PER_SEASON = 30   # > any real week index (NFL <=22, CFB <=21), keeps seasons apart
HFA_SCALE = 10.0        # HFA column is 10 so its penalty is ~1/100th of a team rating's


@dataclass(frozen=True)
class RidgeConfig:
    alpha: float
    tau: float                      # decay time constant, in weeks
    sigma: float                    # margin -> probability scale, in points
    window_seasons: int = 2
    margin_cap: float | None = None  # clip |margin| before fitting (CFB blowouts)
    min_train: int = 50
    covariates: tuple[tuple[str, float], ...] = field(default_factory=tuple)

    def with_(self, **kw) -> "RidgeConfig":
        return replace(self, **kw)


# Tuned on pre-holdout seasons only (see scripts/football/eval_margin_ridge.py):
#   NFL: grid over 2010-2016.   CFB: grid over 2022 FBS (the cache starts 2021).
NFL_RIDGE = RidgeConfig(alpha=3.0, tau=16.0, sigma=12.48)
CFB_RIDGE = RidgeConfig(alpha=1.0, tau=16.0, sigma=15.55, margin_cap=45.0,
                        covariates=(("fbs_diff", 10.0),))


def time_index(season, week) -> np.ndarray:
    return np.asarray(season, dtype=float) * WEEKS_PER_SEASON + np.asarray(week, dtype=float)


def norm_cdf(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def win_prob(margin, sigma: float) -> np.ndarray:
    return norm_cdf(np.asarray(margin, dtype=float) / sigma)


def _require(games: pd.DataFrame, cfg: RidgeConfig) -> None:
    need = {"season", "week", "home_team", "away_team", "neutral"}
    need |= {c for c, _ in cfg.covariates}
    missing = need - set(games.columns)
    if missing:
        raise KeyError(f"margin ridge needs columns {sorted(missing)}")


@dataclass
class MarginRidge:
    """A fitted model: team ratings in points plus the global terms."""

    ratings: dict[str, float]
    hfa: float
    betas: dict[str, float]
    cfg: RidgeConfig
    n_train: int

    def predict_margin(self, games: pd.DataFrame) -> np.ndarray:
        h = games["home_team"].map(self.ratings).fillna(0.0).to_numpy(float)
        a = games["away_team"].map(self.ratings).fillna(0.0).to_numpy(float)
        m = h - a + self.hfa * (1 - games["neutral"].to_numpy(float))
        for col, scale in self.cfg.covariates:
            m = m + self.betas[col] * scale * games[col].fillna(0.0).to_numpy(float)
        return m

    def predict_proba(self, games: pd.DataFrame) -> np.ndarray:
        return win_prob(self.predict_margin(games), self.cfg.sigma)


def _design(games: pd.DataFrame, teams: pd.Index, cfg: RidgeConfig) -> np.ndarray:
    n, T = len(games), len(teams)
    X = np.zeros((n, T + 1 + len(cfg.covariates)))
    rows = np.arange(n)
    X[rows, teams.get_indexer(games["home_team"])] = 1.0
    X[rows, teams.get_indexer(games["away_team"])] = -1.0
    X[:, T] = (1 - games["neutral"].to_numpy(float)) * HFA_SCALE
    for k, (col, scale) in enumerate(cfg.covariates):
        X[:, T + 1 + k] = games[col].fillna(0.0).to_numpy(float) * scale
    return X


def _solve(X: np.ndarray, y: np.ndarray, w: np.ndarray, alpha: float,
           center: np.ndarray | None = None) -> np.ndarray:
    """argmin_b sum w (y - Xb)^2 + alpha |b - center|^2  (no intercept; center None or 0 = sklearn Ridge)."""
    Xw = X * w[:, None]
    A = X.T @ Xw + alpha * np.eye(X.shape[1])
    rhs = Xw.T @ y
    if center is not None:
        rhs = rhs + alpha * center
    return np.linalg.solve(A, rhs)


def _train_mask(t: np.ndarray, t_now: float, has_margin: np.ndarray,
                cfg: RidgeConfig) -> np.ndarray:
    return (t < t_now) & (t >= t_now - cfg.window_seasons * WEEKS_PER_SEASON) & has_margin


def fit(games: pd.DataFrame, t_now: float, cfg: RidgeConfig,
        prior: dict[str, float] | None = None) -> MarginRidge | None:
    """Fit on completed games in the window strictly before time index `t_now`.

    `games` needs a `margin` column (home minus away; NaN for unplayed games).
    Returns None when the window holds fewer than `cfg.min_train` games.

    `prior` (team -> points, e.g. program_prior()) shrinks each team's rating toward its
    prior instead of toward 0, with the same penalty alpha; a team with no game in the
    window gets its prior as its rating. None or {} is exactly the plain ridge.
    """
    _require(games, cfg)
    t = time_index(games["season"], games["week"])
    margin = games["margin"].to_numpy(float)
    tr = _train_mask(t, t_now, ~np.isnan(margin), cfg)
    if tr.sum() < cfg.min_train:
        return None
    g = games[tr]
    teams = pd.Index(sorted(set(g["home_team"]) | set(g["away_team"])))
    X = _design(g, teams, cfg)
    y = margin[tr]
    if cfg.margin_cap is not None:
        y = np.clip(y, -cfg.margin_cap, cfg.margin_cap)
    w = np.exp(-(t_now - t[tr]) / cfg.tau)
    T = len(teams)
    center = None
    if prior:
        center = np.zeros(X.shape[1])
        center[:T] = teams.map(lambda k: prior.get(k, 0.0)).to_numpy(float)
    b = _solve(X, y, w, cfg.alpha, center)
    ratings = dict(zip(teams, b[:T]))
    if prior:
        # No game in the window: the penalised solve would leave the rating at its prior, so that is the rating.
        for team, mu in prior.items():
            ratings.setdefault(team, float(mu))
    return MarginRidge(
        ratings=ratings,
        hfa=float(b[T] * HFA_SCALE),
        betas={col: float(b[T + 1 + k]) for k, (col, _) in enumerate(cfg.covariates)},
        cfg=cfg,
        n_train=int(tr.sum()),
    )


def walk_forward(games: pd.DataFrame, target: np.ndarray, cfg: RidgeConfig) -> np.ndarray:
    """Out-of-sample predicted home margin for every game where `target` is True.

    Refits once per (season, week) block using only earlier blocks — the same
    information production has when it predicts that week. NaN where the window is
    too thin to fit.
    """
    _require(games, cfg)
    games = games.reset_index(drop=True)
    target = np.asarray(target, dtype=bool)
    t = time_index(games["season"], games["week"])
    margin = games["margin"].to_numpy(float)
    has = ~np.isnan(margin)

    # Build the full design once; each block just selects rows. Columns for teams not
    # in a block's window are all-zero there and the ridge penalty pins them at 0.
    teams = pd.Index(sorted(set(games["home_team"]) | set(games["away_team"])))
    X = _design(games, teams, cfg)
    y = np.clip(margin, -cfg.margin_cap, cfg.margin_cap) if cfg.margin_cap else margin

    pred = np.full(len(games), np.nan)
    for t_now in np.unique(t[target]):
        tr = _train_mask(t, t_now, has, cfg)
        if tr.sum() < cfg.min_train:
            continue
        w = np.exp(-(t_now - t[tr]) / cfg.tau)
        b = _solve(X[tr], y[tr], w, cfg.alpha)
        te = target & (t == t_now)
        pred[te] = X[te] @ b
    return pred


# ---------------------------------------------------------------------------------------------- program prior
# A preseason starting point for each team's rating, from our own scores only: last season's one-season rating
# and the two before it, regressed toward the mean. Validated for CFB only (CFB_RIDGE; research/cfb_preseason_prior,
# 2026-11-03): tuned on 2017-19 + 2021, frozen, then 2022-24 log loss -0.0045 [-0.0071, -0.0019] and a one-shot
# 2026 weeks 1-9 check -0.0115 [-0.0186, -0.0035]. The gain is early-season FBS and all of FCS; FBS week 7+ is flat.
PRIOR_MIN_ROWS = 30       # team-seasons a division needs before its regression is trusted


def season_ratings(games: pd.DataFrame, cfg: RidgeConfig, seasons) -> dict[int, dict[str, float]]:
    """One rating per team per season: the same ridge on that season's games alone, no decay."""
    one = cfg.with_(window_seasons=1, tau=1e12)
    out = {}
    for u in seasons:
        m = fit(games[games["season"] == u], (int(u) + 1) * WEEKS_PER_SEASON, one)
        if m is not None:
            out[int(u)] = m.ratings
    return out


def _lag_rows(R: dict[int, dict[str, float]], division_of: dict[str, str], u: int) -> pd.DataFrame:
    """(team, division, rating u-1, u-2, u-3, rating u) for every team rated in u-1. A missing older season repeats
    the newer one, so a new program leans on what it has."""
    rows = []
    for team, r1 in R.get(u - 1, {}).items():
        r2 = R.get(u - 2, {}).get(team, r1)
        r3 = R.get(u - 3, {}).get(team, r2)
        rows.append((team, division_of.get(team, "fcs"), r1, r2, r3, R.get(u, {}).get(team, np.nan)))
    return pd.DataFrame(rows, columns=["team", "div", "l1", "l2", "l3", "target"])


def program_prior(games: pd.DataFrame, season: int, cfg: RidgeConfig,
                  division_of: dict[str, str]) -> dict[str, float]:
    """Preseason prior (team -> points) for `season`, for fit(prior=...).

    Uses only seasons before `season`: per division, an OLS of each season's rating on the three before it, fit on
    every earlier season, applied to the last three. A division with fewer than PRIOR_MIN_ROWS team-seasons gets no
    prior, and so does a team not rated last season: both keep the plain ridge's 0."""
    past = games[(games["season"] < season) & games["margin"].notna()]
    R = season_ratings(past, cfg, sorted(int(u) for u in past["season"].unique()))
    if not R:
        return {}
    frames = [_lag_rows(R, division_of, u) for u in range(min(R) + 1, season) if u in R]
    if not frames:
        return {}
    train = pd.concat(frames, ignore_index=True).dropna(subset=["target"])
    cur = _lag_rows(R, division_of, season)
    prior: dict[str, float] = {}
    for div in ("fbs", "fcs"):
        tr, te = train[train["div"] == div], cur[cur["div"] == div]
        if len(tr) < PRIOR_MIN_ROWS or te.empty:
            continue
        Xtr = np.column_stack([np.ones(len(tr)), tr[["l1", "l2", "l3"]].to_numpy(float)])
        beta, *_ = np.linalg.lstsq(Xtr, tr["target"].to_numpy(float), rcond=None)
        Xte = np.column_stack([np.ones(len(te)), te[["l1", "l2", "l3"]].to_numpy(float)])
        prior.update({t: float(v) for t, v in zip(te["team"], Xte @ beta)})
    return prior


def fit_sigma(margin_pred: np.ndarray, won: np.ndarray,
              lo: float = 5.0, hi: float = 40.0) -> float:
    """Sigma minimising log loss of Phi(margin/sigma) — golden-section, no scipy."""
    ok = ~np.isnan(margin_pred)
    m, y = margin_pred[ok], np.asarray(won, dtype=float)[ok]

    def ll(s):
        return log_loss(y, win_prob(m, s))

    g = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - g * (b - a), a + g * (b - a)
    for _ in range(60):
        if ll(c) < ll(d):
            b = d
        else:
            a = c
        c, d = b - g * (b - a), a + g * (b - a)
    return (a + b) / 2


def log_loss(y, p) -> float:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def tune(games: pd.DataFrame, tune_mask: np.ndarray, won: np.ndarray, base: RidgeConfig,
         alphas=(1, 3, 10, 30), taus=(4, 8, 16, 30)) -> tuple[RidgeConfig, float]:
    """Grid-search alpha/tau (and fit sigma) on `tune_mask` games only."""
    best = None
    for alpha in alphas:
        for tau in taus:
            cfg = base.with_(alpha=float(alpha), tau=float(tau))
            mp = walk_forward(games, tune_mask, cfg)
            ok = tune_mask & ~np.isnan(mp)
            sig = fit_sigma(mp[ok], won[ok])
            ll = log_loss(won[ok], win_prob(mp[ok], sig))
            if best is None or ll < best[1]:
                best = (cfg.with_(sigma=float(sig)), ll)
    return best
