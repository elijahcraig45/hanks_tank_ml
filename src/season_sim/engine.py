"""Rating state and the vectorised rest-of-season game simulator.

The game model is the margin ridge (src/nfl/margin_ridge.py): the best-measured football
winner model we have (NFL 2017-24 log loss 0.6342 vs XGBoost 0.6455; CFB FBS 2025 0.4801 vs
0.5313). A remaining game's home margin is drawn as

    margin ~ Normal(r_home - r_away + hfa * (not neutral) + covariates, noise_sigma)

Three things are layered on top, each switchable so the backtest can measure it:

  * Rating uncertainty (`draw`). The ridge is a Gaussian posterior mean in disguise: with
    y_i ~ N(x_i b, s^2 / w_i) and b ~ N(0, s^2 / alpha), the posterior is
    N(b_hat, s^2 (X'WX + alpha I)^-1). Each simulated season draws its own "true" ratings
    from that posterior, so a team's games are correlated within a path. Without it every
    game is an independent coin with the point estimate, and title odds come out too sharp.
  * Noise deflation (`deflate`). The ridge's sigma was fitted so Phi(margin / sigma) is
    calibrated game by game from the POINT estimate; it already contains rating error.
    When ratings are drawn, per-game noise is shrunk so the marginal variance of a single
    game is unchanged and only the correlation is new.
  * In-season updating (`update`, the "hot hand" in 538's season sims). Before each
    simulated week the ridge is refit on the real results plus that path's simulated ones,
    exactly as production would refit it next Tuesday. Because the ridge solve is linear
    in y and the design does not depend on outcomes, the refit for every path is one
    matrix product: b_w = c_w + G_w y_sim.

Everything is numpy (no scipy), so it stages into either Cloud Function unchanged, and it
takes plain float arrays: BigQuery's nullable Int64/Float64 columns are converted at the
frame boundary (`ridge_frame_from`).
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

try:  # deployed: the Cloud Function tree is flat and margin_ridge sits at top level
    import margin_ridge as mr
except ImportError:  # local: src/season_sim next to src/nfl
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nfl"))
    import margin_ridge as mr  # noqa: E402

MODEL_VERSION = "season_sim_v1"


@dataclass(frozen=True)
class Variant:
    """One way of generating a season. `game_model` picks the per-game mean:
    'ridge' (ours), 'record' (current-record log5, the naive baseline), 'market'
    (closing spread, NFL only, a leaky upper bound) or 'coin' (every game 50/50)."""

    name: str
    game_model: str = "ridge"
    draw: bool = True
    update: bool = False
    deflate: bool = True
    cov_scale: float = 1.0


VARIANTS: dict[str, Variant] = {
    "point": Variant("point", draw=False, update=False, deflate=False),
    "draw": Variant("draw", draw=True, update=False),
    "update": Variant("update", draw=False, update=True, deflate=False),
    "draw_update": Variant("draw_update", draw=True, update=True),
    # sensitivity checks on the rating-uncertainty layer
    "draw_x2": Variant("draw_x2", draw=True, cov_scale=2.0),
    "draw_raw": Variant("draw_raw", draw=True, deflate=False),
    "record": Variant("record", game_model="record", draw=False, deflate=False),
    "market": Variant("market", game_model="market", draw=False, deflate=False),
    "coin": Variant("coin", game_model="coin", draw=False, deflate=False),
}


def norm_ppf(p):
    """Inverse standard normal CDF (Acklam's rational approximation, |err| < 1.2e-9)."""
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    q = np.where(p < 0.5, p, 1 - p)
    out = np.empty_like(q)
    lo = q < 0.02425
    t = np.sqrt(-2 * np.log(q[lo]))
    out[lo] = (((((c[0] * t + c[1]) * t + c[2]) * t + c[3]) * t + c[4]) * t + c[5]) / \
        ((((d[0] * t + d[1]) * t + d[2]) * t + d[3]) * t + 1)
    r = q[~lo] - 0.5
    s = r * r
    out[~lo] = (((((a[0] * s + a[1]) * s + a[2]) * s + a[3]) * s + a[4]) * s + a[5]) * r / \
        (((((b[0] * s + b[1]) * s + b[2]) * s + b[3]) * s + b[4]) * s + 1)
    return np.where(p < 0.5, out, -out)


def ridge_frame_from(games: pd.DataFrame, covariates: tuple[str, ...] = ()) -> pd.DataFrame:
    """Normalise a games frame to the columns the engine needs, as plain numpy dtypes.

    Needs season, week, home_team, away_team, neutral, margin (NaN = unplayed) and a
    boolean `sim` (games to simulate). BigQuery nullable Int64 / Float64 columns become
    float64/int64 here, once: nullable arrays leak pd.NA into linear algebra otherwise.
    """
    g = pd.DataFrame({
        "season": pd.to_numeric(games["season"]).astype("float64").to_numpy().astype(int),
        "week": pd.to_numeric(games["week"]).astype("float64").to_numpy().astype(int),
        "home_team": games["home_team"].astype(str).to_numpy(),
        "away_team": games["away_team"].astype(str).to_numpy(),
        "neutral": pd.to_numeric(games["neutral"]).astype("float64").fillna(0.0).to_numpy(),
        "margin": pd.to_numeric(games["margin"]).astype("float64").to_numpy(),
        "sim": games["sim"].astype(bool).to_numpy(),
    })
    for c in covariates:
        g[c] = pd.to_numeric(games[c]).astype("float64").fillna(0.0).to_numpy()
    for extra in ("game_id", "game_day", "market_margin"):
        if extra in games.columns:
            g[extra] = (games[extra].astype(str).to_numpy() if extra != "market_margin"
                        else pd.to_numeric(games[extra]).astype("float64").to_numpy())
    return g


class RidgeOps:
    """The margin ridge's linear algebra for one season's simulation.

    `frame` holds history (played games of earlier seasons) and the current season's full
    regular-season schedule; rows with sim=True are the games to simulate.
    """

    def __init__(self, frame: pd.DataFrame, cfg: "mr.RidgeConfig"):
        self.frame = frame.reset_index(drop=True)
        self.cfg = cfg
        f = self.frame
        self.t = mr.time_index(f["season"], f["week"])
        self.margin = f["margin"].to_numpy(float)
        self.sim = f["sim"].to_numpy(bool)
        self.teams = pd.Index(sorted(set(f["home_team"]) | set(f["away_team"])))
        self.T = len(self.teams)
        self.X = mr._design(f, self.teams, cfg)
        self.p = self.X.shape[1]
        y = self.margin.copy()
        if cfg.margin_cap is not None:
            y = np.clip(y, -cfg.margin_cap, cfg.margin_cap)
        self.y = y
        self.sim_idx = np.flatnonzero(self.sim)
        self._pos = {int(r): k for k, r in enumerate(self.sim_idx)}

    def team_index(self, names) -> np.ndarray:
        return self.teams.get_indexer(pd.Index(names))

    def solve(self, t_now: float, include_sim: bool = True):
        """(c, G, cols, Ainv, s2): b = c + G @ y_sim[cols] for every path."""
        cfg = self.cfg
        win = (self.t < t_now) & (self.t >= t_now - cfg.window_seasons * mr.WEEKS_PER_SEASON)
        known = win & ~np.isnan(self.margin) & ~self.sim
        simc = win & self.sim if include_sim else np.zeros_like(win)
        w = np.exp(-(t_now - self.t) / cfg.tau)
        rows = known | simc
        Xr = self.X[rows]
        A = Xr.T @ (Xr * w[rows, None]) + cfg.alpha * np.eye(self.p)
        Ainv = np.linalg.inv(A)
        c = Ainv @ (self.X[known].T @ (w[known] * self.y[known]))
        G = Ainv @ (self.X[simc] * w[simc, None]).T
        cols = np.array([self._pos[int(r)] for r in np.flatnonzero(simc)], dtype=int)
        # Posterior noise scale from the known games' weighted residuals.
        r = self.y[known] - self.X[known] @ c
        s2 = float(np.sum(w[known] * r * r) / max(np.sum(w[known]), 1e-9))
        return c, G, cols, Ainv, s2

    def hfa(self, B: np.ndarray) -> np.ndarray:
        return B[..., self.T] * mr.HFA_SCALE


@dataclass
class SimResult:
    """Simulated margins for every sim=True game plus the ratings behind them."""

    ops: RidgeOps
    variant: Variant
    Y: np.ndarray                 # (S, n_sim) simulated home margins (uncapped)
    B_play: np.ndarray            # (S, p) ratings used for postseason games
    B_end: np.ndarray             # (S, p) end-of-regular-season refit (the published board)
    b0: np.ndarray                # (p,) as-of point estimate
    sd0: np.ndarray               # (p,) as-of posterior SD
    noise_sigma: float
    extra: dict = field(default_factory=dict)

    def margins_for(self, rows: np.ndarray) -> np.ndarray:
        """(S, len(rows)) margins for frame rows: known values or simulated draws."""
        S = self.Y.shape[0]
        out = np.empty((S, len(rows)))
        for k, r in enumerate(rows):
            r = int(r)
            if self.ops.sim[r]:
                out[:, k] = self.Y[:, self.ops._pos[r]]
            else:
                out[:, k] = self.ops.margin[r]
        return out

    def game_mean(self, home_idx: np.ndarray, away_idx: np.ndarray,
                  neutral: np.ndarray, covs: np.ndarray | None = None) -> np.ndarray:
        """Mean margin for per-path matchups (postseason). Indices are team columns,
        shape (S,) or (S, k)."""
        return self.extra["mean_fn"](home_idx, away_idx, neutral, covs)


def _record_probs(ops: RidgeOps, season: int) -> np.ndarray:
    """Current win pct per team, shrunk with one win and one loss: (W+1)/(G+2)."""
    f = ops.frame
    cur = (f["season"].to_numpy() == season) & ~ops.sim & ~np.isnan(ops.margin)
    w = np.ones(ops.T)
    g = np.full(ops.T, 2.0)
    hi = ops.team_index(f["home_team"][cur])
    ai = ops.team_index(f["away_team"][cur])
    m = ops.margin[cur]
    np.add.at(w, hi, (m > 0) + 0.5 * (m == 0))
    np.add.at(w, ai, (m < 0) + 0.5 * (m == 0))
    np.add.at(g, hi, 1.0)
    np.add.at(g, ai, 1.0)
    return w / g


def simulate(ops: RidgeOps, season: int, variant: Variant, n_sims: int,
             rng: np.random.Generator, as_of_t: float | None = None) -> SimResult:
    """Simulate every sim=True game of `season`, week by week."""
    cfg = ops.cfg
    f = ops.frame
    sim_rows = ops.sim_idx
    weeks = np.sort(np.unique(f["week"].to_numpy()[sim_rows]))
    if as_of_t is None:
        known_cur = (f["season"].to_numpy() == season) & ~ops.sim & ~np.isnan(ops.margin)
        as_of_t = (ops.t[known_cur].max() + 1) if known_cur.any() else mr.time_index(season, 1)
    as_of_t = float(as_of_t)
    c0, _, _, Ainv0, s2 = ops.solve(as_of_t, include_sim=False)
    cov = s2 * Ainv0 * variant.cov_scale
    sd0 = np.sqrt(np.clip(np.diag(s2 * Ainv0), 0, None))
    S, n = n_sims, len(sim_rows)

    offset = np.zeros((S, ops.p))
    if variant.draw and variant.game_model == "ridge":
        L = np.linalg.cholesky(cov + 1e-9 * np.eye(ops.p))
        offset = rng.standard_normal((S, ops.p)) @ L.T
    B0 = c0[None, :] + offset

    sigma = float(cfg.sigma)
    noise = sigma
    Xs = ops.X[sim_rows]
    if variant.draw and variant.deflate and variant.game_model == "ridge" and n:
        var_mean = float(np.mean(np.einsum("ij,jk,ik->i", Xs, cov, Xs)))
        noise = math.sqrt(max(sigma ** 2 - var_mean, (0.7 * sigma) ** 2))

    rec = _record_probs(ops, season) if variant.game_model == "record" else None
    Y = np.zeros((S, n))
    B = B0
    wk_all = f["week"].to_numpy()[sim_rows]
    for wk in weeks:
        pos = np.flatnonzero(wk_all == wk)
        if variant.update and variant.game_model == "ridge":
            c_w, G_w, cols, _, _ = ops.solve(mr.time_index(season, wk))
            B = c_w[None, :] + (_capped(Y[:, cols], cfg) @ G_w.T if len(cols) else 0.0) + offset
        if variant.game_model == "ridge":
            mean = B @ Xs[pos].T
        elif variant.game_model == "record":
            hi = ops.team_index(f["home_team"].to_numpy()[sim_rows[pos]])
            ai = ops.team_index(f["away_team"].to_numpy()[sim_rows[pos]])
            mean = np.broadcast_to(_log5_margin(rec[hi], rec[ai], sigma), (S, len(pos)))
        elif variant.game_model == "market":
            mm = f["market_margin"].to_numpy(float)[sim_rows[pos]]
            mm = np.where(np.isnan(mm), (B0[0] @ Xs[pos].T), mm)
            mean = np.broadcast_to(mm, (S, len(pos)))
        else:  # coin
            mean = np.zeros((S, len(pos)))
        Y[:, pos] = mean + noise * rng.standard_normal((S, len(pos)))

    # End-of-regular-season refit: the board production would publish after the season.
    t_end = mr.time_index(season, int(weeks.max()) + 1) if len(weeks) else as_of_t
    c_e, G_e, cols_e, _, _ = ops.solve(max(t_end, as_of_t))
    Yc = _capped(Y, cfg)
    B_end = c_e[None, :] + (Yc[:, cols_e] @ G_e.T if len(cols_e) else 0.0)
    if variant.update and variant.game_model == "ridge":
        B_play = B_end + offset
    else:
        B_play = B0

    res = SimResult(ops=ops, variant=variant, Y=Y, B_play=B_play, B_end=B_end,
                    b0=c0, sd0=sd0, noise_sigma=noise)
    res.extra["rec"] = rec
    res.extra["mean_fn"] = _postseason_mean_fn(ops, variant, B_play, rec, sigma)
    return res


def _capped(Y: np.ndarray, cfg) -> np.ndarray:
    return np.clip(Y, -cfg.margin_cap, cfg.margin_cap) if cfg.margin_cap is not None else Y


def _log5_margin(ph: np.ndarray, pa: np.ndarray, sigma: float) -> np.ndarray:
    p = ph * (1 - pa) / (ph * (1 - pa) + pa * (1 - ph))
    return sigma * norm_ppf(p)


def _postseason_mean_fn(ops, variant, B_play, rec, sigma):
    S = B_play.shape[0]
    rows = np.arange(S)

    def fn(hi, ai, neutral, covs=None):
        hi = np.asarray(hi)
        ai = np.asarray(ai)
        r = rows if hi.ndim == 1 else rows[:, None]
        if variant.game_model == "record":
            return _log5_margin(rec[hi], rec[ai], sigma)
        if variant.game_model == "coin":
            return np.zeros(hi.shape)
        m = B_play[r, hi] - B_play[r, ai] + (1 - np.asarray(neutral)) * ops.hfa(B_play)[r]
        if covs is not None:
            for k in range(covs.shape[-1]):
                m = m + B_play[r, ops.T + 1 + k] * ops.cfg.covariates[k][1] * covs[..., k]
        return m

    return fn


def draw_margin(mean: np.ndarray, noise: float, rng: np.random.Generator) -> np.ndarray:
    return mean + noise * rng.standard_normal(np.shape(mean))
