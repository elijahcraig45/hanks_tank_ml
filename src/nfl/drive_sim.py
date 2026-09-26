"""Drive-based Monte Carlo simulator for NFL and college games (SHADOW models).

NFL: `drive_sim_v1` (config NFL, the default everywhere). College: `cfb_drive_sim_v1`
(config CFB), used by src/cfb/cfb_drive_sim.py. Everything league-specific lives in
SimConfig; the NFL default reproduces the original NFL-only module exactly.

Ported from research/football_2026_09/drive_sim/sim_core.py with the configuration
frozen on 2026-09-25 (variant "c_g": field position + game-script states, C=0.03,
tau=16 weeks, kernel tau=6 weeks, pace ridge alpha=200, N=4000). The config was chosen on
2010-16 game-level CRPS and frozen before 2025 was scored. Research result, stated plainly:
the sim ties the margin ridge on winners (2017-24 log loss 0.6335 vs 0.6342, CI spans 0),
adds nothing over the market total, and its one real gain is the *shape* of the margin
distribution (key numbers) once centred on the spread. It is a shadow: nothing served
reads its tables until it earns that.

Everything is fit on drives strictly before the week being predicted:
  1. drive outcome: 9-way penalised multinomial logistic, P(outcome | offense, defense,
     home, start yardline, clock bucket, period x score-differential state), decayed
     sample weights. L2 is the shrinkage.
  2. next-drive start: empirical weighted quantiles of the next start yardline given
     this drive's outcome and start-yardline bin (chains field position).
  3. clock: empirical duration quantiles by (outcome, clock bucket, script) times a team
     pace multiplier (ridge on log-duration residuals). That sets possessions per game.
  4. points: empirical 6/7/8 split on touchdowns; 3 for FG; 2 for a safety.
  5. overtime rules by era (sudden death / modified / both-possess).

Data source (cold Cloud Function safe): one row per drive in
`nfl_historical.drives`, read for the two prior seasons plus the current one (~15k rows).
The weekly ingest (`mode=ingest`) extracts the current season's drives from that
season's pbp only and replaces that season (never WRITE_TRUNCATE); history 2008-2025 is
loaded once from local pbp by `python src/nfl/drives.py --pbp-dir DIR --seasons 2008-2025
--write`. Nothing here reads play-by-play.

Measured locally 2026-09-25, single process: fit 0.5-0.9 s per week, 15-18 ms per game at
N=4000, a 16-game week 1.2 s end to end at 425 MB peak (the research's ~4 s/fit was with
10 jobs in parallel). Inferred, not measured in the cloud: well inside 2 GB / 540 s.
"""
from __future__ import annotations

import json
import logging
import zlib
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MODEL_VERSION = "drive_sim_v1"

# ---------------------------------------------------------------- frozen configuration
# research/football_2026_09/drive_sim/frozen.json, frozen 2026-09-25 before 2025 was scored.
VARIANT = "c"
C = 0.03
TAU_WEEKS = 16.0
KTAU_WEEKS = 6.0
PACE_ALPHA = 200.0
N_SIMS = 4000
SEED = 0
FIT_SEASONS = 2            # drives from the 2 seasons (60 index weeks) before the game week

# Winner calibration: P(home) = sigmoid(a + b * logit(p_raw)), p_raw = P(win | not tied).
# Fit with LogisticRegression(C=1e4) on the research walk-forward sim outputs
# (sim_c_g_N4000_s0_2010_2025.parquet, 4,350 decided games, 2010-2025 regular season +
# playoffs). 2025 is inside this fit, so 2026 is the first out-of-sample season.
PLATT = {"intercept": -0.00679844, "coef": 1.15017185,
         "fit_window": "2010-2025 (4,350 games, research walk-forward sim outputs)"}

# CFB winner calibration, same recipe: LogisticRegression(C=1e4) on logit(p_raw) over the
# research walk-forward outputs 2022-2025 (6,550 decided games); 2026 is out of sample.
CFB_PLATT_INTERCEPT = -0.05149328
CFB_PLATT_COEF = 1.08828684

# Team relocations: schedules carry the old abbreviation, pbp drives the current one.
RELOCATED = {"OAK": "LV", "SD": "LAC", "STL": "LA"}

OUT = ["TD", "FG", "MFG", "PUNT", "TO", "OTD", "TOD", "SAF", "EOH"]
O = {k: i for i, k in enumerate(OUT)}
NO = len(OUT)
NYL, NTB, NPER, NSD = 20, 4, 4, 9
SD_EDGES = np.array([-16.5, -8.5, -3.5, -0.5, 0.5, 3.5, 8.5, 16.5])  # 9 buckets
Q = 101
K_STRUCT = 10.0  # structural columns scaled up => effectively unpenalised
WPS = 30  # season index spacing: t = season * 30 + week

MARGIN_GRID = np.arange(-80, 81)
MARGIN_EXACT_K = range(-21, 22)
OVER_LINE_HALF_WIDTH = 7

DRIVE_COLUMNS = [
    "game_id", "fixed_drive", "season", "week", "season_type", "home_team", "away_team",
    "posteam", "defteam", "qtr", "game_half", "res", "hsr0", "yl", "hs0", "as0", "hs1",
    "as1", "yl_end", "off_home", "dh", "da", "off_pts", "def_pts", "sd0", "hsr_next",
    "yl_next", "dur",
]


# ---------------------------------------------------------------- state helpers
def tb_of(clock):
    return np.where(clock <= 30, 0, np.where(clock <= 120, 1, np.where(clock <= 300, 2, 3)))


def per_of(half, clock):
    # 0 = first half, 1 = Q3, 2 = Q4 > 5min, 3 = Q4 <= 5min
    return np.where(half == 0, 0, np.where(clock > 900, 1, np.where(clock > 300, 2, 3)))


def sd_of(sd, edges=SD_EDGES):
    return np.searchsorted(edges, sd)


@dataclass(frozen=True)
class SimConfig:
    """Everything that differs between leagues. NFL is the default and reproduces the
    module's original behaviour exactly; CFB (below) is the college configuration frozen
    in research/football_2026_09/cfb_drive_sim/frozen.json."""
    name: str = "nfl"
    model_version: str = MODEL_VERSION
    variant: str = VARIANT
    C: float = C
    tau: float = TAU_WEEKS
    ktau: float = KTAU_WEEKS
    pace_alpha: float = PACE_ALPHA
    prior_w: float = 1.0               # extra weight on drives from earlier seasons
    n_sims: int = N_SIMS
    fit_seasons: int = FIT_SEASONS
    min_drives: int = 2000
    sd_edges: tuple = tuple(SD_EDGES.tolist())
    division: bool = False             # off_fbs / def_fbs covariates (CFB)
    ot: str = "nfl"                    # "nfl" (era rules) | "cfb" (2021+ college rules)
    dur_decay: str = "tau"             # decay for duration kernels: "tau" | "ktau"
    margin_lo: int = -80
    margin_hi: int = 80
    sparse: bool = False               # sparse design matrix (CFB: ~400 teams)
    p2pt: float = 0.45                 # CFB overtime two-point rate
    regulation_only_kernels: bool = False
    platt: dict = field(default_factory=lambda: dict(PLATT))
    relocated: dict = field(default_factory=lambda: dict(RELOCATED))

    @property
    def edges(self) -> np.ndarray:
        return np.asarray(self.sd_edges, float)

    @property
    def nsd(self) -> int:
        return len(self.sd_edges) + 1

    @property
    def tied(self) -> int:
        return int(np.searchsorted(self.edges, 0.0))

    @property
    def margin_grid(self) -> np.ndarray:
        return np.arange(self.margin_lo, self.margin_hi + 1)


NFL = SimConfig()

# College, frozen 2026-09-26 on 2022 only (game-level margin CRPS + total CRPS), before
# 2023-2025 were simulated; 21 variants tried. Research result, stated plainly: it LOSES
# to the margin ridge on winners (2025 log loss +0.034, CI [+0.025, +0.044]; stacked with
# the ridge it only ties), loses to the market on totals, and its one validated gain is the
# exact-margin SHAPE centred on the spread (log score -0.087 vs a normal, -0.090 vs the
# league pmf, 2025). Shadow only; label it experimental.
CFB = SimConfig(
    name="cfb", model_version="cfb_drive_sim_v1", variant="c", C=0.1, tau=16.0, ktau=6.0,
    pace_alpha=200.0, prior_w=1.0, n_sims=4000, fit_seasons=2, min_drives=2000,
    sd_edges=(-28.5, -16.5, -8.5, -3.5, -0.5, 0.5, 3.5, 8.5, 16.5, 28.5),
    division=True, ot="cfb", dur_decay="ktau", margin_lo=-100, margin_hi=100, sparse=True,
    p2pt=0.45, regulation_only_kernels=True,
    # Platt on the research walk-forward outputs, 2022-2025 (sim_c_C0.1_t16_pw1_N4000_s0).
    platt={"intercept": CFB_PLATT_INTERCEPT, "coef": CFB_PLATT_COEF,
           "fit_window": "2022-2025 (6,550 games, research walk-forward sim outputs)"},
    relocated={},
)


def time_index(season: int, week: int) -> int:
    return int(season) * WPS + int(week)


def prep_drives(d: pd.DataFrame, games: pd.DataFrame, cfg: SimConfig = NFL) -> pd.DataFrame:
    """Drive rows -> model frame. `games` supplies game_id -> location (Neutral/Home), or
    for college `neutral_site` (0/1)."""
    d = d[d.res.notna()].copy()
    if "location" in games.columns:
        loc = games.set_index("game_id")["location"]
        loc = loc[~loc.index.duplicated()]
        d["neutral"] = (d.game_id.map(loc).fillna("Home") == "Neutral").astype(int)
    else:
        neu = games.set_index("game_id")["neutral_site"]
        neu = neu[~neu.index.duplicated()].fillna(0).astype(int)
        d["neutral"] = d.game_id.map(neu).fillna(0).astype(int)
    d["hadv"] = np.where(d.neutral == 1, 0, np.where(d.off_home.astype(bool), 1, -1))
    d["yl"] = d.yl.fillna(75).clip(1, 99)
    d["yl_next"] = d.yl_next.clip(1, 99)
    d["half"] = (d.game_half != "Half1").astype(int)
    d["ot"] = (d.game_half == "Overtime").astype(int)
    d["tb"] = tb_of(d.hsr0.values)
    d["per"] = per_of(d.half.values, d.hsr0.values)
    d["sdb"] = sd_of(d.sd0.values, cfg.edges)
    d["y"] = d.res.map(O).astype(int)
    d["t"] = d.season * WPS + d.week
    d["ylb"] = np.clip((d.yl // 5).astype(int), 0, NYL - 1)
    d["first_of_half"] = d.groupby(["game_id", "game_half"]).cumcount() == 0
    return d.reset_index(drop=True)


def _features(off, de, hadv, yl, tb, per, sdb, variant, T, cfg: SimConfig = NFL, div=None):
    n = len(off)
    cols = []
    if cfg.sparse:
        from scipy import sparse as sp

        r = np.r_[np.arange(n), np.arange(n)]
        A = sp.csr_matrix((np.ones(2 * n), (r, np.r_[off, T + de])), shape=(n, 2 * T))
    else:
        A = np.zeros((n, 2 * T))
        A[np.arange(n), off] = 1
        A[np.arange(n), T + de] = 1
    cols.append(A)
    if cfg.division:
        offb, defb = div
        cols.append(K_STRUCT * np.stack([offb, defb, offb * defb], 1).astype(float))
    cols.append(K_STRUCT * hadv[:, None].astype(float))
    cols.append(K_STRUCT * np.stack([(tb == k) for k in range(3)], 1).astype(float))
    cols.append(K_STRUCT * (per > 0)[:, None].astype(float))
    if variant in ("b", "c"):
        z = (yl - 70.0) / 25.0
        late = (tb <= 1).astype(float)
        cols.append(K_STRUCT * np.stack([z, z ** 2, z ** 3, (yl <= 40), (yl <= 20), z * late],
                                        1).astype(float))
    if variant == "c":
        nsd, tied = cfg.nsd, cfg.tied
        S = np.zeros((n, NPER * nsd))
        S[np.arange(n), per * nsd + sdb] = 1
        S = np.delete(S, [p * nsd + tied for p in range(NPER)], axis=1)  # tied = base per period
        cols.append(K_STRUCT * S)
    if cfg.sparse:
        from scipy import sparse as sp

        return sp.hstack([cols[0]] + [sp.csr_matrix(c) for c in cols[1:]], format="csr")
    return np.hstack(cols)


def wquant(x, w, q=Q):
    ok = np.isfinite(x) & np.isfinite(w)
    x, w = x[ok], w[ok]
    if len(x) == 0:
        return None
    o = np.argsort(x)
    x, w = x[o], w[o]
    c = np.cumsum(w)
    c = c / c[-1]
    return np.interp(np.linspace(0.005, 0.995, q), c, x)


def _pace_ridge_sparse(off_col, def_col, ncol, y, w, alpha):
    """Weighted ridge with an unpenalised intercept, solved in closed form from sparse
    sums: (Xc' W Xc + alpha I) b = Xc' W yc with weighted centring -- the same problem
    sklearn's Ridge(fit_intercept=True, sample_weight=w) solves, without materialising
    the dense n x 2T design (~1 GB for college's ~400 teams)."""
    from scipy import sparse as sp

    # plain float arrays: a pandas masked array here breaks `w @ y` under pandas 2.3
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    n = len(y)
    X = sp.csr_matrix((np.ones(2 * n), (np.r_[np.arange(n), np.arange(n)],
                                         np.r_[off_col, def_col])), shape=(n, ncol))
    sw = w.sum()
    xbar = np.asarray(X.T @ w).ravel() / sw
    ybar = float(w @ y) / sw
    XtWX = (X.T @ sp.diags(w) @ X).toarray() - sw * np.outer(xbar, xbar)
    XtWy = np.asarray(X.T @ (w * (y - ybar))).ravel()
    return np.linalg.solve(XtWX + alpha * np.eye(ncol), XtWy)


class DriveModel:
    def __init__(self, variant=None, C=None, tau=None, ktau=None, pace_alpha=None,
                 cfg: SimConfig = NFL):
        self.cfg = cfg
        self.variant = cfg.variant if variant is None else variant
        self.C = cfg.C if C is None else C
        self.tau = cfg.tau if tau is None else tau
        self.ktau = cfg.ktau if ktau is None else ktau
        self.pace_alpha = cfg.pace_alpha if pace_alpha is None else pace_alpha

    def _decay(self, t, seasons, tau):
        w = np.exp(-(self.t_now - t) / tau)
        if self.cfg.prior_w != 1.0:
            w = np.where(seasons < self.t_now // WPS, w * self.cfg.prior_w, w)
        return w

    def _div(self, frame):
        if not self.cfg.division:
            return None
        return frame.off_fbs.values.astype(float), frame.def_fbs.values.astype(float)

    def fit(self, tr: pd.DataFrame, t_now: int, teams: pd.Index, fbs: dict | None = None):
        from sklearn.linear_model import LogisticRegression, Ridge

        self.teams = teams
        self.fbs = fbs or {}
        self.t_now = t_now
        T = len(teams)
        if self.cfg.division:
            tr = tr.copy()
            tr["off_fbs"] = tr.posteam.map(self.fbs).fillna(0).astype(int)
            tr["def_fbs"] = tr.defteam.map(self.fbs).fillna(0).astype(int)
        reg = tr[tr.ot == 0]
        w = self._decay(reg.t.values, reg.season.values, self.tau)
        off = teams.get_indexer(reg.posteam)
        de = teams.get_indexer(reg.defteam)
        X = _features(off, de, reg.hadv.values, reg.yl.values.astype(float), reg.tb.values,
                      reg.per.values, reg.sdb.values, self.variant, T, self.cfg, self._div(reg))
        self.clf = LogisticRegression(C=self.C, max_iter=400, tol=1e-4)
        self.clf.fit(X, reg.y.values, sample_weight=w / w.mean())
        self.classes = self.clf.classes_
        # kernels (shorter decay: kickoff/touchback rules change between seasons)
        wk = np.exp(-(t_now - tr.t.values) / self.ktau)
        sb = np.clip((tr.yl.values // 20).astype(int), 0, 4)
        ko_mask = tr.first_of_half.values & (tr.ot.values == 0)
        self.ko = wquant(tr.yl.values[ko_mask].astype(float), wk[ko_mask])
        K = np.zeros((NO, 5, Q))
        has = tr.yl_next.notna().values
        if self.cfg.regulation_only_kernels:
            has = has & (tr.ot.values == 0)
        for o in range(NO):
            for b in range(5):
                m = has & (tr.y.values == o)
                if OUT[o] not in ("TD", "FG", "SAF", "OTD"):
                    m2 = m & (sb == b)
                    if wk[m2].sum() > 5:
                        m = m2
                q = wquant(tr.yl_next.values[m].astype(float), wk[m]) if m.sum() > 0 else None
                K[o, b] = q if q is not None else self.ko
        self.K = K
        # durations
        sg = self._sg(tr.per.values, tr.sdb.values)
        D = np.zeros((NO, NTB, 3, Q))
        dur = tr.dur.values.astype(float)
        w_t = np.exp(-(t_now - tr.t.values) / self.tau)
        # college: short decay, so the 2023 first-down clock change is absorbed in weeks
        w_d = wk if self.cfg.dur_decay == "ktau" else w_t
        for o in range(NO):
            for b in range(NTB):
                base = (tr.y.values == o) & (tr.tb.values == b) & (tr.ot.values == 0)
                qb = wquant(dur[base], w_d[base]) if base.sum() > 0 else np.full(Q, 60.0)
                for g in range(3):
                    m = base & (sg == g)
                    D[o, b, g] = wquant(dur[m], w_d[m]) if m.sum() > 30 else qb
        self.D = D
        # pace: ridge on log-duration residuals of ordinary (tb==3, neutral-script) drives
        m = (tr.tb.values == 3) & (tr.ot.values == 0) & (sg == 0) & (dur > 0)
        sub = tr[m]
        lg = np.log(dur[m])
        mu = pd.Series(lg).groupby(sub.y.values).transform("mean").values
        wp = self._decay(sub.t.values, sub.season.values, self.tau)
        if self.cfg.sparse:
            coef = _pace_ridge_sparse(teams.get_indexer(sub.posteam),
                                      T + teams.get_indexer(sub.defteam), 2 * T,
                                      lg - mu, wp, self.pace_alpha)
        else:
            Xp = np.zeros((m.sum(), 2 * T))
            Xp[np.arange(m.sum()), teams.get_indexer(sub.posteam)] = 1
            Xp[np.arange(m.sum()), T + teams.get_indexer(sub.defteam)] = 1
            coef = Ridge(alpha=self.pace_alpha, fit_intercept=True).fit(
                Xp, lg - mu, sample_weight=wp).coef_
        self.pace_off, self.pace_def = coef[:T], coef[T:]

        # points on TD / OTD
        def pdist(mask, col):
            v = tr[col].values[mask]
            ww = w_t[mask]
            p = np.array([ww[v == k].sum() for k in (6, 7, 8)]) + 1e-3
            return p / p.sum()

        rg = (tr.ot.values == 0) if self.cfg.regulation_only_kernels else np.ones(len(tr), bool)
        self.td_p = pdist((tr.y.values == O["TD"]) & rg, "off_pts")
        self.otd_p = pdist((tr.y.values == O["OTD"]) & rg, "def_pts")
        return self

    def _sg(self, per, sdb):
        if self.variant != "c":
            return np.zeros(len(per), int)
        late = per >= 2
        tied = self.cfg.tied
        return np.where(late & (sdb < tied), 1, np.where(late & (sdb > tied), 2, 0))

    def game_tables(self, home, away, neutral):
        """Cumulative outcome probabilities over the state grid for both offenses."""
        T = len(self.teams)
        hi, ai = self.teams.get_loc(home), self.teams.get_loc(away)
        nsd = self.cfg.nsd
        g = np.meshgrid(np.arange(2), np.arange(NYL), np.arange(NTB), np.arange(NPER),
                        np.arange(nsd), indexing="ij")
        pos, ylb, tb, per, sdb = [x.ravel() for x in g]
        off = np.where(pos == 0, hi, ai)
        de = np.where(pos == 0, ai, hi)
        hadv = np.where(neutral, 0, np.where(pos == 0, 1, -1))
        div = None
        if self.cfg.division:
            hf, af = float(self.fbs.get(home, 0)), float(self.fbs.get(away, 0))
            div = (np.where(pos == 0, hf, af), np.where(pos == 0, af, hf))
        X = _features(off, de, hadv, ylb * 5 + 2.5, tb, per, sdb, self.variant, T, self.cfg, div)
        P = np.zeros((len(pos), NO))
        P[:, self.classes] = self.clf.predict_proba(X)
        cum = np.cumsum(P, 1)
        cum[:, -1] = 1.0
        pace = np.exp(np.array([self.pace_off[hi] + self.pace_def[ai],
                                self.pace_off[ai] + self.pace_def[hi]]))
        return cum.reshape(2, NYL, NTB, NPER, nsd, NO), pace


def ot_rule(season, gtype):
    post = gtype != "REG"
    if post:
        return ("both" if season >= 2022 else "modified" if season >= 2010 else "sudden"), np.inf
    rule = "both" if season >= 2025 else "modified" if season >= 2012 else "sudden"
    return rule, (600.0 if season >= 2017 else 900.0)


def simulate(model: DriveModel, home, away, neutral, season, gtype, N=N_SIMS, rng=None):
    """Simulate N games. Returns (home_points, away_points, went_to_ot) arrays."""
    rng = rng or np.random.default_rng(0)
    cfg = model.cfg
    cum, pace = model.game_tables(home, away, neutral)
    edges, tied_b = cfg.edges, cfg.tied
    score = np.zeros((N, 2))
    ar = np.arange(N)
    recv0 = rng.integers(0, 2, N)

    def step(idx, pos, yl, clock, half, ot=False):
        n = len(idx)
        tb = tb_of(clock)
        per = np.zeros(n, int) if ot else per_of(np.full(n, half), clock)
        sd = score[idx, pos] - score[idx, 1 - pos]
        sdb = np.full(n, tied_b) if ot else sd_of(sd, edges)
        ylb = np.clip((yl // 5).astype(int), 0, NYL - 1)
        c = cum[pos, ylb, tb, per, sdb]
        out = (rng.random(n)[:, None] > c).sum(1)
        out = np.minimum(out, NO - 1)
        sg = model._sg(per, sdb)
        dur = model.D[out, tb, sg, rng.integers(0, Q, n)] * pace[pos]
        td = out == O["TD"]
        otd = out == O["OTD"]
        pts_td = rng.choice([6, 7, 8], n, p=model.td_p)
        pts_otd = rng.choice([6, 7, 8], n, p=model.otd_p)
        score[idx[td], pos[td]] += pts_td[td]
        fg = out == O["FG"]
        score[idx[fg], pos[fg]] += 3
        score[idx[otd], 1 - pos[otd]] += pts_otd[otd]
        sf = out == O["SAF"]
        score[idx[sf], 1 - pos[sf]] += 2
        sb = np.clip((yl // 20).astype(int), 0, 4)
        yl_new = model.K[out, sb, rng.integers(0, Q, n)]
        pos_new = np.where(otd, pos, 1 - pos)
        clock_new = np.where(out == O["EOH"], 0.0, clock - dur)
        return out, pos_new, yl_new, clock_new

    for half in (0, 1):
        idx = ar.copy()
        pos = recv0 if half == 0 else 1 - recv0
        yl = model.ko[rng.integers(0, Q, N)]
        clock = np.full(N, 1800.0)
        for _ in range(40 if cfg.ot == "nfl" else 45):
            if len(idx) == 0:
                break
            _, pos, yl, clock = step(idx, pos, yl, clock, half)
            alive = clock > 0
            idx, pos, yl, clock = idx[alive], pos[alive], yl[alive], clock[alive]
    if cfg.ot == "cfb":
        return (*_cfb_overtime(model, cum, score, rng, N),)
    rule, otclock = ot_rule(season, gtype)
    tied = np.where(score[:, 0] == score[:, 1])[0]
    went_ot = np.zeros(N, bool)
    went_ot[tied] = True
    if len(tied):
        idx = tied
        pos = rng.integers(0, 2, len(idx))
        yl = model.ko[rng.integers(0, Q, len(idx))]
        clock = np.full(len(idx), min(otclock, 1e6))
        ndr = np.zeros(len(idx), int)
        for _ in range(16):
            if len(idx) == 0:
                break
            before = score[idx].copy()
            cl_in = clock if np.isfinite(otclock) else np.full(len(idx), 900.0)
            out, pos, yl, clock = step(idx, pos, yl, cl_in, 1, ot=True)
            ndr += 1
            after = score[idx]
            scored = (after != before).any(1)
            differ = after[:, 0] != after[:, 1]
            if rule == "sudden":
                end = scored
            elif rule == "modified":
                end = np.where(ndr == 1, np.isin(out, [O["TD"], O["OTD"], O["SAF"]]), differ)
            else:
                end = (ndr >= 2) & differ
            if np.isfinite(otclock):
                end = end | (clock <= 0)
            keep = ~end
            idx, pos, yl, clock, ndr = idx[keep], pos[keep], yl[keep], clock[keep], ndr[keep]
        if len(idx) and not np.isfinite(otclock):  # playoff: coin-flip the rare leftovers
            win = rng.integers(0, 2, len(idx))
            score[idx, win] += 3
    return score[:, 0], score[:, 1], went_ot


def _cfb_overtime(model: DriveModel, cum, score, rng, N):
    """College overtime, 2021+ rules: each round both teams get one possession from the
    opponent 25; from OT2 a touchdown must go for two; from OT3 each round is a single
    two-point try per team; a defensive score ends it. (research cfb_drive_sim/sim_core.py)"""
    cfg = model.cfg
    tied = np.where(score[:, 0] == score[:, 1])[0]
    went_ot = np.zeros(N, bool)
    went_ot[tied] = True
    idx = tied
    first = rng.integers(0, 2, len(idx))
    ylb25 = 25 // 5
    for rnd in range(1, 9):
        if len(idx) == 0:
            break
        done = np.zeros(len(idx), bool)
        for k in range(2):
            pos = first if (k == 0) == (rnd % 2 == 1) else 1 - first
            live = ~done
            if rnd >= 3:
                make = rng.random(len(idx)) < cfg.p2pt
                score[idx[live & make], pos[live & make]] += 2
                continue
            c = cum[pos, ylb25, 3, 0, cfg.tied]
            out = np.minimum((rng.random(len(idx))[:, None] > c).sum(1), NO - 1)
            td = live & (out == O["TD"])
            if rnd == 1:
                add = np.where(td, rng.choice([6, 7, 8], len(idx), p=model.td_p), 0)
            else:
                add = np.where(td, 6 + 2 * (rng.random(len(idx)) < cfg.p2pt), 0)
            add = add + np.where(live & (out == O["FG"]), 3, 0)
            score[idx, pos] += add
            dsc = live & np.isin(out, [O["OTD"], O["SAF"]])
            score[idx[dsc], 1 - pos[dsc]] += np.where(out[dsc] == O["OTD"], 6, 2)
            done |= dsc
        end = done | (score[idx, 0] != score[idx, 1])
        idx, first = idx[~end], first[~end]
    if len(idx):
        w = rng.integers(0, 2, len(idx))
        score[idx, w] += 2
    return score[:, 0], score[:, 1], went_ot


def game_rng(game_id: str, seed: int = SEED) -> np.random.Generator:
    """Per-game stream, identical to the research runner's."""
    return np.random.default_rng([seed, zlib.crc32(str(game_id).encode())])


def fit_week(drives: pd.DataFrame, games: pd.DataFrame, season: int, week: int,
             teams: pd.Index | None = None, cfg: SimConfig = NFL,
             fbs: dict | None = None) -> DriveModel:
    """Fit on drives in the fit_seasons before (season, week), strictly earlier weeks."""
    d = drives if "t" in drives.columns else prep_drives(drives, games, cfg)
    t_now = time_index(season, week)
    tr = d[(d.t < t_now) & (d.t >= t_now - cfg.fit_seasons * WPS)]
    if len(tr) < cfg.min_drives:
        raise RuntimeError(f"drive_sim: only {len(tr)} drives before {season} week {week}")
    if teams is None:
        teams = pd.Index(sorted(set(tr.posteam) | set(tr.defteam)))
    return DriveModel(cfg=cfg).fit(tr, t_now, teams, fbs)


# ---------------------------------------------------------------- summaries
def calibrate(p_raw, cfg: SimConfig = NFL):
    p = np.clip(np.asarray(p_raw, float), 1e-4, 1 - 1e-4)
    platt = cfg.platt
    z = platt["intercept"] + platt["coef"] * np.log(p / (1 - p))
    return 1 / (1 + np.exp(-z))


def dist(x: np.ndarray) -> dict:
    """Contract `Dist` for an integer-valued sample: integer percentiles (inverse CDF)."""
    x = np.asarray(x, float)
    q = np.quantile(x, [0.05, 0.25, 0.5, 0.75, 0.95], method="inverted_cdf")
    return {"mean": float(x.mean()), "sd": float(x.std()),
            "p05": int(q[0]), "p25": int(q[1]), "p50": int(q[2]), "p75": int(q[3]),
            "p95": int(q[4]), "min": int(x.min()), "max": int(x.max()), "n": int(len(x))}


def tilt(P: np.ndarray, grid: np.ndarray, target: float) -> np.ndarray:
    """Exponential tilt p'(k) ~ p(k) exp(lam k) moving the mean to `target` while keeping
    the support, so key numbers stay where they are (research eval_sim.tilt, one row)."""
    g = grid.astype(float)
    lo, hi = -1.0, 1.0
    for _ in range(50):
        lam = (lo + hi) / 2
        W = P * np.exp(lam * (g - g.mean()))
        if (W * g).sum() / W.sum() > target:
            hi = lam
        else:
            lo = lam
    W = P * np.exp(((lo + hi) / 2) * (g - g.mean()))
    return W / W.sum()


def _line_key(x: float) -> str:
    return f"{x:g}"


def summarize_game(hs: np.ndarray, aw: np.ndarray, went_ot: np.ndarray,
                   spread_line: float | None = None, total_line: float | None = None,
                   cfg: SimConfig = NFL) -> dict:
    """Per-game outputs. All Dists and probabilities are the RAW simulation, except
    margin_exact, which is the sim shape tilted to the spread when a spread is known —
    the one use the research validated (exact-margin log score -0.107 vs a normal at the
    spread on 2025). `margin_exact_basis` says which one a row carries."""
    mg = hs - aw
    tot = hs + aw
    pw, pl = float((mg > 0).mean()), float((mg < 0).mean())
    p_raw = pw / (pw + pl) if pw + pl > 0 else 0.5
    out = {"home": dist(hs), "away": dist(aw), "total": dist(tot), "margin": dist(mg),
           "p_raw": p_raw, "p_home_win": pw, "p_tie": float((mg == 0).mean()),
           "p_ot": float(went_ot.mean())}
    has_spread = spread_line is not None and np.isfinite(spread_line)
    has_total = total_line is not None and np.isfinite(total_line)
    out["p_home_cover"] = float((mg > spread_line).mean()) if has_spread else None
    centre = float(total_line) if has_total else np.floor(tot.mean()) + 0.5
    lines = centre + np.arange(-OVER_LINE_HALF_WIDTH, OVER_LINE_HALF_WIDTH + 1)
    out["p_over_by_line"] = json.dumps({_line_key(L): round(float((tot > L).mean()), 5)
                                        for L in lines})
    lo, hi = cfg.margin_lo, cfg.margin_hi
    grid = cfg.margin_grid
    H = np.bincount(np.clip(mg, lo, hi).astype(int) - lo, minlength=len(grid)).astype(float)
    P = H / H.sum()
    if has_spread:
        P = tilt(0.995 * P + 0.005 / len(P), grid, float(spread_line))
        basis = "sim_shape_at_spread"
    else:
        basis = "raw_sim"
    out["margin_exact"] = json.dumps({str(k): round(float(P[k - lo]), 5) for k in MARGIN_EXACT_K})
    out["margin_exact_basis"] = basis
    return out


def _flat(prefix: str, d: dict) -> dict:
    return {f"{prefix}_{k}": v for k, v in d.items() if k != "n"}


def _game_date(v):
    t = pd.Timestamp(v)
    return t.date()


def simulate_slate(slate: pd.DataFrame, drives: pd.DataFrame, games: pd.DataFrame,
                   season: int, week: int, n: int | None = None, seed: int = SEED,
                   now: datetime | None = None, cfg: SimConfig = NFL,
                   fbs: dict | None = None,
                   teams: pd.Index | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Fit once for the week and simulate every game in `slate` (already pregame-filtered).

    slate needs game_id, season, week, gameday, home_team, away_team, location,
    game_type, spread_line, total_line. Returns (distribution rows, prediction rows, timing).
    """
    import time

    now = now or datetime.now(timezone.utc)
    n = cfg.n_sims if n is None else n
    t0 = time.time()
    model = fit_week(drives, games, season, week, teams=teams, cfg=cfg, fbs=fbs)
    fit_s = time.time() - t0
    dist_rows, pred_rows = [], []
    t1 = time.time()
    for r in slate.itertuples(index=False):
        home = cfg.relocated.get(r.home_team, r.home_team)
        away = cfg.relocated.get(r.away_team, r.away_team)
        if home not in model.teams or away not in model.teams:
            logger.warning("drive_sim: no drives for %s or %s, skipping %s", home, away, r.game_id)
            continue
        neutral = (bool(getattr(r, "neutral_site", 0) or 0) if cfg.name == "cfb"
                   else getattr(r, "location", "Home") == "Neutral")
        hs, aw, ot = simulate(model, home, away, neutral,
                              int(r.season), getattr(r, "game_type", "REG"), N=n,
                              rng=game_rng(r.game_id, seed))
        spread = pd.to_numeric(getattr(r, "spread_line", None), errors="coerce")
        total = pd.to_numeric(getattr(r, "total_line", None), errors="coerce")
        spread = None if pd.isna(spread) else float(spread)
        total = None if pd.isna(total) else float(total)
        s = summarize_game(hs, aw, ot, spread, total, cfg=cfg)
        gd = _game_date(r.gameday)
        ident = dict(game_id=str(r.game_id), game_date=gd, season=int(r.season), week=int(r.week),
                     predicted_at=now, model_version=cfg.model_version)
        dist_rows.append(dict(
            **ident, n_sims=int(n), home_team=r.home_team, away_team=r.away_team,
            **_flat("home_points", s["home"]), **_flat("away_points", s["away"]),
            **_flat("total", s["total"]), **_flat("margin", s["margin"]),
            p_home_win=s["p_home_win"], p_ot=s["p_ot"], p_tie=s["p_tie"],
            p_home_cover=s["p_home_cover"], p_over_by_line=s["p_over_by_line"],
            margin_exact=s["margin_exact"], margin_exact_basis=s["margin_exact_basis"],
            spread_line=spread, total_line=total))
        p = float(calibrate(s["p_raw"], cfg))
        vegas = None if spread is None else float(1 / (1 + np.exp(-spread / 7.0)))
        extra = {}
        if cfg.name == "cfb":
            # CFB: Phi(spread / 14.92), the market sigma fit on 2022 in the research.
            from math import erf, sqrt

            vegas = None if spread is None else float(0.5 * (1 + erf(spread / 14.92 / sqrt(2))))
            extra = {"division": getattr(r, "division", None)}
        # game_predictions keeps game_date as TIMESTAMP; this table mirrors it.
        pred_rows.append(dict(
            **{**ident, "game_date": pd.Timestamp(gd).tz_localize("UTC")}, game_pk=abs(zlib.crc32(str(r.game_id).encode())),
            home_team_id=r.home_team, away_team_id=r.away_team,
            home_team_name=r.home_team, away_team_name=r.away_team,
            home_win_probability=p, away_win_probability=1 - p,
            predicted_winner=r.home_team if p > 0.5 else r.away_team,
            confidence_tier=_tier(p), sim_p_raw=s["p_raw"],
            predicted_home_score=s["home"]["mean"], predicted_away_score=s["away"]["mean"],
            predicted_home_margin=s["margin"]["mean"], predicted_total=s["total"]["mean"],
            n_sims=int(n), spread_line=spread, total_line=total,
            vegas_implied_home_prob=vegas,
            model_vs_vegas_edge=None if vegas is None else p - vegas, **extra))
        if extra:
            dist_rows[-1].update(extra)
    timing = {"fit_s": round(fit_s, 2), "sim_ms_per_game":
              round(1000 * (time.time() - t1) / max(len(dist_rows), 1), 1),
              "drives_in_window": int(((drives.t if "t" in drives else drives.season * WPS + drives.week)
                                       >= time_index(season, week) - cfg.fit_seasons * WPS).sum())}
    return pd.DataFrame(dist_rows), pd.DataFrame(pred_rows), timing


def _tier(p: float) -> str:
    # Same thresholds as predict_nfl.CONFIDENCE_TIERS (football spreads wider than MLB).
    edge = abs(p - 0.5) + 0.5
    return "high" if edge >= 0.72 else ("medium" if edge >= 0.60 else "low")
