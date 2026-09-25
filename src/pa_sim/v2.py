"""PA simulator v2 (EXPERIMENTAL -- research only, not wired into any pipeline).

Differences from v1 (predict.py / sim.py), each switchable so it can be tested alone:

  rates     time-decayed counts (exp(-age/tau)) instead of equal-weight all-history;
            platoon on BOTH sides with a league (stand x throws) baseline; optional
            Statcast x-outcome substitution for balls in play; optional park
            neutralisation of the counts.
  context   home-field and per-outcome park factors applied inside the rates;
            optional temperature effect on HR / extra-base hits.
  usage     dynamic starter removal (per-pitcher pitch-count hazard, fitted) instead of
            a league per-inning blend; times-through-order multipliers; bullpen
            composite from RECENT appearances.
  chain     empirical base-out transitions (baserunning, SB/WP, DP, sac flies, errors)
            estimated from Savant state, replacing the fixed advancement table and
            the global alpha fudge.
  outputs   full-game, F5, run margin, starter K / outs, batter H / HR / K per slot.

The simulator is batched over games x episodes and advanced PA by PA in numpy.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np, pandas as pd

CLASSES = ["K", "BB", "1B", "2B", "3B", "HR", "E", "GO", "AO"]
NC = len(CLASSES)
K_, BB_, S1, S2, S3, HR_, E_, GO, AO = range(NC)
HITS = [S1, S2, S3, HR_]
BIP = [S1, S2, S3, HR_, E_, GO, AO]
DAY0 = pd.Timestamp("2015-01-01")


@dataclass
class Config:
    tau: float = 365.0            # days; np.inf = equal weight (v1)
    tau_league: float = 60.0
    platoon: str = "both"         # "bat" (v1) | "both"
    home: bool = True             # home-field inside the rates
    park: str = "class"           # "none" | "scalar" | "class"
    park_neutral: bool = False    # neutralise player counts by park
    xw: float = 0.0               # weight on x-outcome (EV/LA) substitution for BIP
    tto: bool = True
    hook: str = "hazard"          # "curve" | "hazard"
    bullpen: str = "recent"       # "all" (v1) | "recent"
    bp_days: int = 30
    trans: str = "empirical"      # "fixed" (v1) | "empirical"
    weather: bool = False
    k_hand_mult: float = 1.0      # shrink split toward platoon-adjusted overall
    alpha: float = 1.0            # v1-style global hit-odds scale (v1 fit 1.155 on 2026)
    env: bool = False             # rescale by short-window league rates (measured: double-counts)
    tag: str = ""


# --------------------------------------------------------------------------- data

class Data:
    """Holds the PA table as flat numpy arrays keyed to integer player indices."""

    def __init__(self, pa: pd.DataFrame, venue_of_game: dict, temp_of_game: dict | None = None,
                 xtable: np.ndarray | None = None):
        pa = pa.sort_values(["game_date", "game_pk"]).reset_index(drop=True)
        self.bat_ids, self.bi = np.unique(pa.batter.values, return_inverse=True)
        self.pit_ids, self.pi = np.unique(pa.pitcher.values, return_inverse=True)
        self.bmap = {int(b): i for i, b in enumerate(self.bat_ids)}
        self.pmap = {int(p): i for i, p in enumerate(self.pit_ids)}
        self.day = ((pa.game_date - DAY0).dt.days).values.astype(np.int32)
        self.game = pa.game_pk.values
        self.stand = pa.stand.values.astype(np.int8)
        self.throws = pa.p_throws.values.astype(np.int8)
        self.cls = pa.cls.values.astype(np.int8)
        self.home_bat = (~pa.top.values.astype(bool))
        self.is_sp = pa.is_starter.values.astype(bool)
        self.inning = pa.inning.values
        self.npitch = pa.n_pitches.fillna(3.8).values.astype(np.float32)
        self.year = pa.game_year.values
        self.home_team = pa.home_team.astype(str).values
        self.away_team = pa.away_team.astype(str).values
        self.def_team = np.where(pa.top.values, self.home_team, self.away_team)
        self.tto = pa.n_thruorder_pitcher.fillna(0).values.astype(np.int8) if "n_thruorder_pitcher" in pa else None
        vv = np.array([venue_of_game.get(int(g), -1) for g in self.game])
        self.venue_ids, self.vi = np.unique(vv, return_inverse=True)
        self.vmap = {int(v): i for i, v in enumerate(self.venue_ids)}
        self.temp = None
        if temp_of_game is not None:
            self.temp = np.array([temp_of_game.get(int(g), np.nan) for g in self.game], dtype=float)
        # per-PA fractional count vectors (one-hot, or x-outcome mixed for BIP)
        self.onehot = np.eye(NC, dtype=np.float32)[self.cls]
        self.xvec = None
        if xtable is not None:
            ls = pa.launch_speed.values.astype(float); la = pa.launch_angle.values.astype(float)
            ok = np.isfinite(ls) & np.isfinite(la) & np.isin(self.cls, BIP)
            xv = self.onehot.copy()
            eb, ab = _xbin(ls[ok], la[ok])
            xv[ok] = xtable[eb, ab]
            self.xvec = xv
        # starter pitch totals per start (for the hook model's expected-pitches feature)
        sp = pa[pa.is_starter].groupby(["game_pk", "pitcher"]).agg(
            p=("n_pitches", "sum"), d=("game_date", "first")).reset_index()
        sp["day"] = (sp.d - DAY0).dt.days
        sp["p"] = sp.p.astype(float)
        sp = sp.sort_values(["day", "game_pk"]).reset_index(drop=True)
        # expected pitches BEFORE each start: shrunk mean of that pitcher's previous 10
        g = sp.groupby("pitcher").p
        s10 = g.transform(lambda x: x.shift(1).rolling(10, min_periods=1).sum()).fillna(0)
        n10 = g.transform(lambda x: x.shift(1).rolling(10, min_periods=1).count()).fillna(0)
        yr = pd.to_datetime(sp.d).dt.year
        lg = sp.groupby(yr).p.mean().shift(1).bfill()
        lgv = yr.map(lg).values
        sp["exp_p"] = (s10 + 4.0 * lgv) / (n10 + 4.0)
        self.starts = sp


def _xbin(ls, la):
    eb = np.clip(((ls - 40) / 4).astype(int), 0, 17)
    ab = np.clip(((la + 60) / 6).astype(int), 0, 24)
    return eb, ab


def build_xtable(pa: pd.DataFrame) -> np.ndarray:
    """P(class | EV bin, LA bin) over balls in play, smoothed toward the LA-bin marginal."""
    d = pa[pa.cls.isin(BIP) & pa.launch_speed.notna() & pa.launch_angle.notna()]
    eb, ab = _xbin(d.launch_speed.values, d.launch_angle.values)
    cnt = np.zeros((18, 25, NC))
    np.add.at(cnt, (eb, ab, d.cls.values), 1.0)
    marg = cnt.sum(0, keepdims=True); marg = marg / np.maximum(marg.sum(-1, keepdims=True), 1)
    tab = (cnt + 20 * marg) / (cnt.sum(-1, keepdims=True) + 20)
    return tab.astype(np.float32)


# --------------------------------------------------------------------------- rates

def _estimate_k(c, n, mean, nmin=50.0):
    m = n >= nmin
    if m.sum() < 30:
        return 300.0
    p = c[m] / n[m]; w = n[m] / n[m].sum()
    vo = float(np.sum(w * (p - mean) ** 2)); vb = float(np.sum(w * mean * (1 - mean) / n[m]))
    pv = vo - vb
    if pv <= 1e-9:
        return 5000.0
    return float(np.clip(mean * (1 - mean) / pv - 1, 20.0, 5000.0))


class Rates:
    """Walk-forward decayed counts; call advance(day) then snapshot()."""

    def __init__(self, D: Data, cfg: Config):
        self.D, self.cfg = D, cfg
        nb, npi, nv = len(D.bat_ids), len(D.pit_ids), len(D.venue_ids)
        self.B = np.zeros((nb, 2, NC)); self.P = np.zeros((npi, 2, NC))
        self.Lc = np.zeros((2, 2, NC))        # league, stand x throws (long tau)
        self.Ls = np.zeros(NC)                # league, short tau (run environment)
        self.H = np.zeros((2, NC))            # by home_bat
        self.V = np.zeros((nv, NC))           # at venue
        self.TR = {}                          # team road counts
        self.cur = None; self.ptr = 0
        self.vec = D.xvec if (cfg.xw > 0 and D.xvec is not None) else None
        self.pf = np.ones((nv, NC))

    def advance(self, day: int):
        """Include all PAs with D.day < day, decayed to `day`."""
        D, c = self.D, self.cfg
        end = int(np.searchsorted(D.day, day, side="left"))
        if self.cur is not None:
            dt = day - self.cur
            f = np.exp(-dt / c.tau) if np.isfinite(c.tau) else 1.0
            fl = np.exp(-dt / c.tau_league)
            fp = np.exp(-dt / (3 * 365.0))
            self.B *= f; self.P *= f; self.Lc *= f; self.H *= fp; self.Ls *= fl; self.V *= fp
            for k in self.TR: self.TR[k] *= fp
        s = slice(self.ptr, end)
        if end > self.ptr:
            age = day - D.day[s]
            w = np.exp(-age / c.tau) if np.isfinite(c.tau) else np.ones(end - self.ptr)
            wl = np.exp(-age / c.tau_league); wp = np.exp(-age / (3 * 365.0))
            v = D.onehot[s].astype(float)
            if self.vec is not None:
                v = (1 - c.xw) * v + c.xw * self.vec[s]
            if c.park_neutral and c.park != "none":
                v = v / self.pf[D.vi[s]]; v = v / v.sum(1, keepdims=True)
            wv = v * w[:, None]
            np.add.at(self.B, (D.bi[s], D.throws[s]), wv)
            np.add.at(self.P, (D.pi[s], D.stand[s]), wv)
            oh = D.onehot[s]
            np.add.at(self.Lc, (D.stand[s], D.throws[s]), oh * w[:, None])
            self.Ls += (oh * wl[:, None]).sum(0)
            np.add.at(self.H, D.home_bat[s].astype(int), oh * wp[:, None])
            np.add.at(self.V, D.vi[s], oh * wp[:, None])
            # road counts per team (both batting and pitching in road games)
            road_bat = D.home_bat[s] == False
            df = pd.DataFrame(oh * wp[:, None])
            for team_arr in (D.away_team[s],):
                g = df.groupby(team_arr).sum()
                for t, row in zip(g.index, g.values):
                    self.TR[t] = self.TR.get(t, 0) + row
        self.ptr = end; self.cur = day

    def snapshot(self, venue_home_team: dict):
        """Shrunk rate tables as of the current day."""
        c = self.cfg
        L = self.Lc.sum((0, 1)); L = L / L.sum()
        Lc = self.Lc / self.Lc.sum(-1, keepdims=True)
        env = self.Ls / self.Ls.sum()
        out = {}
        for role, A in (("bat", self.B), ("pit", self.P)):
            tot = A.sum(1); n = tot.sum(1)
            k = np.array([_estimate_k(tot[:, j], n, L[j]) for j in range(NC)])
            ov = (tot + k * L) / (n[:, None] + k)
            ov /= ov.sum(1, keepdims=True)
            # split prior: overall rate x league platoon ratio for that side
            nh = A.sum(2)                                        # (players, 2)
            if role == "bat":
                # batter vs throws t; league rate by throws, marginalised over stand
                Lt = self.Lc.sum(0); Lt = Lt / Lt.sum(-1, keepdims=True)   # (2 throws, NC)
            else:
                Lt = self.Lc.sum(1); Lt = Lt / Lt.sum(-1, keepdims=True)   # (2 stand, NC)
            mix = nh / np.maximum(nh.sum(1, keepdims=True), 1e-9)
            mix = np.where(nh.sum(1, keepdims=True) > 0, mix, np.array([0.28, 0.72]) if role == "bat" else np.array([0.43, 0.57]))
            Lmix = mix @ Lt                                      # (players, NC)
            prior = ov[:, None, :] * (Lt[None, :, :] / Lmix[:, None, :])
            prior /= prior.sum(-1, keepdims=True)
            kh = k * c.k_hand_mult
            sp = (A + kh * prior) / (nh[:, :, None] + kh)
            sp /= sp.sum(-1, keepdims=True)
            out[role] = dict(ov=ov, split=sp, n=n, k=k)
        out["L"], out["Lc"], out["env"] = L, Lc, env
        Hr = self.H / self.H.sum(-1, keepdims=True)
        out["home_mult"] = np.sqrt(Hr[1] / Hr[0])               # home batters x, away /
        # park factors: rate at venue vs the home team's road rate, moment-shrunk
        pf = np.ones((len(self.D.venue_ids), NC)); nv = self.V.sum(1)
        ratios, ns = [], []
        for vid, vi in self.D.vmap.items():
            t = venue_home_team.get(vid)
            if t is None or t not in self.TR or nv[vi] < 500:
                continue
            road = self.TR[t] / self.TR[t].sum()
            ratios.append((vi, road)); ns.append(nv[vi])
        if ratios:
            vis = np.array([r[0] for r in ratios]); road = np.array([r[1] for r in ratios]); ns = np.array(ns)
            obs = self.V[vis] / ns[:, None]
            m = np.zeros(NC)
            for j in range(NC):
                # between-venue variance of true rate beyond binomial noise (both sides noisy)
                d = obs[:, j] - road[:, j]
                vb = np.mean(road[:, j] * (1 - road[:, j]) / ns) * 2
                tv = max(np.var(d) - vb, 1e-7)
                m[j] = road[:, j].mean() * (1 - road[:, j].mean()) / tv
            shr = (self.V[vis] + m * road) / (ns[:, None] + m)
            pf[vis] = shr / road
            if c.park == "scalar":
                # v1-style: one scalar per park from the hit classes' combined effect
                runw = np.array([0, .3, .45, .75, 1.05, 1.4, .45, -.1, -.1])
                sc = (pf[vis] * road * runw).sum(1) / (road * runw).sum(1)
                pf[vis] = 1.0
                pf[vis[:, None], np.array(HITS)[None, :]] = sc[:, None] ** 0.7
        self.pf = pf
        out["pf"] = pf
        return out


def log5(b, p, L):
    raw = np.clip(b, 1e-9, 1) * np.clip(p, 1e-9, 1) / np.clip(L, 1e-9, 1)
    return raw / raw.sum(-1, keepdims=True)


# --------------------------------------------------------------------------- chain

def fixed_transitions():
    """v1 advancement mapped to 9 classes: E=1B, GO->DP 40% with r1 & <2 outs, AO=out."""
    T = {}
    for b in range(8):
        b1, b2, b3 = b & 1, (b >> 1) & 1, (b >> 2) & 1
        for o in range(3):
            for cl in range(NC):
                res = []
                if cl == BB_:
                    if not b1: res = [(b | 1, o, 0, 1.0)]
                    elif not b2: res = [(1 | 2 | (b3 << 2), o, 0, 1.0)]
                    elif not b3: res = [(7, o, 0, 1.0)]
                    else: res = [(7, o, 1, 1.0)]
                elif cl in (S1, E_): res = [(1 | (b1 << 1) | (b2 << 2), o, b3, 1.0)]
                elif cl == S2: res = [(2 | (b1 << 2), o, b2 + b3, 1.0)]
                elif cl == S3: res = [(4, o, b1 + b2 + b3, 1.0)]
                elif cl == HR_: res = [(0, o, 1 + b1 + b2 + b3, 1.0)]
                elif cl == GO and b1 and o <= 1:
                    res = [(b & ~1, o + 2, 0, 0.4), (b, o + 1, 0, 0.6)]
                else: res = [(b, o + 1, 0, 1.0)]
                T[(b, o, cl)] = [(nb if no < 3 else 0, min(no, 3), r, p) for nb, no, r, p in res]
    return _pack(T)


def empirical_transitions(tr: pd.DataFrame, years, prior_w=20.0):
    """Counts from Savant (bases,outs,cls)->(nb,no,runs) in `years`, smoothed to all years."""
    allc = tr.groupby(["bases", "outs", "cls", "nb", "no", "runs"]).size()
    sub = tr[tr.year.isin(list(years))].groupby(["bases", "outs", "cls", "nb", "no", "runs"]).size()
    T = {}
    for (b, o, cl), g in allc.groupby(level=[0, 1, 2]):
        pa_ = g / g.sum()
        s = sub.reindex(g.index, fill_value=0)
        post = (s + prior_w * pa_) / (s.sum() + prior_w)
        T[(b, o, cl)] = [(int(i[3]) if i[4] < 3 else 0, int(min(i[4], 3)), int(i[5]), float(p))
                         for i, p in post.items() if p > 0]
    for b in range(8):
        for o in range(3):
            for cl in range(NC):
                T.setdefault((b, o, cl), fixed_transitions_dict()[(b, o, cl)])
    return _pack(T)


_FIXED = None
def fixed_transitions_dict():
    global _FIXED
    if _FIXED is None:
        c, nb, no, ra = fixed_transitions()
        _FIXED = {}
        for b in range(8):
            for o in range(3):
                for cl in range(NC):
                    pr = np.diff(np.concatenate([[0], c[b, o, cl]]))
                    _FIXED[(b, o, cl)] = [(int(nb[b, o, cl, i]), int(no[b, o, cl, i]), int(ra[b, o, cl, i]), float(pr[i]))
                                          for i in range(c.shape[-1]) if pr[i] > 0]
    return _FIXED


def _pack(T, K=24):
    cum = np.ones((8, 3, NC, K)); nb = np.zeros((8, 3, NC, K), np.int8)
    no = np.zeros((8, 3, NC, K), np.int8); ra = np.zeros((8, 3, NC, K), np.int8)
    for (b, o, cl), lst in T.items():
        lst = sorted(lst, key=lambda x: -x[3])[:K]
        p = np.array([x[3] for x in lst]); p = p / p.sum()
        n = len(lst)
        cum[b, o, cl, :n] = np.cumsum(p); cum[b, o, cl, n - 1:] = 1.0
        nb[b, o, cl, :n] = [x[0] for x in lst]; no[b, o, cl, :n] = [x[1] for x in lst]
        ra[b, o, cl, :n] = [x[2] for x in lst]
        nb[b, o, cl, n:] = nb[b, o, cl, n - 1]; no[b, o, cl, n:] = no[b, o, cl, n - 1]; ra[b, o, cl, n:] = ra[b, o, cl, n - 1]
    return cum, nb, no, ra


# --------------------------------------------------------------------------- usage

HOOK_FEATS = ["cp", "cp_x", "cp_x2", "inn_start", "bf", "ra", "inn", "tto3", "exp_p", "outs", "on"]


def hook_features(cp, exp_p, inn_start, bf, ra, inn, outs, on):
    x = cp - exp_p
    return np.stack([cp / 100, x / 20, np.maximum(x, 0) ** 2 / 400, inn_start.astype(float),
                     bf / 27, ra / 5, inn / 9, (bf >= 18).astype(float), exp_p / 100,
                     outs / 2, on.astype(float)], -1)


def fit_hook(hook: pd.DataFrame, exp_p_of: np.ndarray):
    from sklearn.linear_model import LogisticRegression
    X = hook_features(hook.cum_p.values.astype(float), exp_p_of, hook.inn_start.values,
                      hook.bf.values.astype(float), hook.ra.values.astype(float),
                      hook.inning.values.astype(float), hook.outs_when_up.values.astype(float),
                      hook.bases.values > 0)
    m = LogisticRegression(C=10.0, max_iter=2000).fit(X, hook.y.values)
    return np.concatenate([m.intercept_, m.coef_[0]])


def expected_pitches(starts: pd.DataFrame, day: int, n_last=10, k=4.0):
    """Per-pitcher trailing mean pitches per start before `day`, shrunk to league."""
    s = starts[starts.day < day]
    s = s[s.day >= day - 400]
    lg = s[s.day >= day - 120].p.mean() if len(s) else 88.0
    if not np.isfinite(lg): lg = s.p.mean() if len(s) else 88.0
    last = s.groupby("pitcher").tail(n_last).groupby("pitcher").p.agg(["sum", "count"])
    ep = (last["sum"] + k * lg) / (last["count"] + k)
    return ep.to_dict(), float(lg)


# --------------------------------------------------------------------------- game tables

@dataclass
class GameSpec:
    h_lineup: list; a_lineup: list; h_sp: int; a_sp: int
    home_team: str; away_team: str; venue: int; sched_inn: int = 9; ghost: bool = False
    temp: float = np.nan


class Engine:
    """Fit on a date, then build per-game probability tables and simulate."""

    def __init__(self, D: Data, cfg: Config, trans_df: pd.DataFrame, hook_df: pd.DataFrame,
                 venue_home_team: dict, tto_mult: np.ndarray | None = None,
                 temp_slope: np.ndarray | None = None):
        self.D, self.cfg = D, cfg
        self.rates = Rates(D, cfg)
        self.trans_df, self.hook_df = trans_df, hook_df
        self.vht = venue_home_team
        self.tto_mult = tto_mult if tto_mult is not None else np.ones((3, NC))
        self.temp_slope = temp_slope
        self._trans_year = None; self._hook_year = None

    def fit(self, date: pd.Timestamp):
        D, c = self.D, self.cfg
        day = int((date - DAY0).days)
        self.rates.advance(day)
        self.R = self.rates.snapshot(self.vht)
        yr = date.year
        if self._trans_year != yr:
            yrs = [y for y in range(yr - 3, yr) if y <= 2025] or [2015]
            self.trans = fixed_transitions() if c.trans == "fixed" else empirical_transitions(self.trans_df, yrs)
            self._trans_year = yr
        # hook
        self.exp_p, self.lg_p = expected_pitches(D.starts, day)
        if c.hook == "hazard" and self._hook_year != yr:
            yrs = [y for y in range(yr - 2, yr) if y <= 2025] or [2015]
            hk = self.hook_df[self.hook_df.game_year.isin(yrs)]
            hk = hk.sample(min(len(hk), 250_000), random_state=0)
            self.hook_beta = fit_hook(hk, hk.exp_p.values.astype(float))
            self._hook_year = yr
        if c.hook == "curve":
            m = (D.day < day) & (D.day >= day - 365) & (D.inning <= 9)
            inn = D.inning[m]; sp = D.is_sp[m]
            curve = np.array([sp[inn == i].mean() for i in range(1, 10)])
            self.curve = np.clip(curve, 1e-3, 1.0)
        # bullpen composites per defensive team, by batter stand
        m = (D.day < day) & ~D.is_sp
        rel = pd.DataFrame(dict(t=D.def_team[m], p=D.pi[m], d=D.day[m]))
        if c.bullpen == "recent":
            # window is relative to each team's most recent game, so it spans the
            # offseason gap instead of going empty in April
            last = rel.groupby("t").d.transform("max")
            rel = rel[rel.d >= last - c.bp_days]
        self.bp = {}
        Psp = self.R["pit"]["split"]
        for t, g in rel.groupby("t"):
            vc = g.p.value_counts()
            if c.bullpen == "all":
                vc = vc.head(12)
            w = vc.values.astype(float); w /= w.sum()
            self.bp[t] = np.einsum("i,ijk->jk", w, Psp[vc.index.values])
        lastd = D.day[D.day < day].max() if (D.day < day).any() else day
        mm = (D.day < day) & (D.day >= lastd - 60) & ~D.is_sp
        lr = np.bincount(D.cls[mm], minlength=NC).astype(float)
        self.bp_league = np.tile(lr / lr.sum(), (2, 1))
        self.day = day
        return self

    def _pit(self, pid, stand):
        i = self.D.pmap.get(int(pid))
        if i is None:
            return self.R["L"] * 0 + self.R["Lc"][stand].mean(0)
        if self.cfg.platoon == "both":
            return self.R["pit"]["split"][i, stand]
        return self.R["pit"]["ov"][i]

    def _bat(self, bid, throws):
        i = self.D.bmap.get(int(bid))
        if i is None:
            # unknown batter: replacement level ~ league minus a bit
            return self.R["Lc"][1, throws]
        return self.R["bat"]["split"][i, throws]

    def _bstand(self, bid, throws):
        """Stand the batter uses against this hand (switch hitters)."""
        i = self.D.bmap.get(int(bid))
        if i is None: return 1
        return self._stand_tab.get((i, throws), 1)

    def tables(self, g: GameSpec):
        """(2 sides, 9 slots, 4 ptypes [SP tto1..3, pen], NC) probabilities."""
        c, R = self.cfg, self.R
        env_adj = R["env"] / R["L"] if c.env else np.ones(NC)
        out = np.zeros((2, 9, 4, NC))
        pf = R["pf"][self.D.vmap[g.venue]] if (c.park != "none" and g.venue in self.D.vmap) else np.ones(NC)
        tadj = np.ones(NC)
        if c.weather and self.temp_slope is not None and np.isfinite(g.temp):
            tadj = np.exp(self.temp_slope * (g.temp - getattr(self, 'temp_center', 72.0)))
        for side in (0, 1):                       # 0 = away batting, 1 = home batting
            lineup = g.a_lineup if side == 0 else g.h_lineup
            sp = g.h_sp if side == 0 else g.a_sp
            pteam = g.home_team if side == 0 else g.away_team
            th = self.hand.get(int(sp), 1)
            for s, b in enumerate(lineup):
                bt = self._bat(b, th)
                st = self._bstand(b, th)
                if c.platoon == "both":
                    Lref = R["Lc"][st, th]
                else:
                    Lref = R["L"]
                bpen = self.bp.get(pteam, self.bp_league)[st] if c.platoon == "both" else \
                    self.bp.get(pteam, self.bp_league).mean(0)
                base_sp = log5(bt, self._pit(sp, st), Lref)
                base_bp = log5(self._bat(b, 1), bpen, R["Lc"][st, 1] if c.platoon == "both" else R["L"])
                for pt in range(4):
                    p = base_sp if pt < 3 else base_bp
                    p = p * env_adj * pf * tadj
                    if c.alpha != 1.0:
                        p = p.copy(); p[HITS] *= c.alpha
                    if c.home:
                        p = p * (R["home_mult"] if side == 1 else 1 / R["home_mult"])
                    if c.tto and pt < 3:
                        p = p * self.tto_mult[pt]
                    out[side, s, pt] = p / p.sum()
        return out

    def prepare(self, pa_hand: dict, stand_tab: dict):
        self.hand = pa_hand; self._stand_tab = stand_tab
        return self


# --------------------------------------------------------------------------- simulator

PITCH_MEAN = np.array([4.85, 5.40, 3.38, 3.37, 3.45, 3.35, 3.47, 3.36, 3.40])


def simulate(engine: Engine, games: list[GameSpec], n: int = 2000, seed: int = 0, stats: bool = True):
    """Batched PA-by-PA Monte Carlo. Returns per-game summary dict of arrays."""
    rng = np.random.default_rng(seed)
    c = engine.cfg
    G = len(games)
    P = np.stack([engine.tables(g) for g in games])           # (G,2,9,4,NC)
    CUM = np.cumsum(P, -1); CUM[..., -1] = 1.0
    tcum, tnb, tno, tra = engine.trans
    L = G * n
    gi = np.repeat(np.arange(G), n)
    sched = np.array([g.sched_inn for g in games])[gi]
    ghost = np.array([g.ghost for g in games])[gi]
    # hook
    if c.hook == "hazard":
        beta = engine.hook_beta
        ep = np.array([[engine.exp_p.get(int(g.a_sp), engine.lg_p), engine.exp_p.get(int(g.h_sp), engine.lg_p)]
                       for g in games])[gi]              # [:,0] away SP (pitches when home bats)
    else:
        curve = engine.curve
    half = np.zeros(L, np.int8)          # 0 top (away bats), 1 bottom
    inn = np.ones(L, np.int16)
    outs = np.zeros(L, np.int8); bases = np.zeros(L, np.int8)
    score = np.zeros((L, 2), np.int16)   # [:,0] away, [:,1] home
    slot = np.zeros((L, 2), np.int8)
    sp_in = np.ones((L, 2), bool)        # indexed by DEFENSIVE side: 0 away pitching, 1 home pitching
    pc = np.zeros((L, 2), np.float32); bf = np.zeros((L, 2), np.int16); ra = np.zeros((L, 2), np.int16)
    inn_start = np.ones(L, bool)
    done = np.zeros(L, bool)
    f5 = np.full((L, 2), -1, np.int16)
    if stats:
        sp_k = np.zeros((L, 2), np.int8); sp_outs = np.zeros((L, 2), np.int8)
        bh = np.zeros((L, 2, 9), np.int8); bhr = np.zeros((L, 2, 9), np.int8)
        bk = np.zeros((L, 2, 9), np.int8); bpa = np.zeros((L, 2, 9), np.int8)
    act = np.arange(L)
    step = 0
    while act.size and step < 400:
        step += 1
        a = act
        bs = half[a].astype(np.int64)             # batting side (0 away,1 home)
        ds = 1 - bs                               # defensive side
        sl = slot[a, bs].astype(np.int64)
        # starter removal decision before the PA
        spin = sp_in[a, ds]
        if c.hook == "hazard":
            X = hook_features(pc[a, ds].astype(float), ep[a, ds], inn_start[a], bf[a, ds].astype(float),
                              ra[a, ds].astype(float), inn[a].astype(float), outs[a].astype(float),
                              bases[a] > 0)
            hz = 1 / (1 + np.exp(-(beta[0] + X @ beta[1:])))
            hz = np.where(inn[a] == 1, np.minimum(hz, 0.002), hz)
        else:
            ii = np.clip(inn[a] - 1, 0, 8)
            prev = np.where(ii > 0, curve[np.maximum(ii - 1, 0)], 1.0)
            hz = np.where(inn_start[a], 1 - np.clip(curve[ii] / prev, 0, 1), 0.0)
        pull = spin & (rng.random(a.size) < hz)
        sp_in[a[pull], ds[pull]] = False
        spin = spin & ~pull
        tto = np.minimum(bf[a, ds] // 9, 2)
        pt = np.where(spin, tto, 3).astype(np.int64)
        cum = CUM[gi[a], bs, sl, pt]                                   # (m, NC)
        u = rng.random(a.size)
        ev = (u[:, None] >= cum).sum(1).clip(0, NC - 1)
        # transition
        b0, o0 = bases[a].astype(np.int64), outs[a].astype(np.int64)
        tc = tcum[b0, o0, ev]
        k = (rng.random(a.size)[:, None] >= tc).sum(1).clip(0, tc.shape[1] - 1)
        nb = tnb[b0, o0, ev, k]; no = tno[b0, o0, ev, k]; r = tra[b0, o0, ev, k].astype(np.int16)
        # walk-off cap: runs beyond the winning run don't count (except HR); approximate by full r
        score[a, bs] += r
        # a non-HR walk-off only scores the winning run
        wo = (bs == 1) & (inn[a] >= sched[a]) & (score[a, 1] > score[a, 0]) & (ev != HR_)
        score[a[wo], 1] = np.minimum(score[a[wo], 1], score[a[wo], 0] + 1)
        # pitcher accounting
        npit = np.maximum(1, rng.poisson(PITCH_MEAN[ev] - 1) + 1)
        pc[a, ds] += np.where(spin, npit, 0); bf[a, ds] += spin; ra[a, ds] += np.where(spin, r, 0)
        if stats:
            sp_k[a, ds] += (spin & (ev == K_))
            sp_outs[a, ds] += np.where(spin, np.minimum(no, 3) - o0, 0).astype(np.int8)
            ishit = np.isin(ev, HITS)
            np.add.at(bh, (a, bs, sl), ishit.astype(np.int8))
            np.add.at(bhr, (a, bs, sl), (ev == HR_).astype(np.int8))
            np.add.at(bk, (a, bs, sl), (ev == K_).astype(np.int8))
            np.add.at(bpa, (a, bs, sl), 1)
        slot[a, bs] = (sl + 1) % 9
        end_half = no >= 3
        bases[a] = np.where(end_half, 0, nb); outs[a] = np.where(end_half, 0, no)
        inn_start[a] = end_half
        sch = sched[a]
        hs, as_ = score[a, 1], score[a, 0]
        # walk-off: bottom half, inning >= sched, home ahead
        walk = (bs == 1) & (inn[a] >= sch) & (hs > as_)
        # end of top half in inning >= sched with home ahead -> game over
        top_end = end_half & (bs == 0) & (inn[a] >= sch) & (hs > as_)
        bot_end = end_half & (bs == 1) & (inn[a] >= sch) & (hs != as_)
        over = walk | top_end | bot_end
        # F5 snapshot: after the bottom of the 5th (or earlier if the game ends)
        f5m = end_half & (bs == 1) & (inn[a] == 5)
        f5[a[f5m], 0] = as_[f5m]; f5[a[f5m], 1] = hs[f5m]
        # advance half-innings
        adv = end_half & ~over
        nh = np.where(adv, 1 - bs, bs)
        ninn = np.where(adv & (bs == 1), inn[a] + 1, inn[a])
        half[a] = nh; inn[a] = ninn
        # ghost runner at the start of each extra half-inning
        gh = adv & ghost[a] & (ninn > sch)
        bases[a[gh]] = 2
        cap = adv & (ninn > 20)
        done[a[over | cap]] = True
        act = a[~(over | cap)]
    # unresolved ties after the cap: coin flip
    tie = score[:, 0] == score[:, 1]
    score[tie, 1] += (rng.random(tie.sum()) < 0.5)
    # games ending before the F5 snapshot (never in regulation) -> use final
    miss = f5[:, 0] < 0
    f5[miss] = score[miss]
    res = dict(gi=gi, n=n, away=score[:, 0], home=score[:, 1], f5a=f5[:, 0], f5h=f5[:, 1])
    if stats:
        res.update(sp_k=sp_k, sp_outs=sp_outs, bh=bh, bhr=bhr, bk=bk, bpa=bpa)
    return res


def summarize(res, G, max_runs=30):
    """Per-game distributions from simulate()'s lane arrays."""
    n = res["n"]; gi = res["gi"]
    h, a = res["home"].astype(int), res["away"].astype(int)
    out = {}
    out["p_home"] = np.bincount(gi, weights=(h > a), minlength=G) / n
    tot = np.clip(h + a, 0, max_runs)
    out["tot_hist"] = np.zeros((G, max_runs + 1))
    np.add.at(out["tot_hist"], (gi, tot), 1.0 / n)
    for side, arr in (("h", h), ("a", a)):
        hh = np.zeros((G, max_runs + 1)); np.add.at(hh, (gi, np.clip(arr, 0, max_runs)), 1.0 / n)
        out[f"{side}_hist"] = hh
    marg = np.clip(h - a, -15, 15) + 15
    out["marg_hist"] = np.zeros((G, 31)); np.add.at(out["marg_hist"], (gi, marg), 1.0 / n)
    f5d = np.sign(res["f5h"].astype(int) - res["f5a"].astype(int)) + 1   # 0 away,1 tie,2 home
    out["f5"] = np.zeros((G, 3)); np.add.at(out["f5"], (gi, f5d), 1.0 / n)
    f5t = np.clip(res["f5h"].astype(int) + res["f5a"].astype(int), 0, 20)
    out["f5_tot"] = np.zeros((G, 21)); np.add.at(out["f5_tot"], (gi, f5t), 1.0 / n)
    if "sp_k" in res:
        for key, arr, mx in (("spk", res["sp_k"], 20), ("spo", res["sp_outs"], 30)):
            hh = np.zeros((G, 2, mx + 1))
            for s in (0, 1):
                np.add.at(hh, (gi, s, np.clip(arr[:, s], 0, mx)), 1.0 / n)
            out[key] = hh                             # [:,0] away SP, [:,1] home SP
        for key, arr, mx in (("bh", res["bh"], 5), ("bhr", res["bhr"], 3), ("bk", res["bk"], 5)):
            hh = np.zeros((G, 2, 9, mx + 1))
            for s in (0, 1):
                for j in range(9):
                    np.add.at(hh, (gi, s, j, np.clip(arr[:, s, j], 0, mx)), 1.0 / n)
            out[key] = hh                             # side index = BATTING side (0 away, 1 home)
    return out
