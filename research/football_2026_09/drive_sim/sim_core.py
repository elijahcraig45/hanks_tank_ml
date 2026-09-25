"""Drive-based Monte Carlo football simulator (NFL).

Pieces, all fit on drives strictly before the game being predicted:
  1. drive-outcome multinomial: P(outcome | offense, defense, home, start yardline,
     clock bucket, [score/period state])  -- penalised multinomial logistic (L2 = shrinkage),
     time-decayed sample weights.
  2. next-drive start kernel: empirical (weighted) distribution of the next drive's start
     yardline given this drive's outcome and start-yardline bin (chains field position).
  3. clock: empirical drive-duration quantiles by (outcome, clock bucket, [script]) times a
     team pace multiplier (ridge on log duration residuals) -> number of possessions.
  4. points: empirical 6/7/8 split for TDs (PAT / 2pt), 3 for FG, 2 for safety.
  5. overtime rules by era.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge

OUT = ["TD", "FG", "MFG", "PUNT", "TO", "OTD", "TOD", "SAF", "EOH"]
O = {k: i for i, k in enumerate(OUT)}
NO = len(OUT)
NYL, NTB, NPER, NSD = 20, 4, 4, 9
SD_EDGES = np.array([-16.5, -8.5, -3.5, -0.5, 0.5, 3.5, 8.5, 16.5])  # 9 buckets
Q = 101
K_STRUCT = 10.0  # structural columns scaled up => effectively unpenalised
WPS = 30  # weeks per season index spacing (same as the ridge baseline)


def tb_of(clock):
    return np.where(clock <= 30, 0, np.where(clock <= 120, 1, np.where(clock <= 300, 2, 3)))


def per_of(half, clock):
    # 0 = first half, 1 = Q3, 2 = Q4 > 5min, 3 = Q4 <= 5min
    return np.where(half == 0, 0, np.where(clock > 900, 1, np.where(clock > 300, 2, 3)))


def sd_of(sd):
    return np.searchsorted(SD_EDGES, sd)


def prep_drives(d: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    d = d[d.res.notna()].copy()
    loc = games.set_index("game_id")["location"]
    d["neutral"] = (d.game_id.map(loc).fillna("Home") == "Neutral").astype(int)
    d["hadv"] = np.where(d.neutral == 1, 0, np.where(d.off_home, 1, -1))
    d["yl"] = d.yl.fillna(75).clip(1, 99)
    d["yl_next"] = d.yl_next.clip(1, 99)
    d["half"] = (d.game_half != "Half1").astype(int)
    d["ot"] = (d.game_half == "Overtime").astype(int)
    d["tb"] = tb_of(d.hsr0.values)
    d["per"] = per_of(d.half.values, d.hsr0.values)
    d["sdb"] = sd_of(d.sd0.values)
    d["y"] = d.res.map(O).astype(int)
    d["t"] = d.season * WPS + d.week
    d["ylb"] = np.clip((d.yl // 5).astype(int), 0, NYL - 1)
    d["first_of_half"] = d.groupby(["game_id", "game_half"]).cumcount() == 0
    return d.reset_index(drop=True)


def _features(off, de, hadv, yl, tb, per, sdb, variant, T):
    n = len(off)
    cols = []
    A = np.zeros((n, 2 * T)); A[np.arange(n), off] = 1; A[np.arange(n), T + de] = 1
    cols.append(A)
    cols.append(K_STRUCT * hadv[:, None].astype(float))
    cols.append(K_STRUCT * np.stack([(tb == k) for k in range(3)], 1).astype(float))
    cols.append(K_STRUCT * (per > 0)[:, None].astype(float))
    if variant in ("b", "c"):
        z = (yl - 70.0) / 25.0
        late = (tb <= 1).astype(float)
        cols.append(K_STRUCT * np.stack([z, z ** 2, z ** 3, (yl <= 40), (yl <= 20), z * late], 1).astype(float))
    if variant == "c":
        S = np.zeros((n, NPER * NSD))
        S[np.arange(n), per * NSD + sdb] = 1
        S = np.delete(S, [p * NSD + 4 for p in range(NPER)], axis=1)  # tied is the base per period
        cols.append(K_STRUCT * S)
    return np.hstack(cols)


def wquant(x, w, q=Q):
    ok = np.isfinite(x) & np.isfinite(w)
    x, w = x[ok], w[ok]
    if len(x) == 0:
        return None
    o = np.argsort(x); x, w = x[o], w[o]
    c = np.cumsum(w); c = c / c[-1]
    return np.interp(np.linspace(0.005, 0.995, q), c, x)


class DriveModel:
    def __init__(self, variant="b", C=0.05, tau=12.0, ktau=6.0, pace_alpha=200.0):
        self.variant, self.C, self.tau, self.ktau, self.pace_alpha = variant, C, tau, ktau, pace_alpha

    def fit(self, tr: pd.DataFrame, t_now: int, teams: pd.Index):
        self.teams = teams; T = len(teams)
        reg = tr[tr.ot == 0]
        w = np.exp(-(t_now - reg.t.values) / self.tau)
        off = teams.get_indexer(reg.posteam); de = teams.get_indexer(reg.defteam)
        X = _features(off, de, reg.hadv.values, reg.yl.values.astype(float), reg.tb.values,
                      reg.per.values, reg.sdb.values, self.variant, T)
        self.clf = LogisticRegression(C=self.C, max_iter=400, tol=1e-4)
        self.clf.fit(X, reg.y.values, sample_weight=w / w.mean())
        self.classes = self.clf.classes_
        # --- kernels (shorter decay: kickoff/touchback rules change between seasons)
        wk = np.exp(-(t_now - tr.t.values) / self.ktau)
        sb = np.clip((tr.yl.values // 20).astype(int), 0, 4)
        ko_mask = tr.first_of_half.values & (tr.ot.values == 0)
        self.ko = wquant(tr.yl.values[ko_mask].astype(float), wk[ko_mask])
        K = np.zeros((NO, 5, Q))
        has = tr.yl_next.notna().values
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
        # --- durations
        sg = self._sg(tr.per.values, tr.sdb.values)
        D = np.zeros((NO, NTB, 3, Q))
        dur = tr.dur.values.astype(float)
        for o in range(NO):
            for b in range(NTB):
                base = (tr.y.values == o) & (tr.tb.values == b) & (tr.ot.values == 0)
                qb = wquant(dur[base], w_all := np.exp(-(t_now - tr.t.values) / self.tau)[base]) if base.sum() > 0 else np.full(Q, 60.0)
                for g in range(3):
                    m = base & (sg == g)
                    D[o, b, g] = wquant(dur[m], np.exp(-(t_now - tr.t.values[m]) / self.tau)) if m.sum() > 30 else qb
        self.D = D
        # --- pace: ridge on log-duration residuals of ordinary (tb==3, neutral-script) drives
        m = (tr.tb.values == 3) & (tr.ot.values == 0) & (sg == 0) & (dur > 0)
        sub = tr[m]
        lg = np.log(dur[m])
        key = sub.y.values
        mu = pd.Series(lg).groupby(key).transform("mean").values
        Xp = np.zeros((m.sum(), 2 * T))
        Xp[np.arange(m.sum()), teams.get_indexer(sub.posteam)] = 1
        Xp[np.arange(m.sum()), T + teams.get_indexer(sub.defteam)] = 1
        wp = np.exp(-(t_now - sub.t.values) / self.tau)
        r = Ridge(alpha=self.pace_alpha, fit_intercept=True).fit(Xp, lg - mu, sample_weight=wp)
        self.pace_off, self.pace_def = r.coef_[:T], r.coef_[T:]
        # --- points on TD / OTD
        wtd = np.exp(-(t_now - tr.t.values) / self.tau)
        def pdist(mask, col):
            v = tr[col].values[mask]; ww = wtd[mask]
            p = np.array([ww[v == k].sum() for k in (6, 7, 8)]) + 1e-3
            return p / p.sum()
        self.td_p = pdist(tr.y.values == O["TD"], "off_pts")
        self.otd_p = pdist(tr.y.values == O["OTD"], "def_pts")
        return self

    def _sg(self, per, sdb):
        if self.variant != "c":
            return np.zeros(len(per), int)
        late = per >= 2
        return np.where(late & (sdb < 4), 1, np.where(late & (sdb > 4), 2, 0))

    def drive_logloss(self, te: pd.DataFrame):
        reg = te[te.ot == 0]
        T = len(self.teams)
        off = self.teams.get_indexer(reg.posteam); de = self.teams.get_indexer(reg.defteam)
        X = _features(off, de, reg.hadv.values, reg.yl.values.astype(float), reg.tb.values,
                      reg.per.values, reg.sdb.values, self.variant, T)
        P = np.full((len(reg), NO), 1e-6)
        P[:, self.classes] = self.clf.predict_proba(X)
        P /= P.sum(1, keepdims=True)
        return -np.log(P[np.arange(len(reg)), reg.y.values])

    def game_tables(self, home, away, neutral):
        """Cumulative outcome probabilities over the state grid for both offenses."""
        T = len(self.teams)
        hi, ai = self.teams.get_loc(home), self.teams.get_loc(away)
        g = np.meshgrid(np.arange(2), np.arange(NYL), np.arange(NTB), np.arange(NPER), np.arange(NSD), indexing="ij")
        pos, ylb, tb, per, sdb = [x.ravel() for x in g]
        off = np.where(pos == 0, hi, ai); de = np.where(pos == 0, ai, hi)
        hadv = np.where(neutral, 0, np.where(pos == 0, 1, -1))
        X = _features(off, de, hadv, ylb * 5 + 2.5, tb, per, sdb, self.variant, T)
        P = np.zeros((len(pos), NO))
        P[:, self.classes] = self.clf.predict_proba(X)
        cum = np.cumsum(P, 1); cum[:, -1] = 1.0
        pace = np.exp(np.array([self.pace_off[hi] + self.pace_def[ai], self.pace_off[ai] + self.pace_def[hi]]))
        return cum.reshape(2, NYL, NTB, NPER, NSD, NO), pace


def ot_rule(season, gtype):
    post = gtype != "REG"
    if post:
        return ("both" if season >= 2022 else "modified" if season >= 2010 else "sudden"), np.inf
    rule = "both" if season >= 2025 else "modified" if season >= 2012 else "sudden"
    return rule, (600.0 if season >= 2017 else 900.0)


def simulate(model: DriveModel, home, away, neutral, season, gtype, N=4000, rng=None):
    rng = rng or np.random.default_rng(0)
    cum, pace = model.game_tables(home, away, neutral)
    sgmap = lambda per, sdb: model._sg(per, sdb)
    score = np.zeros((N, 2))
    ar = np.arange(N)
    recv0 = rng.integers(0, 2, N)

    def step(idx, pos, yl, clock, half, ot=False):
        n = len(idx)
        tb = tb_of(clock)
        per = np.zeros(n, int) if ot else per_of(np.full(n, half), clock)
        sd = score[idx, pos] - score[idx, 1 - pos]
        sdb = np.full(n, 4) if ot else sd_of(sd)
        ylb = np.clip((yl // 5).astype(int), 0, NYL - 1)
        c = cum[pos, ylb, tb, per, sdb]
        out = (rng.random(n)[:, None] > c).sum(1)
        out = np.minimum(out, NO - 1)
        sg = sgmap(per, sdb)
        dur = model.D[out, tb, sg, rng.integers(0, Q, n)] * pace[pos]
        # points
        td = out == O["TD"]; otd = out == O["OTD"]
        pts_td = rng.choice([6, 7, 8], n, p=model.td_p)
        pts_otd = rng.choice([6, 7, 8], n, p=model.otd_p)
        score[idx[td], pos[td]] += pts_td[td]
        fg = out == O["FG"]; score[idx[fg], pos[fg]] += 3
        score[idx[otd], 1 - pos[otd]] += pts_otd[otd]
        sf = out == O["SAF"]; score[idx[sf], 1 - pos[sf]] += 2
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
        for _ in range(40):
            if len(idx) == 0:
                break
            _, pos, yl, clock = step(idx, pos, yl, clock, half)
            alive = clock > 0
            idx, pos, yl, clock = idx[alive], pos[alive], yl[alive], clock[alive]
    # overtime
    rule, otclock = ot_rule(season, gtype)
    tied = np.where(score[:, 0] == score[:, 1])[0]
    if len(tied):
        idx = tied
        pos = rng.integers(0, 2, len(idx))
        yl = model.ko[rng.integers(0, Q, len(idx))]
        clock = np.full(len(idx), min(otclock, 1e6))
        ndr = np.zeros(len(idx), int)
        for k in range(16):
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
            win = rng.integers(0, 2, len(idx)); score[idx, win] += 3
    return score[:, 0], score[:, 1]
