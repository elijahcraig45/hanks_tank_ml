"""Drive-based Monte Carlo simulator, CFB edition. Derived from the NFL research sim
(research/football_2026_09/drive_sim/sim_core.py). What changed for college:

  * division covariates: off_fbs and def_fbs columns, lightly penalised like the other
    structural terms, so FCS teams shrink toward the FCS mean instead of the pooled one
    (the margin ridge's fbs_diff, in drive form);
  * ~340 teams on ~12 games each: shrinkage (C), decay (tau) and a prior-season weight
    are tuned at the GAME level on 2022 only;
  * wider score-differential buckets (college leads reach 28+);
  * CFB overtime (2021+ rules): each round both teams start at the opponent 25; from
    OT2 a TD must go for two; from OT3 each round is a single two-point try; a
    defensive score ends it;
  * duration kernels on a short decay so the 2023 first-down clock change is absorbed
    within weeks rather than seasons;
  * neutral sites carry no home advantage.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge

OUT = ["TD", "FG", "MFG", "PUNT", "TO", "OTD", "TOD", "SAF", "EOH"]
O = {k: i for i, k in enumerate(OUT)}
NO = len(OUT)
SD_EDGES = np.array([-28.5, -16.5, -8.5, -3.5, -0.5, 0.5, 3.5, 8.5, 16.5, 28.5])  # 11 buckets
NYL, NTB, NPER, NSD = 20, 4, 4, len(SD_EDGES) + 1
TIED = int(np.searchsorted(SD_EDGES, 0.0))
Q = 101
K_STRUCT = 10.0
WPS = 30
P2PT = 0.45   # two-point conversion rate: CFBD OT scoring deltas are unreliable, so fixed


def tb_of(clock):
    return np.where(clock <= 30, 0, np.where(clock <= 120, 1, np.where(clock <= 300, 2, 3)))


def per_of(half, clock):
    return np.where(half == 0, 0, np.where(clock > 900, 1, np.where(clock > 300, 2, 3)))


def sd_of(sd):
    return np.searchsorted(SD_EDGES, sd)


def prep_drives(d: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    d = d[d.res.notna()].copy()
    neu = games.set_index("game_id")["neutral_site"].fillna(0).astype(int)
    d["neutral"] = d.game_id.map(neu).fillna(0).astype(int)
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


def _features(off, de, offb, defb, hadv, yl, tb, per, sdb, variant, T):
    n = len(off)
    cols = []
    A = np.zeros((n, 2 * T)); A[np.arange(n), off] = 1; A[np.arange(n), T + de] = 1
    cols.append(A)
    cols.append(K_STRUCT * np.stack([offb, defb, offb * defb], 1).astype(float))
    cols.append(K_STRUCT * hadv[:, None].astype(float))
    cols.append(K_STRUCT * np.stack([(tb == k) for k in range(3)], 1).astype(float))
    cols.append(K_STRUCT * (per > 0)[:, None].astype(float))
    z = (yl - 70.0) / 25.0
    late = (tb <= 1).astype(float)
    cols.append(K_STRUCT * np.stack([z, z ** 2, z ** 3, (yl <= 40), (yl <= 20), z * late], 1).astype(float))
    if variant == "c":
        S = np.zeros((n, NPER * NSD))
        S[np.arange(n), per * NSD + sdb] = 1
        S = np.delete(S, [p * NSD + TIED for p in range(NPER)], axis=1)
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
    def __init__(self, variant="c", C=0.03, tau=16.0, ktau=6.0, pace_alpha=200.0, prior_w=1.0):
        self.variant, self.C, self.tau, self.ktau = variant, C, tau, ktau
        self.pace_alpha, self.prior_w = pace_alpha, prior_w

    def _w(self, t, season_now, seasons, tau):
        w = np.exp(-(self.t_now - t) / tau)
        return np.where(seasons < season_now, w * self.prior_w, w)

    def fit(self, tr: pd.DataFrame, t_now: int, teams: pd.Index, fbs: dict):
        self.teams, self.fbs, self.t_now = teams, fbs, t_now
        T = len(teams); s_now = t_now // WPS
        reg = tr[tr.ot == 0]
        w = self._w(reg.t.values, s_now, reg.season.values, self.tau)
        off = teams.get_indexer(reg.posteam); de = teams.get_indexer(reg.defteam)
        X = _features(off, de, reg.off_fbs.values, reg.def_fbs.values, reg.hadv.values,
                      reg.yl.values.astype(float), reg.tb.values, reg.per.values, reg.sdb.values, self.variant, T)
        self.clf = LogisticRegression(C=self.C, max_iter=400, tol=1e-4)
        self.clf.fit(X, reg.y.values, sample_weight=w / w.mean())
        self.classes = self.clf.classes_
        wk = np.exp(-(t_now - tr.t.values) / self.ktau)
        sb = np.clip((tr.yl.values // 20).astype(int), 0, 4)
        ko_mask = tr.first_of_half.values & (tr.ot.values == 0)
        self.ko = wquant(tr.yl.values[ko_mask].astype(float), wk[ko_mask])
        K = np.zeros((NO, 5, Q))
        has = tr.yl_next.notna().values & (tr.ot.values == 0)
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
        # durations: short decay (ktau) so rule changes to the clock are absorbed quickly
        sg = self._sg(tr.per.values, tr.sdb.values)
        D = np.zeros((NO, NTB, 3, Q))
        dur = tr.dur.values.astype(float)
        for o in range(NO):
            for b in range(NTB):
                base = (tr.y.values == o) & (tr.tb.values == b) & (tr.ot.values == 0)
                qb = wquant(dur[base], wk[base]) if base.sum() > 0 else np.full(Q, 60.0)
                for g in range(3):
                    m = base & (sg == g)
                    D[o, b, g] = wquant(dur[m], wk[m]) if m.sum() > 30 else qb
        self.D = D
        m = (tr.tb.values == 3) & (tr.ot.values == 0) & (sg == 0) & (dur > 0)
        sub = tr[m]
        lg = np.log(dur[m])
        mu = pd.Series(lg).groupby(sub.y.values).transform("mean").values
        Xp = np.zeros((m.sum(), 2 * T))
        Xp[np.arange(m.sum()), teams.get_indexer(sub.posteam)] = 1
        Xp[np.arange(m.sum()), T + teams.get_indexer(sub.defteam)] = 1
        wp = self._w(sub.t.values, s_now, sub.season.values, self.tau)
        r = Ridge(alpha=self.pace_alpha, fit_intercept=True).fit(Xp, lg - mu, sample_weight=wp)
        self.pace_off, self.pace_def = r.coef_[:T], r.coef_[T:]
        wtd = np.exp(-(t_now - tr.t.values) / self.tau)
        def pdist(mask, col):
            v = tr[col].values[mask]; ww = wtd[mask]
            p = np.array([ww[v == k].sum() for k in (6, 7, 8)]) + 1e-3
            return p / p.sum()
        rg = tr.ot.values == 0
        self.td_p = pdist((tr.y.values == O["TD"]) & rg, "off_pts")
        self.otd_p = pdist((tr.y.values == O["OTD"]) & rg, "def_pts")
        return self

    def _sg(self, per, sdb):
        if self.variant != "c":
            return np.zeros(len(per), int)
        late = per >= 2
        return np.where(late & (sdb < TIED), 1, np.where(late & (sdb > TIED), 2, 0))

    def game_tables(self, home, away, neutral):
        T = len(self.teams)
        hi, ai = self.teams.get_loc(home), self.teams.get_loc(away)
        hf, af = int(self.fbs.get(home, 0)), int(self.fbs.get(away, 0))
        g = np.meshgrid(np.arange(2), np.arange(NYL), np.arange(NTB), np.arange(NPER), np.arange(NSD), indexing="ij")
        pos, ylb, tb, per, sdb = [x.ravel() for x in g]
        off = np.where(pos == 0, hi, ai); de = np.where(pos == 0, ai, hi)
        offb = np.where(pos == 0, hf, af); defb = np.where(pos == 0, af, hf)
        hadv = np.where(neutral, 0, np.where(pos == 0, 1, -1))
        X = _features(off, de, offb, defb, hadv, ylb * 5 + 2.5, tb, per, sdb, self.variant, T)
        P = np.zeros((len(pos), NO))
        P[:, self.classes] = self.clf.predict_proba(X)
        cum = np.cumsum(P, 1); cum[:, -1] = 1.0
        pace = np.exp(np.array([self.pace_off[hi] + self.pace_def[ai], self.pace_off[ai] + self.pace_def[hi]]))
        return cum.reshape(2, NYL, NTB, NPER, NSD, NO), pace


def simulate(model: DriveModel, home, away, neutral, N=4000, rng=None, return_ot=False):
    rng = rng or np.random.default_rng(0)
    cum, pace = model.game_tables(home, away, neutral)
    score = np.zeros((N, 2))
    ar = np.arange(N)
    recv0 = rng.integers(0, 2, N)

    def step(idx, pos, yl, clock, half):
        n = len(idx)
        tb = tb_of(clock); per = per_of(np.full(n, half), clock)
        sd = score[idx, pos] - score[idx, 1 - pos]
        sdb = sd_of(sd)
        ylb = np.clip((yl // 5).astype(int), 0, NYL - 1)
        c = cum[pos, ylb, tb, per, sdb]
        out = np.minimum((rng.random(n)[:, None] > c).sum(1), NO - 1)
        sg = model._sg(per, sdb)
        dur = model.D[out, tb, sg, rng.integers(0, Q, n)] * pace[pos]
        td = out == O["TD"]; otd = out == O["OTD"]
        score[idx[td], pos[td]] += rng.choice([6, 7, 8], n, p=model.td_p)[td]
        fg = out == O["FG"]; score[idx[fg], pos[fg]] += 3
        score[idx[otd], 1 - pos[otd]] += rng.choice([6, 7, 8], n, p=model.otd_p)[otd]
        sf = out == O["SAF"]; score[idx[sf], 1 - pos[sf]] += 2
        sb = np.clip((yl // 20).astype(int), 0, 4)
        yl_new = model.K[out, sb, rng.integers(0, Q, n)]
        pos_new = np.where(otd, pos, 1 - pos)
        clock_new = np.where(out == O["EOH"], 0.0, clock - dur)
        return pos_new, yl_new, clock_new

    for half in (0, 1):
        idx = ar.copy()
        pos = recv0 if half == 0 else 1 - recv0
        yl = model.ko[rng.integers(0, Q, N)]
        clock = np.full(N, 1800.0)
        for _ in range(45):
            if len(idx) == 0:
                break
            pos, yl, clock = step(idx, pos, yl, clock, half)
            alive = clock > 0
            idx, pos, yl, clock = idx[alive], pos[alive], yl[alive], clock[alive]

    # ---- college overtime (2021+): rounds of one possession each from the opponent 25
    tied = np.where(score[:, 0] == score[:, 1])[0]
    went_ot = np.zeros(N, bool); went_ot[tied] = True
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
                make = rng.random(len(idx)) < P2PT
                score[idx[live & make], pos[live & make]] += 2
                continue
            c = cum[pos, ylb25, 3, 0, TIED]
            out = np.minimum((rng.random(len(idx))[:, None] > c).sum(1), NO - 1)
            td = live & (out == O["TD"])
            pts = 7 if rnd == 1 else 6
            add = np.where(td, pts, 0)
            if rnd == 1:  # PAT kick: use the regulation PAT split (6/7/8)
                add = np.where(td, rng.choice([6, 7, 8], len(idx), p=model.td_p), 0)
            else:          # must go for two
                add = np.where(td, 6 + 2 * (rng.random(len(idx)) < P2PT), 0)
            add = add + np.where(live & (out == O["FG"]), 3, 0)
            score[idx, pos] += add
            dsc = live & np.isin(out, [O["OTD"], O["SAF"]])   # defensive score ends the game
            score[idx[dsc], 1 - pos[dsc]] += np.where(out[dsc] == O["OTD"], 6, 2)
            done |= dsc
        end = done | (score[idx, 0] != score[idx, 1])
        idx, first = idx[~end], first[~end]
    if len(idx):
        w = rng.integers(0, 2, len(idx)); score[idx, w] += 2
    if return_ot:
        return score[:, 0], score[:, 1], went_ot
    return score[:, 0], score[:, 1]
