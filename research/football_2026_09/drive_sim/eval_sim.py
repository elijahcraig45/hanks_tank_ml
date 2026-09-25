"""Score the drive sim against the margin ridge, a points ridge, naive and market baselines.

usage: eval_sim.py            -> tuning 2010-16 (smoothing / sigmas) + walk-forward 2017-24
       eval_sim.py holdout    -> 2025, only after frozen.json exists (config frozen on 2017-24)
Bootstrap: paired, resampling (season, week) clusters, 2000 reps.
"""
import sys, json, logging, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import norm
from sklearn.linear_model import Ridge, LogisticRegression, LinearRegression
from sklearn.metrics import log_loss

warnings.filterwarnings("ignore"); logging.disable(logging.INFO)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, "/Users/VTNX82W/Documents/personalDev/machineLEARNING/hanks_tank_ml/src/nfl")
from football_eval import ridge_margins, fit_sigma  # exact baseline code
from data import completed_games

MODE = sys.argv[1] if len(sys.argv) > 1 else "wf"
N = 4000
MR = np.arange(-80, 81); TR = np.arange(0, 131)
rng0 = np.random.default_rng(0)
WPS = 30

# ------------------------------------------------------------------ baselines
games = completed_games()
rg = games[["game_id", "season", "week", "game_type", "home_team", "away_team", "result", "location",
            "home_score", "away_score", "spread_line", "total_line", "home_won"]].copy()
rg["margin"] = rg.result; rg["neutral"] = (rg.location != "Home").astype(int)
rg["total"] = rg.home_score + rg.away_score
rg = rg.reset_index(drop=True)
cache = HERE / "baselines.parquet"
if cache.exists():
    B = pd.read_parquet(cache)
else:
    y_all = (rg.result > 0).astype(int).values
    best = None
    for alpha in [1, 3, 10, 30]:
        for tau in [6, 10, 16, 25]:
            mp = ridge_margins(rg, lambda g: g.season.between(2010, 2016), alpha, tau)
            ok = ~np.isnan(mp) & rg.season.between(2010, 2016).values
            sig = fit_sigma(mp[ok], y_all[ok])
            ll = log_loss(y_all[ok], np.clip(norm.cdf(mp[ok] / sig), 1e-6, 1 - 1e-6))
            if best is None or ll < best[0]:
                best = (ll, alpha, tau, sig)
    print("margin ridge (tuned 2010-16):", best)
    rg["ridge_margin"] = ridge_margins(rg, lambda g: g.season.between(2010, 2025), best[1], best[2])
    rg["ridge"] = norm.cdf(rg.ridge_margin / best[3])

    # points ridge: pts = off[team] + def[opp] + hfa + c ; same window/decay machinery
    def pts_ridge(alpha, tau, mask):
        g = rg.copy(); g["t"] = g.season * WPS + g.week
        teams = pd.Index(sorted(set(g.home_team) | set(g.away_team))); T = len(teams)
        n = len(g); hi = teams.get_indexer(g.home_team); ai = teams.get_indexer(g.away_team)
        Xh = np.zeros((n, 2 * T + 1)); Xa = np.zeros((n, 2 * T + 1))
        Xh[np.arange(n), hi] = 1; Xh[np.arange(n), T + ai] = 1; Xh[:, -1] = (1 - g.neutral) * 10
        Xa[np.arange(n), ai] = 1; Xa[np.arange(n), T + hi] = 1
        ph = np.full(n, np.nan); pa = np.full(n, np.nan)
        for (s, w), idx in g[mask(g)].groupby(["season", "week"]).groups.items():
            t_now = s * WPS + w
            tr = (g.t.values < t_now) & (g.t.values >= t_now - 2 * WPS)
            wt = np.exp(-(t_now - g.t.values[tr]) / tau)
            X = np.vstack([Xh[tr], Xa[tr]]); y = np.r_[g.home_score.values[tr], g.away_score.values[tr]]
            m = Ridge(alpha=alpha).fit(X, y, sample_weight=np.r_[wt, wt])
            ii = g.index.get_indexer(idx)
            ph[ii] = m.predict(Xh[ii]); pa[ii] = m.predict(Xa[ii])
        return ph, pa
    tm = lambda g: g.season.between(2010, 2016)
    bt = None
    for alpha in [3, 10, 30, 100]:
        for tau in [6, 10, 16, 25, 40]:
            ph, pa = pts_ridge(alpha, tau, tm)
            ok = tm(rg).values
            mse = np.nanmean((ph[ok] + pa[ok] - rg.total.values[ok]) ** 2)
            if bt is None or mse < bt[0]:
                bt = (mse, alpha, tau)
    print("points ridge (tuned 2010-16):", bt)
    ph, pa = pts_ridge(bt[1], bt[2], lambda g: g.season.between(2010, 2025))
    rg["pr_total"] = ph + pa
    # naive: trailing-2-season mean total (strictly earlier weeks)
    t = rg.season * WPS + rg.week
    rg["naive_total"] = [rg.total[(t < tt) & (t >= tt - 2 * WPS)].mean() for tt in t]
    B = rg
    B.to_parquet(cache)

# ------------------------------------------------------------------ sims
def load(v, seed=0, n=N):
    s = pd.read_parquet(HERE / f"sim_{v}_N{n}_s{seed}_2010_2025.parquet")
    s["mh"] = s.mh.apply(np.asarray); s["th"] = s.th.apply(np.asarray)
    return s.set_index("game_id")

import os
VARS = os.environ.get("VARS", "a b c a_g b_g c_g").split()
VARS = [v for v in VARS if (HERE / f"sim_{v}_N{N}_s0_2010_2025.parquet").exists()]
SIM = {v: load(v) for v in VARS}
D = B[B.season.between(2010, 2025) & B.game_id.isin(SIM[VARS[0]].index)].copy().reset_index(drop=True)
D["cl"] = D.season * 100 + D.week
tune = D.season.between(2010, 2016).values
wf = D.season.between(2017, 2024).values
ho = (D.season == 2025).values


# ------------------------------------------------------------------ scoring helpers
def disc_normal(mu, sd, grid):
    return norm.cdf((grid[None, :] + .5 - mu[:, None]) / sd) - norm.cdf((grid[None, :] - .5 - mu[:, None]) / sd)


def smooth(H, h):
    if h <= 0:
        P = H.astype(float)
    else:
        k = np.arange(-12, 13); ker = np.exp(-.5 * (k / h) ** 2); ker /= ker.sum()
        P = np.apply_along_axis(lambda r: np.convolve(r, ker, mode="same"), 1, H.astype(float))
    P = P / P.sum(1, keepdims=True)
    return 0.995 * P + 0.005 / P.shape[1]


def shift(P, delta):
    """integer shift of each row (keeps the lumpy key-number shape)."""
    out = np.zeros_like(P)
    for i, dl in enumerate(np.round(delta).astype(int)):
        out[i] = np.roll(P[i], dl)
        if dl > 0: out[i, :dl] = P[i, 0] * 0 + 1e-9
        elif dl < 0: out[i, dl:] = 1e-9
    return out / out.sum(1, keepdims=True)


def tilt(P, grid, target):
    """exponential tilt p'(k) ~ p(k) exp(lam k): moves the mean to target while keeping the
    support, so key numbers (3, 7 margins; common totals) stay where they are."""
    g = grid.astype(float)
    lo = np.full(len(P), -1.0); hi = np.full(len(P), 1.0)
    for _ in range(50):
        lam = (lo + hi) / 2
        W = P * np.exp(lam[:, None] * (g[None, :] - g.mean()))
        mu = (W * g).sum(1) / W.sum(1)
        big = mu > target
        hi = np.where(big, lam, hi); lo = np.where(big, lo, lam)
    W = P * np.exp(((lo + hi) / 2)[:, None] * (g[None, :] - g.mean()))
    return W / W.sum(1, keepdims=True)


def league_pmf(vals, seasons, grid, h):
    """walk-forward league-wide empirical pmf: for season s, games from s-3..s-1."""
    out = np.zeros((len(vals), len(grid)))
    for s in np.unique(seasons):
        m = (D.season >= s - 3) & (D.season < s)
        src = (D.total if grid is TR else D.margin)[m].values.astype(int) - grid[0]
        H = np.bincount(np.clip(src, 0, len(grid) - 1), minlength=len(grid))[None, :]
        out[seasons == s] = smooth(H, h)[0]
    return out


def logscore(P, y, grid):
    j = np.clip(np.round(y).astype(int) - grid[0], 0, len(grid) - 1)
    return -np.log(P[np.arange(len(y)), j])


def crps(P, y, grid):
    F = np.cumsum(P, 1)
    ind = (grid[None, :] >= y[:, None]).astype(float)
    return ((F - ind) ** 2).sum(1)


def ll_i(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def boot(d, cl, reps=4000, seed=0):
    r = np.random.default_rng(seed)
    u, inv = np.unique(cl, return_inverse=True)
    s = np.bincount(inv, d); c = np.bincount(inv)
    bs = []
    for _ in range(reps):
        k = r.integers(0, len(u), len(u))
        bs.append(s[k].sum() / c[k].sum())
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), np.percentile(bs, 0.1), np.percentile(bs, 99.9)


def fmt(t):
    return f"{t[0]:+.4f} [{t[1]:+.4f},{t[2]:+.4f}] (Bonf-25 99.8%: [{t[3]:+.4f},{t[4]:+.4f}])"


# ------------------------------------------------------------------ assemble per-game arrays
y = D.home_won.values; mg = D.margin.values; tot = D.total.values
for v in VARS:
    S = SIM[v].loc[D.game_id]
    D[f"p_{v}"] = S.p_raw.values; D[f"m_{v}"] = S.m_mean.values; D[f"t_{v}"] = S.t_mean.values
    D[f"sec_{v}"] = S.sim_sec.values
MH = {v: np.stack(SIM[v].loc[D.game_id].mh.values) for v in VARS}
TH = {v: np.stack(SIM[v].loc[D.game_id].th.values) for v in VARS}
D["vegas"] = norm.cdf(D.spread_line / 13.45)

# sigmas & smoothing chosen on 2010-16 only
sd_mkt_t = np.nanstd((D.total - D.total_line)[tune]); sd_pr_t = np.nanstd((D.total - D.pr_total)[tune])
sd_nv_t = np.nanstd((D.total - D.naive_total)[tune])
sd_spr = np.nanstd((D.margin - D.spread_line)[tune]); sd_rm = np.nanstd((D.margin - D.ridge_margin)[tune])
H_T = {}; H_M = {}
for v in VARS:
    H_T[v] = min([0, .5, 1, 1.5, 2, 3, 4], key=lambda h: logscore(smooth(TH[v][tune], h), tot[tune], TR).mean())
    H_M[v] = min([0, .5, 1, 1.5, 2, 3, 4], key=lambda h: logscore(smooth(MH[v][tune], h), mg[tune], MR).mean())

# walk-forward calibration / stacking: for season s train on 2010..s-1 predictions
def wf_fit(cols_fn, target, kind):
    out = np.full(len(D), np.nan)
    for s in range(2011, 2026):
        tr = (D.season < s).values & (D.season >= 2010).values; te = (D.season == s).values
        Xtr, Xte = cols_fn(tr), cols_fn(te)
        ok = np.isfinite(Xtr).all(1)
        if kind == "logit":
            m = LogisticRegression(C=1e4).fit(Xtr[ok], target[tr][ok]); out[te] = m.predict_proba(Xte)[:, 1]
        else:
            m = LinearRegression().fit(Xtr[ok], target[tr][ok]); out[te] = m.predict(Xte)
    return out

lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
for v in VARS:
    D[f"cal_{v}"] = wf_fit(lambda m: lg(D[f"p_{v}"].values[m])[:, None], y, "logit")
    D[f"stk_{v}"] = wf_fit(lambda m: np.c_[lg(D[f"p_{v}"].values[m]), D.ridge_margin.values[m]], y, "logit")
    D[f"tcal_{v}"] = wf_fit(lambda m: D[f"t_{v}"].values[m][:, None], tot.astype(float), "lin")
    D[f"mcal_{v}"] = wf_fit(lambda m: D[f"m_{v}"].values[m][:, None], mg.astype(float), "lin")
    D[f"tstk_{v}"] = wf_fit(lambda m: np.c_[D[f"t_{v}"].values[m], D.total_line.values[m]], tot.astype(float), "lin")


def section(mask, label):
    b = mask & D.ridge.notna().values & D.total_line.notna().values & D.spread_line.notna().values
    for v in VARS:
        b &= np.isfinite(D[f"stk_{v}"].values)
    Db = D[b]; cl = Db.cl.values; yb = y[b]; mb = mg[b].astype(float); tb = tot[b].astype(float)
    print(f"\n==================== {label}  n={b.sum()} games")
    # ---- winners
    print("\nWINNER  model                 acc     logloss   d(ll) vs ridge [95% CI] (Bonferroni-25 99.8%)")
    base = ll_i(yb, Db.ridge.values)
    res = {}
    for name in ["ridge", "vegas"] + [f"{k}_{v}" for v in VARS for k in ("p", "cal", "stk")]:
        p = Db[name].values; l = ll_i(yb, p)
        res[name] = dict(acc=float(np.mean((p > .5) == yb)), ll=float(l.mean()))
        extra = "" if name == "ridge" else fmt(boot(l - base, cl))
        res[name]["d_vs_ridge"] = None if name == "ridge" else boot(l - base, cl)[:3]
        print(f"  {name:22s} {np.mean((p>.5)==yb):.3f}  {l.mean():.4f}   {extra}")
    # ---- margins
    print("\nMARGIN  model            MAE     logscore   CRPS")
    mres = {}
    rows = [("spread (market)", Db.spread_line.values, disc_normal(Db.spread_line.values, sd_spr, MR)),
            ("margin ridge", Db.ridge_margin.values, disc_normal(Db.ridge_margin.values, sd_rm, MR)),
            ("league pmf@spread", Db.spread_line.values, tilt(league_pmf(mb, Db.season.values, MR, 1.0), MR, Db.spread_line.values))]
    for v in VARS:
        rows.append((f"sim_{v}", Db[f"m_{v}"].values, smooth(MH[v][b], H_M[v])))
        rows.append((f"sim_{v} shape@spread", Db.spread_line.values,
                     tilt(smooth(MH[v][b], H_M[v]), MR, Db.spread_line.values)))
    for nm, pt, P in rows:
        ls, cr = logscore(P, mb, MR), crps(P, mb, MR)
        mres[nm] = dict(mae=float(np.mean(abs(pt - mb))), ls=ls, crps=cr)
        print(f"  {nm:24s} {np.mean(abs(pt-mb)):.3f}  {ls.mean():.4f}  {cr.mean():.4f}")
    for v in VARS:
        print(f"  sim_{v} vs spread-normal: logscore {fmt(boot(mres[f'sim_{v}']['ls']-mres['spread (market)']['ls'], cl))}")
        for ref in ["spread (market)", "league pmf@spread"]:
            print(f"  sim_{v} shape@spread - {ref}: logscore {fmt(boot(mres[f'sim_{v} shape@spread']['ls']-mres[ref]['ls'], cl))}")
            print(f"      crps {fmt(boot(mres[f'sim_{v} shape@spread']['crps']-mres[ref]['crps'], cl))}")
        print(f"  sim_{v} vs ridge MAE: {fmt(boot(abs(Db[f'm_{v}'].values-mb)-abs(Db.ridge_margin.values-mb), cl))}")
    # ---- totals
    print("\nTOTALS  model                  MAE     logscore   CRPS   (bias = mean pred - actual)")
    tres = {}
    rows = [("naive (trailing mean)", Db.naive_total.values, disc_normal(Db.naive_total.values, sd_nv_t, TR)),
            ("points ridge", Db.pr_total.values, disc_normal(Db.pr_total.values, sd_pr_t, TR)),
            ("market total_line", Db.total_line.values, disc_normal(Db.total_line.values, sd_mkt_t, TR)),
            ("league pmf@market", Db.total_line.values, tilt(league_pmf(tb, Db.season.values, TR, 1.0), TR, Db.total_line.values))]
    for v in VARS:
        P = smooth(TH[v][b], H_T[v])
        rows.append((f"sim_{v} raw", Db[f"t_{v}"].values, P))
        rows.append((f"sim_{v} calibrated", Db[f"tcal_{v}"].values, tilt(P, TR, Db[f"tcal_{v}"].values)))
        rows.append((f"sim_{v} shape@market", Db.total_line.values, tilt(P, TR, Db.total_line.values)))
        rows.append((f"sim_{v}+market stack", Db[f"tstk_{v}"].values, tilt(P, TR, Db[f"tstk_{v}"].values)))
    for nm, pt, P in rows:
        ls, cr = logscore(P, tb, TR), crps(P, tb, TR)
        tres[nm] = dict(mae=float(np.mean(abs(pt - tb))), ls=ls, crps=cr, bias=float(np.mean(pt - tb)))
        print(f"  {nm:26s} {np.mean(abs(pt-tb)):.3f}  {ls.mean():.4f}  {cr.mean():.4f}  bias {np.mean(pt-tb):+.2f}")
    for v in VARS:
        for ref in ["naive (trailing mean)", "points ridge", "market total_line", "league pmf@market"]:
            for mine in [f"sim_{v} raw", f"sim_{v} calibrated", f"sim_{v} shape@market", f"sim_{v}+market stack"]:
                if (ref in ("market total_line", "league pmf@market")) != (mine in (f"sim_{v} shape@market", f"sim_{v}+market stack")):
                    continue
                print(f"  {mine} - {ref}: CRPS {fmt(boot(tres[mine]['crps']-tres[ref]['crps'], cl))}")
                print(f"  {'':{len(mine)}s}   logscore {fmt(boot(tres[mine]['ls']-tres[ref]['ls'], cl))}")
    # ---- calibration
    print("\nCALIBRATION")
    for v in VARS:
        p = Db[f"p_{v}"].values
        bins = pd.qcut(p, 5, labels=False)
        print(f"  sim_{v} raw win-prob quintiles (pred->obs):",
              [(round(p[bins == k].mean(), 3), round(yb[bins == k].mean(), 3)) for k in range(5)])
        P = smooth(TH[v][b], H_T[v]); F = np.cumsum(P, 1)
        j = np.clip(tb.astype(int), 0, 130); pit = F[np.arange(len(tb)), j] - P[np.arange(len(tb)), j] / 2
        print(f"  sim_{v} totals PIT: in central 50% {np.mean(abs(pit-.5)<.25):.3f}, central 80% {np.mean(abs(pit-.5)<.4):.3f};"
              f" sim sd {np.sqrt((P*(TR-(P*TR).sum(1,keepdims=True))**2).sum(1)).mean():.2f} vs market-resid sd {np.std(tb-Db.total_line.values):.2f}")
    # ---- betting (secondary)
    print("\nBETTING (secondary; breakeven at -110 is 52.4%)")
    sp = Db.spread_line.values; tl = Db.total_line.values
    for v in VARS:
        P = MH[v][b].astype(float); P /= P.sum(1, keepdims=True)
        pc = (P * (MR[None, :] > sp[:, None])).sum(1); pn = (P * (MR[None, :] < sp[:, None])).sum(1)
        side = np.where(pc > pn, 1, -1); push = mb == sp
        hit = (np.sign(mb - sp) == side)[~push]
        PT = TH[v][b].astype(float); PT /= PT.sum(1, keepdims=True)
        po = (PT * (TR[None, :] > tl[:, None])).sum(1); pu = (PT * (TR[None, :] < tl[:, None])).sum(1)
        ou = np.where(po > pu, 1, -1); pusht = tb == tl
        hito = (np.sign(tb - tl) == ou)[~pusht]
        print(f"  sim_{v}: ATS {hit.mean():.3f} (n={len(hit)})  O/U {hito.mean():.3f} (n={len(hito)})")
    rs = np.where(Db.ridge_margin.values > sp, 1, -1); push = mb == sp
    print(f"  margin ridge ATS {(np.sign(mb-sp)==rs)[~push].mean():.3f};  points ridge O/U "
          f"{(np.sign(tb-tl)==np.where(Db.pr_total.values>tl,1,-1))[tb!=tl].mean():.3f}")
    print(f"\nRUNTIME sim ms/game: " + ", ".join(f"{v} {1000*Db[f'sec_{v}'].mean():.0f}" for v in VARS))
    return res, mres, tres


if MODE == "wf":
    print(f"smoothing bw (tuned 2010-16): totals {H_T}, margins {H_M}; sd market total {sd_mkt_t:.2f}, "
          f"points ridge {sd_pr_t:.2f}, naive {sd_nv_t:.2f}, spread {sd_spr:.2f}, ridge {sd_rm:.2f}")
    section(tune, "TUNING 2010-16 (in-sample for smoothing/sigmas; stack trained on <s)")
    section(wf, "WALK-FORWARD 2017-24")
elif MODE == "holdout":
    fz = json.load(open(HERE / "frozen.json"))
    print("frozen:", fz)
    section(ho, "HOLDOUT 2025")
D.drop(columns=[]).to_parquet(HERE / f"eval_games_{MODE}.parquet")
