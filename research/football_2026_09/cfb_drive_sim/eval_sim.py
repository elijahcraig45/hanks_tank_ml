"""Score the frozen CFB drive sim against the margin ridge, market, FPI, points ridge, naive.
usage: eval_sim.py RUN_TAG   (reads runs/sim_{RUN_TAG}_N4000_s0_2022_2025.parquet; requires frozen.json)
Everything tunable (sigmas, smoothing) is chosen on 2022; calibration/stacking is walk-forward
by season (season s trained on 2022..s-1). Bootstrap: paired, resampling (season, week) clusters."""
import sys, json, warnings, numpy as np, pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression, LinearRegression
warnings.filterwarnings("ignore")
fz = json.load(open("frozen.json")); TAG = sys.argv[1] if len(sys.argv) > 1 else fz["run_tag"]
assert TAG == fz["run_tag"], "scoring a config that is not the frozen one"
MR = np.arange(-100, 101); TR = np.arange(0, 171)
B = pd.read_parquet("baselines.parquet")
S = pd.read_parquet(f"runs/sim_{TAG}_N4000_s0_2022_2025.parquet").set_index("game_id")
D = B[B.game_id.isin(S.index) & B.ridge.notna()].copy().reset_index(drop=True)
S = S.loc[D.game_id]
for c in ["p_raw", "m_mean", "t_mean", "h_mean", "a_mean", "sim_sec", "fit_sec", "p_ot", "m_sd", "t_sd"]:
    D[c] = S[c].values
MH = np.stack(S.mh.values).astype(float); TH = np.stack(S.th.values).astype(float)
D["cl"] = D.season * 100 + D.week
D["fbs_fbs"] = (D.division == "fbs") & (D.fbs_diff == 0) & (D.cross_division.fillna(0) == 0)
y = D.home_won.values.astype(float); mg = D.margin.values; tot = D.total.values
tune = (D.season == 2022).values

def disc_normal(mu, sd, grid):
    return np.clip(norm.cdf((grid[None, :] + .5 - mu[:, None]) / sd) - norm.cdf((grid[None, :] - .5 - mu[:, None]) / sd), 1e-12, 1)
def smooth(H, h):
    P = H.astype(float)
    if h > 0:
        k = np.arange(-12, 13); ker = np.exp(-.5 * (k / h) ** 2); ker /= ker.sum()
        P = np.apply_along_axis(lambda r: np.convolve(r, ker, mode="same"), 1, P)
    P = P / P.sum(1, keepdims=True)
    return 0.995 * P + 0.005 / P.shape[1]
def tilt(P, grid, target):
    g = grid.astype(float); lo = np.full(len(P), -1.0); hi = np.full(len(P), 1.0)
    for _ in range(50):
        lam = (lo + hi) / 2
        W = P * np.exp(lam[:, None] * (g[None, :] - g.mean())); mu = (W * g).sum(1) / W.sum(1)
        big = mu > target; hi = np.where(big, lam, hi); lo = np.where(big, lo, lam)
    W = P * np.exp(((lo + hi) / 2)[:, None] * (g[None, :] - g.mean()))
    return W / W.sum(1, keepdims=True)
def logscore(P, v, grid):
    j = np.clip(np.round(v).astype(int) - grid[0], 0, len(grid) - 1); return -np.log(P[np.arange(len(v)), j])
def crps(P, v, grid):
    F = np.cumsum(P, 1); return ((F - (grid[None, :] >= v[:, None])) ** 2).sum(1)
def ll_i(yy, p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return -(yy * np.log(p) + (1 - yy) * np.log(1 - p))
def boot(dv, cl, reps=4000, seed=0):
    r = np.random.default_rng(seed); u, inv = np.unique(cl, return_inverse=True)
    s = np.bincount(inv, dv); c = np.bincount(inv)
    k = r.integers(0, len(u), (reps, len(u)))
    bs = s[k].sum(1) / c[k].sum(1)
    return dv.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)
def fmt(t): return f"{t[0]:+.4f} [{t[1]:+.4f}, {t[2]:+.4f}]"
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))

# ---- tuned on 2022 only
sd_rm = np.std((D.margin - D.ridge_margin)[tune]); 
mk = tune & D.spread_line.notna().values; sd_spr = np.std((D.margin - D.spread_line)[mk])
tk = tune & D.total_line.notna().values; sd_mt = np.std((D.total - D.total_line)[tk])
sd_pr = np.std((D.total - D.pr_total)[tune]); sd_nv = np.std((D.total - D.naive_total)[tune])
H_M = min([0, .5, 1, 1.5, 2, 3, 4], key=lambda h: logscore(smooth(MH[tune], h), mg[tune], MR).mean())
H_T = min([0, .5, 1, 1.5, 2, 3, 4], key=lambda h: logscore(smooth(TH[tune], h), tot[tune], TR).mean())
print(f"2022-tuned: sd ridge {sd_rm:.2f}, spread {sd_spr:.2f}, mkt total {sd_mt:.2f}, pts ridge {sd_pr:.2f}, naive {sd_nv:.2f}; smooth margin {H_M}, total {H_T}")

def wf_fit(cols_fn, target, kind):
    out = np.full(len(D), np.nan)
    for s in (2023, 2024, 2025):
        tr = (D.season < s).values; te = (D.season == s).values
        Xtr, Xte = cols_fn(tr), cols_fn(te); ok = np.isfinite(Xtr).all(1) & np.isfinite(target[tr])
        m = (LogisticRegression(C=1e4) if kind == "logit" else LinearRegression()).fit(Xtr[ok], target[tr][ok])
        oke = np.isfinite(Xte).all(1); o = np.full(te.sum(), np.nan)
        o[oke] = m.predict_proba(Xte[oke])[:, 1] if kind == "logit" else m.predict(Xte[oke])
        out[te] = o
    return out
D["cal"] = wf_fit(lambda m: lg(D.p_raw.values[m])[:, None], y, "logit")
D["stk"] = wf_fit(lambda m: np.c_[lg(D.p_raw.values[m]), D.ridge_margin.values[m]], y, "logit")
D["mcal"] = wf_fit(lambda m: D.m_mean.values[m][:, None], mg.astype(float), "lin")
D["tcal"] = wf_fit(lambda m: D.t_mean.values[m][:, None], tot.astype(float), "lin")
D["tstk"] = wf_fit(lambda m: np.c_[D.t_mean.values[m], D.total_line.values[m]], tot.astype(float), "lin")
PM = smooth(MH, H_M); PT = smooth(TH, H_T)
# league-empirical pmf baseline (walk-forward: season s uses every game in 2021..s-1), the
# fair comparator for "does the sim's SHAPE add anything" -- a lumpy empirical pmf also has key numbers
def league_pmf(col, grid, h):
    out = np.zeros((len(D), len(grid)))
    for s in D.season.unique():
        src = B[(B.season < s)][col].dropna().values.astype(int) - grid[0]
        H = np.bincount(np.clip(src, 0, len(grid) - 1), minlength=len(grid))[None, :]
        out[(D.season == s).values] = smooth(H, h)[0]
    return out
LH_M = min([0, .5, 1, 2, 3], key=lambda h: logscore(tilt(league_pmf("margin", MR, h)[tune], MR, D.ridge_margin.values[tune]), mg[tune], MR).mean())
LH_T = min([0, .5, 1, 2, 3], key=lambda h: logscore(tilt(league_pmf("total", TR, h)[tune], TR, D.pr_total.values[tune]), tot[tune], TR).mean())
LM = league_pmf("margin", MR, LH_M); LT = league_pmf("total", TR, LH_T)
print(f"league pmf smoothing (2022): margin {LH_M}, total {LH_T}")
R = {}

def section(mask, label):
    out = {"label": label}
    b = mask & np.isfinite(D.cal.values)
    Db = D[b]; cl = Db.cl.values; yb = y[b]; mb = mg[b].astype(float); tb = tot[b].astype(float)
    print(f"\n==================== {label}  n={b.sum()}")
    base = ll_i(yb, Db.ridge.values)
    W = {}
    print("WINNER               acc    logloss  d(ll) vs ridge [95% CI]")
    for nm in ["ridge", "p_raw", "cal", "stk"]:
        l = ll_i(yb, Db[nm].values); W[nm] = dict(acc=float(np.mean((Db[nm].values > .5) == yb)), ll=float(l.mean()), n=int(len(yb)))
        if nm != "ridge": W[nm]["d_ridge"] = boot(l - base, cl)
        print(f"  {nm:18s} {W[nm]['acc']:.3f}  {l.mean():.4f}  {'' if nm=='ridge' else fmt(W[nm]['d_ridge'])}")
    for ref in ["market", "fpi"]:
        m2 = Db[ref].notna().values
        if m2.sum() < 50: continue
        yy = yb[m2]; c2 = cl[m2]; lr = ll_i(yy, Db[ref].values[m2])
        print(f"  -- on {m2.sum()} games with {ref}: {ref} acc {np.mean((Db[ref].values[m2]>.5)==yy):.3f} ll {lr.mean():.4f}; ridge ll {ll_i(yy, Db.ridge.values[m2]).mean():.4f}")
        W[f"{ref}_ref"] = dict(n=int(m2.sum()), ll=float(lr.mean()), acc=float(np.mean((Db[ref].values[m2] > .5) == yy)),
                               ridge_ll=float(ll_i(yy, Db.ridge.values[m2]).mean()))
        for nm in ["ridge", "cal", "stk"]:
            d_ = ll_i(yy, Db[nm].values[m2]) - lr; t = boot(d_, c2)
            W[f"{nm}_minus_{ref}"] = t
            print(f"     {nm} - {ref}: ll {fmt(t)}   (ll {ll_i(yy, Db[nm].values[m2]).mean():.4f})")
    out["winner"] = W
    # ---- margins
    M = {}
    m2 = Db.spread_line.notna().values
    print(f"MARGIN (all n={len(mb)}; market rows n={m2.sum()})   MAE    logscore   CRPS")
    rows = [("ridge normal", Db.ridge_margin.values, disc_normal(Db.ridge_margin.values, sd_rm, MR), None),
            ("sim raw", Db.m_mean.values, PM[b], None),
            ("sim calibrated mean", Db.mcal.values, tilt(PM[b], MR, Db.mcal.values), None),
            ("sim shape@ridge", Db.ridge_margin.values, tilt(PM[b], MR, Db.ridge_margin.values), None),
            ("league pmf@ridge", Db.ridge_margin.values, tilt(LM[b], MR, Db.ridge_margin.values), None)]
    for nm, pt, P, _ in rows:
        ls, cr = logscore(P, mb, MR), crps(P, mb, MR)
        M[nm] = dict(mae=float(np.mean(abs(pt - mb))), ls=float(ls.mean()), crps=float(cr.mean()), _ls=ls, _cr=cr, _ae=abs(pt - mb))
        print(f"  {nm:24s} {M[nm]['mae']:.3f}  {ls.mean():.4f}  {cr.mean():.4f}")
    M["sim shape@ridge"]["ls_vs_league"] = boot(M["sim shape@ridge"]["_ls"] - M["league pmf@ridge"]["_ls"], cl)
    M["sim shape@ridge"]["crps_vs_league"] = boot(M["sim shape@ridge"]["_cr"] - M["league pmf@ridge"]["_cr"], cl)
    print(f"  sim shape@ridge - league pmf@ridge: logscore {fmt(M['sim shape@ridge']['ls_vs_league'])}  CRPS {fmt(M['sim shape@ridge']['crps_vs_league'])}")
    for nm in ["sim raw", "sim calibrated mean", "sim shape@ridge"]:
        M[nm]["mae_vs_ridge"] = boot(M[nm]["_ae"] - M["ridge normal"]["_ae"], cl)
        M[nm]["ls_vs_ridge"] = boot(M[nm]["_ls"] - M["ridge normal"]["_ls"], cl)
        M[nm]["crps_vs_ridge"] = boot(M[nm]["_cr"] - M["ridge normal"]["_cr"], cl)
        print(f"  {nm} - ridge normal: MAE {fmt(M[nm]['mae_vs_ridge'])}  logscore {fmt(M[nm]['ls_vs_ridge'])}  CRPS {fmt(M[nm]['crps_vs_ridge'])}")
    if m2.sum() > 50:
        sp = Db.spread_line.values[m2]; mm = mb[m2]; c2 = cl[m2]
        Pn = disc_normal(sp, sd_spr, MR); Ps = tilt(PM[b][m2], MR, sp); Pl = tilt(LM[b][m2], MR, sp)
        a_sp = abs(sp - mm); a_sim = abs(Db.mcal.values[m2] - mm); a_r = abs(Db.ridge_margin.values[m2] - mm)
        ls_n, ls_s = logscore(Pn, mm, MR), logscore(Ps, mm, MR); cr_n, cr_s = crps(Pn, mm, MR), crps(Ps, mm, MR)
        M["spread"] = dict(n=int(m2.sum()), mae=float(a_sp.mean()), ridge_mae=float(a_r.mean()), sim_mae=float(a_sim.mean()),
                           ls_normal=float(ls_n.mean()), ls_simshape=float(ls_s.mean()),
                           sim_mae_vs_spread=boot(a_sim - a_sp, c2), ridge_mae_vs_spread=boot(a_r - a_sp, c2),
                           shape_ls=boot(ls_s - ls_n, c2), shape_crps=boot(cr_s - cr_n, c2),
                           ls_league=float(logscore(Pl, mm, MR).mean()),
                           shape_ls_vs_league=boot(ls_s - logscore(Pl, mm, MR), c2), shape_crps_vs_league=boot(cr_s - crps(Pl, mm, MR), c2))
        print(f"  market rows: MAE spread {a_sp.mean():.3f}, ridge {a_r.mean():.3f}, sim-cal {a_sim.mean():.3f}; "
              f"sim-cal - spread MAE {fmt(M['spread']['sim_mae_vs_spread'])}")
        print(f"  SHAPE: sim shape@spread - normal@spread: logscore {fmt(M['spread']['shape_ls'])}  CRPS {fmt(M['spread']['shape_crps'])}")
        print(f"  SHAPE: sim shape@spread - league pmf@spread: logscore {fmt(M['spread']['shape_ls_vs_league'])}  CRPS {fmt(M['spread']['shape_crps_vs_league'])}"
              f"   (logscores: normal {ls_n.mean():.4f}, league {M['spread']['ls_league']:.4f}, sim {ls_s.mean():.4f})")
    out["margin"] = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in M.items()}
    # ---- totals
    T = {}
    print(f"TOTALS    MAE    logscore  CRPS   bias")
    rows = [("naive", Db.naive_total.values, disc_normal(Db.naive_total.values, sd_nv, TR)),
            ("points ridge", Db.pr_total.values, disc_normal(Db.pr_total.values, sd_pr, TR)),
            ("sim raw", Db.t_mean.values, PT[b]),
            ("sim calibrated", Db.tcal.values, tilt(PT[b], TR, Db.tcal.values))]
    for nm, pt, P in rows:
        ls, cr = logscore(P, tb, TR), crps(P, tb, TR)
        T[nm] = dict(mae=float(np.mean(abs(pt - tb))), ls=float(ls.mean()), crps=float(cr.mean()), bias=float(np.mean(pt - tb)), _cr=cr, _ae=abs(pt - tb), _ls=ls)
        print(f"  {nm:16s} {T[nm]['mae']:.3f}  {ls.mean():.4f}  {cr.mean():.4f}  {T[nm]['bias']:+.2f}")
    for mine in ["sim raw", "sim calibrated"]:
        for ref in ["naive", "points ridge"]:
            T[mine][f"crps_vs_{ref}"] = boot(T[mine]["_cr"] - T[ref]["_cr"], cl)
            T[mine][f"mae_vs_{ref}"] = boot(T[mine]["_ae"] - T[ref]["_ae"], cl)
            print(f"  {mine} - {ref}: CRPS {fmt(T[mine][f'crps_vs_{ref}'])}  MAE {fmt(T[mine][f'mae_vs_{ref}'])}")
    m3 = Db.total_line.notna().values & np.isfinite(Db.tstk.values)
    if m3.sum() > 50:
        tl = Db.total_line.values[m3]; tt = tb[m3]; c3 = cl[m3]
        Pl = tilt(LT[b][m3], TR, tl)
        Pn = disc_normal(tl, sd_mt, TR); Ps = tilt(PT[b][m3], TR, tl); Pk = tilt(PT[b][m3], TR, Db.tstk.values[m3])
        e = dict(market=abs(tl - tt), simcal=abs(Db.tcal.values[m3] - tt), stack=abs(Db.tstk.values[m3] - tt))
        T["market"] = dict(n=int(m3.sum()), mae=float(e["market"].mean()), simcal_mae=float(e["simcal"].mean()), stack_mae=float(e["stack"].mean()),
                           crps_normal=float(crps(Pn, tt, TR).mean()), crps_simshape=float(crps(Ps, tt, TR).mean()),
                           simcal_mae_vs_mkt=boot(e["simcal"] - e["market"], c3), stack_mae_vs_mkt=boot(e["stack"] - e["market"], c3),
                           shape_crps=boot(crps(Ps, tt, TR) - crps(Pn, tt, TR), c3), shape_ls=boot(logscore(Ps, tt, TR) - logscore(Pn, tt, TR), c3),
                           stack_crps=boot(crps(Pk, tt, TR) - crps(Pn, tt, TR), c3),
                           shape_crps_vs_league=boot(crps(Ps, tt, TR) - crps(Pl, tt, TR), c3), shape_ls_vs_league=boot(logscore(Ps, tt, TR) - logscore(Pl, tt, TR), c3))
        mk_ = T["market"]
        print(f"  market rows n={m3.sum()}: MAE market {mk_['mae']:.3f}, sim-cal {mk_['simcal_mae']:.3f}, sim+market stack {mk_['stack_mae']:.3f}")
        print(f"   sim-cal - market MAE {fmt(mk_['simcal_mae_vs_mkt'])}; stack - market MAE {fmt(mk_['stack_mae_vs_mkt'])}")
        print(f"   sim shape@market - normal@market: CRPS {fmt(mk_['shape_crps'])} logscore {fmt(mk_['shape_ls'])}; stack CRPS {fmt(mk_['stack_crps'])}")
        print(f"   sim shape@market - league pmf@market: CRPS {fmt(mk_['shape_crps_vs_league'])} logscore {fmt(mk_['shape_ls_vs_league'])}")
    out["totals"] = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in T.items()}
    # ---- calibration & runtime
    p = Db.p_raw.values; q = pd.qcut(p, 5, labels=False, duplicates="drop")
    cal = [(round(p[q == k].mean(), 3), round(yb[q == k].mean(), 3)) for k in range(5)]
    slope = LinearRegression().fit(Db[["m_mean"]], mb).coef_[0]
    print(f"raw win-prob quintiles pred->obs {cal}; margin slope (actual on sim mean) {slope:.2f}; "
          f"sim P(OT) {Db.p_ot.mean():.3f}; sim ms/game {1000*Db.sim_sec.mean():.0f}; fit s/block {Db.fit_sec.mean():.1f}")
    out["calib"] = dict(quintiles=cal, margin_slope=float(slope), p_ot=float(Db.p_ot.mean()),
                        ms_per_game=float(1000 * Db.sim_sec.mean()), fit_s=float(Db.fit_sec.mean()))
    R[label] = out

MODE = sys.argv[2] if len(sys.argv) > 2 else "wf"
wf = D.season.between(2023, 2024).values; ho = (D.season == 2025).values
fb = (D.division == "fbs").values; ff = D.fbs_fbs.values
if MODE == "wf":
    section(wf, "2023-24 all"); section(wf & fb, "2023-24 FBS-tagged"); section(wf & ff, "2023-24 FBS-vs-FBS")
else:
    section(ho, "2025 all"); section(ho & fb, "2025 FBS-tagged"); section(ho & ff, "2025 FBS-vs-FBS")
def clean(o):
    if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (tuple, list)): return [clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer)): return o.item()
    return o
json.dump(clean(R), open(f"results_{MODE}.json", "w"), indent=1)
D.drop(columns=[]).to_parquet(f"eval_games_{MODE}.parquet")
