"""FROZEN-CONFIG test report (config in data/backtest_2026/rich/FROZEN_CONFIG.json).

Scores full_x50 (frozen) and v1like (reference) on seasons never used for selection:
2020, 2021 (closing market available), 2022-2025 (no market), 2026 split into the window
an earlier agent already looked at (to 09-07) and the unseen window (09-08 .. 09-23).
Every calibration step is fit on EARLIER seasons of the same config (2016-19 dev runs
seed it). CIs: date-block bootstrap on paired per-game loss differences.
Output: data/backtest_2026/rich/test_report.json
"""
import sys, json, importlib, numpy as np, pandas as pd, warnings
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
sys.path.insert(0, "research/backtest_2026")
E = importlib.import_module("44_eval"); P = importlib.import_module("45_props_series")
ALL = list(range(2016, 2027)); TEST = list(range(2020, 2027))
OUT = {}


def fmt(t):
    return f"{t[0]:+.5f} [{t[1]:+.5f}, {t[2]:+.5f}] P>0={t[3]:.3f}"


def frame(v):
    df, z = E.summarize_variant(v, ALL)
    df = E.totals_calibrated(df, z)
    df["d"] = df.game_date.dt.strftime("%Y%m%d").values
    return df, z


F, Z = frame("full_x50")
V1, _ = frame("v1like")
y = F.y.values
F["v1_cal"] = V1.sim_cal.values
# logit3 walk-forward inside 2026 (as the earlier agent did), expanding from 400 games
F["logit3"] = np.nan
g26 = F.index[(F.year == 2026) & F.elo_differential.notna()].values
X = F.loc[g26, ["elo_differential", "pythag_differential", "sp_quality_composite_diff"]].astype(float)
X = X.fillna(X.median()).values
for s in range(400, len(g26), 100):
    tr, te = np.arange(s), np.arange(s, min(s + 100, len(g26)))
    sc = StandardScaler().fit(X[tr]); m = LogisticRegression(C=0.1, max_iter=3000).fit(sc.transform(X[tr]), y[g26[tr]])
    F.loc[g26[te], "logit3"] = m.predict_proba(sc.transform(X[te]))[:, 1]
# sim + strength stack (no market needed), fit on earlier seasons
F["stack_str"] = E.stack_by_season(F.assign(strength_p=F.strength_p), ["strength_p", "sim_p"], ALL)

sets = {"2020": F.year == 2020, "2021": F.year == 2021, "2020-21": F.year.isin([2020, 2021]),
        "2022-24": F.year.isin([2022, 2023, 2024]), "2025": F.year == 2025,
        "2026 to 09-07 (seen)": (F.year == 2026) & (F.game_date <= "2026-09-07"),
        "2026 08-08..09-07 (prior holdout)": (F.year == 2026) & (F.game_date >= "2026-08-08") & (F.game_date <= "2026-09-07"),
        "2026 09-08..09-23 (unseen)": (F.year == 2026) & (F.game_date >= "2026-09-08"),
        "ALL TEST 2020-26": F.year.isin(TEST)}
L = lambda p: np.where(np.isfinite(p), E.ll(y, np.nan_to_num(p, nan=.5)), np.nan)
cols = {"sim_raw": F.sim_p.values, "sim_cal": F.sim_cal.values, "v1like_cal": F.v1_cal.values,
        "strength": F.strength_p.values, "sim+strength": F.stack_str.values, "market": F.mkt_p.values,
        "market_refit": F.mkt_only.values, "market+sim": F.stack_mkt.values, "V10": F.v10_p.values,
        "logit3": F.logit3.values, "home_rate": np.full(len(F), 0.53)}
LL = {k: L(v) for k, v in cols.items()}
print("=" * 100); print("WINNER log loss (lower is better); paired deltas are ref - model (positive = model better)")
OUT["win"] = {}
for nm, m in sets.items():
    m = m.values & np.isfinite(LL["sim_cal"])
    row = {"n": int(m.sum())}
    for k in cols:
        mm = m & np.isfinite(LL[k])
        if mm.sum() == m.sum() and m.sum() > 0:
            row[k] = float(LL[k][m].mean())
    row["acc_sim_cal"] = float(((F.sim_cal.values[m] > .5) == y[m]).mean())
    comps = [("sim_cal", "strength"), ("sim_cal", "v1like_cal"), ("sim+strength", "strength"),
             ("sim_cal", "market"), ("market+sim", "market_refit"), ("sim_cal", "V10"), ("sim_cal", "logit3")]
    for a, b in comps:
        mm = m & np.isfinite(LL[a]) & np.isfinite(LL[b])
        if mm.sum() == m.sum() and m.sum() > 0:
            row[f"{a} vs {b}"] = E.boot((LL[b] - LL[a])[mm], F.d.values[mm])
    OUT["win"][nm] = row
    print(f"\n{nm}  n={row['n']}  acc(sim_cal)={row['acc_sim_cal']*100:.2f}%")
    print("   " + "  ".join(f"{k} {row[k]:.5f}" for k in cols if k in row))
    for a, b in comps:
        if f"{a} vs {b}" in row: print(f"   {a:>12} vs {b:<13} {fmt(row[f'{a} vs {b}'])}")

# ---------------- totals ----------------
print("\n" + "=" * 100); print("TOTALS (lower is better)")
OUT["totals"] = {}
for nm, m in sets.items():
    m = m.values & F.crps_simcal.notna().values
    if m.sum() == 0: continue
    r = {"n": int(m.sum()), "bias_raw": float((F.mean_sim - F.tot)[m].mean()),
         "crps_simcal": float(F.crps_simcal[m].mean()), "crps_base": float(F.crps_base[m].mean()),
         "ls_simcal": float(F.ls_simcal[m].mean()), "ls_base": float(F.ls_base[m].mean()),
         "simcal vs base CRPS": E.boot((F.crps_base - F.crps_simcal).values[m], F.d.values[m]),
         "simcal vs base LS": E.boot((F.ls_base - F.ls_simcal).values[m], F.d.values[m])}
    mk = m & F.crps_mkt.notna().values & F.crps_simmkt.notna().values
    if mk.sum() > 100:
        r.update({"n_mkt": int(mk.sum()), "crps_mkt": float(F.crps_mkt[mk].mean()), "ls_mkt": float(F.ls_mkt[mk].mean()),
                  "ls_simmkt": float(F.ls_simmkt[mk].mean()), "crps_simmkt": float(F.crps_simmkt[mk].mean()),
                  "simcal vs mkt CRPS": E.boot((F.crps_mkt - F.crps_simcal).values[mk], F.d.values[mk]),
                  "sim-shape@mkt-mean vs mkt NB LS": E.boot((F.ls_mkt - F.ls_simmkt).values[mk], F.d.values[mk])})
        ou = mk & F.over_p.notna().values & F.sim_over.notna().values
        # O/U: sim P(over) Platt-calibrated on earlier seasons
        cal = np.full(len(F), np.nan)
        for yr in TEST:
            tr = (F.year < yr).values & F.sim_over.notna().values & F.over_p.notna().values
            te = (F.year == yr).values & ou
            if tr.sum() < 500 or te.sum() == 0: continue
            mm_ = LogisticRegression(C=1e4).fit(E.lg(F.sim_over.values[tr]).reshape(-1, 1), F.over_y.values[tr])
            cal[te] = mm_.predict_proba(E.lg(F.sim_over.values[te]).reshape(-1, 1))[:, 1]
        ok = ou & np.isfinite(cal)
        if ok.sum():
            ls_s = E.ll(F.over_y.values[ok], cal[ok]); ls_m = E.ll(F.over_y.values[ok], F.over_p.values[ok])
            r.update({"ou_n": int(ok.sum()), "ou_ll_sim": float(ls_s.mean()), "ou_ll_mkt": float(ls_m.mean()),
                      "ou sim vs mkt": E.boot(ls_m - ls_s, F.d.values[ok])})
    OUT["totals"][nm] = r
    print(f"\n{nm} n={r['n']} bias(raw mean) {r['bias_raw']:+.3f}")
    print(f"   CRPS simcal {r['crps_simcal']:.4f} base {r['crps_base']:.4f} | LS simcal {r['ls_simcal']:.4f} base {r['ls_base']:.4f}")
    print(f"   simcal vs base CRPS {fmt(r['simcal vs base CRPS'])}   LS {fmt(r['simcal vs base LS'])}")
    if "crps_mkt" in r:
        print(f"   market: CRPS {r['crps_mkt']:.4f} LS {r['ls_mkt']:.4f} | sim-shape@mkt-mean CRPS {r['crps_simmkt']:.4f} LS {r['ls_simmkt']:.4f}")
        print(f"   simcal vs mkt CRPS {fmt(r['simcal vs mkt CRPS'])}   sim-shape@mkt vs mkt LS {fmt(r['sim-shape@mkt-mean vs mkt NB LS'])}")
        if "ou_n" in r:
            print(f"   O/U n={r['ou_n']} ll sim {r['ou_ll_sim']:.5f} mkt {r['ou_ll_mkt']:.5f}  {fmt(r['ou sim vs mkt'])}")

# ---------------- run line ----------------
print("\n" + "=" * 100); print("RUN LINE (home -1.5 / +1.5 cover), sim Platt-calibrated on earlier seasons")
mh = Z["marg_hist"]                                           # index = margin + 15
rl = F.rl_home_spread.values
cover_p = np.full(len(F), np.nan)
ok = np.isfinite(rl)
thr = np.where(ok, -rl, 0)                                     # need margin > -spread
for i in np.flatnonzero(ok):
    cover_p[i] = mh[i, int(np.floor(thr[i])) + 16:].sum()
F["rl_sim"] = cover_p; F["rl_y"] = np.where(ok, ((F.h_runs - F.a_runs).values > -rl).astype(float), np.nan)
cal = np.full(len(F), np.nan)
for yr in TEST:
    tr = (F.year < yr).values & ok & F.rl_home_p.notna().values
    te = (F.year == yr).values & ok & F.rl_home_p.notna().values
    if tr.sum() < 500 or te.sum() == 0: continue
    X_ = np.column_stack([E.lg(F.rl_sim.values), rl])
    mm_ = LogisticRegression(C=1e4).fit(X_[tr], F.rl_y.values[tr]); cal[te] = mm_.predict_proba(X_[te])[:, 1]
okr = np.isfinite(cal)
if okr.sum():
    a_ = E.ll(F.rl_y.values[okr], cal[okr]); b_ = E.ll(F.rl_y.values[okr], F.rl_home_p.values[okr])
    OUT["runline"] = {"n": int(okr.sum()), "ll_sim": float(a_.mean()), "ll_mkt": float(b_.mean()), "sim vs mkt": E.boot(b_ - a_, F.d.values[okr])}
    print(OUT["runline"])

# ---------------- F5 ----------------
print("\n" + "=" * 100); print("F5 3-way (away lead / tie / home lead) log score")
OUT["f5"] = {}
for nm, m in sets.items():
    m = m.values & F.f5_ls_simcal.notna().values & F.f5_ls_mkt.notna().values
    if m.sum() == 0: continue
    r = {"n": int(m.sum()), "sim_cal": float(F.f5_ls_simcal[m].mean()), "mkt_or_strength_derived": float(F.f5_ls_mkt[m].mean()),
         "clim": float(F.f5_ls_clim[m].mean()), "sim vs mkt/strength": E.boot((F.f5_ls_mkt - F.f5_ls_simcal).values[m], F.d.values[m])}
    OUT["f5"][nm] = r
    print(f"{nm:34} n={r['n']} sim {r['sim_cal']:.5f} mkt/strength-derived {r['mkt_or_strength_derived']:.5f} clim {r['clim']:.5f}  {fmt(r['sim vs mkt/strength'])}")

# ---------------- props ----------------
print("\n" + "=" * 100); print("PROPS vs naive season-rate baselines (gain = baseline loss - sim loss)")
K_, B_ = P.score_props("full_x50", TEST)
OUT["props"] = {"starter_K": {"n": len(K_), "crps_sim": float(K_.crps_sim.mean()), "crps_base": float(K_.crps_base.mean()),
                              "ls_sim": float(K_.ls_sim.mean()), "ls_base": float(K_.ls_base.mean()),
                              "mae_sim": float((K_.mu_sim - K_.k).abs().mean()), "mae_base": float((K_.mu_base - K_.k).abs().mean()),
                              "crps gain": E.boot((K_.crps_base - K_.crps_sim).values, K_.date.values),
                              "ls gain": E.boot((K_.ls_base - K_.ls_sim).values, K_.date.values)}}
print("starter K", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in OUT["props"]["starter_K"].items()})
for t in ("h", "hr", "k"):
    ok = B_[f"p_{t}_base"].notna()
    ls = E.ll(B_[f"y_{t}"][ok].values, B_[f"p_{t}_sim"][ok].values); lb = E.ll(B_[f"y_{t}"][ok].values, B_[f"p_{t}_base"][ok].values)
    OUT["props"][f"batter_{t}>=1"] = {"n": int(ok.sum()), "ll_sim": float(ls.mean()), "ll_base": float(lb.mean()),
                                      "gain": E.boot(lb - ls, B_.date[ok].values),
                                      "rate": float(B_[f"y_{t}"][ok].mean()), "mean_p_sim": float(B_[f"p_{t}_sim"][ok].mean())}
    print(f"batter {t}>=1", OUT["props"][f"batter_{t}>=1"])

# ---------------- series ----------------
print("\n" + "=" * 100); print("SERIES (>=3 games inside one weekly fit; game-1 information only for the benchmarks)")
S = P.series_frame("full_x50", ALL)
for c in ("mkt_g1", "strength_g1", "sim_g1"):
    S[c + "_ser"] = np.nan
    for yr in TEST:
        tr = (S.year < yr) & S[c].notna(); te = (S.year == yr) & S[c].notna()
        if tr.sum() < 200 or te.sum() == 0: continue
        m = LogisticRegression(C=1e4).fit(E.lg(S[c][tr].values).reshape(-1, 1), S.y[tr])
        S.loc[te, c + "_ser"] = m.predict_proba(E.lg(S[c][te].values).reshape(-1, 1))[:, 1]
S["sim_series_cal"] = np.nan
for yr in TEST:
    tr = S.year < yr; te = S.year == yr
    m = LogisticRegression(C=1e4).fit(E.lg(S.sim_series[tr].values).reshape(-1, 1), S.y[tr])
    S.loc[te, "sim_series_cal"] = m.predict_proba(E.lg(S.sim_series[te].values).reshape(-1, 1))[:, 1]
OUT["series"] = {}
for nm, m in {"2020-21 (market)": S.year.isin([2020, 2021]), "ALL TEST 2020-26": S.year.isin(TEST)}.items():
    r = {"n": int(m.sum())}
    for c in ("sim_series_cal", "sim_g1_ser", "strength_g1_ser", "mkt_g1_ser"):
        mm = m & S[c].notna()
        if mm.sum() == m.sum():
            r[c] = {"ll": float(E.ll(S.y[mm].values, S[c][mm].values).mean()), "acc": float(((S[c][mm] > .5) == S.y[mm]).mean())}
    for a, b in (("sim_series_cal", "strength_g1_ser"), ("sim_series_cal", "mkt_g1_ser"), ("sim_series_cal", "sim_g1_ser")):
        if a in r and b in r:
            mm = m
            r[f"{a} vs {b}"] = E.boot(E.ll(S.y[mm].values, S[b][mm].values) - E.ll(S.y[mm].values, S[a][mm].values), S.date[mm].dt.strftime("%Y%m%d").values)
    OUT["series"][nm] = r
    print(nm, json.dumps(r, default=float))

# ---------------- lineup fallback ----------------
print("\n" + "=" * 100); print("LINEUP FALLBACK: previous game's lineup instead of the posted one")
OUT["lineup_fallback"] = {}
for yr in (2025, 2026):
    try:
        zp = E.load_runs("full_x50_prevlu", [yr])
    except FileNotFoundError:
        continue
    m = (F.year == yr).values & np.isfinite(F.sim_cal.values)
    pp = pd.Series(zp["p_home"], index=zp["game_pk"]).reindex(F.game_pk[m]).values
    # same calibration map as the actual-lineup model (fit on earlier seasons)
    tr = (F.year < yr).values
    cm = LogisticRegression(C=1e4).fit(E.lg(F.sim_p.values[tr]).reshape(-1, 1), y[tr])
    pc = cm.predict_proba(E.lg(pp).reshape(-1, 1))[:, 1]
    la, lp = LL["sim_cal"][m], E.ll(y[m], pc)
    OUT["lineup_fallback"][yr] = {"n": int(m.sum()), "ll_actual": float(la.mean()), "ll_prev": float(lp.mean()),
                                  "cost": E.boot(lp - la, F.d.values[m]), "corr_p": float(np.corrcoef(pp, F.sim_p.values[m])[0, 1])}
    print(yr, OUT["lineup_fallback"][yr])

json.dump(OUT, open("data/backtest_2026/rich/test_report.json", "w"), indent=1, default=float)
F.drop(columns=["h_lineup", "a_lineup"]).to_parquet("data/backtest_2026/rich/test_frame.parquet", index=False)
print("\nwrote data/backtest_2026/rich/test_report.json")
