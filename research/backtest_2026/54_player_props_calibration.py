"""Player stat projections from the frozen v2 sim: calibration backtest and correction.

    python3 research/backtest_2026/54_player_props_calibration.py exposure
    python3 research/backtest_2026/54_player_props_calibration.py sim YEAR [N]
    python3 research/backtest_2026/54_player_props_calibration.py eval PA_STATE_PARQUET

exposure  Slot survival S[s, k] = P(the starting batter still holds lineup slot s at the
          slot's (k+1)-th turn | that turn happens), from actual 2022-23 PAs (universal-DH
          era, never scored). Writes $PLAYER_OUT/exposure_2022_23.json.
sim       Frozen full_x50 config, weekly refit exactly as 42_run_v2.py (same seeds), with
          the new per-lane accumulators. Saves per-player pmfs, raw and exposure-adjusted,
          to $PLAYER_OUT/runs/players_full_x50_{YEAR}.npz.
eval      Joins actual box lines and scores calibration per stat:
            fit season 2024 -> per-stat thinning factor on the exposure-adjusted pmf;
            test seasons 2025 and 2026 (to 09-23) -> the gate.
          Starter IP_outs and ER actuals need per-PA outs/runs, which the 2026 statcast
          table lacks, so those two are tested on 2025 only. PA_STATE_PARQUET is a per-PA
          (game_pk, at_bat_number, inning, topbot, pitcher, outs0, runs, n_on0) extract of
          mlb_historical_data.statcast_pitches for 2024-25 (a read-only SELECT).
          Writes src/pa_sim/player_calibration.json.

Gate (all on the test seasons, per stat): |mean ratio - 1| <= 0.03, randomized-PIT
coverage of the central 90% within 0.02 of 0.90 and of the central 50% within 0.03 of
0.50; batter stats also |mean P(>=1) - observed| <= 0.01 and 10-bin ECE <= 0.01.
"""
import importlib
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, "src"); sys.path.insert(0, "research/backtest_2026")
warnings.filterwarnings("ignore")
from pa_sim import v2, dists  # noqa: E402

R = "data/backtest_2026/rich/"
OUT = os.environ.get("PLAYER_OUT", R)             # R is a read-only cache: point this elsewhere
OUT_RUNS = os.path.join(OUT, "runs/")
EXPO = os.path.join(OUT, "exposure_2022_23.json")
KX = 8
BAT = list(dists.BATTER_STATS)                      # PA H HR TB BB K
SP = list(dists.STARTER_STATS)                      # K BF IP_outs ER H_allowed BB_allowed
SPCAP = {s: dists.CAPS["sp_K" if s == "K" else s] for s in SP}
FIT_YEARS, TEST_YEARS = [2024], [2025, 2026]


# ------------------------------------------------------------------ exposure
def exposure(years=(2022, 2023)):
    pa = pd.read_parquet(R + "pa_all.parquet", columns=["game_pk", "game_year", "batter", "top", "at_bat_number"])
    pa = pa[pa.game_year.isin(years)].sort_values(["game_pk", "top", "at_bat_number"])
    g = pd.read_parquet(R + "games.parquet")
    g = g[g.year.isin(years)].set_index("game_pk")
    pa["i"] = pa.groupby(["game_pk", "top"]).cumcount()
    pa["slot"] = pa.i % 9; pa["k"] = np.minimum(pa.i // 9, KX - 1)
    pa = pa[pa.game_pk.isin(g.index)]
    lu_h = g.h_lineup.to_dict(); lu_a = g.a_lineup.to_dict()
    starter = [ (lu_a if t else lu_h)[pk][s] for pk, t, s in zip(pa.game_pk.values, pa.top.values, pa.slot.values)]
    pa["is_st"] = pa.batter.values == np.array(starter)
    t = pa.groupby(["slot", "k"]).is_st.agg(["mean", "size"]).reset_index()
    S = np.ones((9, KX))
    for r in t.itertuples():
        S[r[1], r[2]] = r[3] if r[4] >= 200 else np.nan
    for s in range(9):                               # thin cells: carry the last value
        for k in range(KX):
            if not np.isfinite(S[s, k]):
                S[s, k] = S[s, k - 1] if k else 1.0
    S = np.minimum.accumulate(S, axis=1)
    out = dict(years=list(years), n_pa=int(len(pa)), S=np.round(S, 5).tolist(),
               note="S[slot][k] = P(starting batter holds slot at its (k+1)-th turn | turn happens); k>=7 pooled")
    json.dump(out, open(EXPO, "w"), indent=1)
    print(json.dumps(out, indent=1))


# ------------------------------------------------------------------ sim
def sim(year, n_ep=3000):
    run = importlib.import_module("42_run_v2")
    t0 = time.time()
    pa, meta, games = run.load()
    pa15 = pa[pa.game_year == 2015]
    venue_of = dict(zip(meta.game_pk, meta.venue_id))
    fz = json.load(open(R + "FROZEN_CONFIG.json"))
    cfg = v2.Config(**{k: v for k, v in fz["config"].items()}, tag=fz["variant"])
    xtab = v2.build_xtable(pa15)
    D = v2.Data(pa[pa.game_date < pd.Timestamp(f"{year}-12-31")], venue_of, None, xtab)
    vh = games.groupby("venue_id").home_team.agg(lambda s: s.value_counts().index[0]).to_dict()
    hook = pd.read_parquet(R + "hook.parquet").merge(
        D.starts[["game_pk", "pitcher", "exp_p"]].rename(columns={"pitcher": "sp"}), on=["game_pk", "sp"], how="left")
    hook["exp_p"] = hook.exp_p.fillna(88.0)
    trans = pd.read_parquet(R + "trans.parquet")
    eng = v2.Engine(D, cfg, trans, hook, vh, run.est_tto(pa15), None)
    hand = pa.groupby("pitcher").p_throws.agg(lambda s: int(s.mean() >= 0.5)).to_dict()
    st = pd.DataFrame(dict(b=D.bi, t=D.throws, s=D.stand)).groupby(["b", "t"]).s.mean()
    eng.prepare(hand, {k: int(v >= 0.5) for k, v in st.items()})
    S = np.array(json.load(open(EXPO))["S"])
    g = games[games.year == year].copy()
    g["week"] = g.game_date.dt.to_period("W").dt.start_time
    outs = []
    for wk, gw in g.groupby("week"):
        eng.fit(wk)
        specs = [v2.GameSpec(list(r.h_lineup), list(r.a_lineup), r.h_sp, r.a_sp, r.home_team, r.away_team,
                             int(r.venue_id) if pd.notna(r.venue_id) else -1,
                             int(r.sched_inn) if pd.notna(r.sched_inn) else 9, year >= 2020)
                 for r in gw.itertuples()]
        res = v2.simulate(eng, specs, n=n_ep, seed=int(wk.value // 10**9) % 100000, exposure=S)
        G = len(specs); gi = res["gi"]
        o = {"game_pk": gw.game_pk.values}
        for s, key in dists.BATTER_STATS.items():
            o[f"b_{s}"] = dists.pmfs(res[key], gi, G, dists.CAPS[s]).astype(np.float32)
            o[f"bx_{s}"] = dists.pmfs(res[key + "_x"], gi, G, dists.CAPS[s]).astype(np.float32)
        for s, key in dists.STARTER_STATS.items():
            o[f"sp_{s}"] = dists.pmfs(res[key], gi, G, SPCAP[s]).astype(np.float32)
        outs.append(o)
    cat = {k: np.concatenate([o[k] for o in outs]) for k in outs[0]}
    os.makedirs(OUT_RUNS, exist_ok=True)
    np.savez_compressed(OUT_RUNS + f"players_full_x50_{year}.npz", **cat)
    print(f"players {year}: {len(cat['game_pk'])} games, {time.time() - t0:.0f}s", flush=True)


# ------------------------------------------------------------------ actuals
def batter_actuals():
    pa = pd.read_parquet(R + "pa_all.parquet", columns=["game_pk", "game_year", "batter", "cls"])
    pa = pa[pa.game_year >= 2024]
    tb = v2.TB_OF[pa.cls.values]
    a = pd.DataFrame(dict(game_pk=pa.game_pk.values, batter=pa.batter.values, PA=1,
                          H=np.isin(pa.cls.values, v2.HITS).astype(int), HR=(pa.cls.values == v2.HR_).astype(int),
                          TB=tb.astype(int), BB=(pa.cls.values == v2.BB_).astype(int), K=(pa.cls.values == v2.K_).astype(int)))
    return a.groupby(["game_pk", "batter"]).sum().reset_index(), set(pa.game_pk.unique())


def starter_actuals(pa_state):
    pa = pd.read_parquet(R + "pa_all.parquet", columns=["game_pk", "game_year", "pitcher", "cls", "is_starter"])
    pa = pa[(pa.game_year >= 2024) & pa.is_starter]
    c = pa.cls.values
    a = pd.DataFrame(dict(game_pk=pa.game_pk.values, pitcher=pa.pitcher.values, K=(c == v2.K_).astype(int), BF=1,
                          H_allowed=np.isin(c, v2.HITS).astype(int), BB_allowed=(c == v2.BB_).astype(int)))
    a = a.groupby(["game_pk", "pitcher"]).sum().reset_index()
    # outs and charged runs from per-PA states (2024-25 only)
    s = pd.read_parquet(pa_state).sort_values(["game_pk", "at_bat_number"])
    s["half"] = s.inning.astype(str) + s.topbot
    s["outs1"] = s.groupby(["game_pk", "half"]).outs0.shift(-1).fillna(3)
    s["n_on1"] = s.groupby(["game_pk", "half"]).n_on0.shift(-1).fillna(0)
    s["side"] = np.where(s.topbot == "Top", 1, 0)                 # defensive side: Top -> home pitches
    first = s.groupby(["game_pk", "side"]).pitcher.transform("first")
    s["sp"] = s.pitcher == first
    rows = []
    for (pk, side), g in s.groupby(["game_pk", "side"], sort=False):
        spid = g.pitcher.iloc[0]
        m = g.sp.values
        outs = int(np.clip(g.outs1.values[m] - g.outs0.values[m], 0, 3).sum())
        runs = int(g.runs.values[m].sum())
        last = np.nonzero(m)[0].max()
        if last + 1 < len(g) and g.half.values[last + 1] == g.half.values[last]:
            inh = int(g.n_on1.values[last])
            j = last + 1
            while inh > 0 and j < len(g) and g.half.values[j] == g.half.values[last]:
                ch = min(int(g.runs.values[j]), inh); runs += ch
                inh = min(inh - ch, int(g.n_on1.values[j]) if (j + 1 < len(g) and g.half.values[j + 1] == g.half.values[j]) else 0)
                j += 1
        rows.append(dict(game_pk=pk, pitcher=spid, IP_outs=outs, ER=runs))
    b = pd.DataFrame(rows)
    return a.merge(b, on=["game_pk", "pitcher"], how="left")


# ------------------------------------------------------------------ metrics
def pit_cov(P, y, seed=0):
    """Randomized PIT for integer outcomes; returns (cov90, cov50)."""
    rng = np.random.default_rng(seed)
    y = np.clip(y.astype(int), 0, P.shape[1] - 1)
    F = np.cumsum(P, 1)
    lo = np.where(y > 0, F[np.arange(len(y)), np.maximum(y - 1, 0)], 0.0)
    hi = F[np.arange(len(y)), y]
    u = lo + rng.random(len(y)) * (hi - lo)
    return float(np.mean((u > 0.05) & (u < 0.95))), float(np.mean((u > 0.25) & (u < 0.75)))


def ece(p, y, bins=10):
    q = pd.qcut(p, bins, labels=False, duplicates="drop")
    d = pd.DataFrame(dict(p=p, y=y, q=q)).groupby("q").agg(p=("p", "mean"), y=("y", "mean"), n=("p", "size"))
    return float((d.n * (d.p - d.y).abs()).sum() / d.n.sum())


def metrics(P, y, batter):
    k = np.arange(P.shape[1])
    mu = (P * k).sum(1)
    out = dict(n=int(len(y)), mean_pred=float(mu.mean()), mean_obs=float(y.mean()),
               ratio=float(mu.mean() / max(y.mean(), 1e-9)))
    out["cov90"], out["cov50"] = pit_cov(P, y)
    if batter:
        p1 = 1 - P[:, 0]; y1 = (y >= 1).astype(float)
        out.update(p1_pred=float(p1.mean()), p1_obs=float(y1.mean()), ece=ece(p1, y1),
                   logloss=float(-np.mean(y1 * np.log(np.clip(p1, 1e-4, 1)) + (1 - y1) * np.log(np.clip(1 - p1, 1e-4, 1)))))
    return out


def passes(m, batter):
    ok = abs(m["ratio"] - 1) <= 0.03 and abs(m["cov90"] - 0.90) <= 0.02 and abs(m["cov50"] - 0.50) <= 0.03
    if batter:
        ok = ok and abs(m["p1_pred"] - m["p1_obs"]) <= 0.01 and m["ece"] <= 0.01
    return bool(ok)


def fit_thin(P, y):
    """Largest-likelihood-free choice: the m that matches the mean, capped at 1."""
    k = np.arange(P.shape[1])
    return float(min(1.0, y.mean() / max((P * k).sum(1).mean(), 1e-9)))


def thin_all(P, m):
    if m >= 1:
        return P
    from scipy.stats import binom
    k = np.arange(P.shape[1])
    M = binom.pmf(k[None, :], k[:, None], m)
    Q = P @ M
    return Q / Q.sum(1, keepdims=True)


def frames(years, pa_state):
    """Row-aligned (batter frame, starter frame) with pmf row indices into each season's npz."""
    ba, have = batter_actuals()
    sa = starter_actuals(pa_state)
    g = pd.read_parquet(R + "games.parquet").set_index("game_pk")
    Bs, Ss = [], []
    for y in years:
        pks = np.load(OUT_RUNS + f"players_full_x50_{y}.npz")["game_pk"]
        idx = pd.DataFrame({"game_pk": pks, "i": np.arange(len(pks))})
        idx = idx[idx.game_pk.isin(g.index) & idx.game_pk.isin(have)]
        gg = g.loc[idx.game_pk]
        rows = []
        for side, col in ((0, "a_lineup"), (1, "h_lineup")):
            lu = np.stack(gg[col].map(lambda v: np.asarray(v, np.int64)).values)
            for s_ in range(9):
                rows.append(pd.DataFrame({"year": y, "game_pk": idx.game_pk.values, "i": idx.i.values,
                                          "side": side, "slot": s_, "batter": lu[:, s_]}))
        b = pd.concat(rows).merge(ba, on=["game_pk", "batter"], how="inner")
        Bs.append(b)
        sp = pd.concat([pd.DataFrame({"year": y, "game_pk": idx.game_pk.values, "i": idx.i.values,
                                      "side": side, "pitcher": gg[col].astype(np.int64).values})
                        for side, col in ((0, "a_sp"), (1, "h_sp"))])
        Ss.append(sp.merge(sa, on=["game_pk", "pitcher"], how="inner"))
    return pd.concat(Bs, ignore_index=True), pd.concat(Ss, ignore_index=True)


def evaluate(pa_state):
    years = FIT_YEARS + TEST_YEARS
    B, SPR = frames(years, pa_state)
    report = {"fit_years": FIT_YEARS, "test_years": TEST_YEARS, "n_batter_games": int(len(B)),
              "n_starts": int(len(SPR)), "batter": {}, "starter": {}}
    print("batter-games", len(B), "starts", len(SPR), flush=True)

    def gather(frame, key, batter):
        out = []
        for y in years:
            m = (frame.year == y).values
            arr = np.load(OUT_RUNS + f"players_full_x50_{y}.npz")[key]
            f = frame[m]
            out.append((np.flatnonzero(m), arr[f.i.values, f.side.values, f.slot.values] if batter
                        else arr[f.i.values, f.side.values]))
        P = np.zeros((len(frame), out[0][1].shape[-1]))
        for ii, v in out:
            P[ii] = v
        return P

    yb = B.year.values; ys = SPR.year.values
    for s in BAT:
        y = B[s].values.astype(float)
        P_raw, P_x = gather(B, f"b_{s}", True), gather(B, f"bx_{s}", True)
        fit = np.isin(yb, FIT_YEARS); te = np.isin(yb, TEST_YEARS)
        m = fit_thin(P_x[fit], y[fit])
        P_c = thin_all(P_x, m)
        rep = {"thin": round(m, 4)}
        for tag, P in (("raw", P_raw), ("exposure", P_x), ("exposure+thin", P_c)):
            rep[tag] = {"fit": metrics(P[fit], y[fit], True), "test": metrics(P[te], y[te], True)}
            for yy in TEST_YEARS:
                rep[tag][str(yy)] = metrics(P[yb == yy], y[yb == yy], True)
        rep["calibrated"] = passes(rep["exposure+thin"]["test"], True) and all(
            passes(rep["exposure+thin"][str(yy)], True) for yy in TEST_YEARS)
        report["batter"][s] = rep
        print("BAT", s, json.dumps(rep, default=float), flush=True)
    for s in SP:
        y = SPR[s].values.astype(float); ok = np.isfinite(y)
        P = gather(SPR, f"sp_{s}", False)
        fit = np.isin(ys, FIT_YEARS) & ok; te = np.isin(ys, TEST_YEARS) & ok
        m = fit_thin(P[fit], y[fit])
        P_c = thin_all(P, m)
        rep = {"thin": round(m, 4)}
        tys = [yy for yy in TEST_YEARS if ((ys == yy) & ok).sum() > 0]
        for tag, PP in (("raw", P), ("thinned", P_c)):
            rep[tag] = {"fit": metrics(PP[fit], y[fit], False), "test": metrics(PP[te], y[te], False)}
            for yy in tys:
                mm = (ys == yy) & ok
                rep[tag][str(yy)] = metrics(PP[mm], y[mm], False)
        rep["test_years"] = tys
        rep["calibrated"] = passes(rep["thinned"]["test"], False) and all(passes(rep["thinned"][str(yy)], False) for yy in tys)
        report["starter"][s] = rep
        print("SP", s, json.dumps(rep, default=float), flush=True)
    json.dump(report, open(os.path.join(OUT, "player_calibration_report.json"), "w"), indent=1, default=float)
    json.dump(calibration_file(report), open(os.path.join(OUT, "player_calibration.json"), "w"), indent=1)
    return report


def _fmt(m, batter):
    s = f"mean {m['mean_pred']:.3f} vs {m['mean_obs']:.3f} (ratio {m['ratio']:.3f}), PIT90 {m['cov90']:.3f}, PIT50 {m['cov50']:.3f}"
    if batter and np.isfinite(m.get("ece", np.nan)):
        s += f", P(>=1) {m['p1_pred']:.3f} vs {m['p1_obs']:.3f}, ECE {m['ece']:.4f}"
    return s


def _why(m, batter):
    bad = []
    if abs(m["ratio"] - 1) > 0.03: bad.append(f"mean ratio {m['ratio']:.3f}")
    if abs(m["cov90"] - 0.90) > 0.02: bad.append(f"PIT90 {m['cov90']:.3f}")
    if abs(m["cov50"] - 0.50) > 0.03: bad.append(f"PIT50 {m['cov50']:.3f}")
    if batter and np.isfinite(m.get("ece", np.nan)):
        if abs(m["p1_pred"] - m["p1_obs"]) > 0.01: bad.append(f"P(>=1) off {m['p1_pred'] - m['p1_obs']:+.3f}")
        if m["ece"] > 0.01: bad.append(f"ECE {m['ece']:.4f}")
    return bad


def calibration_file(report):
    """The production calibration (src/pa_sim/player_calibration.json) from a report."""
    expo = json.load(open(EXPO))
    out = {"version": "player_cal_v1",
           "fitted_at": time.strftime("%Y-%m-%d"),
           "source": "research/backtest_2026/54_player_props_calibration.py (frozen full_x50 sim, N=3000, weekly refit)",
           "gate": "test seasons, each: |mean ratio-1|<=0.03, randomized-PIT central 90% within 0.02 of 0.90 and "
                   "central 50% within 0.03 of 0.50; batters also |P(>=1) - observed|<=0.01 and 10-bin ECE<=0.01",
           "exposure": {"years": expo["years"], "S": expo["S"], "note": expo["note"]},
           "batter": {}, "starter": {}}
    for role, batter, final in (("batter", True, "exposure+thin"), ("starter", False, "thinned")):
        for st, rep in report[role].items():
            tys = [str(y) for y in (TEST_YEARS if batter else rep["test_years"])]
            fails = sorted({f"{y}: {b}" for y in tys for b in _why(rep[final][y], batter)})
            ok = rep["calibrated"]
            method = ("slot-survival exposure (late substitutions, fit 2022-23) + " if batter else "") + \
                (f"binomial thinning x{rep['thin']:.4f} (fit {FIT_YEARS[0]})" if rep["thin"] < 1 else "no thinning (sim under-predicts; left as is)")
            note = f"{st}: {method}. Out of sample " + "; ".join(f"{y} {_fmt(rep[final][y], batter)}" for y in tys) + \
                f". Raw sim {'/'.join(tys)}: " + "; ".join(_fmt(rep['raw'][y], batter) for y in tys) + "."
            if role == "starter" and st == "ER":
                ok = False
                note += (" Checked against runs CHARGED to the starter (own PAs + inherited runners, earned and "
                         "unearned); official earned runs are not in the backtest data, so not marked calibrated.")
            elif ok:
                note += " Passes the gate: calibrated."
            else:
                note += " Fails the gate (" + "; ".join(fails) + "): not calibrated."
            if role == "starter" and st in ("IP_outs", "ER"):
                note += " Tested on 2025 only: the 2026 statcast table has no per-PA outs/runs."
            if st == "BB":
                note += " BB here includes HBP and catcher's interference."
            out[role][st] = {"calibrated": bool(ok), "apply_thin": bool(rep["thin"] < 1), "thin": float(rep["thin"]),
                             "note": note}
    return out


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "exposure":
        exposure()
    elif cmd == "sim":
        sim(int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 3000)
    elif cmd == "eval":
        evaluate(sys.argv[2])
