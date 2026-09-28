"""Starter strikeouts: re-examination of the "not calibrated" flag and a spread fix.

    python3 research/backtest_2026/55_starter_k_recal.py [--write]

Reads the per-start K pmfs saved by 54_player_props_calibration.py sim
($PLAYER_OUT/runs/players_full_x50_{2024,2025,2026}.npz) and the actual starter strikeouts
from data/backtest_2026/rich/pa_all.parquet (read-only caches, no BigQuery).

Protocol is 54_'s: fit on 2024 only, gate on 2025 and 2026 (to 09-23) separately.

What it found (2026-09-28, 13,613 starts):
  * The mean is fine. After the existing thinning (x0.9631) the mean ratio is 0.986 (2025)
    and 0.993 (2026). No calendar effect: September errors are -0.06/+0.10/-0.08 by season
    and the final week of 2024/2025 is -0.37/+0.33 (opposite signs), so no late-season term.
  * The spread is not. The simulated pmf is too narrow: mean predictive sd 1.99 vs a
    residual sd of 2.25, randomized-PIT central 90% covers 0.84-0.86 and central 50% covers
    0.45, and the calibration slope of actual on projected is 1.07-1.19. Random thinning
    (beta mixture) cannot fix this: it only shrinks counts, and the upper tail is short too.
  * Fix: after thinning, recalibrate the CDF, F' = BetaCDF(F; a, b), fitted by maximum
    likelihood on 2024. It passes the gate in both test seasons (see RESULTS printed).

--write patches starter.K in src/pa_sim/player_calibration.json (run after 54_ eval,
whose calibration_file() does not know about cdf_recal).
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, "src")
from pa_sim import dists  # noqa: E402

R = "data/backtest_2026/rich/"
RUNS = os.path.join(os.environ.get("PLAYER_OUT", R), "runs/")
CAL = "src/pa_sim/player_calibration.json"
FIT, TEST = [2024], [2025, 2026]


def frame():
    pa = pd.read_parquet(R + "pa_all.parquet", columns=["game_pk", "game_year", "pitcher", "cls", "is_starter"])
    pa = pa[(pa.game_year >= 2024) & pa.is_starter]
    act = pd.DataFrame(dict(game_pk=pa.game_pk.values, pitcher=pa.pitcher.values,
                            K=(pa.cls.values == 0).astype(int))).groupby(["game_pk", "pitcher"]).K.sum()
    g = pd.read_parquet(R + "games.parquet").set_index("game_pk")
    Ps, ys, yrs = [], [], []
    for y in FIT + TEST:
        z = np.load(RUNS + f"players_full_x50_{y}.npz")
        pks = z["game_pk"]
        for side, col in ((0, "a_sp"), (1, "h_sp")):
            for i in np.flatnonzero(np.isin(pks, g.index)):
                key = (int(pks[i]), int(g.at[pks[i], col]))
                if key in act.index:
                    Ps.append(z["sp_K"][i, side]); ys.append(int(act.loc[key])); yrs.append(y)
    P = np.asarray(Ps, float)
    return P / P.sum(1, keepdims=True), np.asarray(ys), np.asarray(yrs)


def pit(P, y, seed=0):
    rng = np.random.default_rng(seed)
    C = np.cumsum(P, 1)
    lo = np.where(y > 0, C[np.arange(len(y)), np.maximum(y - 1, 0)], 0.0)
    hi = C[np.arange(len(y)), y]
    u = lo + rng.random(len(y)) * (hi - lo)
    return float(np.mean((u > .05) & (u < .95))), float(np.mean((u > .25) & (u < .75)))


def metrics(P, y):
    k = np.arange(P.shape[1])
    c90, c50 = pit(P, y)
    ratio = float((P * k).sum(1).mean() / y.mean())
    logs = float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-9, 1))))
    ok = abs(ratio - 1) <= .03 and abs(c90 - .9) <= .02 and abs(c50 - .5) <= .03
    return dict(n=int(len(y)), ratio=ratio, cov90=c90, cov50=c50, logscore=logs, passes=bool(ok))


def apply_all(P, cal):
    return np.stack([dists.apply_calibration(p, cal) for p in P])


def main(write=False):
    P, y, yr = frame()
    base = json.load(open(CAL))["starter"]["K"]
    thin_only = {"thin": base["thin"], "apply_thin": True}
    Pt = apply_all(P, thin_only)
    fit = np.isin(yr, FIT)

    def nll(t):
        Q = apply_all(Pt[fit], {"apply_thin": False, "cdf_recal": {"a": np.exp(t[0]), "b": np.exp(t[1])}})
        return -np.mean(np.log(np.clip(Q[np.arange(fit.sum()), y[fit]], 1e-12, None)))

    a, b = np.exp(minimize(nll, [0.0, 0.0], method="Nelder-Mead").x)
    new = {**thin_only, "cdf_recal": {"a": round(float(a), 4), "b": round(float(b), 4)}}
    Pn = apply_all(P, new)
    res = {}
    for tag, PP in (("raw", P), ("thinned", Pt), ("thinned+recal", Pn)):
        res[tag] = {str(s): metrics(PP[yr == s], y[yr == s]) for s in FIT + TEST}
    print(json.dumps({"cdf_recal": new["cdf_recal"], "results": res}, indent=1))
    if write:
        cal = json.load(open(CAL))
        ok = all(res["thinned+recal"][str(s)]["passes"] for s in TEST)
        f = lambda m: (f"mean ratio {m['ratio']:.3f}, PIT90 {m['cov90']:.3f}, PIT50 {m['cov50']:.3f}, "
                       f"log score {m['logscore']:.4f}")
        cal["starter"]["K"].update(
            calibrated=ok, cdf_recal=new["cdf_recal"],
            note=(f"K: binomial thinning x{base['thin']} (mean, fit 2024) + CDF recalibration "
                  f"BetaCDF(a={new['cdf_recal']['a']}, b={new['cdf_recal']['b']}) (spread, fit 2024; "
                  "the raw pmf is too narrow). Out of sample, thinned+recal: "
                  + "; ".join(f"{s} {f(res['thinned+recal'][str(s)])}" for s in TEST)
                  + ". Thinning alone: "
                  + "; ".join(f"{s} {f(res['thinned'][str(s)])}" for s in TEST)
                  + (". Passes the gate: calibrated." if ok else ". Fails the gate: not calibrated.")
                  + " research/backtest_2026/55_starter_k_recal.py"))
        json.dump(cal, open(CAL, "w"), indent=1)
    return res


if __name__ == "__main__":
    main(write="--write" in sys.argv)
