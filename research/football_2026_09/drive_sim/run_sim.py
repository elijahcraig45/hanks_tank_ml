"""Walk-forward drive sim: for each (season, week) block, fit on the prior 2 seasons of drives
(strictly earlier weeks) and simulate every game in the block N times.
usage: run_sim.py VARIANT [N] [seed] [first_season] [last_season]"""
import zlib, sys, time, warnings, numpy as np, pandas as pd
from joblib import Parallel, delayed
warnings.filterwarnings("ignore")
import sim_core as sc
V = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 0
S0 = int(sys.argv[4]) if len(sys.argv) > 4 else 2010; S1 = int(sys.argv[5]) if len(sys.argv) > 5 else 2025
import os
CFG = dict(C=float(os.environ.get("SIM_C", 0.01)), tau=float(os.environ.get("SIM_TAU", 64.0)), ktau=6.0, pace_alpha=200.0)
TAG = os.environ.get("SIM_TAG", "")   # default config frozen from tune.py (2010-16 drive log loss)
FR = {"OAK": "LV", "SD": "LAC", "STL": "LA"}
g = pd.read_csv("../nfl.csv")
d = sc.prep_drives(pd.read_parquet("nfl_drives.parquet"), g)
teams = pd.Index(sorted(d.posteam.unique()))
G = g[g.season.between(S0, S1) & g.result.notna() & (g.result != 0)].copy()
G["t"] = G.season * sc.WPS + G.week
MR = np.arange(-80, 81); TR = np.arange(0, 131)

def block(t_now, gb):
    import warnings; warnings.filterwarnings("ignore")
    tr = d[(d.t < t_now) & (d.t >= t_now - 2 * sc.WPS)]
    t0 = time.time()
    m = sc.DriveModel(V, **CFG).fit(tr, t_now, teams)
    tf = time.time() - t0
    rows = []
    for _, r in gb.iterrows():
        rng = np.random.default_rng([SEED, zlib.crc32(r.game_id.encode())])
        t1 = time.time()
        hs, aw = sc.simulate(m, FR.get(r.home_team, r.home_team), FR.get(r.away_team, r.away_team),
                             r.location == "Neutral", r.season, r.game_type, N=N, rng=rng)
        ts = time.time() - t1
        mg = hs - aw; tot = hs + aw
        pw, pl_ = (mg > 0).mean(), (mg < 0).mean()
        rows.append(dict(game_id=r.game_id, p_raw=pw / (pw + pl_), p_tie=(mg == 0).mean(),
                         m_mean=mg.mean(), m_med=np.median(mg), t_mean=tot.mean(), t_med=np.median(tot),
                         t_sd=tot.std(), h_mean=hs.mean(), a_mean=aw.mean(), sim_sec=ts, fit_sec=tf,
                         mh=np.bincount(np.clip(mg, -80, 80).astype(int) + 80, minlength=161).astype(np.int32),
                         th=np.bincount(np.clip(tot, 0, 130).astype(int), minlength=131).astype(np.int32)))
    return rows

import os
os.environ.setdefault("PYTHONHASHSEED", "0")
t0 = time.time()
res = Parallel(n_jobs=10)(delayed(block)(t, gb) for t, gb in G.groupby("t"))
out = pd.DataFrame([r for b in res for r in b])
out["mh"] = out.mh.apply(list); out["th"] = out.th.apply(list)
out.to_parquet(f"sim_{V}{TAG}_N{N}_s{SEED}_{S0}_{S1}.parquet")
print(V, len(out), f"{time.time()-t0:.0f}s total; sim ms/game {1000*out.sim_sec.mean():.0f}; fit s/block {out.fit_sec.mean():.2f}")
