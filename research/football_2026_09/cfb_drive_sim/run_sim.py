"""Walk-forward CFB drive sim: per (season, week) block, fit on the prior 2 seasons of drives
strictly before the block, simulate every game N times.
usage: run_sim.py VARIANT C TAU PRIOR_W S0 S1 [N] [SEED]"""
import sys, time, zlib, warnings, numpy as np, pandas as pd
from joblib import Parallel, delayed
warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/VTNX82W/Documents/personalDev/mlb/hanks_tank_ml/src/cfb")
from espn_data import resolve_team_divisions
import sim_core as sc
V, C, TAU, PW = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
S0, S1 = int(sys.argv[5]), int(sys.argv[6])
N = int(sys.argv[7]) if len(sys.argv) > 7 else 4000
SEED = int(sys.argv[8]) if len(sys.argv) > 8 else 0
g = pd.read_parquet("cfb_games_bq.parquet"); g = g[(g.season <= 2025)]
d = sc.prep_drives(pd.read_parquet("cfb_drives.parquet"), g)
teams = pd.Index(sorted(set(g.home_team) | set(g.away_team)))
fbs = {k: int(v == "fbs") for k, v in resolve_team_divisions(g).items()}
G = g[g.season.between(S0, S1) & g.home_won.notna()].copy()
G["t"] = G.season * sc.WPS + G.week
MR0, NM, NT = -100, 201, 171

def block(t_now, gb):
    import warnings; warnings.filterwarnings("ignore")
    tr = d[(d.t < t_now) & (d.t >= t_now - 2 * sc.WPS)]
    t0 = time.time(); m = sc.DriveModel(V, C=C, tau=TAU, prior_w=PW).fit(tr, t_now, teams, fbs); tf = time.time() - t0
    rows = []
    for r in gb.itertuples():
        rng = np.random.default_rng([SEED, zlib.crc32(r.game_id.encode())])
        t1 = time.time()
        hs, aw, ot = sc.simulate(m, r.home_team, r.away_team, bool(r.neutral_site), N=N, rng=rng, return_ot=True)
        ts = time.time() - t1
        mg = hs - aw; tot = hs + aw
        rows.append(dict(game_id=r.game_id, p_raw=(mg > 0).mean(), p_ot=ot.mean(), m_mean=mg.mean(), m_sd=mg.std(),
                         t_mean=tot.mean(), t_sd=tot.std(), h_mean=hs.mean(), a_mean=aw.mean(), sim_sec=ts, fit_sec=tf,
                         mh=np.bincount(np.clip(mg - MR0, 0, NM - 1).astype(int), minlength=NM).astype(np.int32).tolist(),
                         th=np.bincount(np.clip(tot, 0, NT - 1).astype(int), minlength=NT).astype(np.int32).tolist()))
    return rows

t0 = time.time()
res = Parallel(n_jobs=11)(delayed(block)(t, gb) for t, gb in G.groupby("t"))
out = pd.DataFrame([r for b in res for r in b])
tag = f"{V}_C{C:g}_t{TAU:g}_pw{PW:g}_N{N}_s{SEED}_{S0}_{S1}"
out.to_parquet(f"runs/sim_{tag}.parquet")
print(tag, len(out), f"{time.time()-t0:.0f}s; sim ms/game {1000*out.sim_sec.mean():.0f}; fit s/block {out.fit_sec.mean():.1f}", flush=True)
