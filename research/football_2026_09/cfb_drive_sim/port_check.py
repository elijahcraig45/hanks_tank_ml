import sys, time, warnings, dataclasses, resource, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
W = "/Users/VTNX82W/Documents/personalDev/worktrees/ml-cfb-sim/src"
sys.path[:0] = [W + "/nfl", W + "/cfb", W]
import drive_sim as ds
from espn_data import resolve_team_divisions
g = pd.read_parquet("cfb_games_bq.parquet"); g = g[g.season <= 2025]
fbs = {k: int(v == "fbs") for k, v in resolve_team_divisions(g).items()}
teams = pd.Index(sorted(set(g.home_team) | set(g.away_team)))
ref = pd.read_parquet("runs/sim_c_C0.1_t16_pw1_N4000_s0_2022_2025.parquet").set_index("game_id")
raw = pd.read_parquet("cfb_drives.parquet")
S, Wk = int(sys.argv[1]), int(sys.argv[2])
G = g[(g.season == S) & (g.week == Wk) & g.home_won.notna()]
for sparse in (False, True):
    cfg = dataclasses.replace(ds.CFB, sparse=sparse)
    d = ds.prep_drives(raw, g, cfg)
    t0 = time.time(); m = ds.fit_week(d, g, S, Wk, teams=teams, cfg=cfg, fbs=fbs); tf = time.time() - t0
    exact = 0; dp = []
    for r in G.itertuples():
        hs, aw, _ = ds.simulate(m, r.home_team, r.away_team, bool(r.neutral_site), S, "REG", N=4000, rng=ds.game_rng(r.game_id))
        mg = hs - aw
        h = np.bincount(np.clip(mg + 100, 0, 200).astype(int), minlength=201)
        exact += np.array_equal(h, np.asarray(ref.loc[r.game_id].mh)); dp.append((mg > 0).mean() - ref.loc[r.game_id].p_raw)
    print(f"sparse={sparse}: fit {tf:.1f}s, exact histograms {exact}/{len(G)}, max |dp| {np.max(np.abs(dp)):.4f}, "
          f"peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6:.0f} MB", flush=True)
