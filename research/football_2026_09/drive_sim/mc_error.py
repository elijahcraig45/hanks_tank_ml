"""Monte Carlo error vs model error: seed-to-seed and N=4000 vs N=16000."""
import numpy as np, pandas as pd
from sklearn.metrics import log_loss
B = pd.read_parquet("baselines.parquet").set_index("game_id")
a = pd.read_parquet("sim_b_N4000_s0_2010_2025.parquet").set_index("game_id")
b = pd.read_parquet("sim_b_N4000_s1_2010_2025.parquet").set_index("game_id")
c = pd.read_parquet("sim_b_N16000_s0_2023_2024.parquet").set_index("game_id")
j = a.join(b, rsuffix="_1").join(B[["season", "home_won", "margin", "total", "ridge"]])
w = j[j.season.between(2017, 2024)]
dp = w.p_raw - w.p_raw_1
print(f"seed-to-seed (N=4000): sd(p diff)={dp.std():.4f} -> per-run MC sd {dp.std()/np.sqrt(2):.4f}; "
      f"theory sqrt(p(1-p)/N)={np.sqrt((w.p_raw*(1-w.p_raw)/4000)).mean():.4f}")
for col in ["m_mean", "t_mean"]:
    print(f"  {col}: MC sd per run {(w[col]-w[col+'_1']).std()/np.sqrt(2):.3f} pts")
ll0 = log_loss(w.home_won, w.p_raw); ll1 = log_loss(w.home_won, w.p_raw_1)
print(f"log loss seed0 {ll0:.5f} seed1 {ll1:.5f} diff {ll0-ll1:+.5f}")
print(f"model error scale: sd(p_sim - p_ridge)={(w.p_raw-w.ridge).std():.4f}; RMSE of t_mean vs actual {np.sqrt(((w.t_mean-w.total)**2).mean()):.2f}")
k = c.join(a, rsuffix="_4k", how="inner").join(B[["home_won"]])
print(f"N=16000 vs N=4000 (2023-24, n={len(k)}): sd(p diff)={(k.p_raw-k.p_raw_4k).std():.4f}; "
      f"ll {log_loss(k.home_won,k.p_raw):.5f} vs {log_loss(k.home_won,k.p_raw_4k):.5f}; ms/game {1000*c.sim_sec.mean():.0f} vs {1000*a.loc[k.index].sim_sec.mean():.0f}")
