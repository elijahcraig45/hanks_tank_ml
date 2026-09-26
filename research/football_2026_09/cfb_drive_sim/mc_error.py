import numpy as np, pandas as pd
B = pd.read_parquet("baselines.parquet").set_index("game_id")
a = pd.read_parquet("runs/sim_c_C0.1_t16_pw1_N4000_s0_2022_2025.parquet").set_index("game_id")
b = pd.read_parquet("runs/sim_c_C0.1_t16_pw1_N4000_s1_2023_2025.parquet").set_index("game_id")
j = a.join(b, rsuffix="_1", how="inner").join(B[["home_won", "ridge", "margin", "total"]])
ll = lambda y, p: float(-np.mean(y * np.log(np.clip(p, 1e-4, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-4, 1))))
dp = j.p_raw - j.p_raw_1
print(f"n={len(j)}; MC sd per run: p {dp.std()/np.sqrt(2):.4f} (theory {np.sqrt(j.p_raw*(1-j.p_raw)/4000).mean():.4f}), "
      f"margin mean {(j.m_mean-j.m_mean_1).std()/np.sqrt(2):.3f} pts, total mean {(j.t_mean-j.t_mean_1).std()/np.sqrt(2):.3f} pts")
print(f"log loss seed0 {ll(j.home_won, j.p_raw):.5f} seed1 {ll(j.home_won, j.p_raw_1):.5f}")
print(f"model error scale: sd(p_sim - p_ridge) {(j.p_raw-j.ridge).std():.4f}; RMSE margin {np.sqrt(((j.m_mean-j.margin)**2).mean()):.2f}, total {np.sqrt(((j.t_mean-j.total)**2).mean()):.2f}")
print(f"runtime: sim ms/game {1000*a.sim_sec.mean():.0f} (N=4000), fit s/block {a.fit_sec.mean():.0f} (11 parallel fits)")
