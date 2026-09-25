"""Game-level shrinkage check on TUNING years only (2010-16): raw sim win-prob log loss and
margin-slope (how compressed the sim's margins are) for several (C, tau)."""
import glob, numpy as np, pandas as pd
from sklearn.metrics import log_loss
from sklearn.linear_model import LinearRegression
B = pd.read_parquet("baselines.parquet").set_index("game_id")
for f in sorted(glob.glob("sim_c*_N4000_s0_2010_2016.parquet")) + ["sim_c_N4000_s0_2010_2025.parquet"]:
    s = pd.read_parquet(f).set_index("game_id")
    s = s.join(B[["season", "home_won", "margin", "total", "ridge"]], how="inner")
    s = s[s.season.between(2010, 2016)]
    ll = log_loss(s.home_won, s.p_raw.clip(1e-4, 1 - 1e-4))
    slope = LinearRegression().fit(s[["m_mean"]], s.margin).coef_[0]
    tslope = LinearRegression().fit(s[["t_mean"]], s.total).coef_[0]
    print(f"{f:40s} n={len(s)} ll={ll:.4f} ridge_ll={log_loss(s.home_won, s.ridge):.4f} "
          f"margin slope={slope:.2f} total slope={tslope:.2f} sd(m_mean)={s.m_mean.std():.2f}")
