"""Score every 2022 tuning run at the GAME level: margin CRPS + total CRPS (objective),
plus winner log loss and the margin regression slope (over/under-shrinkage check)."""
import glob, numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
B = pd.read_parquet("baselines.parquet").set_index("game_id")
MR = np.arange(-100, 101); TR = np.arange(0, 171)
def crps(H, y, grid):
    P = H / H.sum(1, keepdims=True); F = np.cumsum(P, 1)
    return ((F - (grid[None, :] >= y[:, None])) ** 2).sum(1)
def ll(y, p):
    p = np.clip(p, 1e-4, 1 - 1e-4); return -(y * np.log(p) + (1 - y) * np.log(1 - p))
rows = []
for f in sorted(glob.glob("runs/sim_*_2022_2022.parquet")):
    s = pd.read_parquet(f).set_index("game_id").join(B, how="inner")
    s = s[s.ridge.notna()]
    mh = np.stack(s.mh.values).astype(float); th = np.stack(s.th.values).astype(float)
    mc = crps(mh, s.margin.values, MR).mean(); tc = crps(th, s.total.values, TR).mean()
    rows.append(dict(run=f.split("sim_")[1].split("_N")[0], n=len(s), obj=mc + tc, margin_crps=mc, total_crps=tc,
                     win_ll=ll(s.home_won.values, s.p_raw.values).mean(), ridge_ll=ll(s.home_won.values, s.ridge.values).mean(),
                     m_slope=LinearRegression().fit(s[["m_mean"]], s.margin).coef_[0],
                     t_bias=(s.t_mean - s.total).mean()))
r = pd.DataFrame(rows).sort_values("obj")
print(r.round(4).to_string(index=False))
r.to_csv("runs/tune_scores.csv", index=False)
