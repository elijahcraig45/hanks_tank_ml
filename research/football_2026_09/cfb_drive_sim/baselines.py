import sys, warnings, logging, numpy as np, pandas as pd
warnings.filterwarnings("ignore"); logging.disable(logging.INFO)
SRC = "/Users/VTNX82W/Documents/personalDev/mlb/hanks_tank_ml/src"
sys.path[:0] = [SRC, SRC + "/nfl", SRC + "/cfb"]
import margin_ridge as mr
import pipeline as cp
from sklearn.linear_model import Ridge
WPS = 30
g = pd.read_parquet("cfb_games_bq.parquet"); g = g[(g.season <= 2025) & g.home_won.notna()].reset_index(drop=True)
rg = cp.ridge_frame(g)
tgt = (rg.season >= 2022).to_numpy()
rg["ridge_margin"] = mr.walk_forward(rg, tgt, mr.CFB_RIDGE)
rg["ridge"] = mr.win_prob(rg.ridge_margin, mr.CFB_RIDGE.sigma)
B = g[["game_id", "season", "week", "division", "is_postseason", "home_team", "away_team", "neutral_site",
       "home_score", "away_score", "home_won", "result", "cross_division"]].copy()
B["margin"] = B.result.astype(float); B["total"] = (B.home_score + B.away_score).astype(float)
B = B.merge(rg[["game_id", "ridge_margin", "ridge", "fbs_diff"]], on="game_id")
L = pd.read_parquet("cfb_lines.parquet")[["game_id", "spread_line", "total_line", "home_ml", "away_ml"]]
B = B.merge(L, on="game_id", how="left")
chk = B.dropna(subset=["spread_line"])
print("market sign check corr(spread, margin):", np.corrcoef(chk.spread_line, chk.margin)[0, 1].round(3))
tune = B.season == 2022
ok = tune & B.spread_line.notna()
B.attrs["sig_mkt"] = sig = mr.fit_sigma(B.spread_line[ok].values, B.home_won[ok].values.astype(float))
B["market"] = mr.win_prob(B.spread_line, sig)
print("market sigma (fit 2022):", round(sig, 2))
# FPI 2025 FBS (cached by the FPI backtest)
F = pd.read_parquet("../cmp/bt_cfb.parquet")[["game_id", "fpi"]]
B = B.merge(F, on="game_id", how="left")
# points ridge for totals: pts = off[team] + def[opp] + hfa + fbs terms; tuned on 2022
teams = pd.Index(sorted(set(B.home_team) | set(B.away_team))); T = len(teams)
B["t"] = B.season * WPS + B.week
n = len(B); hi = teams.get_indexer(B.home_team); ai = teams.get_indexer(B.away_team)
def X_side(o, d_, home):
    X = np.zeros((n, 2 * T + 2)); X[np.arange(n), o] = 1; X[np.arange(n), T + d_] = 1
    X[:, -2] = home * (1 - B.neutral_site.fillna(0).values) * 10; X[:, -1] = 10
    return X
Xh, Xa = X_side(hi, ai, 1), X_side(ai, hi, 0)
def pts_ridge(alpha, tau, mask):
    ph = np.full(n, np.nan); pa = np.full(n, np.nan)
    for t_now in np.unique(B.t[mask]):
        tr = (B.t.values < t_now) & (B.t.values >= t_now - 2 * WPS)
        wt = np.exp(-(t_now - B.t.values[tr]) / tau)
        m = Ridge(alpha=alpha, fit_intercept=False).fit(np.vstack([Xh[tr], Xa[tr]]),
              np.r_[B.home_score.values[tr], B.away_score.values[tr]].astype(float), sample_weight=np.r_[wt, wt])
        te = mask & (B.t.values == t_now)
        ph[te] = m.predict(Xh[te]); pa[te] = m.predict(Xa[te])
    return ph, pa
best = None
for alpha in [1, 3, 10, 30]:
    for tau in [8, 16, 32]:
        ph, pa = pts_ridge(alpha, tau, tune.values)
        mse = np.nanmean((ph + pa - B.total.values)[tune.values] ** 2)
        if best is None or mse < best[0]: best = (mse, alpha, tau)
print("points ridge tuned 2022 (mse, alpha, tau):", best)
ph, pa = pts_ridge(best[1], best[2], (B.season >= 2022).values)
B["pr_total"] = ph + pa; B["pr_home"] = ph; B["pr_away"] = pa
t = B.t.values
B["naive_total"] = [B.total.values[(t < tt) & (t >= tt - 2 * WPS)].mean() for tt in t]
B.to_parquet("baselines.parquet")
fb = B[(B.season == 2025) & (B.division == "fbs")].dropna(subset=["ridge", "fpi"])
print("sanity FBS 2025 ridge ll", mr.log_loss(fb.home_won, fb.ridge), "fpi ll", mr.log_loss(fb.home_won, fb.fpi), len(fb))
for lab, m in [("FBS 2023-24", B.season.between(2023, 2024) & (B.division == "fbs")), ("FBS 2025", (B.season == 2025) & (B.division == "fbs"))]:
    b = B[m].dropna(subset=["ridge"]); print(lab, "ridge ll", round(mr.log_loss(b.home_won, b.ridge), 4), len(b))
