"""Expanding-window walk-forward backtest over the 2026 season.

Protocol: sort games by date, train on everything before the test block, predict
the block, refit, advance. This mimics "predicting games going forward" -- no
model ever sees a game played on or after the day it predicts.

All candidates are scored on the IDENTICAL test games, including the incumbent
V10's stored production predictions, so the comparison is apples to apples.
Imputation and scaling are fit on the train fold only.
"""
import warnings, json
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
import xgboost as xgb, lightgbm as lgb

warnings.filterwarnings("ignore")
rng = 42
MIN_TRAIN, STEP = 600, 50

d = pd.read_parquet("data/backtest_2026/games_2026_pregame.parquet").reset_index(drop=True)
d = d.sort_values(["game_date", "game_pk"]).reset_index(drop=True)
y = d.home_win.values

SMALL = ["elo_differential", "pythag_differential", "sp_quality_composite_diff"]
MID = SMALL + ["matchup_advantage_home", "home_park_factor", "h2h_win_pct_3yr",
               "home_sp_k_pct", "away_sp_k_pct", "home_run_diff_10g", "away_run_diff_10g"]
DROP = {"home_win", "home_score", "away_score", "game_pk", "venue_id", "home_team_id",
        "away_team_id", "home_starter_id", "away_starter_id",
        "home_win_probability", "away_win_probability"}
ALL = [c for c in d.columns if d[c].dtype.kind in "fiub" and c not in DROP]
for c in set(ALL) | set(MID):
    d[c] = pd.to_numeric(d[c], errors="coerce")
d["rest_diff"] = d.home_days_rest.fillna(4) - d.away_days_rest.fillna(4)
MID = MID + ["rest_diff"]
ALL = ALL + ["rest_diff"]

def logit(cols, C):
    def f(tr, te):
        Xtr, Xte = d.loc[tr, cols].astype(float), d.loc[te, cols].astype(float)
        med = Xtr.median()
        Xtr, Xte = Xtr.fillna(med).fillna(0), Xte.fillna(med).fillna(0)
        sc = StandardScaler().fit(Xtr)
        m = LogisticRegression(C=C, max_iter=2000).fit(sc.transform(Xtr), y[tr])
        return m.predict_proba(sc.transform(Xte))[:, 1]
    return f

def tree(kind, cols, **kw):
    def f(tr, te):
        Xtr, Xte = d.loc[tr, cols].astype(float), d.loc[te, cols].astype(float)
        if kind == "xgb":
            m = xgb.XGBClassifier(eval_metric="logloss", random_state=rng,
                                  verbosity=0, **kw)
        else:
            m = lgb.LGBMClassifier(random_state=rng, verbose=-1, **kw)
        m.fit(Xtr, y[tr])
        return m.predict_proba(Xte)[:, 1]
    return f

CANDS = {
    "always_home":          lambda tr, te: np.full(len(te), y[tr].mean()),
    "elo_raw (no training)": lambda tr, te: d.loc[te, "elo_home_win_prob"].fillna(.5).values,
    "logit_small_3feat":    logit(SMALL, 1.0),
    "logit_small_C0.1":     logit(SMALL, 0.1),
    "logit_mid_11feat":     logit(MID, 0.1),
    "logit_all_C0.01":      logit(ALL, 0.01),
    "logit_all_C0.1":       logit(ALL, 0.1),
    "xgb_shallow_reg":      tree("xgb", MID, max_depth=2, n_estimators=150,
                                 learning_rate=0.03, subsample=0.8,
                                 colsample_bytree=0.8, reg_lambda=10.0, min_child_weight=20),
    "xgb_deep_like_v10":    tree("xgb", ALL, max_depth=6, n_estimators=400, learning_rate=0.05),
    "lgb_shallow_reg":      tree("lgb", MID, num_leaves=4, n_estimators=200,
                                 learning_rate=0.03, min_child_samples=40,
                                 reg_lambda=10.0, subsample=0.8, colsample_bytree=0.8),
}

n = len(d)
starts = list(range(MIN_TRAIN, n, STEP))
preds = {k: np.full(n, np.nan) for k in CANDS}
for s in starts:
    tr = np.arange(0, s)
    te = np.arange(s, min(s + STEP, n))
    for k, fn in CANDS.items():
        preds[k][te] = fn(tr, te)

mask = ~np.isnan(preds["logit_small_3feat"])
preds["prod_v10 (incumbent)"] = d.home_win_probability.values.copy()
yy = y[mask]
print(f"walk-forward: {len(d)} games, min_train={MIN_TRAIN}, step={STEP}")
print(f"test games   : {mask.sum()}  ({d.loc[mask,'game_date'].min().date()} -> {d.loc[mask,'game_date'].max().date()})")
print(f"home_win rate on test = {yy.mean():.4f}\n")

rows = []
for k, p in preds.items():
    pp = np.clip(p[mask].astype(float), 1e-6, 1 - 1e-6)
    conf = np.maximum(pp, 1 - pp)
    corr = ((pp >= .5).astype(int) == yy)
    hi = conf >= 0.60
    rows.append(dict(model=k, acc=corr.mean() * 100, auc=roc_auc_score(yy, pp),
                     brier=brier_score_loss(yy, pp), logloss=log_loss(yy, pp),
                     avg_p=pp.mean(),
                     acc60=(corr[hi].mean() * 100 if hi.sum() >= 20 else np.nan),
                     cov60=hi.mean() * 100))
r = pd.DataFrame(rows).sort_values("auc", ascending=False)
print(r.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
r.to_csv("data/backtest_2026/walkforward_results.csv", index=False)
np.save("data/backtest_2026/wf_mask.npy", mask)
pd.DataFrame({k: v for k, v in preds.items()}).to_parquet("data/backtest_2026/wf_preds.parquet")
