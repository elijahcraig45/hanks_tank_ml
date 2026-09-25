"""A proper test of whether 2015-2025 history can be made to transfer to 2026.

The earlier claim ("more history hurts") rested on 2 raw features and one model --
too weak to support a ceiling claim. This version uses all 54 features common to the
historical V8 parquet and the 2026 table, and tries the standard fixes for the domain
shift that made raw transfer fail:

  within-season z-scoring   -- removes season-level scale/level shift
  recency weighting         -- exponential decay by season age
  regularisation sweep      -- history is 20x larger, so the optimum differs

2026 rows are z-scored with statistics from games BEFORE the block being predicted,
never the full season, so nothing looks ahead.
"""
import warnings
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from sklearn.ensemble import HistGradientBoostingClassifier
warnings.filterwarnings("ignore")

HOLD_FROM = pd.Timestamp("2026-08-08"); STEP = 100
h = pd.concat([pd.read_parquet("data/training/train_v8_2015_2024.parquet"),
               pd.read_parquet("data/training/val_v8_2025.parquet")], ignore_index=True)
c = pd.read_parquet("data/backtest_2026/full_features_2026.parquet")
c = c.sort_values(["game_date","game_pk"]).reset_index(drop=True)
common = sorted(set(h.columns) & set(c.columns))
F = [x for x in common if h[x].dtype.kind in "fiub" and c[x].dtype.kind in "fiub"
     and x not in {"game_pk","home_team_id","away_team_id","home_won","home_win","year"}]
h["season"] = pd.to_datetime(h.game_date).dt.year
yh = h.home_won.values.astype(int)
yc = c.home_win.values
Xh = h[F].astype(float).replace([np.inf,-np.inf], np.nan)
Xc = c[F].astype(float).replace([np.inf,-np.inf], np.nan)
Xh = Xh.fillna(Xh.median()); Xc = Xc.fillna(Xh.median())
print(f"history {len(h)} games (2015-2025), 2026 {len(c)} games, {len(F)} shared features")

# within-season z-scoring of history
Zh = Xh.copy()
for s, g in h.groupby("season"):
    mu, sd = Xh.loc[g.index].mean(), Xh.loc[g.index].std().replace(0, 1)
    Zh.loc[g.index] = (Xh.loc[g.index] - mu) / sd
w_by_season = {s: 0.85 ** (2026 - s) for s in sorted(h.season.unique())}
wt = h.season.map(w_by_season).values

hold = (c.game_date >= HOLD_FROM).values
sidx, hidx = np.where(~hold)[0], np.where(hold)[0]

def z_2026(upto, block):
    """z-score the block using only 2026 games before it"""
    ref = Xc.iloc[upto]
    mu, sd = ref.mean(), ref.std().replace(0, 1)
    return (Xc.iloc[block] - mu) / sd

def eval_on(idx, get_pred):
    p = np.full(len(c), np.nan)
    start = idx[0]
    for s in range(start, len(c), STEP):
        block = np.arange(s, min(s+STEP, len(c)))
        block = block[np.isin(block, idx)]
        if len(block) == 0: continue
        upto = np.arange(0, s)
        if len(upto) < 120: continue
        p[block] = get_pred(upto, block)
    m = ~np.isnan(p[idx])
    pp = np.clip(p[idx][m], 1e-6, 1-1e-6); t = yc[idx][m]
    return dict(n=int(m.sum()), acc=((pp>=.5).astype(int)==t).mean()*100,
                auc=roc_auc_score(t,pp), brier=brier_score_loss(t,pp), ll=log_loss(t,pp))

# ---- candidate mechanisms ----
def hist_raw(C=0.03):
    m = LogisticRegression(C=C, max_iter=4000).fit(
        (Xh - Xh.mean())/Xh.std().replace(0,1), yh)
    mu, sd = Xh.mean(), Xh.std().replace(0,1)
    return lambda upto, block: m.predict_proba((Xc.iloc[block]-mu)/sd)[:,1]

def hist_z(C=0.03, weighted=False, gbm=False):
    if gbm:
        m = HistGradientBoostingClassifier(max_depth=3, max_iter=300,
              learning_rate=0.05, l2_regularization=1.0, random_state=0)
        m.fit(Zh, yh, sample_weight=wt if weighted else None)
    else:
        m = LogisticRegression(C=C, max_iter=4000)
        m.fit(Zh, yh, sample_weight=wt if weighted else None)
    return lambda upto, block: m.predict_proba(z_2026(upto, block))[:,1]

def season_only(C=0.1):
    def fn(upto, block):
        A = Xc.iloc[upto]; mu, sd = A.mean(), A.std().replace(0,1)
        Az = (A-mu)/sd
        m = LogisticRegression(C=C, max_iter=4000).fit(Az, yc[upto])
        return m.predict_proba((Xc.iloc[block]-mu)/sd)[:,1]
    return fn

def hist_plus_season(C=0.03):
    def fn(upto, block):
        A = Xc.iloc[upto]; mu, sd = A.mean(), A.std().replace(0,1)
        Az = (A-mu)/sd
        Xcomb = pd.concat([Zh, Az], ignore_index=True)
        ycomb = np.concatenate([yh, yc[upto]])
        wcomb = np.concatenate([wt, np.full(len(upto), 3.0)])   # upweight in-season rows
        m = LogisticRegression(C=C, max_iter=4000).fit(Xcomb, ycomb, sample_weight=wcomb)
        return m.predict_proba((Xc.iloc[block]-mu)/sd)[:,1]
    return fn

CANDS = {
 "history RAW, 54 feat":                 hist_raw(),
 "history Z-SCORED, 54 feat":            hist_z(),
 "history Z + recency weights":          hist_z(weighted=True),
 "history Z + recency, C=0.3":           hist_z(C=0.3, weighted=True),
 "history Z + recency, GBM":             hist_z(weighted=True, gbm=True),
 "history Z + 2026 in-season combined":  hist_plus_season(),
 "2026 SEASON ONLY (control)":           season_only(),
}
for tag, idx in [("2026 SEARCH WINDOW", sidx), ("UNTOUCHED HOLDOUT", hidx)]:
    print(f"\n{'='*86}\n{tag}  (n={len(idx)})\n{'='*86}")
    print(f"{'mechanism':40}{'n':>6}{'acc':>8}{'auc':>9}{'brier':>9}{'logloss':>10}")
    for k, fn in CANDS.items():
        r = eval_on(idx, fn)
        print(f"{k:40}{r['n']:>6}{r['acc']:>8.2f}{r['auc']:>9.4f}{r['brier']:>9.4f}{r['ll']:>10.5f}")
    bs_ = yc[idx].mean()
    print(f"{'   always home':40}{len(idx):>6}{max(bs_,1-bs_)*100:>8.2f}{0.5:>9.4f}"
          f"{np.mean((bs_-yc[idx])**2):>9.4f}{log_loss(yc[idx],np.full(len(idx),bs_)):>10.5f}")
