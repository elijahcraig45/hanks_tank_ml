"""Round 2: forward feature selection + blending + shrinkage.

Feature selection runs ONLY on the first MIN_TRAIN games (the initial training
window, which is never part of the walk-forward test set), via internal CV.
The selected set is then evaluated on the untouched walk-forward test games.
"""
import warnings, itertools
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
warnings.filterwarnings("ignore")

MIN_TRAIN, STEP = 600, 50
d = pd.read_parquet("data/backtest_2026/games_2026_pregame.parquet")
d = d.sort_values(["game_date", "game_pk"]).reset_index(drop=True)
y = d.home_win.values
DROP = {"home_win","home_score","away_score","game_pk","venue_id","home_team_id","away_team_id",
        "home_starter_id","away_starter_id","home_win_probability","away_win_probability"}
CAND = [c for c in d.columns if d[c].dtype.kind in "fiub" and c not in DROP]
for c in CAND: d[c] = pd.to_numeric(d[c], errors="coerce")
d["rest_diff"] = d.home_days_rest.fillna(4) - d.away_days_rest.fillna(4)
CAND.append("rest_diff")
CAND = [c for c in CAND if d[c].notna().mean() > 0.85 and d[c].nunique() > 3]

def fit_predict(cols, tr, te, C=0.1):
    Xtr, Xte = d.loc[tr, cols].astype(float), d.loc[te, cols].astype(float)
    med = Xtr.median()
    Xtr, Xte = Xtr.fillna(med).fillna(0), Xte.fillna(med).fillna(0)
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(C=C, max_iter=2000).fit(sc.transform(Xtr), y[tr])
    return m.predict_proba(sc.transform(Xte))[:, 1]

# ---- forward selection on the first 600 games only, scored by CV log-loss ----
sel_idx = np.arange(MIN_TRAIN)
def cv_score(cols):
    X = d.loc[sel_idx, cols].astype(float)
    X = X.fillna(X.median()).fillna(0)
    sc = StandardScaler().fit_transform(X)
    return cross_val_score(LogisticRegression(C=0.1, max_iter=2000), sc, y[sel_idx],
                           cv=KFold(5, shuffle=True, random_state=0),
                           scoring="neg_log_loss").mean()

chosen, best = [], -np.inf
while len(chosen) < 6:
    scores = [(cv_score(chosen + [c]), c) for c in CAND if c not in chosen]
    s, c = max(scores)
    if s <= best + 1e-4: break
    chosen.append(c); best = s
    print(f"  + {c:32s} cv_logloss={-s:.5f}")
print(f"\nselected ({len(chosen)}): {chosen}\n")

# ---- walk-forward evaluate ----
n = len(d)
SMALL = ["elo_differential", "pythag_differential", "sp_quality_composite_diff"]
variants = {"fwd_selected": chosen, "small_3feat": SMALL,
            "fwd+small_union": sorted(set(chosen) | set(SMALL))}
preds = {k: np.full(n, np.nan) for k in variants}
for s in range(MIN_TRAIN, n, STEP):
    tr, te = np.arange(0, s), np.arange(s, min(s + STEP, n))
    for k, cols in variants.items():
        preds[k][te] = fit_predict(cols, tr, te)

mask = ~np.isnan(preds["small_3feat"])
yy = y[mask]
elo = d.elo_home_win_prob.fillna(.5).values
v10 = d.home_win_probability.values

def shrink(p, lam, base):  # pull toward base rate
    return base + lam * (p - base)

base = y[:MIN_TRAIN].mean()
out = {}
for k, p in preds.items(): out[k] = p
out["prod_v10"] = v10
out["elo_raw"] = elo
out["blend_small+elo"] = 0.5 * preds["small_3feat"] + 0.5 * elo
out["blend_small+v10"] = 0.5 * preds["small_3feat"] + 0.5 * v10
out["blend_sel+elo+v10"] = (preds["fwd_selected"] + elo + v10) / 3
out["v10_shrunk_0.5"] = shrink(v10, 0.5, base)
out["elo_shrunk_0.6"] = shrink(elo, 0.6, base)

rows = []
for k, p in out.items():
    pp = np.clip(p[mask].astype(float), 1e-6, 1 - 1e-6)
    conf = np.maximum(pp, 1 - pp); corr = ((pp >= .5).astype(int) == yy)
    hi = conf >= 0.58
    rows.append(dict(model=k, acc=corr.mean()*100, auc=roc_auc_score(yy, pp),
                     brier=brier_score_loss(yy, pp), logloss=log_loss(yy, pp),
                     acc58=(corr[hi].mean()*100 if hi.sum() >= 20 else np.nan),
                     cov58=hi.mean()*100))
r = pd.DataFrame(rows).sort_values("logloss")
print(f"walk-forward test: {mask.sum()} games, home rate {yy.mean():.4f}")
print(r.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# paired bootstrap: best vs incumbent on log-loss
bestk = r.iloc[0].model
pa = np.clip(out[bestk][mask], 1e-6, 1-1e-6); pb = np.clip(v10[mask], 1e-6, 1-1e-6)
la = -(yy*np.log(pa)+(1-yy)*np.log(1-pa)); lb = -(yy*np.log(pb)+(1-yy)*np.log(1-pb))
diff = lb - la
bs = np.array([diff[np.random.randint(0, len(diff), len(diff))].mean() for _ in range(4000)])
print(f"\n{bestk} vs prod_v10  mean logloss gain = {diff.mean():+.5f}")
print(f"  95% CI [{np.percentile(bs,2.5):+.5f}, {np.percentile(bs,97.5):+.5f}]  "
      f"P(better) = {(bs>0).mean():.3f}")
