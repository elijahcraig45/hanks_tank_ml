"""Model RUNS instead of WINS, then derive P(home win).

The earlier searches all trained on a binary label. A game's score carries far more
information than one bit, so the same 1,300 games give a much richer training signal
if the target is runs / margin. Three mechanisms, same nested protocol and the same
untouched holdout (game_date >= 2026-08-08) as script 16:

  A  margin regression  -> P(margin >= 1) under a fitted residual distribution
  B  two Poisson models -> P(home runs > away runs) via the Skellam sum
  C  the binary classifier (control, what we already have)
"""
import warnings
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge, LogisticRegression, PoissonRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from scipy.stats import norm, skellam, t as tdist
warnings.filterwarnings("ignore")

HOLD_FROM = pd.Timestamp("2026-08-08"); MIN_TRAIN, STEP = 500, 100
b = pd.read_parquet("data/backtest_2026/full_features_2026.parquet")
n_ = pd.read_parquet("data/backtest_2026/statcast_new_features.parquet").drop(columns=["game_date"])
d = b.merge(n_, on="game_pk", how="inner").sort_values(["game_date","game_pk"]).reset_index(drop=True)
d["sp_form_k_diff"] = d.home_sp_form_k - d.away_sp_form_k
d["sp_form_xwoba_diff"] = d.away_sp_form_xwoba - d.home_sp_form_xwoba

y   = d.home_win.values
mrg = (d.home_score - d.away_score).values.astype(float)
hr, ar = d.home_score.values.astype(float), d.away_score.values.astype(float)
print(f"{len(d)} games | margin mean {mrg.mean():+.3f} sd {mrg.std():.3f} | "
      f"runs: home {hr.mean():.2f} away {ar.mean():.2f} | ties {int((mrg==0).sum())}")

F3 = ["elo_differential","pythag_differential","sp_quality_composite_diff"]
F5 = F3 + ["sp_form_k_diff","sp_form_xwoba_diff"]
# team-oriented features for the Poisson arms (offense of one side, pitching of the other)
OFF_H = ["home_pythag_season","home_run_diff_10g","home_fg_ops","home_fg_woba","home_runs_scored_10g"]
OFF_A = ["away_pythag_season","away_run_diff_10g","away_fg_ops","away_fg_woba","away_runs_scored_10g"]
DEF_H = ["home_sp_xera","home_sp_k_pct","home_fg_era","home_runs_allowed_10g","home_park_factor"]
DEF_A = ["away_sp_xera","away_sp_k_pct","away_fg_era","away_runs_allowed_10g"]
def have(cs): return [c for c in cs if c in d.columns]
HOME_RUNS_F = have(OFF_H + DEF_A + ["home_park_factor"])
AWAY_RUNS_F = have(OFF_A + DEF_H)

hold = (d.game_date >= HOLD_FROM).values
sidx, hidx = np.where(~hold)[0], np.where(hold)[0]
X = d.copy()

def prep(cols, tr, te):
    A, B = X.loc[tr, cols].astype(float), X.loc[te, cols].astype(float)
    A = A.replace([np.inf,-np.inf], np.nan); B = B.replace([np.inf,-np.inf], np.nan)
    med = A.median(); return A.fillna(med).fillna(0), B.fillna(med).fillna(0)

def run_wf(idx, fn):
    """fn(tr, te) -> P(home win) for te; expanding window inside idx"""
    p = np.full(len(X), np.nan)
    for s in range(MIN_TRAIN, len(idx), STEP):
        tr, te = idx[:s], idx[s:min(s+STEP, len(idx))]
        p[te] = fn(tr, te)
    return p

# ---- A: margin regression -> P(margin >= 1) ----
def margin_ridge(cols, alpha=10.0, dist="t"):
    def fn(tr, te):
        A, B = prep(cols, tr, te)
        sc = StandardScaler().fit(A)
        m = Ridge(alpha=alpha).fit(sc.transform(A), mrg[tr])
        mu_tr = m.predict(sc.transform(A)); mu = m.predict(sc.transform(B))
        resid = mrg[tr] - mu_tr
        s = resid.std(ddof=len(cols)+1)
        if dist == "t":
            df = max(3, len(tr) - len(cols) - 1)
            return 1 - tdist.cdf((0.5 - mu)/s, df)
        return 1 - norm.cdf((0.5 - mu)/s)
    return fn

def margin_gbm(cols):
    def fn(tr, te):
        A, B = prep(cols, tr, te)
        m = GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=0.03,
                                      subsample=0.8, random_state=0).fit(A, mrg[tr])
        mu_tr, mu = m.predict(A), m.predict(B)
        s = (mrg[tr] - mu_tr).std()
        return 1 - norm.cdf((0.5 - mu)/s)
    return fn

# ---- B: two Poisson arms -> Skellam ----
def poisson_pair(hcols, acols):
    def fn(tr, te):
        Ah, Bh = prep(hcols, tr, te); Aa, Ba = prep(acols, tr, te)
        sh = StandardScaler().fit(Ah); sa = StandardScaler().fit(Aa)
        mh = PoissonRegressor(alpha=1.0, max_iter=800).fit(sh.transform(Ah), hr[tr])
        ma = PoissonRegressor(alpha=1.0, max_iter=800).fit(sa.transform(Aa), ar[tr])
        lh = np.clip(mh.predict(sh.transform(Bh)), 0.5, 15)
        la = np.clip(ma.predict(sa.transform(Ba)), 0.5, 15)
        # P(home > away) with the tie mass split by the observed extra-innings home edge
        p_gt = 1 - skellam.cdf(0, lh, la)
        p_tie = skellam.pmf(0, lh, la)
        return p_gt + 0.5 * p_tie
    return fn

# ---- C: binary control ----
def clf(cols, C=0.1):
    def fn(tr, te):
        A, B = prep(cols, tr, te)
        sc = StandardScaler().fit(A)
        m = LogisticRegression(C=C, max_iter=3000).fit(sc.transform(A), y[tr])
        return m.predict_proba(sc.transform(B))[:,1]
    return fn

CANDS = {
    "C  binary logistic, 3 feat":      clf(F3),
    "C  binary logistic, 5 feat":      clf(F5),
    "A  margin ridge, 3 feat (t)":     margin_ridge(F3),
    "A  margin ridge, 5 feat (t)":     margin_ridge(F5),
    "A  margin ridge, 5 feat (norm)":  margin_ridge(F5, dist="norm"),
    "A  margin GBM, 5 feat":           margin_gbm(F5),
    "B  two-Poisson -> Skellam":       poisson_pair(HOME_RUNS_F, AWAY_RUNS_F),
}

def report(tag, idx):
    print(f"\n{'='*86}\n{tag}  (n={len(idx)})\n{'='*86}")
    print(f"{'mechanism':34}{'acc':>8}{'auc':>9}{'brier':>9}{'logloss':>10}")
    out = {}
    for k, fn in CANDS.items():
        p = run_wf(idx, fn)
        m = ~np.isnan(p[idx])
        pp = np.clip(p[idx][m], 1e-6, 1-1e-6); t = y[idx][m]
        acc = ((pp>=.5).astype(int)==t).mean()*100
        print(f"{k:34}{acc:>8.2f}{roc_auc_score(t,pp):>9.4f}"
              f"{brier_score_loss(t,pp):>9.4f}{log_loss(t,pp):>10.5f}")
        out[k] = p
    base = y[idx].mean()
    print(f"{'   always home':34}{max(base,1-base)*100:>8.2f}{0.5:>9.4f}"
          f"{np.mean((base-y[idx])**2):>9.4f}{log_loss(y[idx],np.full(len(idx),base)):>10.5f}")
    return out

# search-set walk-forward first (selection view), then the untouched holdout
_ = report("SEARCH SET — walk-forward", sidx)

# holdout: refit on everything before each block
def serve(fn):
    p = np.full(len(X), np.nan)
    for s in range(len(sidx), len(X), STEP):
        tr, te = np.arange(0,s), np.arange(s, min(s+STEP, len(X)))
        p[te] = fn(tr, te)
    return p
print(f"\n{'='*86}\nUNTOUCHED HOLDOUT  ({d.game_date[hidx[0]].date()} -> {d.game_date[hidx[-1]].date()}, n={len(hidx)})\n{'='*86}")
print(f"{'mechanism':34}{'acc':>8}{'auc':>9}{'brier':>9}{'logloss':>10}")
res={}
for k, fn in CANDS.items():
    p = serve(fn); m = ~np.isnan(p[hidx])
    pp = np.clip(p[hidx][m],1e-6,1-1e-6); t = y[hidx][m]
    acc=((pp>=.5).astype(int)==t).mean()*100
    res[k]=dict(acc=acc,auc=roc_auc_score(t,pp),brier=brier_score_loss(t,pp),ll=log_loss(t,pp),p=pp,t=t)
    print(f"{k:34}{acc:>8.2f}{res[k]['auc']:>9.4f}{res[k]['brier']:>9.4f}{res[k]['ll']:>10.5f}")
bh=y[hidx].mean()
print(f"{'   always home':34}{max(bh,1-bh)*100:>8.2f}{0.5:>9.4f}"
      f"{np.mean((bh-y[hidx])**2):>9.4f}{log_loss(y[hidx],np.full(len(hidx),bh)):>10.5f}")

# paired bootstrap: best margin model vs the binary control
kb = min([k for k in res if k.startswith("A") or k.startswith("B")], key=lambda k: res[k]["ll"])
kc = "C  binary logistic, 3 feat"
pa, pb, t = res[kb]["p"], res[kc]["p"], res[kc]["t"]
la=-(t*np.log(pa)+(1-t)*np.log(1-pa)); lb=-(t*np.log(pb)+(1-t)*np.log(1-pb))
dd=lb-la; rng=np.random.default_rng(0)
bs=np.array([dd[rng.integers(0,len(dd),len(dd))].mean() for _ in range(4000)])
print(f"\nbest score-based ({kb.strip()}) vs binary control:")
print(f"  logloss {dd.mean():+.5f}  CI[{np.percentile(bs,2.5):+.5f},{np.percentile(bs,97.5):+.5f}]"
      f"  P(better)={(bs>0).mean():.3f}")
print(f"  accuracy {res[kb]['acc']-res[kc]['acc']:+.2f}pp   AUC {res[kb]['auc']-res[kc]['auc']:+.4f}")
