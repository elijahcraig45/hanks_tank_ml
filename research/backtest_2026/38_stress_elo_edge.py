"""Stress-test the one positive signal: does Elo ALONE really add on top of the line?

Experiment 3 threw up orthogonal gain +0.00077 nats at P(helps)=0.970 for Elo alone,
while the full baseline scored P=0.033. That pattern -- simplest model holds edge, extra
features destroy it -- is either a real and useful finding or a multiple-comparisons
artifact. I have now run ~15 variants, so at P=0.97 roughly 0.45 false positives are
expected by chance. This script tries to break it four ways:

  1. use the FULL test set instead of the half the probe trained on
  2. walk-forward the probe instead of one arbitrary split
  3. hold out entire seasons
  4. check whether the ROI is positive, which is the only unarguable test
"""
import sys, warnings
import numpy as np, pandas as pd
sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss

a = pd.read_parquet("data/odds/arena_ss.parquet")
a["game_date"] = pd.to_datetime(a.game_date)
a["season"] = a.game_date.dt.year
a = a.sort_values("game_date").reset_index(drop=True)
y = a.home_won.values.astype(int)
mkt = a.market_home_prob.values.astype(float)
LOGIT = lambda p: np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6)))

def elo_pred(train_mask):
    X = a[["elo_differential"]].astype(float)
    X = X.fillna(X[train_mask].median()).fillna(0)
    sc = StandardScaler().fit(X[train_mask])
    m = LogisticRegression(C=0.1, max_iter=4000).fit(sc.transform(X[train_mask]), y[train_mask])
    return m.predict_proba(sc.transform(X))[:, 1]

def probe(p_model, fit_mask, eval_mask):
    """Fit logit(market) and logit(market)+logit(model) on fit_mask, compare on eval_mask."""
    A = np.column_stack([LOGIT(mkt), LOGIT(p_model)])
    s2 = StandardScaler().fit(A[fit_mask])
    s1 = StandardScaler().fit(A[fit_mask][:, :1])
    m2 = LogisticRegression(C=0.1, max_iter=4000).fit(s2.transform(A[fit_mask]), y[fit_mask])
    m1 = LogisticRegression(C=0.1, max_iter=4000).fit(s1.transform(A[fit_mask][:, :1]), y[fit_mask])
    pb = m2.predict_proba(s2.transform(A[eval_mask]))[:, 1]
    pm = m1.predict_proba(s1.transform(A[eval_mask][:, :1]))[:, 1]
    yy = y[eval_mask]
    la = -(yy*np.log(np.clip(pb,1e-9,1))+(1-yy)*np.log(np.clip(1-pb,1e-9,1)))
    lb = -(yy*np.log(np.clip(pm,1e-9,1))+(1-yy)*np.log(np.clip(1-pm,1e-9,1)))
    dd = lb - la
    rng = np.random.default_rng(0)
    bs = np.array([dd[rng.integers(0,len(dd),len(dd))].mean() for _ in range(4000)])
    return dd.mean(), (bs>0).mean(), m2.coef_[0][1], len(yy)

print("="*90)
print("TEST 1 -- the original split, then the FULL test set")
print("="*90)
tr = (a.season <= 2019).values
p = elo_pred(tr)
te_idx = np.flatnonzero(~tr)
half_fit = np.zeros(len(a), bool); half_fit[te_idx[:len(te_idx)//2]] = True
half_ev  = np.zeros(len(a), bool); half_ev[te_idx[len(te_idx)//2:]] = True
for nm, fm, em in [("probe fit on 1st half of 2020-21, eval on 2nd", half_fit, half_ev),
                   ("probe fit on 2015-19, eval on ALL of 2020-21", tr, ~tr)]:
    g_, pb_, co, n_ = probe(p, fm, em)
    print(f"  {nm:46} gain {g_:+.5f}  P={pb_:.3f}  coef {co:+.4f}  n={n_}")

print("\n" + "="*90)
print("TEST 2 -- walk-forward probe (expanding window, refit every 300 games)")
print("="*90)
gains, ps, ns = [], [], []
for s in range(2000, len(a) - 300, 300):
    fm = np.zeros(len(a), bool); fm[:s] = True
    em = np.zeros(len(a), bool); em[s:s+300] = True
    pw = elo_pred(fm)
    g_, pb_, co, n_ = probe(pw, fm, em)
    gains.append(g_); ps.append(pb_); ns.append(n_)
gains = np.array(gains)
print(f"  {len(gains)} folds   mean gain {gains.mean():+.5f} nats   "
      f"folds positive: {(gains>0).sum()}/{len(gains)}")
print(f"  fold gains: " + " ".join(f"{x:+.4f}" for x in gains))
from scipy.stats import binomtest
print(f"  sign test on folds: p={binomtest((gains>0).sum(), len(gains), 0.5).pvalue:.3f}")

print("\n" + "="*90)
print("TEST 3 -- hold out entire seasons")
print("="*90)
for s in [2018, 2019, 2020, 2021]:
    fm = (a.season < s).values; em = (a.season == s).values
    if fm.sum() < 1500 or em.sum() < 300: continue
    pw = elo_pred(fm)
    g_, pb_, co, n_ = probe(pw, fm, em)
    print(f"  holdout {s}: gain {g_:+.5f}  P={pb_:.3f}  coef {co:+.4f}  n={n_}")

print("\n" + "="*90)
print("TEST 4 -- the unarguable one: would betting Elo's disagreements have made money?")
print("="*90)
dec = lambda ml: np.where(ml > 0, 1 + ml/100.0, 1 + 100.0/np.abs(ml))
hml, aml = a.home_moneyline.values.astype(float), a.away_moneyline.values.astype(float)
for lo_s, hi_s, lab in [(2020, 2021, "2020-21 (test period)"), (2015, 2021, "all seasons")]:
    m = (a.season >= lo_s) & (a.season <= hi_s)
    m = m.values & np.isfinite(p)
    edge = p - mkt
    for thr in (0.02, 0.04, 0.06):
        bh = m & (edge > thr); ba = m & (edge < -thr)
        n = int(bh.sum() + ba.sum())
        if n < 40: continue
        stake = bh | ba
        won = np.where(bh, y == 1, np.where(ba, y == 0, False))
        pay = np.where(bh, dec(hml), dec(aml))
        prof = float(np.where(won & stake, pay - 1, np.where(stake, -1.0, 0.0)).sum())
        print(f"  {lab:22} edge>{thr*100:.0f}%: {n:5d} bets  hit {won[stake].mean()*100:5.2f}%  "
              f"ROI {prof/n*100:+6.2f}%  {prof:+7.1f}u")
