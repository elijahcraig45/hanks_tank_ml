"""Does the simulator add information the 3-feature logistic does not have?

The simulator reaches a game through 1000 PA-by-PA episodes off 1.9M plate appearances;
the logistic reaches it through three season-level team aggregates. If their errors are
different, combining should beat both -- and it has to do so in BOTH windows.
"""
import warnings
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
warnings.filterwarnings("ignore")

HOLD_FROM = pd.Timestamp("2026-08-08")
d = pd.read_parquet("data/backtest_2026/sim_backtest.parquet")
d = d[np.isfinite(d.sim) & np.isfinite(d.logit3) & np.isfinite(d.home_win_probability)]
d = d.sort_values("game_date").reset_index(drop=True)
y = d.home_win.values
lg = lambda p: np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6)))

print(f"{len(d)} games with all three predictions")
print(f"prediction correlations (low = independent information):")
print(f"  sim  vs logit3 : r={np.corrcoef(d.sim, d.logit3)[0,1]:+.4f}")
print(f"  sim  vs V10    : r={np.corrcoef(d.sim, d.home_win_probability)[0,1]:+.4f}")
print(f"  logit3 vs V10  : r={np.corrcoef(d.logit3, d.home_win_probability)[0,1]:+.4f}")

# walk-forward stacks, all weights fit on the training window only
combos = {
 "sim + logit3 (avg)":        lambda a: 0.5*a["sim"] + 0.5*a["logit3"],
 "sim + logit3 (stack)":      None,
 "sim + logit3 + V10 (stack)": None,
}
S = {k: np.full(len(d), np.nan) for k in list(combos) + ["sim_cal2"]}
for s in range(400, len(d), 100):
    tr, te = np.arange(0, s), np.arange(s, min(s+100, len(d)))
    # recalibrate the simulator on the training window
    c = LogisticRegression(C=1e6, max_iter=3000).fit(lg(d.sim.values[tr]).reshape(-1,1), y[tr])
    sc_tr = c.predict_proba(lg(d.sim.values[tr]).reshape(-1,1))[:,1]
    sc_te = c.predict_proba(lg(d.sim.values[te]).reshape(-1,1))[:,1]
    S["sim_cal2"][te] = sc_te
    S["sim + logit3 (avg)"][te] = 0.5*sc_te + 0.5*d.logit3.values[te]
    A_tr = np.column_stack([lg(sc_tr), lg(d.logit3.values[tr])])
    A_te = np.column_stack([lg(sc_te), lg(d.logit3.values[te])])
    m = LogisticRegression(C=1.0, max_iter=3000).fit(A_tr, y[tr])
    S["sim + logit3 (stack)"][te] = m.predict_proba(A_te)[:,1]
    B_tr = np.column_stack([A_tr, lg(d.home_win_probability.values[tr])])
    B_te = np.column_stack([A_te, lg(d.home_win_probability.values[te])])
    m2 = LogisticRegression(C=1.0, max_iter=3000).fit(B_tr, y[tr])
    S["sim + logit3 + V10 (stack)"][te] = m2.predict_proba(B_te)[:,1]
    if s == 400:
        pass
print(f"\nstack coefficients on the final refit: sim {m.coef_[0][0]:+.3f}, logit3 {m.coef_[0][1]:+.3f}")

CAND = {"PA simulator (calibrated)": S["sim_cal2"], "3-feature logistic": d.logit3.values,
        "V10 (production)": d.home_win_probability.values,
        "sim + logit3 (avg)": S["sim + logit3 (avg)"],
        "sim + logit3 (stack)": S["sim + logit3 (stack)"],
        "sim + logit3 + V10 (stack)": S["sim + logit3 + V10 (stack)"]}
res = {}
for tag, mask in [("SEARCH WINDOW", (d.game_date < HOLD_FROM).values),
                  ("UNTOUCHED HOLDOUT", (d.game_date >= HOLD_FROM).values)]:
    print(f"\n{'='*86}\n{tag}\n{'='*86}")
    print(f"{'model':32}{'n':>6}{'acc':>8}{'auc':>9}{'brier':>9}{'logloss':>10}")
    res[tag] = {}
    for nm, p in CAND.items():
        m = mask & np.isfinite(p)
        pp = np.clip(p[m],1e-6,1-1e-6); t = y[m]
        r = dict(acc=((pp>=.5).astype(int)==t).mean()*100, auc=roc_auc_score(t,pp),
                 brier=brier_score_loss(t,pp), ll=log_loss(t,pp), n=int(m.sum()), p=pp, t=t)
        res[tag][nm] = r
        print(f"{nm:32}{r['n']:>6}{r['acc']:>8.2f}{r['auc']:>9.4f}{r['brier']:>9.4f}{r['ll']:>10.5f}")
    b = y[mask].mean()
    print(f"{'   always home':32}{int(mask.sum()):>6}{max(b,1-b)*100:>8.2f}{0.5:>9.4f}"
          f"{np.mean((b-y[mask])**2):>9.4f}{log_loss(y[mask],np.full(int(mask.sum()),b)):>10.5f}")

print(f"\n{'='*86}\nFINAL GATE -- log-loss gain vs the 3-feature logistic, required in BOTH windows\n{'='*86}")
print(f"{'model':32}{'search':>11}{'holdout':>11}   verdict")
winners = []
for nm in CAND:
    if nm == "3-feature logistic": continue
    gs = res["SEARCH WINDOW"]["3-feature logistic"]["ll"] - res["SEARCH WINDOW"][nm]["ll"]
    gh = res["UNTOUCHED HOLDOUT"]["3-feature logistic"]["ll"] - res["UNTOUCHED HOLDOUT"][nm]["ll"]
    v = "BETTER IN BOTH" if (gs>0 and gh>0) else ("worse in both" if (gs<0 and gh<0) else "inconsistent")
    if gs>0 and gh>0: winners.append(nm)
    print(f"{nm:32}{gs:>+11.5f}{gh:>+11.5f}   {v}")

for nm in winners:
    for tag in ["SEARCH WINDOW", "UNTOUCHED HOLDOUT"]:
        a, b_ = res[tag][nm], res[tag]["3-feature logistic"]
        n = min(len(a["p"]), len(b_["p"])); t = a["t"][:n]
        la = -(t*np.log(a["p"][:n])+(1-t)*np.log(1-a["p"][:n]))
        lb = -(t*np.log(b_["p"][:n])+(1-t)*np.log(1-b_["p"][:n]))
        dd = lb-la; rng = np.random.default_rng(0)
        bs = np.array([dd[rng.integers(0,len(dd),len(dd))].mean() for _ in range(4000)])
        print(f"\n{nm} vs 3-feature [{tag}]: logloss {dd.mean():+.5f} "
              f"CI[{np.percentile(bs,2.5):+.5f},{np.percentile(bs,97.5):+.5f}] P(better)={(bs>0).mean():.3f}")
pd.DataFrame({k: v for k, v in S.items()}).assign(
    game_pk=d.game_pk.values, game_date=d.game_date.values, home_win=y,
    logit3=d.logit3.values, v10=d.home_win_probability.values
).to_parquet("data/backtest_2026/ensemble_preds.parquet", index=False)
