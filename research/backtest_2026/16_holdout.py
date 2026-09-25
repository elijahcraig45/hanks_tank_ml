"""One single evaluation of the search winner on the untouched final holdout.

Reports the overfit gap: the search's best internal walk-forward log-loss vs what
the same configuration achieves on games it never influenced. Also scores the
incumbent V10 and the simple baselines on the identical holdout games.
"""
import warnings, json
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
exec(open("research/backtest_2026/15_autosearch.py").read()
     .split("t0 = time.time()")[0])            # reuse data + model builders

B = json.load(open("data/backtest_2026/autosearch_best.json"))
best_p = sample(optuna.trial.FixedTrial(B["best_params"]))
search_ll = B["search_ll"]

def wilson(k,nn):
    z=1.96; ph=k/nn; den=1+z*z/nn
    c=(ph+z*z/(2*nn))/den; h=z*np.sqrt(ph*(1-ph)/nn+z*z/(4*nn*nn))/den
    return round((c-h)*100,2), round((c+h)*100,2)

# walk-forward across the holdout, refitting on everything before each block
def serve(p, seeds=(0,)):
    out = np.full(n, np.nan)
    for s in range(split, n, STEP):
        tr = np.arange(0, s); te = np.arange(s, min(s+STEP, n))
        ps = [fit_predict(p, tr, te, seed) for seed in seeds]
        out[te] = np.mean(ps, axis=0)
    return out

p_best = serve(best_p)
p_m3   = serve({"family":"logit","fsel":"manual3","C":0.1})

# incumbent V10 predictions, aligned by game_pk
pred = pd.read_parquet("data/backtest_2026/games_2026_pregame.parquet")[["game_pk","home_win_probability"]]
v10 = d[["game_pk"]].merge(pred, on="game_pk", how="left").home_win_probability.values

H = HOLD
yy = y[H]
base = y[:split].mean()
print(f"FINAL HOLDOUT: {len(H)} games, {d.game_date[split].date()} -> {d.game_date[n-1].date()}")
print(f"holdout home rate {yy.mean():.4f}   (search-set base rate {base:.4f})\n")

cands = {
    "SEARCH WINNER": p_best[H],
    "3-feature logistic": p_m3[H],
    "V10 (incumbent)": v10[H],
    "always home": np.full(len(H), base),
}
rows=[]
for k,p in cands.items():
    m = ~pd.isna(p)
    pp = np.clip(np.asarray(p,dtype=float)[m],1e-6,1-1e-6); t=yy[m]
    corr = (pp>=.5).astype(int)==t
    lo,hi = wilson(int(corr.sum()), int(m.sum()))
    rows.append(dict(model=k, n=int(m.sum()), acc=round(corr.mean()*100,2),
                     ci=f"[{lo}, {hi}]", auc=round(roc_auc_score(t,pp),4),
                     brier=round(brier_score_loss(t,pp),4),
                     logloss=round(log_loss(t,pp),4)))
R = pd.DataFrame(rows)
print(R.to_string(index=False))

hold_ll = R.loc[R.model=="SEARCH WINNER","logloss"].iloc[0]
print(f"\n{'='*74}\nOVERFIT GAP\n{'='*74}")
print(f"  search's best internal walk-forward log-loss : {search_ll:.5f}")
print(f"  same config on the untouched holdout         : {hold_ll:.5f}")
print(f"  gap                                          : {hold_ll-search_ll:+.5f}")
print(f"  trials run                                   : {B['n_trials']}")
alw = R.loc[R.model=='always home','logloss'].iloc[0]
print(f"  always-home log-loss on the same games       : {alw:.5f}")
verdict = ("the search winner does NOT beat always-home out of sample"
           if hold_ll >= alw else
           f"the search winner beats always-home by {alw-hold_ll:.5f} nats")
print(f"  -> {verdict}")

# paired bootstrap vs incumbent
m = ~pd.isna(v10[H])
pa=np.clip(p_best[H][m],1e-6,1-1e-6); pb=np.clip(v10[H][m].astype(float),1e-6,1-1e-6); t=yy[m]
la=-(t*np.log(pa)+(1-t)*np.log(1-pa)); lb=-(t*np.log(pb)+(1-t)*np.log(1-pb))
dd=lb-la; rng=np.random.default_rng(0)
bs=np.array([dd[rng.integers(0,len(dd),len(dd))].mean() for _ in range(4000)])
print(f"\nsearch winner vs V10 on holdout: logloss {dd.mean():+.5f} "
      f"CI[{np.percentile(bs,2.5):+.5f},{np.percentile(bs,97.5):+.5f}] P(better)={(bs>0).mean():.3f}")
R.to_csv("data/backtest_2026/holdout_results.csv", index=False)
