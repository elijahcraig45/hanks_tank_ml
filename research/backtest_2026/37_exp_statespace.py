"""Experiment 3 -- replace Elo with a Kalman-filtered latent strength.

PREDICTION REGISTERED: this should beat Elo on the same information, because the update
size is derived rather than hand-set and the offseason is handled properly. But it uses
NO new information, so the gain should be modest -- a better estimator of the same
quantity, not a new quantity. It should NOT produce orthogonal edge over the market.
"""
import sys, warnings, itertools
import numpy as np, pandas as pd
sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, roc_auc_score
from edge.benchmark import score
from edge.statespace import TeamStateSpace
from google.cloud import bigquery

g = bigquery.Client(project="hankstank").query("""
  SELECT game_pk, game_date, home_team_id, away_team_id, home_score, away_score
  FROM `hankstank.mlb_historical_data.games_historical`
  WHERE game_type='R' AND home_score IS NOT NULL
    AND EXTRACT(YEAR FROM game_date) BETWEEN 2015 AND 2021
  ORDER BY game_date, game_pk""").to_dataframe()
g["game_date"] = pd.to_datetime(g.game_date)
g["margin"] = (g.home_score - g.away_score).astype(float)
g["season"] = g.game_date.dt.year
g["home_won"] = (g.margin > 0).astype(int)
teams = sorted(set(g.home_team_id) | set(g.away_team_id))
print(f"filtering {len(g)} games, {len(teams)} teams, {g.season.min()}-{g.season.max()}")

tr_g = (g.game_date < pd.Timestamp("2020-01-01")).values

def fit_and_score(q, r, phi, inflate, ha):
    ss = TeamStateSpace(teams, q=q, r=r, phi=phi, season_inflate=inflate, home_adv=ha)
    mu, var, unc = ss.run(g.home_team_id.values, g.away_team_id.values,
                          g.margin.values, g.season.values)
    # map predicted margin -> win probability, fitted on the training period only
    X = np.column_stack([mu, mu / np.sqrt(var)])
    sc = StandardScaler().fit(X[tr_g])
    lr = LogisticRegression(C=1.0, max_iter=3000).fit(sc.transform(X[tr_g]), g.home_won.values[tr_g])
    p = lr.predict_proba(sc.transform(X))[:, 1]
    ll_tr = log_loss(g.home_won.values[tr_g], p[tr_g])
    return ll_tr, mu, var, unc, p

print("\ntuning process/observation noise on the TRAINING period only (2015-2019):")
best = None
for q, phi, inf in itertools.product([0.0008, 0.002, 0.005, 0.012],
                                     [1.0, 0.999, 0.997],
                                     [1.0, 4.0, 10.0]):
    ll, *_ = fit_and_score(q, 9.0, phi, inf, 0.20)
    if best is None or ll < best[0]:
        best = (ll, q, phi, inf)
print(f"  best: q={best[1]}  phi={best[2]}  season_inflate={best[3]}  train log-loss={best[0]:.5f}")
_, q, phi, inf = best
ll_tr, mu, var, unc, p_ss = fit_and_score(q, 9.0, phi, inf, 0.20)

# join to the arena and compare against Elo on identical games
a = pd.read_parquet("data/odds/arena_il.parquet")
a["game_date"] = pd.to_datetime(a.game_date)
ssdf = g[["game_pk"]].copy()
ssdf["ss_mu"] = mu; ssdf["ss_z"] = mu / np.sqrt(var); ssdf["ss_unc"] = unc; ssdf["ss_p"] = p_ss
a = a.merge(ssdf, on="game_pk", how="inner")
print(f"\narena games with a state-space prediction: {len(a)}")

y = a.home_won.values.astype(int)
tr = (a.game_date < pd.Timestamp("2020-01-01")).values
te = ~tr
half = np.zeros(int(te.sum()), bool); half[:int(te.sum()*0.5)] = True
BASE = [c for c in ["elo_differential","pythag_differential","win_pct_diff",
                    "run_diff_differential","home_pythag_last30","away_pythag_last30",
                    "era_proxy_differential","streak_differential","luck_differential"]
        if c in a.columns]
NO_ELO = [c for c in BASE if c != "elo_differential"]
SS = ["ss_mu", "ss_z", "ss_unc"]

def run(cols, name):
    X = a[cols].astype(float); X = X.fillna(X[tr].median()).fillna(0)
    sc = StandardScaler().fit(X[tr])
    m = LogisticRegression(C=0.1, max_iter=4000).fit(sc.transform(X[tr]), y[tr])
    p = m.predict_proba(sc.transform(X))[:, 1]
    return score(y[te], p[te], a.market_home_prob.values[te], a.home_moneyline.values[te],
                 a.away_moneyline.values[te], name=name, train_mask=half, verbose=False)

res = [
  run(["elo_differential"], "A. Elo alone"),
  run(SS, "B. state-space alone"),
  run(BASE, "C. full baseline (with Elo)"),
  run(NO_ELO + SS, "D. baseline, Elo REPLACED by state-space"),
  run(BASE + SS, "E. baseline + both"),
]
print(f"\n{'='*94}\nEXPERIMENT 3 -- STATE-SPACE TEAM STRENGTH  (test 2020-21, n={int(te.sum())})\n{'='*94}")
print(f"  {'model':44}{'acc':>8}{'AUC':>9}{'log-loss':>11}{'gap':>12}")
for r in res:
    print(f"  {r['name']:44}{r['acc']:>7.2f}%{r['auc']:>9.4f}{r['logloss']:>11.5f}{r['gap_nats']:>+12.5f}")
print(f"  {'closing line':44}{res[0]['market_acc']:>7.2f}%{res[0]['market_auc']:>9.4f}"
      f"{res[0]['market_logloss']:>11.5f}{0.0:>+12.5f}")
print(f"\n  head-to-head on the same information:")
print(f"    Elo alone          AUC {res[0]['auc']:.4f}  log-loss {res[0]['logloss']:.5f}")
print(f"    state-space alone  AUC {res[1]['auc']:.4f}  log-loss {res[1]['logloss']:.5f}"
      f"   -> {res[0]['logloss']-res[1]['logloss']:+.5f} nats vs Elo")
print(f"\n  orthogonal edge over the line:")
for r in res:
    if "orth_gain_nats" in r:
        print(f"    {r['name']:44}{r['orth_gain_nats']:>+10.5f}  P(helps)={r['orth_p_better']:.3f}")
a.to_parquet("data/odds/arena_ss.parquet", index=False)
