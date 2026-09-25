"""The real test of edge: does our signal add anything ON TOP of the closing line?

If the market already contains everything our features know, then adding them to a
model that starts from the line will not improve it, and we have no independent
information -- no edge, by definition. If they do add, that increment IS the edge, and
it is measurable in nats and in win rate against the line.

Train on 2015-2019, test on 2020-2021 (temporal split, never shuffled).
"""
import warnings
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
warnings.filterwarnings("ignore")

mk = pd.read_parquet("data/odds/market_joined.parquet")
mk["game_date"] = pd.to_datetime(mk.game_date)
h = pd.concat([pd.read_parquet("data/training/train_v8_2015_2024.parquet"),
               pd.read_parquet("data/training/val_v8_2025.parquet")], ignore_index=True)
h["game_date"] = pd.to_datetime(h.game_date)

F = ["elo_differential", "pythag_differential", "elo_home_win_prob",
     "win_pct_diff", "run_diff_differential", "home_pythag_last30", "away_pythag_last30"]
F = [c for c in F if c in h.columns]
keep = ["game_date", "home_team_id", "away_team_id", "home_won"] + F
hh = h[keep].copy()

# join on date + our own game identity via the games table's names is already in mk;
# match on date and the two pythag/elo rows by nearest game -- use date + home_won parity
# safer: join through the historical games table on date + team names
from google.cloud import bigquery
gid = bigquery.Client(project="hankstank").query("""
  SELECT game_date, home_team_name, away_team_name, home_team_id, away_team_id
  FROM `hankstank.mlb_historical_data.games_historical`
  WHERE game_type='R' AND EXTRACT(YEAR FROM game_date) BETWEEN 2015 AND 2021
""").to_dataframe()
gid["game_date"] = pd.to_datetime(gid.game_date)
mk2 = mk.merge(gid, on=["game_date", "home_team_name", "away_team_name"], how="inner")
d = mk2.merge(hh, on=["game_date", "home_team_id", "away_team_id"], how="inner")
d = d.drop_duplicates(subset=["game_date", "home_team_id", "away_team_id"]).reset_index(drop=True)
d = d.sort_values("game_date").reset_index(drop=True)
print(f"games with BOTH market odds and our features: {len(d)}  "
      f"{d.game_date.min().date()} -> {d.game_date.max().date()}")

y = d.home_won.values.astype(int)
mp = d.market_home_prob.values.astype(float)
X = d[F].astype(float); X = X.fillna(X.median())
lg = lambda p: np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6)))

tr = (d.game_date < pd.Timestamp("2020-01-01")).values
te = ~tr
print(f"train {tr.sum()} (2015-2019)   test {te.sum()} (2020-2021)\n")

def fit(cols_market, cols_feat):
    parts = []
    if cols_market: parts.append(lg(mp).reshape(-1,1))
    if cols_feat:   parts.append(X[cols_feat].values)
    A = np.hstack(parts)
    sc = StandardScaler().fit(A[tr])
    m = LogisticRegression(C=0.1, max_iter=4000).fit(sc.transform(A[tr]), y[tr])
    return m.predict_proba(sc.transform(A[te]))[:,1], m, sc

def rep(name, pp):
    pp = np.clip(pp, 1e-6, 1-1e-6); t = y[te]
    corr = (pp>=.5).astype(int)==t
    print(f"  {name:32} acc={corr.mean()*100:6.2f}%  auc={roc_auc_score(t,pp):.4f}  "
          f"brier={brier_score_loss(t,pp):.4f}  logloss={log_loss(t,pp):.5f}")
    return log_loss(t,pp)

print("="*88)
print("DOES OUR SIGNAL ADD ANYTHING ON TOP OF THE LINE?  (test = 2020-2021)")
print("="*88)
ll_raw   = rep("raw closing line (untouched)", mp[te])
p_m, _, _ = fit(True, None);      ll_m = rep("line, recalibrated", p_m)
p_f, mf, _ = fit(False, F);       ll_f = rep("our features only", p_f)
p_b, mb, _ = fit(True, F);        ll_b = rep("line + our features", p_b)

print(f"\n  gain from adding our features to the line: {ll_m - ll_b:+.5f} nats")
rng = np.random.default_rng(0); t = y[te]
la = -(t*np.log(np.clip(p_b,1e-6,1-1e-6))+(1-t)*np.log(1-np.clip(p_b,1e-6,1-1e-6)))
lb = -(t*np.log(np.clip(p_m,1e-6,1-1e-6))+(1-t)*np.log(1-np.clip(p_m,1e-6,1-1e-6)))
dd = lb-la
bs = np.array([dd[rng.integers(0,len(dd),len(dd))].mean() for _ in range(4000)])
print(f"  95% CI [{np.percentile(bs,2.5):+.5f}, {np.percentile(bs,97.5):+.5f}]  "
      f"P(our features help) = {(bs>0).mean():.3f}")

print(f"\n  coefficients in 'line + our features' (standardised):")
names = ["logit(market)"] + F
for n_, c in sorted(zip(names, mb.coef_[0]), key=lambda kv: -abs(kv[1])):
    print(f"    {n_:26} {c:+.4f}")

print(f"\n{'='*88}\nBETTING TEST -- would disagreeing with the line have made money?\n{'='*88}")
edge = p_b - mp[te]
hold = (d.raw_home_implied_prob + d.raw_away_implied_prob - 1).values[te]
for thr in [0.02, 0.03, 0.05, 0.08]:
    bet_home = edge > thr; bet_away = edge < -thr
    n = int(bet_home.sum() + bet_away.sum())
    if n < 40: continue
    # decimal payout from the American moneyline actually offered
    hml = d.home_moneyline.values[te]; aml = d.away_moneyline.values[te]
    dec = lambda ml: np.where(ml > 0, 1 + ml/100.0, 1 + 100.0/np.abs(ml))
    won = np.where(bet_home, t == 1, np.where(bet_away, t == 0, False))
    pay = np.where(bet_home, dec(hml), dec(aml))
    stake = (bet_home | bet_away).astype(float)
    profit = np.where(won & (stake > 0), pay - 1, np.where(stake > 0, -1.0, 0.0)).sum()
    hitrate = won[stake > 0].mean() * 100
    print(f"  edge > {thr*100:.0f}%: {n:5d} bets  hit {hitrate:5.2f}%  "
          f"ROI {profit/n*100:+6.2f}%  profit {profit:+8.1f} units")
print(f"\n  average hold in this data: {hold.mean()*100:.2f}%  "
      f"-> break-even needs ROI > 0 after that spread")
