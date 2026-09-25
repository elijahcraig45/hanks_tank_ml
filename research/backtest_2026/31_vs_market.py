"""How good is the market, measured on the same games as our models?

Joins 2012-2021 historical moneylines to our game results and scores the market the
same way we score ourselves. This is the benchmark: it says how far our ~54% actually
is from the closing line, in the units that matter.
"""
import warnings
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from google.cloud import bigquery
warnings.filterwarnings("ignore")

o = pd.read_csv("data/odds/kaggle_market_odds.csv")
o["game_date"] = pd.to_datetime(o.game_date)
o = o[~o.is_synthetic_odds].copy()
print(f"odds rows (real, non-synthetic): {len(o)}  {o.game_date.min().date()} -> {o.game_date.max().date()}")

g = bigquery.Client(project="hankstank").query("""
  SELECT game_date, home_team_name, away_team_name, home_score, away_score
  FROM `hankstank.mlb_historical_data.games_historical`
  WHERE game_type='R' AND home_score IS NOT NULL
    AND EXTRACT(YEAR FROM game_date) BETWEEN 2015 AND 2021
""").to_dataframe()
g["game_date"] = pd.to_datetime(g.game_date)
g["home_win"] = (g.home_score > g.away_score).astype(int)
print(f"our games 2015-2021: {len(g)}")

m = g.merge(o, on=["game_date", "home_team_name", "away_team_name"], how="inner") \
     if "home_team_name" in o.columns else \
     g.merge(o.rename(columns={"home_team": "home_team_name", "away_team": "away_team_name"}),
             on=["game_date", "home_team_name", "away_team_name"], how="inner")
# a doubleheader shares a date+teams; keep one odds row per game
m = m.drop_duplicates(subset=["game_date", "home_team_name", "away_team_name"])
print(f"joined: {len(m)} games ({len(m)/len(g)*100:.1f}% of our 2015-2021 games)\n")

y = m.home_win.values
p = m.market_home_prob.values.astype(float)
def rep(name, pp, t):
    pp = np.clip(pp, 1e-6, 1-1e-6)
    corr = (pp >= .5).astype(int) == t
    print(f"  {name:34} acc={corr.mean()*100:6.2f}%  auc={roc_auc_score(t,pp):.4f}  "
          f"brier={brier_score_loss(t,pp):.4f}  logloss={log_loss(t,pp):.5f}")

print("="*88)
print(f"THE MARKET, scored the way we score ourselves (n={len(m)}, 2015-2021)")
print("="*88)
rep("closing line (vig removed)", p, y)
rep("always pick home", np.full(len(y), y.mean()), y)
print(f"\n  home win rate {y.mean()*100:.2f}%   market mean P(home) {p.mean()*100:.2f}%")
print(f"  market picks the home team {(p>=.5).mean()*100:.1f}% of the time")

print(f"\n{'='*88}\nMARKET CALIBRATION -- is the line honest?\n{'='*88}")
print(f"  {'implied P(home)':22}{'n':>7}{'mean pred':>12}{'actual':>10}{'gap':>9}")
for lo, hi in [(0,.35),(.35,.42),(.42,.47),(.47,.53),(.53,.58),(.58,.65),(.65,1.01)]:
    k = (p >= lo) & (p < hi)
    if k.sum() < 30: continue
    print(f"  {f'{lo:.2f}-{hi:.2f}':22}{k.sum():>7}{p[k].mean()*100:>11.2f}%{y[k].mean()*100:>9.2f}%"
          f"{(y[k].mean()-p[k].mean())*100:>+8.2f}pp")

print(f"\n{'='*88}\nSELECTIVITY -- the market's own confidence curve\n{'='*88}")
conf = np.maximum(p, 1-p); corr = (p >= .5).astype(int) == y
print(f"  {'threshold':14}{'n':>7}{'coverage':>10}{'accuracy':>10}")
for t in [.50,.55,.60,.65,.70]:
    k = conf >= t
    print(f"  conf >= {t:.2f}  {k.sum():>7}{k.mean()*100:>9.1f}%{corr[k].mean()*100:>9.2f}%")

print(f"\n{'='*88}\nWHAT BEATING IT WOULD REQUIRE\n{'='*88}")
vig = (m.raw_home_implied_prob + m.raw_away_implied_prob - 1).mean()
print(f"  average bookmaker hold (vig) in this data: {vig*100:.2f}%")
print(f"  break-even win rate on a coin-flip-priced bet: {(1/(2-vig))*100:.2f}%")
print(f"  our best model's log-loss on 2026: 0.6817   |  market's here: {log_loss(y,np.clip(p,1e-6,1-1e-6)):.4f}")
print(f"  market log-loss advantage over always-home: "
      f"{log_loss(y,np.full(len(y),y.mean())) - log_loss(y,np.clip(p,1e-6,1-1e-6)):+.5f} nats")
print(f"  our best model's advantage over always-home (2026): +0.00717 nats")
m.to_parquet("data/odds/market_joined.parquet", index=False)
print(f"\nwrote data/odds/market_joined.parquet ({len(m)} games)")
