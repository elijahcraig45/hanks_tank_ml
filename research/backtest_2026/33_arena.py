"""Build the arena: games where we have BOTH closing odds and our features, then
score the current baseline on it. This is the fixed testbed every experiment runs on.
"""
import sys, warnings
import numpy as np, pandas as pd
sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from edge.benchmark import score
from google.cloud import bigquery

d = pd.read_parquet("data/odds/market_joined.parquet"); d["game_date"] = pd.to_datetime(d.game_date)
h = pd.concat([pd.read_parquet("data/training/train_v8_2015_2024.parquet"),
               pd.read_parquet("data/training/val_v8_2025.parquet")], ignore_index=True)
h["game_date"] = pd.to_datetime(h.game_date)
gid = bigquery.Client(project="hankstank").query("""
  SELECT game_date, game_pk, home_team_name, away_team_name, home_team_id, away_team_id, venue_id
  FROM `hankstank.mlb_historical_data.games_historical`
  WHERE game_type='R' AND EXTRACT(YEAR FROM game_date) BETWEEN 2015 AND 2021""").to_dataframe()
gid["game_date"] = pd.to_datetime(gid.game_date)

F = [c for c in ["elo_differential","pythag_differential","win_pct_diff","run_diff_differential",
                 "home_pythag_last30","away_pythag_last30","era_proxy_differential",
                 "streak_differential","luck_differential"] if c in h.columns]
m = (d.merge(gid, on=["game_date","home_team_name","away_team_name"], how="inner")
       .merge(h[["game_date","home_team_id","away_team_id","home_won"]+F],
              on=["game_date","home_team_id","away_team_id"], how="inner"))
m = m.drop_duplicates(["game_date","home_team_id","away_team_id"]).sort_values("game_date").reset_index(drop=True)
m.to_parquet("data/odds/arena.parquet", index=False)
print(f"ARENA: {len(m)} games, {m.game_date.min().date()} -> {m.game_date.max().date()}")
print(f"  features carried: {F}")
print(f"  seasons: {sorted(m.game_date.dt.year.unique())}")

y = m.home_won.values.astype(int)
X = m[F].astype(float); X = X.fillna(X.median())
tr = (m.game_date < pd.Timestamp("2020-01-01")).values
sc = StandardScaler().fit(X[tr])
mod = LogisticRegression(C=0.1, max_iter=4000).fit(sc.transform(X[tr]), y[tr])
p = mod.predict_proba(sc.transform(X))[:, 1]

te = ~tr
half = np.zeros(int(te.sum()), bool); half[:int(te.sum()*0.5)] = True
score(y[te], p[te], m.market_home_prob.values[te],
      m.home_moneyline.values[te], m.away_moneyline.values[te],
      name="BASELINE  team-strength features only  (train 2015-19, test 2020-21)",
      train_mask=half)
