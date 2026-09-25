"""Extract the FULL 166-column v10 feature table with a pregame guard.

game_v10_features carries computed_at; like game_predictions it is also written by
backfills, so rows computed after first pitch must be dropped or the search trains
on post-hoc features.
"""
import pandas as pd
from google.cloud import bigquery

SQL = """
WITH gt AS (            -- first pitch time per game, from the predictions table
  SELECT game_pk, MIN(game_time_utc) AS game_time_utc
  FROM `hankstank.mlb_2026_season.game_predictions`
  WHERE game_time_utc IS NOT NULL GROUP BY game_pk
),
f AS (
  SELECT * EXCEPT(rn) FROM (
    SELECT v.*, ROW_NUMBER() OVER (PARTITION BY v.game_pk ORDER BY v.computed_at DESC) rn
    FROM `hankstank.mlb_2026_season.game_v10_features` v
    JOIN gt USING (game_pk)
    WHERE v.computed_at < gt.game_time_utc          -- pregame only
  ) WHERE rn = 1
)
SELECT f.*, g.home_score, g.away_score,
       IF(g.home_score > g.away_score, 1, 0) AS home_win
FROM f
JOIN `hankstank.mlb_2026_season.games` g USING (game_pk)
WHERE g.game_type = 'R' AND g.status LIKE '%Final%' AND g.home_score IS NOT NULL
ORDER BY g.game_date, f.game_pk
"""
df = bigquery.Client(project="hankstank").query(SQL).to_dataframe()
df["game_date"] = pd.to_datetime(df["game_date"])
out = "data/backtest_2026/full_features_2026.parquet"
df.to_parquet(out, index=False)
num = df.select_dtypes("number").shape[1]
print(f"rows={len(df)}  cols={len(df.columns)}  numeric={num}")
print(f"{df.game_date.min().date()} -> {df.game_date.max().date()}  home_win={df.home_win.mean():.4f}")
nn = df.isna().mean().sort_values(ascending=False)
print(f"cols >30% null: {(nn>0.3).sum()}   >5% null: {(nn>0.05).sum()}")
print(f"wrote {out}")
