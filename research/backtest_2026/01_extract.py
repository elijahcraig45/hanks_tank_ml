"""Extract the honest 2026 backtest dataset.

Only *pregame* prediction rows are used: game_predictions is also written by
post-hoc backfills (418 rows in 2026), and those carry features computed after
the game finished. Taking the latest row per game_pk without the
predicted_at < game_time_utc guard silently pulls those in.
"""
import pandas as pd
from google.cloud import bigquery

SQL = """
WITH p AS (
  SELECT * EXCEPT(rn) FROM (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY game_pk ORDER BY predicted_at DESC) rn
    FROM `hankstank.mlb_2026_season.game_predictions`
    WHERE predicted_at < game_time_utc
  ) WHERE rn = 1
)
SELECT
  p.*,
  g.home_score, g.away_score, g.venue_id,
  IF(g.home_score > g.away_score, 1, 0) AS home_win
FROM p
JOIN `hankstank.mlb_2026_season.games` g USING (game_pk)
WHERE g.game_type = 'R'
  AND g.status LIKE '%Final%'
  AND g.home_score IS NOT NULL
  AND p.home_win_probability IS NOT NULL
ORDER BY g.game_date, p.game_pk
"""

df = bigquery.Client(project="hankstank").query(SQL).to_dataframe()
df["game_date"] = pd.to_datetime(df["game_date"])
out = "data/backtest_2026/games_2026_pregame.parquet"
df.to_parquet(out, index=False)
print(f"rows={len(df)}  cols={len(df.columns)}  {df.game_date.min().date()} -> {df.game_date.max().date()}")
print(f"home_win rate = {df.home_win.mean():.4f}")
print(f"prod v10 acc  = {(((df.home_win_probability>=.5)).astype(int)==df.home_win).mean():.4f}")
print(f"wrote {out}")
