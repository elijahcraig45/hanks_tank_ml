"""Pull 2026 lineups for simulation input.

The lineups table holds ~3 pregame snapshots per game (27 batting-order rows, not 9),
so this takes the LATEST snapshot that is still strictly before first pitch.
"""
import pandas as pd
from google.cloud import bigquery

SQL = """
WITH snap AS (
  SELECT game_pk, game_date, team_id, team_type, player_id, batting_order,
         bat_side, fetched_at, game_time_utc, lineup_confirmed,
         ROW_NUMBER() OVER (PARTITION BY game_pk, team_type, batting_order
                            ORDER BY fetched_at DESC) rn
  FROM `hankstank.mlb_2026_season.lineups`
  WHERE batting_order BETWEEN 1 AND 9
    AND fetched_at < game_time_utc          -- pregame only
),
lu AS (SELECT * FROM snap WHERE rn = 1),
sp AS (
  SELECT * EXCEPT(rn) FROM (
    SELECT game_pk, home_starter_id, away_starter_id, game_time_utc,
           ROW_NUMBER() OVER (PARTITION BY game_pk ORDER BY predicted_at DESC) rn
    FROM `hankstank.mlb_2026_season.game_predictions`
    WHERE predicted_at < game_time_utc) WHERE rn = 1
)
SELECT lu.game_pk, lu.game_date, lu.team_type, lu.batting_order, lu.player_id,
       lu.bat_side, lu.lineup_confirmed,
       sp.home_starter_id, sp.away_starter_id,
       g.home_team_name, g.away_team_name, g.venue_id,
       g.home_score, g.away_score,
       IF(g.home_score > g.away_score, 1, 0) AS home_win
FROM lu
JOIN sp USING (game_pk)
JOIN `hankstank.mlb_2026_season.games` g USING (game_pk)
WHERE g.game_type = 'R' AND g.status LIKE '%Final%' AND g.home_score IS NOT NULL
ORDER BY lu.game_date, lu.game_pk, lu.team_type, lu.batting_order
"""
df = bigquery.Client(project="hankstank").query(SQL).to_dataframe()
df["game_date"] = pd.to_datetime(df["game_date"])
out = "data/backtest_2026/lineups_2026.parquet"
df.to_parquet(out, index=False)
cnt = df.groupby(["game_pk", "team_type"]).size()
full = df.groupby("game_pk").size()
print(f"rows={len(df)}  games={df.game_pk.nunique()}")
print(f"sides with exactly 9 batters: {(cnt==9).mean()*100:.1f}%")
print(f"games with both sides complete (18 rows): {(full==18).sum()} of {df.game_pk.nunique()}")
print(f"lineup_confirmed: {df.lineup_confirmed.mean()*100:.1f}%")
print(f"date range {df.game_date.min().date()} -> {df.game_date.max().date()}")
print(f"wrote {out}")
