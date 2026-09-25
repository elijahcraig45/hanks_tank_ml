"""Phase 1 data: pull plate-appearance level rows for 2015-2026.

One row per PA (the pitch that carries `events`), with batter, pitcher, both
handednesses, the outcome class, and whether the pitcher was that game's starter.
Only columns the model needs are selected, to bound bytes scanned.
"""
import pandas as pd
from google.cloud import bigquery

EVENT_CASE = """
  CASE
    WHEN events IN ('strikeout','strikeout_double_play') THEN 'K'
    WHEN events IN ('walk','intent_walk','hit_by_pitch','catcher_interf') THEN 'BB'
    WHEN events = 'single' THEN '1B'
    WHEN events = 'double' THEN '2B'
    WHEN events = 'triple' THEN '3B'
    WHEN events = 'home_run' THEN 'HR'
    WHEN events = 'field_error' THEN '1B'          -- reaches base
    WHEN events IN ('grounded_into_double_play','double_play',
                    'strikeout_double_play','sac_fly_double_play','triple_play') THEN 'DP'
    WHEN events IN ('field_out','force_out','sac_fly','sac_bunt',
                    'fielders_choice','fielders_choice_out') THEN 'OUT'
    ELSE NULL
  END
"""

SQL = f"""
WITH base AS (
  SELECT game_year, game_date, game_pk, batter, pitcher, stand, p_throws, inning,
         inning_topbot, home_team, away_team, {EVENT_CASE} AS ev
  FROM `hankstank.mlb_historical_data.statcast_pitches`
  WHERE game_type = 'R' AND events IS NOT NULL AND events != ''
  UNION ALL
  SELECT game_year, game_date, game_pk, batter, pitcher, stand, p_throws, inning,
         inning_topbot, home_team, away_team, {EVENT_CASE} AS ev
  FROM `hankstank.mlb_2026_season.statcast_pitches`
  WHERE game_type = 'R' AND events IS NOT NULL AND events != ''
),
firstinn AS (      -- a pitcher who appeared in inning 1 of a game is its starter
  SELECT game_pk, pitcher, MIN(inning) AS first_inning
  FROM base GROUP BY game_pk, pitcher
)
SELECT b.game_year, b.game_date, b.game_pk, b.batter, b.pitcher,
       b.stand, b.p_throws, b.inning, b.ev,
       b.inning_topbot, b.home_team, b.away_team,
       IF(b.inning_topbot = 'Top', b.away_team, b.home_team) AS batting_team,
       (b.inning_topbot = 'Bot') AS home_batting,
       (f.first_inning = 1) AS pitcher_is_starter
FROM base b
JOIN firstinn f ON f.game_pk = b.game_pk AND f.pitcher = b.pitcher
WHERE b.ev IS NOT NULL
"""

df = bigquery.Client(project="hankstank").query(SQL).to_dataframe()
df["game_date"] = pd.to_datetime(df["game_date"])
df = df.sort_values(["game_date", "game_pk", "inning"]).reset_index(drop=True)
for c in ("batter", "pitcher", "inning", "game_year"):
    df[c] = df[c].astype("int32")
df["ev"] = df["ev"].astype("category")
for c in ("stand", "p_throws", "inning_topbot", "home_team", "away_team", "batting_team"):
    df[c] = df[c].astype("category")
out = "data/backtest_2026/pa_2015_2026.parquet"
df.to_parquet(out, index=False, compression="zstd")
print(f"rows={len(df):,}  seasons={df.game_year.min()}-{df.game_year.max()}")
print(df.ev.value_counts().to_string())
print(f"\nleague rates:\n{(df.ev.value_counts(normalize=True)*100).round(3).to_string()}")
print(f"\nstarter PAs {df.pitcher_is_starter.mean()*100:.1f}%  "
      f"batters {df.batter.nunique():,}  pitchers {df.pitcher.nunique():,}")
import os
print(f"wrote {out} ({os.path.getsize(out)/1e6:.1f} MB)")
