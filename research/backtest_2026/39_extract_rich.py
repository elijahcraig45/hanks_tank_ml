"""Rich PA-level extract for the simulator exploration (READ-ONLY BigQuery).

Adds what the original 23_pa_extract.py dropped: at-bat order, base/out state before
the PA, runs scored on the play, pitches in the PA, batted-ball quality (EV/LA/xwOBA),
times-through-order, and raw event names (baserunning outs, errors, sac flies).

2015-2025 comes from mlb_historical_data (full Savant schema). 2026 comes from
mlb_2026_season, whose schema lacks at_bat_number / on_Xb / post scores, so for 2026 we
get the per-PA outcome + quality + pitch count only. Also pulls the post-2026-09-07
evaluation slate (lineups, starters, scores, V10 pregame predictions, v10 features).

Nothing is written to BigQuery. Outputs go to data/backtest_2026/rich/.
"""
import os
import pandas as pd
from google.cloud import bigquery

OUT = "data/backtest_2026/rich"
os.makedirs(OUT, exist_ok=True)
bq = bigquery.Client(project="hankstank")

HIST = """
WITH p AS (
  SELECT game_pk, game_date, game_year, at_bat_number, pitch_number, batter, pitcher,
         stand, p_throws, inning, inning_topbot, outs_when_up,
         on_1b IS NOT NULL AS r1, on_2b IS NOT NULL AS r2, on_3b IS NOT NULL AS r3,
         events, bb_type, launch_speed, launch_angle,
         estimated_woba_using_speedangle AS xwoba, woba_value, woba_denom,
         bat_score, post_bat_score, home_team, away_team, n_thruorder_pitcher,
         post_home_score, post_away_score
  FROM `hankstank.mlb_historical_data.statcast_pitches`
  WHERE game_type = 'R'
),
np AS (SELECT game_pk, at_bat_number, COUNT(*) AS n_pitches FROM p GROUP BY 1, 2)
SELECT p.* EXCEPT(pitch_number), np.n_pitches
FROM p JOIN np USING (game_pk, at_bat_number)
WHERE p.events IS NOT NULL AND p.events != ''
"""

CUR = """
WITH p AS (
  SELECT game_pk, game_date, game_year, batter, pitcher, stand, p_throws, inning,
         inning_topbot, outs_when_up, events, launch_speed, launch_angle,
         estimated_woba_using_speedangle AS xwoba, woba_value, woba_denom,
         home_team, away_team, home_score, away_score
  FROM `hankstank.mlb_2026_season.statcast_pitches`
  WHERE game_type = 'R'
),
np AS (SELECT game_pk, inning, inning_topbot, batter, pitcher, COUNT(*) AS n_pitches
       FROM p GROUP BY 1, 2, 3, 4, 5)
SELECT p.*, np.n_pitches
FROM p JOIN np USING (game_pk, inning, inning_topbot, batter, pitcher)
WHERE p.events IS NOT NULL AND p.events != ''
"""

LINEUPS = """
WITH snap AS (
  SELECT game_pk, game_date, team_type, player_id, batting_order, fetched_at, game_time_utc,
         lineup_confirmed,
         ROW_NUMBER() OVER (PARTITION BY game_pk, team_type, batting_order
                            ORDER BY fetched_at DESC) rn
  FROM `hankstank.mlb_2026_season.lineups`
  WHERE batting_order BETWEEN 1 AND 9 AND fetched_at < game_time_utc
    AND game_date > '2026-09-07'
),
lu AS (SELECT * FROM snap WHERE rn = 1),
sp AS (
  SELECT * EXCEPT(rn) FROM (
    SELECT game_pk, home_starter_id, away_starter_id, home_win_probability, predicted_at,
           game_time_utc AS gt,
           ROW_NUMBER() OVER (PARTITION BY game_pk ORDER BY predicted_at DESC) rn
    FROM `hankstank.mlb_2026_season.game_predictions`
    WHERE predicted_at < game_time_utc AND game_date > '2026-09-07') WHERE rn = 1
)
SELECT lu.game_pk, lu.game_date, lu.team_type, lu.batting_order, lu.player_id,
       lu.lineup_confirmed, sp.home_starter_id, sp.away_starter_id,
       sp.home_win_probability, g.home_team_name, g.away_team_name, g.venue_id,
       g.home_score, g.away_score, IF(g.home_score > g.away_score, 1, 0) AS home_win
FROM lu JOIN sp USING (game_pk)
JOIN `hankstank.mlb_2026_season.games` g USING (game_pk)
WHERE g.game_type = 'R' AND g.status LIKE '%Final%' AND g.home_score IS NOT NULL
"""

FEATS = """
WITH gt AS (SELECT game_pk, MIN(game_time_utc) t FROM `hankstank.mlb_2026_season.game_predictions`
            WHERE game_time_utc IS NOT NULL GROUP BY 1)
SELECT * EXCEPT(rn) FROM (
  SELECT v.game_pk, v.game_date, v.elo_differential, v.pythag_differential,
         v.sp_quality_composite_diff, v.home_park_factor,
         ROW_NUMBER() OVER (PARTITION BY v.game_pk ORDER BY v.computed_at DESC) rn
  FROM `hankstank.mlb_2026_season.game_v10_features` v JOIN gt USING (game_pk)
  WHERE v.computed_at < gt.t) WHERE rn = 1
"""

GAMES26 = """
SELECT game_pk, game_date, home_team_name, away_team_name, home_score, away_score, venue_id
FROM `hankstank.mlb_2026_season.games`
WHERE game_type='R' AND status LIKE '%Final%' AND home_score IS NOT NULL
"""


def run(sql, name):
    df = bq.query(sql).to_dataframe()
    if "game_date" in df:
        df["game_date"] = pd.to_datetime(df["game_date"])
    df.to_parquet(f"{OUT}/{name}.parquet", index=False)
    print(name, df.shape)
    return df


if __name__ == "__main__":
    run(HIST, "pa_hist")
    run(CUR, "pa_2026")
    run(LINEUPS, "lineups_post0907")
    run(FEATS, "feats_2026_all")
    run(GAMES26, "games_2026")
