"""Build NEW features from pitch-level Statcast: SP recent form + bullpen fatigue.

These are V10's own untested "Tier 1 / Tier 2" next steps. Everything is windowed
strictly BEFORE the game in question -- rolling windows use ROWS BETWEEN n PRECEDING
AND 1 PRECEDING, and bullpen load counts only prior calendar days.
"""
import pandas as pd
from google.cloud import bigquery

SQL = """
WITH p AS (
  SELECT game_pk, game_date, pitcher,
         IF(inning_topbot = 'Top', home_team, away_team) AS pitch_team,
         inning, events, description, release_speed,
         estimated_woba_using_speedangle AS xwoba, woba_denom
  FROM `hankstank.mlb_2026_season.statcast_pitches`
  WHERE game_type = 'R'
),
pg AS (                       -- one row per pitcher per game
  SELECT pitcher, game_pk, game_date, ANY_VALUE(pitch_team) AS pitch_team,
         COUNT(*) AS pitches,
         MIN(inning) AS first_inning,
         AVG(release_speed) AS velo,
         SAFE_DIVIDE(COUNTIF(events = 'strikeout'), NULLIF(COUNTIF(woba_denom > 0), 0)) AS k_rate,
         AVG(IF(woba_denom > 0, xwoba, NULL)) AS xwoba
  FROM p GROUP BY pitcher, game_pk, game_date
),
sp AS (                       -- starter rolling form over the previous 3 starts
  SELECT pitcher, game_pk, game_date,
    AVG(xwoba)   OVER w AS sp_form_xwoba,
    AVG(k_rate)  OVER w AS sp_form_k,
    AVG(velo)    OVER w AS sp_form_velo,
    AVG(pitches) OVER w AS sp_form_pitches,
    COUNT(1)     OVER w AS sp_form_n
  FROM pg WHERE first_inning = 1
  WINDOW w AS (PARTITION BY pitcher ORDER BY game_date
               ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING)
),
relief AS (                   -- per team per day, bullpen workload only
  SELECT pitch_team, game_date,
         SUM(pitches) AS rp_pitches,
         AVG(IF(xwoba IS NOT NULL, xwoba, NULL)) AS rp_xwoba
  FROM pg WHERE first_inning > 1
  GROUP BY pitch_team, game_date
),
bull AS (                     -- prior-3-day and prior-7-day bullpen load
  SELECT pitch_team, game_date,
    SUM(rp_pitches) OVER w3 AS bp_pitches_3d,
    AVG(rp_xwoba)   OVER w7 AS bp_xwoba_7d
  FROM relief
  WINDOW
    w3 AS (PARTITION BY pitch_team ORDER BY UNIX_DATE(game_date)
           RANGE BETWEEN 3 PRECEDING AND 1 PRECEDING),
    w7 AS (PARTITION BY pitch_team ORDER BY UNIX_DATE(game_date)
           RANGE BETWEEN 7 PRECEDING AND 1 PRECEDING)
),
gmap AS (   -- team abbreviations straight from statcast, no name matching needed
  SELECT game_pk, ANY_VALUE(home_team) AS home_abbr, ANY_VALUE(away_team) AS away_abbr
  FROM `hankstank.mlb_2026_season.statcast_pitches`
  WHERE game_type = 'R' GROUP BY game_pk
),
g AS (
  SELECT gp.game_pk, gp.game_date, gp.home_starter_id, gp.away_starter_id,
         gmap.home_abbr, gmap.away_abbr
  FROM (SELECT * EXCEPT(rn) FROM (
          SELECT game_pk, game_date, home_starter_id, away_starter_id, game_time_utc,
                 ROW_NUMBER() OVER (PARTITION BY game_pk ORDER BY predicted_at DESC) rn
          FROM `hankstank.mlb_2026_season.game_predictions`
          WHERE predicted_at < game_time_utc) WHERE rn = 1) gp
  JOIN `hankstank.mlb_2026_season.games` gm USING (game_pk)
  JOIN gmap USING (game_pk)
  WHERE gm.game_type = 'R'
)
SELECT g.game_pk, g.game_date,
  hs.sp_form_xwoba AS home_sp_form_xwoba, hs.sp_form_k AS home_sp_form_k,
  hs.sp_form_velo  AS home_sp_form_velo,  hs.sp_form_n AS home_sp_form_n,
  as_.sp_form_xwoba AS away_sp_form_xwoba, as_.sp_form_k AS away_sp_form_k,
  as_.sp_form_velo  AS away_sp_form_velo,  as_.sp_form_n AS away_sp_form_n,
  hb.bp_pitches_3d AS home_bp_pitches_3d, hb.bp_xwoba_7d AS home_bp_xwoba_7d,
  ab.bp_pitches_3d AS away_bp_pitches_3d, ab.bp_xwoba_7d AS away_bp_xwoba_7d
FROM g
LEFT JOIN sp hs  ON hs.game_pk = g.game_pk  AND hs.pitcher = g.home_starter_id
LEFT JOIN sp as_ ON as_.game_pk = g.game_pk AND as_.pitcher = g.away_starter_id
LEFT JOIN bull hb ON hb.game_date = g.game_date AND hb.pitch_team = g.home_abbr
LEFT JOIN bull ab ON ab.game_date = g.game_date AND ab.pitch_team = g.away_abbr
ORDER BY g.game_date, g.game_pk
"""
df = bigquery.Client(project="hankstank").query(SQL).to_dataframe()
df["game_date"] = pd.to_datetime(df["game_date"])
out = "data/backtest_2026/statcast_new_features.parquet"
df.to_parquet(out, index=False)
print(f"rows={len(df)}")
print("coverage (non-null %):")
for c in df.columns:
    if c in ("game_pk","game_date"): continue
    print(f"  {c:24s} {df[c].notna().mean()*100:5.1f}%")
print(f"wrote {out}")
