"""Read-only pulls for eval_v10_fixes.py. Writes parquet into <scratch_dir> only.

  full_features_2026.parquet  game_v10_features rows computed before first pitch
  served_2026.parquet         the genuine pregame game_predictions row per game
  games_2026_R.parquet        deduped regular-season results (for games-played counts)

The team Statcast pull is team_statcast.sql, run by the same entry point.
Usage: python research/v10_fixes/fetch_inputs.py <scratch_dir>
"""
import sys
from pathlib import Path

from google.cloud import bigquery

OUT = Path(sys.argv[1])
bq = bigquery.Client(project="hankstank")

GAMES = """
SELECT game_pk, ANY_VALUE(game_date) AS game_date,
       ANY_VALUE(home_team_id) AS home_team_id, ANY_VALUE(away_team_id) AS away_team_id,
       ANY_VALUE(home_score) AS home_score, ANY_VALUE(away_score) AS away_score
FROM `hankstank.mlb_2026_season.games`
WHERE game_type = 'R' AND status LIKE '%Final%' AND home_score IS NOT NULL
GROUP BY game_pk
"""

FEATURES = f"""
WITH gt AS (
  SELECT game_pk, MIN(game_time_utc) AS game_time_utc
  FROM `hankstank.mlb_2026_season.game_predictions`
  WHERE game_time_utc IS NOT NULL GROUP BY game_pk
),
f AS (
  SELECT * EXCEPT(rn) FROM (
    SELECT v.*, ROW_NUMBER() OVER (PARTITION BY v.game_pk ORDER BY v.computed_at DESC) rn
    FROM `hankstank.mlb_2026_season.game_v10_features` v
    JOIN gt USING (game_pk)
    WHERE v.computed_at < gt.game_time_utc
  ) WHERE rn = 1
)
SELECT f.*, g.home_score, g.away_score, IF(g.home_score > g.away_score, 1, 0) AS home_win
FROM f JOIN ({GAMES}) g USING (game_pk)
ORDER BY g.game_date, f.game_pk
"""

SERVED = f"""
SELECT * EXCEPT(rn) FROM (
  SELECT p.game_pk, p.home_win_probability, p.predicted_at, p.game_time_utc, p.model_version,
         ROW_NUMBER() OVER (PARTITION BY p.game_pk ORDER BY p.predicted_at DESC) rn
  FROM `hankstank.mlb_2026_season.game_predictions` p
  WHERE p.predicted_at < p.game_time_utc AND p.home_win_probability IS NOT NULL
) WHERE rn = 1
"""

for name, sql in [("games_2026_R", GAMES), ("full_features_2026", FEATURES),
                  ("served_2026", SERVED),
                  ("team_statcast_asof", (Path(__file__).parent / "team_statcast.sql").read_text())]:
    df = bq.query(sql).to_dataframe()
    df.to_parquet(OUT / f"{name}.parquet", index=False)
    print(f"{name}: {df.shape}")
