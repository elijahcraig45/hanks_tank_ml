-- NFL drive simulator (shadow, model_version 'drive_sim_v1'): three tables.
--
-- Written by src/nfl (drive_sim.py, drives.py). The two prediction tables are loaded
-- with CREATE_NEVER, so they must exist before the shadow is switched on; the explicit
-- schema also avoids the autodetect trap that turned a naive game_date into INT64.
--
-- Order of operations:
--   1. run this file;
--   2. backfill drives once, from local nflverse pbp (not in the function):
--        python src/nfl/drives.py --pbp-dir <dir of pbp_YYYY.parquet> --seasons 2008-2025 --write
--      (2024-2025 are the only seasons the 2026 fits read; the rest are for backtests);
--   3. the Tuesday ingest then refreshes the current season's drives each week;
--   4. deploy with scripts/gcp/nfl/deploy_nfl.sh --shadow (sets NFL_DRIVE_SIM_SHADOW=1).

-- One row per drive (research/football_2026_09/drive_sim/nfl_drives.py, verbatim).
-- Replaced one season at a time (bq_io.replace_seasons); never truncated.
CREATE TABLE IF NOT EXISTS `hankstank.nfl_historical.drives` (
  game_id STRING NOT NULL,
  fixed_drive FLOAT64,
  season INT64 NOT NULL,
  week INT64 NOT NULL,
  season_type STRING,
  home_team STRING,
  away_team STRING,
  posteam STRING,
  defteam STRING,
  qtr FLOAT64,
  game_half STRING,          -- Half1 | Half2 | Overtime
  res STRING,                -- TD FG MFG PUNT TO OTD TOD SAF EOH
  hsr0 FLOAT64,              -- half seconds remaining at drive start
  yl FLOAT64,                -- start yardline_100
  hs0 FLOAT64, as0 FLOAT64,  -- score before the drive
  hs1 FLOAT64, as1 FLOAT64,  -- score after
  yl_end FLOAT64,
  off_home BOOL,
  dh FLOAT64, da FLOAT64,
  off_pts FLOAT64, def_pts FLOAT64,
  sd0 FLOAT64,               -- offense score differential at start
  hsr_next FLOAT64,
  yl_next FLOAT64,           -- next drive's start yardline (field-position chaining)
  dur FLOAT64                -- seconds until the next drive starts
)
CLUSTER BY season, posteam;

-- Contract table (predictions_contract.md, "nfl_season.game_sim_distributions").
-- One row per (game_id, model_version, predicted_at); the writer upserts by game_id, so
-- each pregame rerun replaces that game's earlier row.
-- All *_points / total / margin Dists and all probabilities are the RAW simulation.
-- margin_exact is the sim shape exponentially tilted to spread_line when a spread is
-- known (margin_exact_basis = 'sim_shape_at_spread'), else raw ('raw_sim').
CREATE TABLE IF NOT EXISTS `hankstank.nfl_season.game_sim_distributions` (
  game_id STRING NOT NULL,
  game_date DATE NOT NULL,
  season INT64 NOT NULL,
  week INT64 NOT NULL,
  predicted_at TIMESTAMP NOT NULL,
  model_version STRING NOT NULL,
  n_sims INT64,
  home_team STRING,
  away_team STRING,
  home_points_mean FLOAT64, home_points_sd FLOAT64,
  home_points_p05 INT64, home_points_p25 INT64, home_points_p50 INT64,
  home_points_p75 INT64, home_points_p95 INT64, home_points_min INT64, home_points_max INT64,
  away_points_mean FLOAT64, away_points_sd FLOAT64,
  away_points_p05 INT64, away_points_p25 INT64, away_points_p50 INT64,
  away_points_p75 INT64, away_points_p95 INT64, away_points_min INT64, away_points_max INT64,
  total_mean FLOAT64, total_sd FLOAT64,
  total_p05 INT64, total_p25 INT64, total_p50 INT64,
  total_p75 INT64, total_p95 INT64, total_min INT64, total_max INT64,
  margin_mean FLOAT64, margin_sd FLOAT64,
  margin_p05 INT64, margin_p25 INT64, margin_p50 INT64,
  margin_p75 INT64, margin_p95 INT64, margin_min INT64, margin_max INT64,
  p_home_win FLOAT64,        -- P(home wins), ties count as not a win
  p_ot FLOAT64,              -- P(tied after regulation)
  p_tie FLOAT64,             -- P(final tie)
  p_home_cover FLOAT64,      -- P(margin > spread_line); NULL without a spread
  p_over_by_line STRING,     -- JSON {"44.5": 0.51, ...}: total_line +-7 (sim mean if no line)
  margin_exact STRING,       -- JSON {"-21": p, ..., "21": p}
  margin_exact_basis STRING, -- sim_shape_at_spread | raw_sim
  spread_line FLOAT64,       -- nflverse sign: positive = home favoured
  total_line FLOAT64
)
PARTITION BY game_date
CLUSTER BY season, week;

-- Win probability and predicted scores, shaped like nfl_season.game_predictions
-- (same names for the shared columns) so the model-comparison readers can treat it like
-- the ridge shadow. home_win_probability = Platt(logit(sim_p_raw)), frozen, fit 2010-2025.
CREATE TABLE IF NOT EXISTS `hankstank.nfl_season.game_predictions_drive_sim` (
  game_id STRING NOT NULL,
  game_pk INT64,             -- crc32(game_id): stable across processes
  game_date TIMESTAMP NOT NULL,
  season INT64 NOT NULL,
  week INT64 NOT NULL,
  home_team_id STRING,
  away_team_id STRING,
  home_team_name STRING,
  away_team_name STRING,
  home_win_probability FLOAT64,
  away_win_probability FLOAT64,
  predicted_winner STRING,
  confidence_tier STRING,
  model_version STRING,
  predicted_at TIMESTAMP,
  sim_p_raw FLOAT64,         -- P(home win | not tied), uncalibrated
  predicted_home_score FLOAT64,
  predicted_away_score FLOAT64,
  predicted_home_margin FLOAT64,
  predicted_total FLOAT64,
  n_sims INT64,
  spread_line FLOAT64,
  total_line FLOAT64,
  vegas_implied_home_prob FLOAT64,
  model_vs_vegas_edge FLOAT64
)
PARTITION BY TIMESTAMP_TRUNC(game_date, DAY)
CLUSTER BY season, week;
