-- College drive simulator (shadow, model_version 'cfb_drive_sim_v1'): three tables.
--
-- Written by src/cfb (cfb_drives.py, cfb_drive_sim.py) with the shared simulator
-- src/nfl/drive_sim.py (config CFB). All three are loaded with CREATE_NEVER, so they must
-- exist before the backfill or the shadow runs. The two prediction tables mirror
-- nfl_season.game_sim_distributions / game_predictions_drive_sim column for column (same
-- names, same types) plus one additive column, `division` ('fbs' | 'fcs', the schedule's
-- group tag), so the backend can read CFB and NFL with the same code.
--
-- Order of operations:
--   1. run this file (bq query --use_legacy_sql=false < this file);
--   2. backfill drives once, locally (CFBD responses are cached; ~0 new API calls):
--        CFBD_API_KEY="$(cat ~/.cfbd_key)" python src/cfb/cfb_drives.py --seasons 2021-2026 \
--            --cfbd-cache <repo>/data/cfbd --write
--   3. deploy with scripts/gcp/cfb/deploy_cfb.sh --shadow (sets CFB_DRIVE_SIM_SHADOW=1 and
--      adds the Sunday cfb-weekly-drives job that keeps drives current).

-- One row per drive (research/football_2026_09/cfb_drive_sim/build_drives.py transform).
-- Replaced by game_id (backfill_cfb.replace_game_ids); never truncated.
CREATE TABLE IF NOT EXISTS `hankstank.cfb_historical.drives` (
  game_id STRING NOT NULL,   -- ESPN id (= CFBD gameId)
  season INT64 NOT NULL,
  week INT64 NOT NULL,       -- the schedule's week (cfb_historical.games), not CFBD's
  season_type STRING,        -- REG | POST
  home_team STRING,          -- ESPN abbreviations, as in cfb_historical.games
  away_team STRING,
  posteam STRING,
  defteam STRING,
  qtr FLOAT64,               -- start period; 5+ = overtime
  game_half STRING,          -- Half1 | Half2 | Overtime
  res STRING,                -- TD FG MFG PUNT TO OTD TOD SAF EOH
  hsr0 FLOAT64,              -- half seconds remaining at drive start
  yl FLOAT64,                -- start yards to goal
  sd0 FLOAT64,               -- offense score differential at start
  off_pts FLOAT64,
  def_pts FLOAT64,
  off_home BOOL,
  yl_next FLOAT64,           -- next drive's start (field-position chaining)
  dur FLOAT64,               -- seconds until the next drive starts
  drive_number FLOAT64,
  cfbd_offense STRING,       -- CFBD school names, kept for auditing the crosswalk
  cfbd_defense STRING,
  division STRING
)
CLUSTER BY season, posteam;

-- Contract shape of nfl_season.game_sim_distributions (predictions_contract.md), + division.
-- One row per (game_id, model_version, predicted_at); the writer replaces by game_id.
-- All *_points / total / margin Dists and probabilities are the RAW simulation;
-- margin_exact is the sim shape tilted to spread_line when one is known
-- (margin_exact_basis = 'sim_shape_at_spread'), else raw ('raw_sim'). p_tie is 0: college
-- overtime always produces a winner.
CREATE TABLE IF NOT EXISTS `hankstank.cfb_season.game_sim_distributions` (
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
  spread_line FLOAT64,       -- positive = home favoured (cfb_season.betting_lines)
  total_line FLOAT64,
  division STRING
)
PARTITION BY game_date
CLUSTER BY season, week;

-- Shaped like cfb_season.game_predictions for the shared columns, mirroring
-- nfl_season.game_predictions_drive_sim, + division. game_date is the kickoff instant.
-- home_win_probability = Platt(logit(sim_p_raw)), frozen, fit 2022-2025 research outputs.
-- vegas_implied_home_prob = Phi(spread_line / 14.92) (market sigma fit on 2022).
CREATE TABLE IF NOT EXISTS `hankstank.cfb_season.game_predictions_drive_sim` (
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
  model_vs_vegas_edge FLOAT64,
  division STRING
)
PARTITION BY TIMESTAMP_TRUNC(game_date, DAY)
CLUSTER BY season, week;
