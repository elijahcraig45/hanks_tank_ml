-- Additive migration for the per-game season-sim projections (2026-09-28). Safe to rerun.
-- Run BEFORE deploying the code that writes them: store.write checks every table's schema
-- first and refuses (writing nothing) if a column is missing.
--
--   CLOUDSDK_CORE_ACCOUNT=elijahcraig45@gmail.com bq query --use_legacy_sql=false \
--     --project_id=hankstank < scripts/gcp/football/alter_season_sim_per_game.sql
--
-- New columns are NULL on rows written before the migration; readers treat NULL as
-- "not computed for that run".

ALTER TABLE `hankstank.nfl_season.season_sim_team`
  ADD COLUMN IF NOT EXISTS rem_wins_mean FLOAT64,
  ADD COLUMN IF NOT EXISTS rem_wins_dist STRING,
  ADD COLUMN IF NOT EXISTS projected_wins_games STRING,
  ADD COLUMN IF NOT EXISTS modal_sequence STRING,
  ADD COLUMN IF NOT EXISTS modal_sequence_freq FLOAT64,
  ADD COLUMN IF NOT EXISTS modal_sequence_record_p FLOAT64;

ALTER TABLE `hankstank.cfb_season.season_sim_team`
  ADD COLUMN IF NOT EXISTS rem_wins_mean FLOAT64,
  ADD COLUMN IF NOT EXISTS rem_wins_dist STRING,
  ADD COLUMN IF NOT EXISTS projected_wins_games STRING,
  ADD COLUMN IF NOT EXISTS modal_sequence STRING,
  ADD COLUMN IF NOT EXISTS modal_sequence_freq FLOAT64,
  ADD COLUMN IF NOT EXISTS modal_sequence_record_p FLOAT64;

-- The per-game table (same definition as create_season_sim_tables.sql).

CREATE TABLE IF NOT EXISTS `hankstank.nfl_season.season_sim_games` (
  sport            STRING    NOT NULL,
  season           INT64     NOT NULL,
  as_of_week       INT64     NOT NULL,
  computed_at      TIMESTAMP NOT NULL,
  model_version    STRING    NOT NULL,
  n_sims           INT64     NOT NULL,
  game_id          STRING    NOT NULL,  -- nflverse game_id / ESPN event id
  week             INT64,
  game_date        DATE,                -- NFL local date; CFB US Eastern date
  home             STRING,              -- team key as in season_sim_team.team (FCS: ESPN abbr)
  away             STRING,
  home_name        STRING,
  away_name        STRING,
  neutral          BOOL,
  p_home_win       FLOAT64,             -- share of simulated seasons the home side won (rating draws included)
  margin_mean      FLOAT64,             -- simulated home margin, points
  margin_p10       FLOAT64,
  margin_p90       FLOAT64
)
PARTITION BY DATE(computed_at)
CLUSTER BY season, as_of_week, home
OPTIONS (description = "Rest-of-season Monte Carlo: one row per remaining regular-season game, from the same simulated seasons as season_sim_team. Shadow experiment.");

CREATE TABLE IF NOT EXISTS `hankstank.cfb_season.season_sim_games` (
  sport            STRING    NOT NULL,
  season           INT64     NOT NULL,
  as_of_week       INT64     NOT NULL,
  computed_at      TIMESTAMP NOT NULL,
  model_version    STRING    NOT NULL,
  n_sims           INT64     NOT NULL,
  game_id          STRING    NOT NULL,  -- nflverse game_id / ESPN event id
  week             INT64,
  game_date        DATE,                -- NFL local date; CFB US Eastern date
  home             STRING,              -- team key as in season_sim_team.team (FCS: ESPN abbr)
  away             STRING,
  home_name        STRING,
  away_name        STRING,
  neutral          BOOL,
  p_home_win       FLOAT64,             -- share of simulated seasons the home side won (rating draws included)
  margin_mean      FLOAT64,             -- simulated home margin, points
  margin_p10       FLOAT64,
  margin_p90       FLOAT64
)
PARTITION BY DATE(computed_at)
CLUSTER BY season, as_of_week, home
OPTIONS (description = "Rest-of-season Monte Carlo: one row per remaining regular-season game, from the same simulated seasons as season_sim_team. Shadow experiment.");
