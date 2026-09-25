-- One-time DDL for the ESPN FPI pregame snapshot tables. NOT run by any pipeline:
-- src/rankings/fpi_games.py loads with CREATE_NEVER, so snapshots fail (non-fatally)
-- until a person has approved and run this.
--
--   bq query --use_legacy_sql=false --project_id=hankstank \
--     < scripts/gcp/football/create_fpi_game_predictions.sql
--
-- Append-only: one row per game per snapshot. Readers take the latest row whose
-- predicted_at is before kickoff. Rows with source = 'backfill_after_kickoff' were
-- fetched after the game and are excluded from any pregame scoreboard by that rule.
--
-- The first four value columns (home_win_probability, predicted_home_margin,
-- predicted_at, game_id) are the contract every model table on the comparison page
-- shares, so the backend reads them all with one query shape.

CREATE TABLE IF NOT EXISTS `hankstank.nfl_season.fpi_game_predictions` (
  game_id               STRING    NOT NULL,  -- nflverse game_id, e.g. 2026_03_KC_MIA
  season                INT64     NOT NULL,
  week                  INT64     NOT NULL,
  division              STRING,              -- NULL for the NFL
  home_team             STRING,
  away_team             STRING,
  kickoff               TIMESTAMP,
  home_win_probability  FLOAT64   NOT NULL,  -- gameProjection, renormalised over no-tie
  predicted_home_margin FLOAT64,             -- teamPredPtDiff, + = home favoured
  home_game_projection  FLOAT64,             -- raw ESPN value, 0-100
  away_game_projection  FLOAT64,
  matchup_quality       FLOAT64,
  espn_event_id         STRING,
  espn_home_team_id     STRING,
  espn_last_modified    TIMESTAMP,
  predicted_at          TIMESTAMP NOT NULL,  -- when WE captured it
  source                STRING    NOT NULL,  -- pregame_snapshot | backfill_after_kickoff
  model_version         STRING
)
PARTITION BY DATE(predicted_at)
CLUSTER BY season, week;

CREATE TABLE IF NOT EXISTS `hankstank.cfb_season.fpi_game_predictions` (
  game_id               STRING    NOT NULL,  -- ESPN event id (same key as game_predictions)
  season                INT64     NOT NULL,
  week                  INT64     NOT NULL,
  division              STRING,              -- fbs | fcs, from the ESPN scoreboard group
  home_team             STRING,
  away_team             STRING,
  kickoff               TIMESTAMP,
  home_win_probability  FLOAT64   NOT NULL,
  predicted_home_margin FLOAT64,
  home_game_projection  FLOAT64,
  away_game_projection  FLOAT64,
  matchup_quality       FLOAT64,
  espn_event_id         STRING,
  espn_home_team_id     STRING,
  espn_last_modified    TIMESTAMP,
  predicted_at          TIMESTAMP NOT NULL,
  source                STRING    NOT NULL,
  model_version         STRING
)
PARTITION BY DATE(predicted_at)
CLUSTER BY season, week;
