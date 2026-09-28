-- One-time DDL for the rest-of-season Monte Carlo (EXPERIMENT, shadow). NOT run by any
-- pipeline: src/season_sim/store.py loads with CREATE_NEVER and the table's own schema,
-- so the season_sim mode fails (with an error in its response) until this has been run.
--
--   CLOUDSDK_CORE_ACCOUNT=elijahcraig45@gmail.com bq query --use_legacy_sql=false \
--     --project_id=hankstank < scripts/gcp/football/create_season_sim_tables.sql
--
-- Keyed by (season, as_of_week, computed_at). The writer DELETEs exactly one
-- (season, as_of_week) slice and appends, so a rerun replaces only itself; history of
-- earlier weeks is kept (that is what a later calibration check will score).
-- Readers (the backend's /api/season-sim) take the latest computed_at per slice.
-- Written with load jobs, never streaming inserts, so the DELETE never meets a
-- streaming buffer.

CREATE TABLE IF NOT EXISTS `hankstank.nfl_season.season_sim_team` (
  sport            STRING    NOT NULL,  -- 'nfl' / 'cfb'
  season           INT64     NOT NULL,
  as_of_week       INT64     NOT NULL,  -- results through this week are fixed
  computed_at      TIMESTAMP NOT NULL,
  model_version    STRING    NOT NULL,  -- season_sim_v1
  n_sims           INT64     NOT NULL,
  team             STRING    NOT NULL,  -- NFL code / ESPN abbreviation
  team_name        STRING,
  conference       STRING,
  division         STRING,              -- NFL division; CFB Sun Belt East/West, else NULL
  wins             INT64,               -- current record
  losses           INT64,
  ties             INT64,
  conf_wins        INT64,
  conf_losses      INT64,
  rating           FLOAT64,             -- margin-ridge rating, points vs average
  rating_sd        FLOAT64,             -- posterior SD of that rating
  power_rank       INT64,
  remaining_games  INT64,
  remaining_sos    FLOAT64,             -- mean rating of remaining opponents
  mean_wins        FLOAT64,             -- final regular-season record
  mean_losses      FLOAT64,
  wins_p10         FLOAT64,
  wins_p50         FLOAT64,
  wins_p90         FLOAT64,
  wins_dist        STRING,              -- JSON array: P(final wins == k), k = 0..games
  p_division       FLOAT64,             -- NFL only
  p_conf_game      FLOAT64,             -- CFB title game / NFL conference championship
  p_conf_title     FLOAT64,             -- CFB conference title / NFL AFC-NFC title
  p_playoffs       FLOAT64,             -- NFL 14-team playoff / CFP 12-team field
  p_bye            FLOAT64,
  p_seed           STRING,              -- JSON array: P(seed == k), k = 1..7 or 1..12
  p_quarters       FLOAT64,
  p_semis          FLOAT64,
  p_final          FLOAT64,
  p_champion       FLOAT64,
  exp_final_rank   FLOAT64,             -- NFL rating rank / CFB committee-proxy rank
  rank_p10         FLOAT64,
  rank_p90         FLOAT64
)
PARTITION BY DATE(computed_at)
CLUSTER BY season, as_of_week, team
OPTIONS (description = "Rest-of-season Monte Carlo, one row per team per run. Shadow experiment; see hanks_tank_ml docs/MODEL_CARDS.md.");

CREATE TABLE IF NOT EXISTS `hankstank.nfl_season.season_sim_bracket` (
  sport            STRING    NOT NULL,
  season           INT64     NOT NULL,
  as_of_week       INT64     NOT NULL,
  computed_at      TIMESTAMP NOT NULL,
  model_version    STRING    NOT NULL,
  n_sims           INT64     NOT NULL,
  bracket          STRING    NOT NULL,  -- AFC / NFC / NFL (Super Bowl) / CFP
  round            STRING    NOT NULL,  -- seed, wild_card, divisional, ... / first_round ...
  round_order      INT64     NOT NULL,  -- 0 = seed
  slot             INT64     NOT NULL,  -- seed number, or game number within the round
  slot_label       STRING,
  team             STRING    NOT NULL,
  team_name        STRING,
  p_slot           FLOAT64,             -- P(team holds this seed / plays this game)
  p_win            FLOAT64,             -- P(team plays AND wins it)
  is_modal         BOOL,                -- in the most-likely bracket at this slot
  modal_opponent   STRING
)
PARTITION BY DATE(computed_at)
CLUSTER BY season, as_of_week, bracket
OPTIONS (description = "Rest-of-season Monte Carlo: per-slot bracket probabilities and the most-likely bracket. Shadow experiment.");

CREATE TABLE IF NOT EXISTS `hankstank.cfb_season.season_sim_team` (
  sport            STRING    NOT NULL,  -- 'nfl' / 'cfb'
  season           INT64     NOT NULL,
  as_of_week       INT64     NOT NULL,  -- results through this week are fixed
  computed_at      TIMESTAMP NOT NULL,
  model_version    STRING    NOT NULL,  -- season_sim_v1
  n_sims           INT64     NOT NULL,
  team             STRING    NOT NULL,  -- NFL code / ESPN abbreviation
  team_name        STRING,
  conference       STRING,
  division         STRING,              -- NFL division; CFB Sun Belt East/West, else NULL
  wins             INT64,               -- current record
  losses           INT64,
  ties             INT64,
  conf_wins        INT64,
  conf_losses      INT64,
  rating           FLOAT64,             -- margin-ridge rating, points vs average
  rating_sd        FLOAT64,             -- posterior SD of that rating
  power_rank       INT64,
  remaining_games  INT64,
  remaining_sos    FLOAT64,             -- mean rating of remaining opponents
  mean_wins        FLOAT64,             -- final regular-season record
  mean_losses      FLOAT64,
  wins_p10         FLOAT64,
  wins_p50         FLOAT64,
  wins_p90         FLOAT64,
  wins_dist        STRING,              -- JSON array: P(final wins == k), k = 0..games
  p_division       FLOAT64,             -- NFL only
  p_conf_game      FLOAT64,             -- CFB title game / NFL conference championship
  p_conf_title     FLOAT64,             -- CFB conference title / NFL AFC-NFC title
  p_playoffs       FLOAT64,             -- NFL 14-team playoff / CFP 12-team field
  p_bye            FLOAT64,
  p_seed           STRING,              -- JSON array: P(seed == k), k = 1..7 or 1..12
  p_quarters       FLOAT64,
  p_semis          FLOAT64,
  p_final          FLOAT64,
  p_champion       FLOAT64,
  exp_final_rank   FLOAT64,             -- NFL rating rank / CFB committee-proxy rank
  rank_p10         FLOAT64,
  rank_p90         FLOAT64
)
PARTITION BY DATE(computed_at)
CLUSTER BY season, as_of_week, team
OPTIONS (description = "Rest-of-season Monte Carlo, one row per team per run. Shadow experiment; see hanks_tank_ml docs/MODEL_CARDS.md.");

CREATE TABLE IF NOT EXISTS `hankstank.cfb_season.season_sim_bracket` (
  sport            STRING    NOT NULL,
  season           INT64     NOT NULL,
  as_of_week       INT64     NOT NULL,
  computed_at      TIMESTAMP NOT NULL,
  model_version    STRING    NOT NULL,
  n_sims           INT64     NOT NULL,
  bracket          STRING    NOT NULL,  -- AFC / NFC / NFL (Super Bowl) / CFP
  round            STRING    NOT NULL,  -- seed, wild_card, divisional, ... / first_round ...
  round_order      INT64     NOT NULL,  -- 0 = seed
  slot             INT64     NOT NULL,  -- seed number, or game number within the round
  slot_label       STRING,
  team             STRING    NOT NULL,
  team_name        STRING,
  p_slot           FLOAT64,             -- P(team holds this seed / plays this game)
  p_win            FLOAT64,             -- P(team plays AND wins it)
  is_modal         BOOL,                -- in the most-likely bracket at this slot
  modal_opponent   STRING
)
PARTITION BY DATE(computed_at)
CLUSTER BY season, as_of_week, bracket
OPTIONS (description = "Rest-of-season Monte Carlo: per-slot bracket probabilities and the most-likely bracket. Shadow experiment.");
