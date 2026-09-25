-- One-time DDL for the simulator's distribution outputs (totals and starter strikeouts).
-- NOT run by any pipeline: src/pa_sim/blend.py loads with CREATE_NEVER.
--
--   bq query --use_legacy_sql=false --project_id=hankstank \
--     < scripts/gcp/2026_season/create_game_props_sim.sql
--
-- Append-only, same reader rule as the prediction tables (latest row before first pitch).
-- Batter props are NOT written by default: P(>=1 hit) is over-predicted in backtest
-- (64.5% vs 60.8% actual; the sim has no substitutions). batter_props_json is filled only
-- when a run passes experimental_props=true, and carries "calibrated": false.

CREATE TABLE IF NOT EXISTS `hankstank.mlb_2026_season.game_props_sim` (
  game_pk             INT64     NOT NULL,
  game_date           DATE      NOT NULL,
  game_time_utc       TIMESTAMP,
  home_team_name      STRING,
  away_team_name      STRING,
  model_version       STRING    NOT NULL,
  n_episodes          INT64,
  mean_home_runs      FLOAT64   OPTIONS(description = "raw sim mean"),
  mean_away_runs      FLOAT64,
  mean_total_runs     FLOAT64   OPTIONS(description = "mean of total_runs_pmf, i.e. AFTER the bias shift"),
  total_runs_pmf      ARRAY<FLOAT64> OPTIONS(description = "P(total runs = index), index 0..30 (30 = 30+), tilted so its mean is raw mean + total_bias_shift"),
  totals_calibrated   BOOL      OPTIONS(description = "true: pmf tilted by the fit-window bias"),
  total_bias_shift    FLOAT64   OPTIONS(description = "runs added to the raw mean (negative: raw sim over-predicts totals)"),
  market_total_line   FLOAT64   OPTIONS(description = "NULL: no live totals feed. The research tilts the sim shape to the market mean where a line exists"),
  p_over_market       FLOAT64   OPTIONS(description = "NULL until a market total exists"),
  home_starter_id     INT64,
  home_starter_name   STRING,
  home_starter_k_mean FLOAT64,
  home_starter_k_pmf  ARRAY<FLOAT64> OPTIONS(description = "P(home starter strikeouts = index), 0..20"),
  away_starter_id     INT64,
  away_starter_name   STRING,
  away_starter_k_mean FLOAT64,
  away_starter_k_pmf  ARRAY<FLOAT64>,
  batter_props_json   STRING    OPTIONS(description = "EXPERIMENTAL, uncalibrated, NULL by default"),
  predicted_at        TIMESTAMP NOT NULL
)
PARTITION BY game_date
CLUSTER BY game_pk
OPTIONS(description = "SHADOW. Simulator totals and starter-K distributions for the Models page Totals & props panel.");
