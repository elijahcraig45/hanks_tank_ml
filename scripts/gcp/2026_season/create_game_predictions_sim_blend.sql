-- One-time DDL for the PA simulator + team-strength blend shadow table. NOT run by any
-- pipeline: src/pa_sim/blend.py loads with CREATE_NEVER, so runs report status
-- "table_missing" (non-fatal) until a person has approved and run this.
--
--   bq query --use_legacy_sql=false --project_id=hankstank \
--     < scripts/gcp/2026_season/create_game_predictions_sim_blend.sql
--
-- Append-only: one row per game per run. Readers take the latest row with
-- predicted_at < game_time_utc; the writer skips games that have already started.

CREATE TABLE IF NOT EXISTS `hankstank.mlb_2026_season.game_predictions_sim_blend` (
  game_pk              INT64     NOT NULL,
  game_date            DATE      NOT NULL,
  game_time_utc        TIMESTAMP,
  home_team_id         INT64,
  away_team_id         INT64,
  home_team_name       STRING,
  away_team_name       STRING,
  home_starter_id      INT64,
  away_starter_id      INT64,
  home_win_probability FLOAT64   NOT NULL OPTIONS(description = "the blend: sigmoid(a + b1*logit(strength_p) + b2*logit(sim_p_raw)), frozen coefficients in src/pa_sim/blend_coefs.json"),
  away_win_probability FLOAT64,
  sim_p_raw            FLOAT64   OPTIONS(description = "v2 simulator home win share over n_episodes, uncalibrated"),
  sim_p_cal            FLOAT64   OPTIONS(description = "Platt-calibrated sim alone (not the blend)"),
  strength_p           FLOAT64   OPTIONS(description = "Elo + season pythag logistic, fit on the 3 prior seasons"),
  predicted_winner     STRING,
  confidence_tier      STRING,
  model_version        STRING    NOT NULL OPTIONS(description = "sim_blend_v2"),
  n_episodes           INT64,
  mean_home_runs       FLOAT64   OPTIONS(description = "raw sim mean; raw totals run hot, see game_props_sim.total_bias_shift"),
  mean_away_runs       FLOAT64,
  predicted_at         TIMESTAMP NOT NULL
)
PARTITION BY game_date
CLUSTER BY game_pk
OPTIONS(description = "SHADOW. v2 plate-appearance simulator stacked with team strength. Never read by the live predictions page.");
