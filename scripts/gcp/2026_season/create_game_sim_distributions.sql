-- One-time DDL for the simulator's per-game run distributions (unified predictions contract v1).
-- NOT run by any pipeline: src/pa_sim/blend.py appends with CREATE_NEVER.
--
--   bq query --use_legacy_sql=false --project_id=hankstank \
--     < scripts/gcp/2026_season/create_game_sim_distributions.sql
--
-- One row per (game_pk, model_version, predicted_at); append-only. Readers take the latest
-- row per game and model with predicted_at < first pitch (pregame), else the latest row
-- flagged pregame=false. Percentiles of counts are integers (inverted-CDF).
-- totals_calibrated = true: every field below is computed with each episode reweighted by
-- exp(theta * total) so the weighted mean total equals the raw sim mean + total_bias_shift
-- (raw totals run hot, +0.2..+0.8 runs/game in 2022-26; blend_coefs.json totals_bias_runs).

CREATE TABLE IF NOT EXISTS `hankstank.mlb_2026_season.game_sim_distributions` (
  game_pk           INT64     NOT NULL,
  game_date         DATE      NOT NULL,
  game_time_utc     TIMESTAMP OPTIONS(description = "scheduled first pitch, for the pregame test"),
  predicted_at      TIMESTAMP NOT NULL,
  model_version     STRING    NOT NULL,
  n_sims            INT64     OPTIONS(description = "simulated games (the Dist n)"),
  home_runs_mean  FLOAT64 OPTIONS(description = "home runs: mean over simulated games"),
  home_runs_sd    FLOAT64,
  home_runs_p05   INT64,
  home_runs_p25   INT64,
  home_runs_p50   INT64,
  home_runs_p75   INT64,
  home_runs_p95   INT64,
  home_runs_min   INT64,
  home_runs_max   INT64,
  away_runs_mean  FLOAT64 OPTIONS(description = "away runs: mean over simulated games"),
  away_runs_sd    FLOAT64,
  away_runs_p05   INT64,
  away_runs_p25   INT64,
  away_runs_p50   INT64,
  away_runs_p75   INT64,
  away_runs_p95   INT64,
  away_runs_min   INT64,
  away_runs_max   INT64,
  total_mean  FLOAT64 OPTIONS(description = "total runs: mean over simulated games"),
  total_sd    FLOAT64,
  total_p05   INT64,
  total_p25   INT64,
  total_p50   INT64,
  total_p75   INT64,
  total_p95   INT64,
  total_min   INT64,
  total_max   INT64,
  margin_mean  FLOAT64 OPTIONS(description = "home minus away runs: mean over simulated games"),
  margin_sd    FLOAT64,
  margin_p05   INT64,
  margin_p25   INT64,
  margin_p50   INT64,
  margin_p75   INT64,
  margin_p95   INT64,
  margin_min   INT64,
  margin_max   INT64,
  p_home_win        FLOAT64   OPTIONS(description = "share of (tilt-weighted) simulated games the home team wins; the featured win prob is game_predictions_sim_blend.home_win_probability"),
  p_extra_innings   FLOAT64,
  p_home_cover_rl   FLOAT64   OPTIONS(description = "P(home wins by 2+), i.e. covers -1.5"),
  p_over_by_line    STRING    OPTIONS(description = "JSON {\"6.5\": p, ..., \"11.5\": p}: P(total > line)"),
  totals_calibrated BOOL      OPTIONS(description = "true: the totals correction (tilt) is applied to every field"),
  total_bias_shift  FLOAT64   OPTIONS(description = "runs added to the raw sim mean total (negative)")
)
PARTITION BY game_date
CLUSTER BY game_pk
OPTIONS(description = "SHADOW. v2 PA simulator game distributions (runs, total, margin) written by mode=sim_blend.");
