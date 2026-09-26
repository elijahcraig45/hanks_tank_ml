-- One-time DDL for the simulator's player stat projections (unified predictions contract v1).
-- NOT run by any pipeline: src/pa_sim/blend.py appends with CREATE_NEVER.
--
--   bq query --use_legacy_sql=false --project_id=hankstank \
--     < scripts/gcp/2026_season/create_player_sim_projections.sql
--
-- One row per (game_pk, player_id, stat, predicted_at); append-only; ~120 rows per game
-- (18 batters x PA,H,HR,TB,BB,K + 2 starters x K,BF,IP_outs,ER,H_allowed,BB_allowed).
-- R and RBI are deliberately absent: the sim tracks bases as a bitmask without runner
-- identities, so it cannot attribute who scored or drove a run in.
-- BB counts walks + HBP + catcher's interference (the sim's BB class). ER is runs charged
-- to the starter (own PAs + inherited runners scoring later); earned/unearned not separated.
-- calibrated/calibration_note come from src/pa_sim/player_calibration.json
-- (research/backtest_2026/54_player_props_calibration.py).

CREATE TABLE IF NOT EXISTS `hankstank.mlb_2026_season.player_sim_projections` (
  game_pk          INT64     NOT NULL,
  game_date        DATE      NOT NULL,
  predicted_at     TIMESTAMP NOT NULL,
  model_version    STRING    NOT NULL,
  player_id        INT64     NOT NULL,
  player_name      STRING,
  team_id          INT64,
  role             STRING    NOT NULL OPTIONS(description = "batter | starter"),
  batting_order    INT64     OPTIONS(description = "1-9 for batters, NULL for starters"),
  stat             STRING    NOT NULL OPTIONS(description = "batter: PA H HR TB BB K; starter: K BF IP_outs ER H_allowed BB_allowed"),
  mean             FLOAT64,
  sd               FLOAT64,
  p05              INT64,
  p25              INT64,
  p50              INT64,
  p75              INT64,
  p95              INT64,
  min              INT64,
  max              INT64,
  n_sims           INT64,
  p_at_least_1     FLOAT64   OPTIONS(description = "P(stat >= 1); batters only, NULL for starters"),
  calibrated       BOOL      NOT NULL OPTIONS(description = "true only if this stat passed the out-of-sample calibration gate"),
  calibration_note STRING
)
PARTITION BY game_date
CLUSTER BY game_pk, player_id
OPTIONS(description = "SHADOW. v2 PA simulator per-player stat distributions written by mode=sim_blend.");
