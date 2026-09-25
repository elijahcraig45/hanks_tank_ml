-- One-time DDL for the 3-feature logistic shadow table. NOT run by any pipeline:
-- src/logit3_shadow.py loads with CREATE_NEVER, so runs report status "table_missing"
-- (non-fatal) until a person has approved and run this.
--
--   bq query --use_legacy_sql=false --project_id=hankstank \
--     < scripts/gcp/2026_season/create_game_predictions_logit3.sql
--
-- Append-only: one row per game per run. Readers take the latest row with
-- predicted_at < game_time_utc; the writer skips games that have already started.

CREATE TABLE IF NOT EXISTS `hankstank.mlb_2026_season.game_predictions_logit3` (
  game_pk                   INT64     NOT NULL,
  game_date                 DATE      NOT NULL,
  home_team_id              INT64,
  away_team_id              INT64,
  home_team_name            STRING,
  away_team_name            STRING,
  home_win_probability      FLOAT64   NOT NULL OPTIONS(description = "P(home win), 3-feature L1 logistic"),
  away_win_probability      FLOAT64,
  predicted_winner          STRING,
  confidence_tier           STRING    OPTIONS(description = "high >= 0.64, medium >= 0.57, else low (same absolute cut-offs as game_predictions)"),
  model_version             STRING    NOT NULL OPTIONS(description = "logit3_l1_v1"),
  n_train                   INT64     OPTIONS(description = "2026 Final regular-season games the model was refit on, all before game_date"),
  coef_json                 STRING    OPTIONS(description = "features, intercept, standardised coefficients, scaler mu/sd, C, penalty"),
  elo_differential          FLOAT64,
  pythag_differential       FLOAT64,
  sp_quality_composite_diff FLOAT64,
  features_computed_at      TIMESTAMP OPTIONS(description = "computed_at of the game_v10_features row used; always < game_time_utc"),
  game_time_utc             TIMESTAMP,
  predicted_at              TIMESTAMP NOT NULL
)
PARTITION BY game_date
CLUSTER BY game_pk
OPTIONS(description = "SHADOW. 3-feature L1 logistic (elo, pythag, SP quality diffs), refit in-season each run. Never read by the live predictions page.");
