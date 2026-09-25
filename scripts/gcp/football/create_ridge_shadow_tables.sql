-- Shadow tables for the football margin ridge.
--
-- Created explicitly rather than on first write: the first load from a dataframe had
-- to autodetect the schema, and in the Cloud Function's pandas/pyarrow the naive
-- microsecond game_date came through as INT64, which DAY partitioning rejects. Same
-- columns, partitioning and clustering as production, plus the ridge's four outputs.

CREATE TABLE IF NOT EXISTS `hankstank.nfl_season.game_predictions_ridge_shadow`
LIKE `hankstank.nfl_season.game_predictions`;

ALTER TABLE `hankstank.nfl_season.game_predictions_ridge_shadow`
  ADD COLUMN IF NOT EXISTS predicted_home_margin FLOAT64,
  ADD COLUMN IF NOT EXISTS home_power_rating FLOAT64,
  ADD COLUMN IF NOT EXISTS away_power_rating FLOAT64,
  ADD COLUMN IF NOT EXISTS home_field_points FLOAT64;

CREATE TABLE IF NOT EXISTS `hankstank.cfb_season.game_predictions_ridge_shadow`
LIKE `hankstank.cfb_season.game_predictions`;

ALTER TABLE `hankstank.cfb_season.game_predictions_ridge_shadow`
  ADD COLUMN IF NOT EXISTS predicted_home_margin FLOAT64,
  ADD COLUMN IF NOT EXISTS home_power_rating FLOAT64,
  ADD COLUMN IF NOT EXISTS away_power_rating FLOAT64,
  ADD COLUMN IF NOT EXISTS home_field_points FLOAT64;
