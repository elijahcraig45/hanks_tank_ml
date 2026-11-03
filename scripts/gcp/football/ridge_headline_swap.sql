-- The margin ridge becomes the football headline (NFL from 2026 week 9, CFB from week 11).
--
-- Run this BEFORE the pipeline release that writes the ridge into game_predictions: the
-- ridge's rows carry four columns the headline table did not have, and a load with an
-- unknown column fails. Everything here is additive and idempotent, and harmless to the
-- old code (its XGBoost rows just leave the new columns NULL).
--
-- 1. The former headline (XGBoost) gets its own table, same columns, partitioning and
--    clustering as game_predictions, seeded with every row it wrote there (2024-25
--    walk-forward backfills and the live 2026 weeks), so its record stays whole after the
--    swap. Created before the ALTER, so it keeps the XGBoost's own columns only.
-- 2. game_predictions gains the ridge's outputs. Rows written before the swap read NULL.
--
-- Undo: DROP TABLE the two *_xgb_shadow tables; the four columns can stay (all NULL on
-- XGBoost rows) or be dropped with ALTER TABLE ... DROP COLUMN.

CREATE TABLE IF NOT EXISTS `hankstank.nfl_season.game_predictions_xgb_shadow`
LIKE `hankstank.nfl_season.game_predictions`;

INSERT INTO `hankstank.nfl_season.game_predictions_xgb_shadow`
SELECT * FROM `hankstank.nfl_season.game_predictions` p
WHERE p.model_version IN ('nfl_v1_pure', 'nfl_v1_pure_epa')
  AND NOT EXISTS (SELECT 1 FROM `hankstank.nfl_season.game_predictions_xgb_shadow` s
                  WHERE s.game_id = p.game_id);

CREATE TABLE IF NOT EXISTS `hankstank.cfb_season.game_predictions_xgb_shadow`
LIKE `hankstank.cfb_season.game_predictions`;

INSERT INTO `hankstank.cfb_season.game_predictions_xgb_shadow`
SELECT * FROM `hankstank.cfb_season.game_predictions` p
WHERE p.model_version = 'cfb_v1'
  AND NOT EXISTS (SELECT 1 FROM `hankstank.cfb_season.game_predictions_xgb_shadow` s
                  WHERE s.game_id = p.game_id);

ALTER TABLE `hankstank.nfl_season.game_predictions`
  ADD COLUMN IF NOT EXISTS predicted_home_margin FLOAT64,
  ADD COLUMN IF NOT EXISTS home_power_rating FLOAT64,
  ADD COLUMN IF NOT EXISTS away_power_rating FLOAT64,
  ADD COLUMN IF NOT EXISTS home_field_points FLOAT64;

ALTER TABLE `hankstank.cfb_season.game_predictions`
  ADD COLUMN IF NOT EXISTS predicted_home_margin FLOAT64,
  ADD COLUMN IF NOT EXISTS home_power_rating FLOAT64,
  ADD COLUMN IF NOT EXISTS away_power_rating FLOAT64,
  ADD COLUMN IF NOT EXISTS home_field_points FLOAT64;
