-- Shadow table for the CFB margin ridge with a preseason program prior (cfb_v3_ridge_prior).
--
-- Created explicitly, LIKE the ridge's own table (game_predictions' columns, partitioning and
-- clustering plus the ridge's four outputs), for the same reason as the ridge shadow: a first
-- load from a dataframe would autodetect game_date badly. Additive and idempotent.
-- Undo: DROP TABLE.

CREATE TABLE IF NOT EXISTS `hankstank.cfb_season.game_predictions_ridge_prior_shadow`
LIKE `hankstank.cfb_season.game_predictions_ridge_shadow`;
