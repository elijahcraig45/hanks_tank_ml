-- One-off cleanup of duplicate prediction rows written before the 2026-09-28 fixes.
-- REVIEWED BY HAND, RUN BY HAND. Nothing in the pipeline runs this file.
--
--   CLOUDSDK_CORE_ACCOUNT=elijahcraig45@gmail.com bq query --project_id=hankstank \
--     --use_legacy_sql=false < scripts/gcp/2026_season/cleanup_duplicate_predictions_2026_09_28.sql
--
-- Cause: from 2026-09-11 both App Engine cron and the mlb-2026-pregame-schedule job
-- called /api/lineup/schedule-today, so every Cloud Task was enqueued twice. The sim_blend
-- twins ran one after the other on its single instance (identical rows 1-6 min apart,
-- same seed); the V10 twins ran concurrently, both DELETEd, then both INSERTed (identical
-- rows < 1 s apart). Fixed at the source (named tasks) and at the writers (content-keyed
-- load jobs); see docs/SHADOW_MODELS.md "Operations".
--
-- Rule: a row (for player_sim_projections, a whole snapshot) is a duplicate when an
-- EARLIER row for the same game_pk and model_version has identical content in every
-- column except predicted_at. The earliest copy is kept. For game_predictions the copy
-- must also be within 120 s of the first: identical V10 rows hours apart are separate
-- checkpoints, and they are left alone. Readers take the latest pregame row, so no scored
-- probability changes; only COUNT/AVG over raw rows stop double counting.
--
-- Counted with the SELECTs below on 2026-09-28 (read-only):
--   game_predictions_sim_blend    31 rows ->   20   (11 deleted, 11 games)
--   game_props_sim                31 rows ->   20   (11 deleted, 11 games)
--   game_sim_distributions        30 rows ->   20   (10 deleted, 10 games)
--   player_sim_projections      3600 rows -> 2400   (1200 deleted = 10 snapshots x 120)
--   game_predictions (2026)     3746 rows -> 3528   (218 deleted, 170 games; 133 since 09-11)
-- Game 823650 keeps two sim snapshots: they differ (a lineup changed), so neither is a copy.
-- Each (game_pk, model_version, predicted_at) key identifies exactly one row (one snapshot
-- for players), checked 2026-09-28, so a DELETE by key cannot hit a kept row; the ASSERTs
-- re-check that and the expected counts before anything is deleted.
--
-- Every row involved was written by a load job more than a day ago, so none is in the
-- streaming buffer; the 3-hour guard keeps any in-flight run out regardless.

DECLARE guard TIMESTAMP DEFAULT TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 3 HOUR);

-- ------------------------------------------------------------------ duplicate keys
CREATE TEMP TABLE dup_sim_blend AS
WITH s AS (
  SELECT game_pk, model_version, predicted_at,
         TO_JSON_STRING((SELECT AS STRUCT t.* EXCEPT(predicted_at))) AS fp
  FROM `hankstank.mlb_2026_season.game_predictions_sim_blend` t WHERE predicted_at < guard)
SELECT game_pk, model_version, predicted_at FROM s
QUALIFY ROW_NUMBER() OVER (PARTITION BY game_pk, model_version, fp ORDER BY predicted_at) > 1;

CREATE TEMP TABLE dup_props AS
WITH s AS (
  SELECT game_pk, model_version, predicted_at,
         TO_JSON_STRING((SELECT AS STRUCT t.* EXCEPT(predicted_at))) AS fp
  FROM `hankstank.mlb_2026_season.game_props_sim` t WHERE predicted_at < guard)
SELECT game_pk, model_version, predicted_at FROM s
QUALIFY ROW_NUMBER() OVER (PARTITION BY game_pk, model_version, fp ORDER BY predicted_at) > 1;

CREATE TEMP TABLE dup_dist AS
WITH s AS (
  SELECT game_pk, model_version, predicted_at,
         TO_JSON_STRING((SELECT AS STRUCT t.* EXCEPT(predicted_at))) AS fp
  FROM `hankstank.mlb_2026_season.game_sim_distributions` t WHERE predicted_at < guard)
SELECT game_pk, model_version, predicted_at FROM s
QUALIFY ROW_NUMBER() OVER (PARTITION BY game_pk, model_version, fp ORDER BY predicted_at) > 1;

CREATE TEMP TABLE dup_players AS
WITH snap AS (
  SELECT game_pk, model_version, predicted_at,
         FARM_FINGERPRINT(STRING_AGG(TO_JSON_STRING((SELECT AS STRUCT t.* EXCEPT(predicted_at))), '|'
                                     ORDER BY player_id, stat)) AS fp
  FROM `hankstank.mlb_2026_season.player_sim_projections` t WHERE predicted_at < guard
  GROUP BY 1, 2, 3)
SELECT game_pk, model_version, predicted_at FROM snap
QUALIFY ROW_NUMBER() OVER (PARTITION BY game_pk, model_version, fp ORDER BY predicted_at) > 1;

CREATE TEMP TABLE dup_v10 AS
WITH s AS (
  SELECT game_pk, model_version, predicted_at,
         TO_JSON_STRING((SELECT AS STRUCT t.* EXCEPT(predicted_at))) AS fp
  FROM `hankstank.mlb_2026_season.game_predictions` t
  WHERE game_date >= '2026-01-01' AND predicted_at < guard),
d AS (
  SELECT *, FIRST_VALUE(predicted_at) OVER w AS first_at, ROW_NUMBER() OVER w AS k
  FROM s WINDOW w AS (PARTITION BY game_pk, model_version, fp ORDER BY predicted_at))
SELECT game_pk, model_version, predicted_at FROM d
WHERE k > 1 AND TIMESTAMP_DIFF(predicted_at, first_at, SECOND) <= 120;

-- ------------------------------------------------------------------ preview (read-only)
SELECT 'game_predictions_sim_blend' AS tbl, COUNT(*) AS rows_to_delete, COUNT(DISTINCT game_pk) AS games FROM dup_sim_blend
UNION ALL SELECT 'game_props_sim', COUNT(*), COUNT(DISTINCT game_pk) FROM dup_props
UNION ALL SELECT 'game_sim_distributions', COUNT(*), COUNT(DISTINCT game_pk) FROM dup_dist
UNION ALL SELECT 'player_sim_projections (snapshots)', COUNT(*), COUNT(DISTINCT game_pk) FROM dup_players
UNION ALL SELECT 'game_predictions', COUNT(*), COUNT(DISTINCT game_pk) FROM dup_v10;

-- Stop if the table moved since the counts above were taken (re-review, then edit these).
ASSERT (SELECT COUNT(*) FROM dup_sim_blend) = 11 AS 'sim_blend duplicate count changed: re-review';
ASSERT (SELECT COUNT(*) FROM dup_props) = 11 AS 'game_props_sim duplicate count changed: re-review';
ASSERT (SELECT COUNT(*) FROM dup_dist) = 10 AS 'game_sim_distributions duplicate count changed: re-review';
ASSERT (SELECT COUNT(*) FROM dup_players) = 10 AS 'player snapshot duplicate count changed: re-review';
ASSERT (SELECT COUNT(*) FROM dup_v10) = 218 AS 'game_predictions duplicate count changed: re-review';
-- A key must identify one row, or deleting it would also remove the kept copy.
ASSERT NOT EXISTS (
  SELECT 1 FROM `hankstank.mlb_2026_season.game_predictions` t JOIN dup_v10 USING (game_pk, model_version, predicted_at)
  GROUP BY game_pk, model_version, predicted_at HAVING COUNT(*) > 1) AS 'game_predictions key not unique';
ASSERT NOT EXISTS (
  SELECT 1 FROM `hankstank.mlb_2026_season.game_predictions_sim_blend` t JOIN dup_sim_blend USING (game_pk, model_version, predicted_at)
  GROUP BY game_pk, model_version, predicted_at HAVING COUNT(*) > 1) AS 'sim_blend key not unique';

-- ------------------------------------------------------------------ delete
BEGIN TRANSACTION;

DELETE FROM `hankstank.mlb_2026_season.game_predictions_sim_blend`
WHERE STRUCT(game_pk, model_version, predicted_at) IN (SELECT AS STRUCT game_pk, model_version, predicted_at FROM dup_sim_blend);

DELETE FROM `hankstank.mlb_2026_season.game_props_sim`
WHERE STRUCT(game_pk, model_version, predicted_at) IN (SELECT AS STRUCT game_pk, model_version, predicted_at FROM dup_props);

DELETE FROM `hankstank.mlb_2026_season.game_sim_distributions`
WHERE STRUCT(game_pk, model_version, predicted_at) IN (SELECT AS STRUCT game_pk, model_version, predicted_at FROM dup_dist);

DELETE FROM `hankstank.mlb_2026_season.player_sim_projections`
WHERE STRUCT(game_pk, model_version, predicted_at) IN (SELECT AS STRUCT game_pk, model_version, predicted_at FROM dup_players);

DELETE FROM `hankstank.mlb_2026_season.game_predictions`
WHERE game_date >= '2026-01-01'
  AND STRUCT(game_pk, model_version, predicted_at) IN (SELECT AS STRUCT game_pk, model_version, predicted_at FROM dup_v10);

COMMIT TRANSACTION;

-- ------------------------------------------------------------------ verify (read-only)
SELECT 'game_predictions_sim_blend' AS tbl, COUNT(*) AS rows_now FROM `hankstank.mlb_2026_season.game_predictions_sim_blend`
UNION ALL SELECT 'game_props_sim', COUNT(*) FROM `hankstank.mlb_2026_season.game_props_sim`
UNION ALL SELECT 'game_sim_distributions', COUNT(*) FROM `hankstank.mlb_2026_season.game_sim_distributions`
UNION ALL SELECT 'player_sim_projections', COUNT(*) FROM `hankstank.mlb_2026_season.player_sim_projections`
UNION ALL SELECT 'game_predictions (2026)', COUNT(*) FROM `hankstank.mlb_2026_season.game_predictions` WHERE game_date >= '2026-01-01';
-- expected: 20, 20, 20, 2400, 3528 (plus anything written after 2026-09-28)
