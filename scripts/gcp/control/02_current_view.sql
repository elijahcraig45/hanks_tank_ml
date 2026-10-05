-- The current control state: the latest EFFECTIVE value of each field per (sport, target). '__unset__' supersedes older values and then
-- disappears, so the field falls back to its default. Kept to portable SQL (no BigQuery-only syntax) so tests can run it in DuckDB.
CREATE OR REPLACE VIEW `hankstank.control.model_control_current` AS
WITH ranked AS (
  -- Legacy events written under the old field name `role` count as `lifecycle`, so the latest of either wins.
  SELECT sport, target, CASE WHEN field = 'role' THEN 'lifecycle' ELSE field END AS field, value, event_ts,
         ROW_NUMBER() OVER (PARTITION BY sport, target, CASE WHEN field = 'role' THEN 'lifecycle' ELSE field END ORDER BY event_ts DESC, recorded_at DESC, event_id DESC) AS rn
  FROM `hankstank.control.model_control_events`
  WHERE event_ts <= CURRENT_TIMESTAMP()
), latest AS (
  SELECT * FROM ranked WHERE rn = 1 AND value != '__unset__'
)
SELECT sport, target,
  MAX(IF(field = 'site_visible',   value, NULL)) AS site_visible,
  MAX(IF(field = 'run_state',      value, NULL)) AS run_state,
  MAX(IF(field = 'lifecycle',      value, NULL)) AS lifecycle,
  -- TRANSITION ALIAS: `role` is the same value as `lifecycle`, kept only until every deployed consumer reads `lifecycle`.
  MAX(IF(field = 'lifecycle',      value, NULL)) AS role,
  MAX(IF(field = 'artifact_uri',   value, NULL)) AS artifact_uri,
  MAX(IF(field = 'artifact_sha256',value, NULL)) AS artifact_sha256,
  MAX(IF(field = 'display_label',  value, NULL)) AS display_label,
  MAX(IF(field = 'public_note',    value, NULL)) AS public_note,
  MAX(IF(field = 'sort_order',     value, NULL)) AS sort_order,
  MAX(IF(field = 'tier_high',      value, NULL)) AS tier_high,
  MAX(IF(field = 'tier_medium',    value, NULL)) AS tier_medium,
  MAX(IF(field = 'banner',         value, NULL)) AS banner,
  MAX(IF(field = 'banner_level',   value, NULL)) AS banner_level,
  -- Which power-rankings columns the site draws (sport-wide row, target '*'): ordered comma lists, or 'none'. NULL = today's behaviour.
  MAX(IF(field = 'rankings_show',  value, NULL)) AS rankings_show,
  MAX(IF(field = 'rankings_media', value, NULL)) AS rankings_media,
  MAX(event_ts) AS updated_at
FROM latest
GROUP BY sport, target;
