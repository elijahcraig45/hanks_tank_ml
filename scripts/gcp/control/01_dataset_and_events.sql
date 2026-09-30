-- Model control plane: an APPEND-ONLY event log. Nothing is ever updated or deleted, so every change is auditable and any state can be
-- restored by appending the previous value. A row takes effect when event_ts <= now, so a future-dated row is a scheduled change.
CREATE SCHEMA IF NOT EXISTS `hankstank.control` OPTIONS (location = 'US');

CREATE TABLE IF NOT EXISTS `hankstank.control.model_control_events` (
  event_id    STRING    NOT NULL,   -- uuid; the final tie-breaker
  event_ts    TIMESTAMP NOT NULL,   -- when the change takes effect (may be in the future)
  recorded_at TIMESTAMP NOT NULL,   -- when it was written
  sport       STRING    NOT NULL,   -- mlb | nfl | cfb | *
  target      STRING    NOT NULL,   -- a model key, or '*' for sport-wide settings (banner)
  field       STRING    NOT NULL,   -- site_visible | run_state | role | artifact_uri | artifact_sha256 | display_label | public_note | sort_order | tier_high | tier_medium | banner | banner_level
  value       STRING,               -- '__unset__' removes the field (returns to the default)
  actor       STRING,               -- who or what: a person, `remedy:<rule>`, ...
  note        STRING
)
CLUSTER BY sport, target;
