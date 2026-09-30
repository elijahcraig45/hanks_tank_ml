# Model control plane: the contract

Goal: change what Hank's Tank SHOWS and RUNS without editing code or redeploying, and be able to see and heal model health.
The lab (`mllab`, private, tailnet-only) is the ONLY writer. The site backend, the frontend and the ML pipeline only READ.
Base of all work: branch `model-control` cut from `deploy/2026-09-28b` (the live line) in each repo. Nothing here is deployed until approved.

## Non-negotiables (every consumer)
1. **Fail open.** If the control data is missing, empty, slow or unreadable, behave EXACTLY as before this feature existed. Log one WARNING per
   read failure (not per request). A control outage must never take the site or the pipeline down.
2. **Read-only.** Consumers never write control data. There are NO admin/write endpoints on the public backend.
3. **Plain text only.** `display_label`, `public_note`, `banner` are inert text. Never render as HTML (React text nodes only).
4. **Bounded cost.** Cache the control read (backend: 30 s per instance; ML: read once per invocation). One small query, never per request/row.
5. **A pause returns success.** ML pauses return HTTP 200 with status `paused` and write nothing, or Cloud Tasks will retry forever.

## Data (BigQuery, project `hankstank`, dataset `control`, location US)
DDL lives in `mllab/sql/control/` and is copied to `hanks_tank_ml/scripts/gcp/control/`.

`control.model_control_events` (append-only): `event_id, event_ts, recorded_at, sport, target, field, value, actor, note`.
`control.model_control_current` (view): one row per `(sport, target)` with the latest EFFECTIVE (`event_ts <= now`) value of each field:
`sport, target, site_visible, run_state, lifecycle, role (transition alias of lifecycle), artifact_uri, artifact_sha256, display_label, public_note, sort_order, tier_high, tier_medium, banner, banner_level, updated_at`.
All value columns are STRING (NULL = not set = default). A row with `target = '*'` carries sport-wide fields (`banner`, `banner_level`).

Consumers read ONLY the view:
`SELECT * FROM \`hankstank.control.model_control_current\` WHERE sport = @sport OR sport = '*'` (dataset name from env `CONTROL_DATASET`, default `control`).

## Targets = the registry model KEYS (not `model_version` strings)
- MLB (`backend/src/config/models.config.ts` MLB_MODELS): `v10` (production), `logit3`, `sim_blend`, `elo` (derived), `market` (derived).
  ML-only shadow with no site card: `pa_sim`.
- NFL and CFB (FOOTBALL_MODELS): `xgb` (production), `ridge`, `fpi`, `drive_sim`, `market` (derived).
- `elo` and `market` are derived from stored columns, so they are not runnable: only visibility, label, note and order apply to them.
- Production keys (`v10` for MLB, `xgb` for football) are the SPINE of the slate/compare pages (their table is the games list).

## Fields
| field | values | default | who acts |
|---|---|---|---|
| `site_visible` | `true`/`false` | `true` | backend: hide the model everywhere on the site |
| `run_state` | `active`/`paused` | `active` | ML: stop producing this model (200, no writes) |
| `lifecycle` | `live`/`shadow`/`archived` | (unset) | ML: `live` + artifact = pinned production artifact; others informational. Renamed from `role`; the view keeps a `role` alias during the transition |
| `artifact_uri` | `gs://bucket/path` | unset | ML: pinned artifact location (production key only) |
| `artifact_sha256` | 64 hex | unset | ML: REQUIRED with a pin; verified before use |
| `display_label` | plain text <= 60 | registry label | backend: overrides the label |
| `public_note` | plain text <= 300 | registry note | backend: overrides the note |
| `sort_order` | int -1000..1000 | registry order | backend: lower first; unset keeps registry order |
| `tier_high`, `tier_medium` | 0.5 < x < 1 | code constants | ML: confidence tier cut-offs for the production key |
| `banner` (target `*`) | plain text <= 240 | none | backend `/api/site-status` -> frontend banner |
| `banner_level` (target `*`) | `info`/`warn`/`error` | `info` | frontend styling |

## Backend behaviour (`hanks_tank_backend`)
- `services/model-control.service.ts`: `getControl(sport)` -> `{available: boolean, version: string, models: Record<key, {visible, paused, role, label?, note?, sortOrder?}>, banner?: {text, level}}`.
  30 s cache per instance; on error return `{available:false, version:'none', models:{}}` (fail open).
  `version` = short stable hash of the state, used in cache keys so a change invalidates cached responses on every instance.
- Overlay of the model registry (`modelsForSport`): hidden keys are REMOVED from `models[]`, `predictions{}` per game, the scoreboard and
  `featured_default`; label/note/order overrides applied. Applies to `/api/models/:sport/compare`, the unified slate builders, and the legacy
  `/api/football/:sport/models/compare`. If the production key is hidden the spine (games list) is still served, with the production model
  unavailable and no predictions for it; legacy prediction endpoints (`/api/predictions`, `/api/football/:sport/predictions`) then return the
  games with prediction fields null and `hidden: true`.
- Caching: the control `version` goes into the cache key (prefix) of every route above. Browser/edge `Cache-Control: max-age` is capped at 60 s for
  these routes so a hide is visible within about a minute, not 15.
- `GET /api/site-status` -> `{success:true, data:{generated_at, control_available, sports:{mlb:{banner:{text,level}|null}, nfl:{...}, cfb:{...}}}}`, `Cache-Control: public, max-age=30`.
- No other new public surface.

## Frontend behaviour (`hanks_tank`)
- Site-wide banner from `/api/site-status` in `AppShell` (between navbar and main), shown for the current sport (routes) or when any sport has one on
  non-sport pages; `warn`/`error`/`info` styles; plain text; dismissible per session (sessionStorage) only for `info`.
- `ModelsPage` renders from the API `models[]` (already ordered/filtered by the backend). `MODEL_ORDER`/`MODEL_CARDS` become ENRICHMENT, not the source of
  truth: a model in the API with no card gets a generic card built from API `label` + `note`; a model with a card but absent from the API is not shown.
- `PredictionsPage`: unknown `model_version` strings must not print a wrong description.
- Everything tolerant of the API not having the new fields (old backend).

## ML behaviour (`hanks_tank_ml`)
- New `src/model_control.py`: `get_state(sport, client=None)` reads the view once (fail-open, one warning), returns an object with
  `is_paused(key)`, `pin(key)` -> `(uri, sha256)|None`, `tiers(key)` -> `(high, medium)|None`. Injectable BigQuery client for tests.
- Pause: `_shadow()` (logit3, sim_blend) and `_run_pa_sim`, the production `_run_daily_prediction`/`predict_today`, and the NFL/CFB predict modes and
  their shadow wrappers return `{"status":"paused"}` with HTTP 200 and write nothing when their key is paused. One structured log line per pause.
- Pin (MLB production key `v10`): if `role=live` with `artifact_uri` + `artifact_sha256`, `load_model` loads that artifact FIRST, verifies the sha256 of
  the bytes BEFORE unpickling, and raises on mismatch (never fall back silently after a hash mismatch). If the object is missing/unreadable, log ERROR and fall
  back to the normal chain. `model_version` written to rows comes from the artifact payload's `version`.
- Artifacts self-describe: payload key `feature_set` (`"v10"`, `"v8"`) selects the feature path instead of string-matching the label. Keep the old
  label/name detection as the fallback for artifacts without it.
- `model_sha256` STRING column on `game_predictions` (nullable): written only if the table schema already has it (`ALTER TABLE ... ADD COLUMN` is a
  separate rollout step), so deploy order cannot break writes.
- Tiers: `tier_high`/`tier_medium` for the production key override the constants used in `predict_game` (MLB) and the football `confidence_tier`
  helpers; invalid or absent -> constants.
- DDL: `scripts/gcp/control/*.sql` + `setup_control.sh` (idempotent, `--dry-run`), never run by a pipeline.

## Lab behaviour (`mllab`, private)
- `mllab control ...`: the only writer (uses the caller's own gcloud identity via the `bq` CLI: no service-account grants needed).
- `mllab health`: builds the private dashboard and `health.json`; `mllab remedy`: self-healing engine (dry-run by default, per-rule opt-in).
- Production key rules: hiding a production key requires an active banner in the same change; pin requires the sha256; live-affecting changes need the
  typed confirmation phrase; the last visible model cannot be hidden by accident.

## Health dashboard and self-healing (lab only)
- `mllab health [--sports mlb,nfl,cfb] [--build] [--remedy] [--json]` collects facts (BigQuery prediction tables, the live site registry, control state,
  Cloud Scheduler, lab disk/services), evaluates the pure rules in `health.py`, and with `--build` writes `<root>/health/{index.html,health.json,history.jsonl}`.
  The page is static, escapes all text and is served only behind the lab's auth at `/dash/`. Exit code 5 means something is critical.
- A collector that cannot read its source reports `unavailable: <why>`; it never turns a model green or red by itself.
- `mllab-modelhealth.timer` runs it every 10 minutes (`--build --remedy`).
- Self-healing (`remedy.py`) reads `/etc/mllab/remedy.json`. Default is dry-run with every rule off:
  `{"dry_run": false, "rules": {"coverage_gap": {"enabled": true, "max_per_day": 2, "cooldown_minutes": 90}}}`.
  Actions are a closed set (run an existing scheduler job by mode, hide a model behind an auto banner only when another healthy model is visible,
  prune idle envs, alert). Nothing is ever un-hidden or redeployed automatically. Failed attempts count toward the limits. Log: `mllab remedy log`.
- Access: `scripts/grant_health_access.sh SA [--remedy]` (you run it).

## Addendum 2026-09-29: stand-in, last-known-good, lifecycle, pin-mismatch
Spec: `docs/specs/model-control-remaining.md`. Vocabulary: `GLOSSARY.md`. Six items in that spec are still OPEN; the defaults below are what is built, and each is reversible.

### `role` -> `lifecycle` (field rename)
- The control field is now `lifecycle` (`live` | `shadow` | `archived`), informational. The site registry's `role` (production/shadow/reference/benchmark) is unrelated and unchanged.
- TRANSITION: deployed consumers still read the column `role`. The current-state view therefore exposes BOTH columns, `lifecycle` and `role` (same value), until every consumer is redeployed. Consumers read `lifecycle`, falling back to `role`. The lab writes only `lifecycle` and rejects the field name `role` with a message pointing to `lifecycle`. Dropping the `role` column is a later step.
- Pin rule is unchanged (default for open question 5): a pin needs `lifecycle=live`, `artifact_uri` and `artifact_sha256`.

### Last known good (backend only)
- After a failed control read, the backend keeps applying the last successfully read state for up to 6 hours, then fails open. One WARNING per failure window and one when the remembered state expires. A successful read replaces it at once. ML is unchanged (one read per run, fails open).

### Stand-in model (backend + frontend)
- Applies when the PRODUCTION key of a sport is hidden (`site_visible=false`) or paused (`run_state=paused`) by the control state. Never otherwise. (Open question 2 is whether pause should trigger it; built as the user chose: hidden or paused.)
- Scope: compare and slate routes for MLB only in this release (football and the legacy prediction endpoints are open questions 3 and 4 and are NOT changed).
- Candidates: visible models, excluding the production key and the derived models (`elo`, `market`), that have predictions for the games being served. Choose the lowest season-to-date log loss (point estimate) on games every candidate was scored on, when there are at least 50 such graded games. Otherwise use the fixed order `sim_blend`, then `logit3`. If no candidate exists there is no stand-in.
- Response contract: `stand_in: { model: <key>, label: <string>, reason: "production_hidden" | "production_paused", basis: "season_log_loss" | "fixed_order", n_games: <int|null> } | null`. `featured_default` becomes the stand-in key. The field is always present on the compare and slate responses (null when unused).
- Frontend: when `stand_in` is non-null, show an automatic notice (plain text, not dismissible, cannot be turned off by control settings) such as "Showing <label> while <production label> is offline." It is separate from, and shown alongside, the owner banner.

### Pin mismatch (lab)
- New critical health finding `pin_mismatch` (names model and artifact, no remedy). Signal (default for open question 1, reversible): the lab's collector looks in Cloud Logging for the ML pipeline's pin-mismatch error line; if it cannot read logs the source shows `unavailable` and no finding is raised. The pipeline itself is unchanged: it fails the run and writes nothing.
