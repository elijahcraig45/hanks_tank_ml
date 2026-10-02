#!/usr/bin/env bash
# One-time (idempotent) setup of the model control plane in BigQuery. Creates dataset `control`, the append-only events table and the current-state
# view, and adds the nullable model_sha256 column to the MLB predictions table. Nothing here deletes or rewrites data; re-running is a no-op.
# Run it yourself, with your own gcloud account:
#   CLOUDSDK_CORE_ACCOUNT=elijahcraig45@gmail.com scripts/gcp/control/setup_control.sh --dry-run     # print exactly what would run
#   CLOUDSDK_CORE_ACCOUNT=elijahcraig45@gmail.com scripts/gcp/control/setup_control.sh               # do it
#   ... --skip-sha-column   to leave game_predictions untouched
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PROJECT=${GCP_PROJECT:-hankstank}
DRY=0; SHA=1
for a in "$@"; do case "$a" in --dry-run) DRY=1 ;; --skip-sha-column) SHA=0 ;; *) echo "unknown arg $a" >&2; exit 2 ;; esac; done
case "${CLOUDSDK_CORE_ACCOUNT:-}" in *homedepot.com) echo "refusing the work account" >&2; exit 1 ;; esac
run_sql() {
  local f=$1
  echo "== $(basename "$f")"
  if [ "$DRY" = 1 ]; then sed 's/^/   | /' "$f"; else bq query --project_id="$PROJECT" --use_legacy_sql=false --quiet < "$f"; fi
}
run_sql "$HERE/01_dataset_and_events.sql"
run_sql "$HERE/02_current_view.sql"
[ "$SHA" = 1 ] && run_sql "$HERE/03_add_model_sha256.sql"
if [ "$DRY" = 0 ]; then
  echo "== verify"
  bq query --project_id="$PROJECT" --use_legacy_sql=false --quiet "SELECT COUNT(*) AS events FROM \`$PROJECT.control.model_control_events\`"
  bq query --project_id="$PROJECT" --use_legacy_sql=false --quiet "SELECT COUNT(*) AS current_rows FROM \`$PROJECT.control.model_control_current\`"
fi
echo "done. Grant readers with:  bq add-iam-policy-binding is not needed for the view when the reader already has project-level bigquery.dataViewer"
