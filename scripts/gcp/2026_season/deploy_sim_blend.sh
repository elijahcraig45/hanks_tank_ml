#!/bin/bash
set -euo pipefail
#
# Deploy the v2 PA-simulator + strength blend as its own Cloud Function.
#
# It is the same source and entry point as mlb-2026-daily-pipeline, deployed separately
# because the simulator peaks at 1.3-3.4 GB and the daily function has 1 GB. It is a
# SHADOW: mode=sim_blend writes game_predictions_sim_blend and game_props_sim only.
#
# Triggered per game by the backend (lineup-scheduler.service.ts), which enqueues one
# {"mode":"sim_blend","game_pks":[pk],"date":...} task at the 90-minute checkpoint when
# SIM_BLEND_FUNCTION_URL is set in app.yaml. No Scheduler job.
#
# Usage:
#   ./deploy_sim_blend.sh --dry-run
#   ./deploy_sim_blend.sh

PROJECT="hankstank"
REGION="us-central1"
FUNCTION_NAME="mlb-2026-sim-blend"
RUNTIME="python312"
ENTRY_POINT="daily_pipeline"
MEMORY="4Gi"
CPU="2"
TIMEOUT="540s"
MAX_INSTANCES="1"
SERVICE_ACCOUNT="$PROJECT@appspot.gserviceaccount.com"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

DRY_RUN=false
for arg in "$@"; do
    case $arg in
        --dry-run) DRY_RUN=true ;;
        *) echo "unknown flag: $arg" >&2; exit 2 ;;
    esac
done

_dry() { if [ "$DRY_RUN" = true ]; then echo "  [DRY RUN] $*"; else "$@"; fi }

echo "=============================================="
echo " Sim blend (shadow) — deploy"
echo " Function: $FUNCTION_NAME  ($MEMORY, $CPU vCPU, max $MAX_INSTANCES instance)"
[ "$DRY_RUN" = true ] && echo " MODE:     DRY RUN"
echo "=============================================="

# Deploy only what git tracks under src/ (same staging as deploy_v10.sh).
STAGE="$(mktemp -d)/src"
mkdir -p "$STAGE"
git -C "$REPO_DIR" ls-files src/ | while IFS= read -r f; do
    rel="${f#src/}"
    mkdir -p "$STAGE/$(dirname "$rel")"
    cp "$REPO_DIR/$f" "$STAGE/$rel"
done
echo "▸ Staged $(find "$STAGE" -type f | wc -l | tr -d ' ') tracked files into $STAGE"
[ -d "$STAGE/pa_sim" ] || { echo "ERROR: pa_sim is not tracked" >&2; exit 1; }

cd "$STAGE"
_dry gcloud functions deploy "$FUNCTION_NAME" \
    --gen2 --region="$REGION" --runtime="$RUNTIME" \
    --source="." --entry-point="$ENTRY_POINT" \
    --trigger-http --no-allow-unauthenticated \
    --memory="$MEMORY" --cpu="$CPU" --timeout="$TIMEOUT" \
    --max-instances="$MAX_INSTANCES" \
    --service-account="$SERVICE_ACCOUNT" \
    --set-env-vars="GCP_PROJECT=$PROJECT,PYTHONPATH=/workspace,SIM_BLEND_MEMORY_MB=4096" \
    --quiet

if [ "$DRY_RUN" = false ]; then
    URL=$(gcloud functions describe "$FUNCTION_NAME" --gen2 --region="$REGION" \
        --format="value(serviceConfig.uri)")
    echo ""
    echo "  ✓ Deployed: $URL"
    echo "  Set SIM_BLEND_FUNCTION_URL=$URL in hanks_tank_backend/app.yaml, then deploy the backend."
fi
