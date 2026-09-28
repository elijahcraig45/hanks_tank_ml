#!/bin/bash
set -euo pipefail
#
# Deploy the v2 PA-simulator + strength blend as its own Cloud Function, and create or
# update its own Cloud Tasks queue.
#
# It is the same source and entry point as mlb-2026-daily-pipeline, deployed separately
# because the simulator peaks near 3 GB cold and the daily function has 1 GB. It is a
# SHADOW: mode=sim_blend writes game_predictions_sim_blend, game_props_sim,
# game_sim_distributions and player_sim_projections only. The last two are appended with
# CREATE_NEVER and reported per table (a missing one never blocks the first two); create
# them first with create_game_sim_distributions.sql and create_player_sim_projections.sql.
#
# Triggered per game by the backend (lineup-scheduler.service.ts) when
# SIM_BLEND_FUNCTION_URL is set in app.yaml: {"mode":"sim_blend",...} at T-80 (after the
# T-90 pregame task has fetched lineups) and a retry at T-35 with "lineup_fallback":true.
# Both go to the SIM_BLEND_TASK_QUEUE queue created here, never the pregame queue, so a
# backlog or failure of the shadow cannot delay a real pregame task. No Scheduler job.
#
# Sizing (measured 2026-09-28, docs/SHADOW_MODELS.md "Operations"):
#   memory     6 GiB. Peak RSS 2.80 GB cold, 2.36 GB for later tasks the same day (the
#              engine stays warm), 3.53 GB when a warm instance changes date; production
#              ran ~10% above local (OOM at 4,271 MiB where local peaked at 3,863 MiB).
#   instances  3, with the queue dispatching at most 2 at once: every dispatched task
#              finds an idle or startable instance, so no 429 "no available instance".
#
# Usage (with CLOUDSDK_CORE_ACCOUNT=elijahcraig45@gmail.com):
#   ./deploy_sim_blend.sh --dry-run
#   ./deploy_sim_blend.sh                # function + queue
#   ./deploy_sim_blend.sh --only-queue   # queue only

PROJECT="hankstank"
REGION="us-central1"
FUNCTION_NAME="mlb-2026-sim-blend"
RUNTIME="python312"
ENTRY_POINT="daily_pipeline"
MEMORY="6Gi"
MEMORY_MB="6144"
CPU="2"
TIMEOUT="540s"
MAX_INSTANCES="3"
QUEUE="sim-blend"
QUEUE_MAX_CONCURRENT="2"   # < MAX_INSTANCES: one spare instance
QUEUE_MAX_ATTEMPTS="3"
QUEUE_MIN_BACKOFF="60s"
QUEUE_MAX_BACKOFF="300s"
SERVICE_ACCOUNT="$PROJECT@appspot.gserviceaccount.com"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

DRY_RUN=false
ONLY_QUEUE=false
for arg in "$@"; do
    case $arg in
        --dry-run) DRY_RUN=true ;;
        --only-queue) ONLY_QUEUE=true ;;
        *) echo "unknown flag: $arg" >&2; exit 2 ;;
    esac
done

_dry() { if [ "$DRY_RUN" = true ]; then echo "  [DRY RUN] $*"; else "$@"; fi }

echo "=============================================="
echo " Sim blend (shadow) — deploy"
echo " Function: $FUNCTION_NAME  ($MEMORY, $CPU vCPU, max $MAX_INSTANCES instances)"
echo " Queue:    $QUEUE  (max $QUEUE_MAX_CONCURRENT concurrent, $QUEUE_MAX_ATTEMPTS attempts)"
[ "$DRY_RUN" = true ] && echo " MODE:     DRY RUN"
echo "=============================================="

# Cloud Tasks queue for the shadow's tasks only.
QUEUE_FLAGS=(--location="$REGION" --project="$PROJECT"
    --max-concurrent-dispatches="$QUEUE_MAX_CONCURRENT" --max-dispatches-per-second=1
    --max-attempts="$QUEUE_MAX_ATTEMPTS" --min-backoff="$QUEUE_MIN_BACKOFF" --max-backoff="$QUEUE_MAX_BACKOFF")
if gcloud tasks queues describe "$QUEUE" --location="$REGION" --project="$PROJECT" >/dev/null 2>&1; then
    echo "▸ Updating queue $QUEUE"
    _dry gcloud tasks queues update "$QUEUE" "${QUEUE_FLAGS[@]}"
else
    echo "▸ Creating queue $QUEUE"
    _dry gcloud tasks queues create "$QUEUE" "${QUEUE_FLAGS[@]}"
fi
[ "$ONLY_QUEUE" = true ] && { echo "  (--only-queue: function not deployed)"; exit 0; }

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
    --set-env-vars="GCP_PROJECT=$PROJECT,PYTHONPATH=/workspace,SIM_BLEND_MEMORY_MB=$MEMORY_MB" \
    --quiet

if [ "$DRY_RUN" = false ]; then
    URL=$(gcloud functions describe "$FUNCTION_NAME" --gen2 --region="$REGION" \
        --format="value(serviceConfig.uri)")
    echo ""
    echo "  ✓ Deployed: $URL"
    echo "  Set SIM_BLEND_FUNCTION_URL=$URL and SIM_BLEND_TASK_QUEUE=$QUEUE in"
    echo "  hanks_tank_backend/app.yaml, then deploy the backend."
fi
