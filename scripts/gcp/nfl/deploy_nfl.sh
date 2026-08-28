#!/bin/bash
set -euo pipefail
#
# Deploy the NFL weekly pipeline.
#
# Deliberately a SEPARATE Cloud Function from mlb-2026-daily-pipeline, deployed with
# --source=src/nfl. Consequences that matter:
#   * an NFL deploy never redeploys the code predicting MLB games at 4am
#   * the NFL image carries no pybaseball / MLB-StatsAPI / catboost
#   * the two functions do not share a cold start
#
# Cadence: NFL results settle Thursday->Monday, so ingest runs Tuesday and predictions
# Wednesday. This is week-shaped, not date-shaped like MLB.
#
# Usage:
#   ./deploy_nfl.sh                  # function + scheduler
#   ./deploy_nfl.sh --only-scheduler
#   ./deploy_nfl.sh --dry-run

PROJECT="hankstank"
REGION="us-central1"
FUNCTION_NAME="nfl-weekly-pipeline"
RUNTIME="python312"
ENTRY_POINT="nfl_pipeline"
MEMORY="2048MB"
TIMEOUT="540s"
SERVICE_ACCOUNT="$PROJECT@appspot.gserviceaccount.com"
SCHEDULER_TZ="America/New_York"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd "$SCRIPT_DIR/../../../src/nfl" && pwd)"

DRY_RUN=false
ONLY_SCHEDULER=false
for arg in "$@"; do
    case $arg in
        --dry-run)        DRY_RUN=true ;;
        --only-scheduler) ONLY_SCHEDULER=true ;;
    esac
done

echo "=============================================="
echo " NFL Weekly Pipeline — deploy"
echo " Project:  $PROJECT"
echo " Function: $FUNCTION_NAME"
echo " Source:   $SRC_DIR   (NFL only — no MLB code)"
[ "$DRY_RUN" = true ] && echo " MODE:     DRY RUN"
echo "=============================================="

_dry() {
    if [ "$DRY_RUN" = true ]; then echo "  [DRY RUN] $*"; else "$@"; fi
}

gcloud config set project "$PROJECT" --quiet

if [ "$ONLY_SCHEDULER" = false ]; then
    echo ""
    echo "▸ Deploying Cloud Function..."
    _dry gcloud functions deploy "$FUNCTION_NAME" \
        --gen2 \
        --region="$REGION" \
        --runtime="$RUNTIME" \
        --source="$SRC_DIR" \
        --entry-point="$ENTRY_POINT" \
        --trigger-http \
        --no-allow-unauthenticated \
        --memory="$MEMORY" \
        --timeout="$TIMEOUT" \
        --service-account="$SERVICE_ACCOUNT" \
        --set-env-vars="GCP_PROJECT=$PROJECT,NFL_DATASET=nfl_season,NFL_HIST_DATASET=nfl_historical" \
        --quiet
    echo "  ✓ deployed"
fi

FUNCTION_URL="https://$REGION-$PROJECT.cloudfunctions.net/$FUNCTION_NAME"
if [ "$DRY_RUN" = false ]; then
    FUNCTION_URL=$(gcloud functions describe "$FUNCTION_NAME" --gen2 --region="$REGION" \
        --format="value(serviceConfig.uri)" 2>/dev/null || echo "$FUNCTION_URL")
fi
echo "  URL: $FUNCTION_URL"

_sched() {
    local JOB="$1" SCHEDULE="$2" BODY="$3" DESC="$4"
    _dry gcloud scheduler jobs delete "$JOB" --location="$REGION" --quiet 2>/dev/null || true
    _dry gcloud scheduler jobs create http "$JOB" \
        --location="$REGION" \
        --schedule="$SCHEDULE" \
        --uri="$FUNCTION_URL" \
        --http-method=POST \
        --headers="Content-Type=application/json" \
        --message-body="$BODY" \
        --time-zone="$SCHEDULER_TZ" \
        --oidc-service-account-email="$SERVICE_ACCOUNT" \
        --oidc-token-audience="$FUNCTION_URL" \
        --attempt-deadline="$TIMEOUT" \
        --description="$DESC" \
        --quiet
}

echo ""
echo "▸ Scheduler jobs (Sep–Feb only; NFL season plus playoffs)..."

# Tuesday 6 AM ET — MNF is final by now, so the week's results are complete.
_sched "nfl-weekly-ingest" "0 6 * 9-12,1,2 2" \
    '{"mode":"ingest"}' \
    "NFL: refresh schedules + EPA into nfl_historical"
echo "  ✓ nfl-weekly-ingest (Tue 6:00 AM ET)"

# Tuesday 7 AM ET — score last week's predictions once results have landed.
_sched "nfl-weekly-score" "0 7 * 9-12,1,2 2" \
    '{"mode":"score"}' \
    "NFL: score stored predictions against final results"
echo "  ✓ nfl-weekly-score (Tue 7:00 AM ET)"

# Wednesday 6 AM ET — predict the upcoming slate.
_sched "nfl-weekly-predict" "0 6 * 9-12,1,2 3" \
    '{"mode":"predict_week"}' \
    "NFL: predict the next unplayed week"
echo "  ✓ nfl-weekly-predict (Wed 6:00 AM ET)"

echo ""
echo "=============================================="
echo " Done. Verify with:"
echo "   gcloud functions logs read $FUNCTION_NAME --gen2 --region=$REGION --limit=30"
echo "   bq query --use_legacy_sql=false 'SELECT season, week, COUNT(*) FROM \`$PROJECT.nfl_season.game_predictions\` GROUP BY 1,2 ORDER BY 1 DESC, 2 DESC LIMIT 5'"
echo "=============================================="
