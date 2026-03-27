#!/bin/bash
set -euo pipefail
#
# Deploy the 2026 season pipeline infrastructure to GCP.
#
# What this sets up:
#   1. BigQuery dataset + tables  (mlb_2026_season)
#   2. Cloud Function             (mlb-2026-daily-pipeline)
#   3. Cloud Scheduler jobs:
#        - Daily 4 AM ET: data collection + validation + features
#        - Friday 5 AM ET: weekly batch predictions
#        - Monday 3 AM ET: roster refresh
#   4. Backfill spring training + opening days data
#
# Prerequisites:
#   - gcloud CLI authenticated with hankstank project
#   - APIs enabled: cloudfunctions, cloudscheduler, bigquery, aiplatform
#
# Usage:
#   ./deploy_2026_pipeline.sh              # full deploy
#   ./deploy_2026_pipeline.sh --skip-backfill
#   ./deploy_2026_pipeline.sh --only-scheduler

PROJECT="hankstank"
REGION="us-central1"
FUNCTION_NAME="mlb-2026-daily-pipeline"
RUNTIME="python312"
ENTRY_POINT="daily_pipeline"
MEMORY="1024MB"
TIMEOUT="540s"
SERVICE_ACCOUNT="$PROJECT@appspot.gserviceaccount.com"
SCHEDULER_TZ="America/New_York"
SRC_DIR="$(cd "$(dirname "$0")/../../src" && pwd)"

SKIP_BACKFILL=false
ONLY_SCHEDULER=false
for arg in "$@"; do
    case $arg in
        --skip-backfill) SKIP_BACKFILL=true ;;
        --only-scheduler) ONLY_SCHEDULER=true ;;
    esac
done

echo "=============================================="
echo " 2026 MLB Season Pipeline Deployment"
echo "=============================================="
echo " Project:  $PROJECT"
echo " Region:   $REGION"
echo " Function: $FUNCTION_NAME"
echo " Source:   $SRC_DIR"
echo "=============================================="

# ------------------------------------------------------------------
# 0. Verify gcloud
# ------------------------------------------------------------------
echo ""
echo "▸ Checking gcloud..."
gcloud config set project "$PROJECT" --quiet
ACTIVE=$(gcloud config get-value project 2>/dev/null)
if [ "$ACTIVE" != "$PROJECT" ]; then
    echo "ERROR: gcloud project is $ACTIVE, expected $PROJECT"
    exit 1
fi
echo "  ✓ Project: $PROJECT"

# ------------------------------------------------------------------
# 1. Create BigQuery dataset + tables
# ------------------------------------------------------------------
if [ "$ONLY_SCHEDULER" = false ]; then
    echo ""
    echo "▸ Creating BigQuery dataset and tables..."
    cd "$SRC_DIR/../scripts/gcp/2026_season"
    python3 create_dataset.py
    echo "  ✓ BigQuery ready"
fi

# ------------------------------------------------------------------
# 2. Deploy Cloud Function
# ------------------------------------------------------------------
if [ "$ONLY_SCHEDULER" = false ]; then
    echo ""
    echo "▸ Deploying Cloud Function..."
    cd "$SRC_DIR"

    gcloud functions deploy "$FUNCTION_NAME" \
        --gen2 \
        --region="$REGION" \
        --runtime="$RUNTIME" \
        --source="." \
        --entry-point="$ENTRY_POINT" \
        --trigger-http \
        --allow-unauthenticated=false \
        --memory="$MEMORY" \
        --timeout="$TIMEOUT" \
        --service-account="$SERVICE_ACCOUNT" \
        --set-env-vars="GCP_PROJECT=$PROJECT,PYTHONPATH=/workspace" \
        --quiet

    FUNCTION_URL=$(gcloud functions describe "$FUNCTION_NAME" \
        --gen2 --region="$REGION" --format="value(serviceConfig.uri)")
    echo "  ✓ Deployed: $FUNCTION_URL"
fi

# ------------------------------------------------------------------
# 3. Cloud Scheduler jobs
# ------------------------------------------------------------------
echo ""
echo "▸ Setting up Cloud Scheduler..."

# Get function URL if not already set
if [ -z "${FUNCTION_URL:-}" ]; then
    FUNCTION_URL=$(gcloud functions describe "$FUNCTION_NAME" \
        --gen2 --region="$REGION" --format="value(serviceConfig.uri)")
fi

# Get OIDC token audience
OIDC_AUDIENCE="$FUNCTION_URL"

# Daily pipeline — 4 AM ET every day during season (Mar-Nov)
gcloud scheduler jobs delete mlb-2026-daily --location="$REGION" --quiet 2>/dev/null || true
gcloud scheduler jobs create http mlb-2026-daily \
    --location="$REGION" \
    --schedule="0 4 * 3-11 *" \
    --uri="$FUNCTION_URL" \
    --http-method=POST \
    --headers="Content-Type=application/json" \
    --message-body='{"mode":"daily"}' \
    --time-zone="$SCHEDULER_TZ" \
    --oidc-service-account-email="$SERVICE_ACCOUNT" \
    --oidc-token-audience="$OIDC_AUDIENCE" \
    --attempt-deadline="$TIMEOUT" \
    --description="Daily 2026 MLB data collection, validation, and feature build" \
    --quiet
echo "  ✓ mlb-2026-daily (4 AM ET, Mar-Nov)"

# Weekly predictions — Friday 5 AM ET
gcloud scheduler jobs delete mlb-2026-weekly-predict --location="$REGION" --quiet 2>/dev/null || true
gcloud scheduler jobs create http mlb-2026-weekly-predict \
    --location="$REGION" \
    --schedule="0 5 * 3-11 5" \
    --uri="$FUNCTION_URL" \
    --http-method=POST \
    --headers="Content-Type=application/json" \
    --message-body='{"mode":"predict"}' \
    --time-zone="$SCHEDULER_TZ" \
    --oidc-service-account-email="$SERVICE_ACCOUNT" \
    --oidc-token-audience="$OIDC_AUDIENCE" \
    --attempt-deadline="$TIMEOUT" \
    --description="Weekly Friday predictions for next week of MLB games" \
    --quiet
echo "  ✓ mlb-2026-weekly-predict (Friday 5 AM ET)"

# Validation-only — 6 AM ET daily (separate check)
gcloud scheduler jobs delete mlb-2026-validate --location="$REGION" --quiet 2>/dev/null || true
gcloud scheduler jobs create http mlb-2026-validate \
    --location="$REGION" \
    --schedule="0 6 * 3-11 *" \
    --uri="$FUNCTION_URL" \
    --http-method=POST \
    --headers="Content-Type=application/json" \
    --message-body='{"mode":"validate"}' \
    --time-zone="$SCHEDULER_TZ" \
    --oidc-service-account-email="$SERVICE_ACCOUNT" \
    --oidc-token-audience="$OIDC_AUDIENCE" \
    --attempt-deadline="300s" \
    --description="Daily 2026 data validation check" \
    --quiet
echo "  ✓ mlb-2026-validate (6 AM ET)"

# ------------------------------------------------------------------
# 4. Backfill (spring training through today)
# ------------------------------------------------------------------
if [ "$SKIP_BACKFILL" = false ] && [ "$ONLY_SCHEDULER" = false ]; then
    echo ""
    echo "▸ Running backfill (spring training → today)..."
    cd "$SRC_DIR"

    # Spring training started ~Feb 20
    YESTERDAY=$(date -v-1d +%Y-%m-%d 2>/dev/null || date -d "yesterday" +%Y-%m-%d)
    python3 season_2026_pipeline.py --backfill --start 2026-02-20 --end "$YESTERDAY"

    echo ""
    echo "▸ Building features from backfilled data..."
    python3 build_2026_features.py

    echo ""
    echo "▸ Running validation..."
    python3 data_validation.py --year 2026 || true

    echo "  ✓ Backfill complete"
fi

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
echo ""
echo "=============================================="
echo " ✅ Deployment Complete"
echo "=============================================="
echo ""
echo " BigQuery:  $PROJECT.mlb_2026_season"
echo " Function:  $FUNCTION_NAME ($REGION)"
echo " Schedules:"
echo "   • mlb-2026-daily          4 AM ET daily (Mar-Nov)"
echo "   • mlb-2026-weekly-predict 5 AM ET Fridays"
echo "   • mlb-2026-validate       6 AM ET daily"
echo ""
echo " Next steps:"
echo "   1. Train the Vertex AI model:"
echo "      python3 src/train_vertex_model.py"
echo ""
echo "   2. Test weekly predictions:"
echo "      python3 src/predict_2026_weekly.py --dry-run"
echo ""
echo "   3. Monitor logs:"
echo "      gcloud functions logs read $FUNCTION_NAME --region=$REGION --limit=50"
echo ""
echo "   4. Trigger manual run:"
echo "      gcloud scheduler jobs run mlb-2026-daily --location=$REGION"
echo ""
echo "=============================================="
