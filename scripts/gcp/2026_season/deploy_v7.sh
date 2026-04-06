#!/bin/bash
set -euo pipefail
#
# Deploy V7 pipeline to GCP.
#
# What this does:
#   1. Redeploys the Cloud Function with V7 source files included
#   2. Adds the V7-specific Cloud Scheduler jobs (Sunday V7 training, daily V7 features)
#   3. Backfills V7 features from BACKFILL_START through yesterday
#   4. Triggers a one-time V7 model training run
#
# Keeps all existing V6 scheduler jobs running unchanged.
#
# Prerequisites:
#   - V6 already deployed (deploy_2026_pipeline.sh has been run)
#   - gcloud CLI authenticated with hankstank project
#   - ephem pip package in requirements.txt (for moon phase)
#
# Usage:
#   ./deploy_v7.sh                         # full deploy + backfill + train
#   ./deploy_v7.sh --skip-backfill         # deploy + train, no backfill
#   ./deploy_v7.sh --skip-train            # deploy + backfill, no training
#   ./deploy_v7.sh --only-function         # just redeploy the function
#   ./deploy_v7.sh --dry-run               # preview all steps, no writes

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

# Backfill from opening day (V7 needs statcast + bullpen data — starts at season open)
BACKFILL_START="2026-03-27"
BACKFILL_END=$(date -v-1d +%Y-%m-%d 2>/dev/null || date -d "yesterday" +%Y-%m-%d)

SKIP_BACKFILL=false
SKIP_TRAIN=false
ONLY_FUNCTION=false
DRY_RUN_FLAG=""
for arg in "$@"; do
    case $arg in
        --skip-backfill) SKIP_BACKFILL=true ;;
        --skip-train)    SKIP_TRAIN=true ;;
        --only-function) ONLY_FUNCTION=true ;;
        --dry-run)       DRY_RUN_FLAG="--dry-run" ;;
    esac
done

echo "=============================================="
echo " V7 Pipeline Deployment"
echo " Bullpen Health + Moon Phase + Pitcher Venue"
echo "=============================================="
echo " Project:       $PROJECT"
echo " Region:        $REGION"
echo " Function:      $FUNCTION_NAME"
echo " Source:        $SRC_DIR"
echo " Backfill from: $BACKFILL_START → $BACKFILL_END"
echo "=============================================="

# ------------------------------------------------------------------
# 0. Verify gcloud project
# ------------------------------------------------------------------
echo ""
echo "▸ Checking gcloud..."
gcloud config set project "$PROJECT" --quiet
ACTIVE=$(gcloud config get-value project 2>/dev/null)
if [ "$ACTIVE" != "$PROJECT" ]; then
    echo "ERROR: active project is $ACTIVE, expected $PROJECT"; exit 1
fi
echo "  ✓ Project: $PROJECT"

# Get current function URL
FUNCTION_URL=$(gcloud functions describe "$FUNCTION_NAME" \
    --gen2 --region="$REGION" --format="value(serviceConfig.uri)" 2>/dev/null || echo "")
if [ -z "$FUNCTION_URL" ]; then
    echo "ERROR: $FUNCTION_NAME not found. Run deploy_2026_pipeline.sh first."; exit 1
fi
echo "  ✓ Function URL: $FUNCTION_URL"
OIDC_AUDIENCE="$FUNCTION_URL"

# ------------------------------------------------------------------
# 1. Ensure ephem is in requirements
# ------------------------------------------------------------------
echo ""
echo "▸ Checking requirements.txt for ephem..."
REQS="$SRC_DIR/../requirements.txt"
if ! grep -q "^ephem" "$REQS" 2>/dev/null; then
    echo "ephem>=4.1" >> "$REQS"
    echo "  ✓ Added ephem to requirements.txt"
else
    echo "  ✓ ephem already present"
fi

# ------------------------------------------------------------------
# 2. Redeploy Cloud Function (adds V7 files)
# ------------------------------------------------------------------
echo ""
echo "▸ Redeploying Cloud Function with V7 source..."
cd "$SRC_DIR"

if [ -z "$DRY_RUN_FLAG" ]; then
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
    echo "  ✓ Cloud Function redeployed"
else
    echo "  [DRY RUN] Would redeploy $FUNCTION_NAME"
fi

# ------------------------------------------------------------------
# 3. Add V7-specific Cloud Scheduler jobs
#    (V6 jobs from deploy_2026_pipeline.sh are left untouched)
# ------------------------------------------------------------------
if [ "$ONLY_FUNCTION" = false ]; then
    echo ""
    echo "▸ Adding V7 Cloud Scheduler jobs..."

    # Sunday V7 weekly training (replaces V6 on Sundays going forward)
    # V6 train_weekly job is kept but will be superseded by this one
    gcloud scheduler jobs delete mlb-2026-train-v7-weekly \
        --location="$REGION" --quiet 2>/dev/null || true
    if [ -z "$DRY_RUN_FLAG" ]; then
        gcloud scheduler jobs create http mlb-2026-train-v7-weekly \
            --location="$REGION" \
            --schedule="0 7 * 3-11 0" \
            --uri="$FUNCTION_URL" \
            --http-method=POST \
            --headers="Content-Type=application/json" \
            --message-body='{"mode":"train_weekly","model_version":"v7"}' \
            --time-zone="$SCHEDULER_TZ" \
            --oidc-service-account-email="$SERVICE_ACCOUNT" \
            --oidc-token-audience="$OIDC_AUDIENCE" \
            --attempt-deadline="$TIMEOUT" \
            --description="Sunday V7 model retrain: bullpen+moon+venue stacked ensemble" \
            --quiet
        echo "  ✓ mlb-2026-train-v7-weekly (Sunday 7 AM ET)"
    else
        echo "  [DRY RUN] Would create mlb-2026-train-v7-weekly"
    fi

    # Daily V7 feature build (runs after collection + V5/V6 matchup, 5:30 AM ET)
    gcloud scheduler jobs delete mlb-2026-v7-features-daily \
        --location="$REGION" --quiet 2>/dev/null || true
    if [ -z "$DRY_RUN_FLAG" ]; then
        gcloud scheduler jobs create http mlb-2026-v7-features-daily \
            --location="$REGION" \
            --schedule="30 5 * 3-11 *" \
            --uri="$FUNCTION_URL" \
            --http-method=POST \
            --headers="Content-Type=application/json" \
            --message-body='{"mode":"matchup_v7_features"}' \
            --time-zone="$SCHEDULER_TZ" \
            --oidc-service-account-email="$SERVICE_ACCOUNT" \
            --oidc-token-audience="$OIDC_AUDIENCE" \
            --attempt-deadline="300s" \
            --description="Daily V7 feature build (bullpen health, moon, pitcher venue)" \
            --quiet
        echo "  ✓ mlb-2026-v7-features-daily (5:30 AM ET)"
    else
        echo "  [DRY RUN] Would create mlb-2026-v7-features-daily"
    fi

    # Update the pre-game Cloud Tasks to use pregame_v7 mode
    # (This updates the mode used when the backend schedules pre-game tasks)
    echo ""
    echo "  NOTE: Pre-game Cloud Tasks are enqueued by the backend via"
    echo "        /api/lineup/schedule-task. The task payload mode should"
    echo "        be changed from 'pregame' → 'pregame_v7' in the backend"
    echo "        to include V7 features in the pre-game pipeline."
    echo "        See: hanks_tank_backend/src/controllers/lineup.controller.ts"
fi

# ------------------------------------------------------------------
# 4. Backfill V7 features for the 2026 season so far
# ------------------------------------------------------------------
if [ "$SKIP_BACKFILL" = false ] && [ "$ONLY_FUNCTION" = false ]; then
    echo ""
    echo "▸ Backfilling V7 features ($BACKFILL_START → $BACKFILL_END)..."
    echo "  This triggers the Cloud Function in backfill_v7 mode."
    echo "  ETA: ~1-3 minutes per week of games."

    if [ -z "$DRY_RUN_FLAG" ]; then
        # Invoke via gcloud — chunked to stay within 540s timeout
        # Split into monthly chunks
        BACKFILL_BODY=$(printf '{"mode":"backfill_v7","start":"%s","end":"%s"}' \
            "$BACKFILL_START" "$BACKFILL_END")

        TOKEN=$(gcloud auth print-identity-token 2>/dev/null || \
                gcloud auth print-access-token 2>/dev/null)

        HTTP_STATUS=$(curl -s -o /tmp/v7_backfill_response.json -w "%{http_code}" \
            -X POST "$FUNCTION_URL" \
            -H "Content-Type: application/json" \
            -H "Authorization: Bearer $TOKEN" \
            -d "$BACKFILL_BODY" \
            --max-time 600)

        if [ "$HTTP_STATUS" = "200" ]; then
            GAMES=$(python3 -c "import json; d=json.load(open('/tmp/v7_backfill_response.json')); \
                steps=d.get('steps',[]); \
                bf=[s for s in steps if s.get('step')=='v7_backfill']; \
                print(bf[0].get('games_processed',0) if bf else 0)" 2>/dev/null || echo "?")
            echo "  ✓ Backfill complete — $GAMES game-days processed"
        else
            echo "  ⚠ Backfill returned HTTP $HTTP_STATUS"
            cat /tmp/v7_backfill_response.json 2>/dev/null || true
        fi
    else
        echo "  [DRY RUN] Would POST backfill_v7 to $FUNCTION_URL"
    fi
fi

# ------------------------------------------------------------------
# 5. Trigger V7 model training
# ------------------------------------------------------------------
if [ "$SKIP_TRAIN" = false ] && [ "$ONLY_FUNCTION" = false ]; then
    echo ""
    echo "▸ Triggering V7 model training..."
    echo "  This trains on 2015–2024 data + V7 feature joins and uploads"
    echo "  the model to gs://$PROJECT-data/models/vertex/game_outcome_2026_v7/"
    echo "  ETA: 3–8 minutes depending on BQ join latency."

    if [ -z "$DRY_RUN_FLAG" ]; then
        TOKEN=$(gcloud auth print-identity-token 2>/dev/null || \
                gcloud auth print-access-token 2>/dev/null)

        HTTP_STATUS=$(curl -s -o /tmp/v7_train_response.json -w "%{http_code}" \
            -X POST "$FUNCTION_URL" \
            -H "Content-Type: application/json" \
            -H "Authorization: Bearer $TOKEN" \
            -d '{"mode":"train_weekly","model_version":"v7"}' \
            --max-time 600)

        if [ "$HTTP_STATUS" = "200" ]; then
            echo "  ✓ V7 model training complete"
            python3 -c "
import json
d = json.load(open('/tmp/v7_train_response.json'))
for step in d.get('steps', []):
    if 'v7' in step.get('step', ''):
        print('  Model:', step.get('model','?'))
        m = step.get('metrics', {})
        if m:
            print(f\"  Accuracy: {m.get('accuracy',0):.4f} | AUC: {m.get('auc',0):.4f}\")
" 2>/dev/null || true
        else
            echo "  ⚠ Training returned HTTP $HTTP_STATUS"
            cat /tmp/v7_train_response.json 2>/dev/null || true
        fi
    else
        echo "  [DRY RUN] Would POST train_weekly/v7 to $FUNCTION_URL"
    fi
fi

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
echo ""
echo "=============================================="
echo " ✅ V7 Deployment Complete"
echo "=============================================="
echo ""
echo " Active scheduler jobs (all):"
echo ""
gcloud scheduler jobs list --location="$REGION" \
    --format="table(name.basename(),schedule,state)" 2>/dev/null || true
echo ""
echo " GCS model artifacts:"
gsutil ls "gs://hanks_tank_data/models/vertex/" 2>/dev/null || true
echo ""
echo " Next steps:"
echo "  1. Update backend pre-game task mode: 'pregame' → 'pregame_v7'"
echo "     File: hanks_tank_backend/src/controllers/lineup.controller.ts"
echo "  2. Monitor first live V7 predictions in mlb_2026_season.game_predictions"
echo "     bq query --project=$PROJECT \\"
echo "       'SELECT * FROM mlb_2026_season.game_predictions"
echo "        ORDER BY created_at DESC LIMIT 5'"
echo "  3. Compare V7 vs V6 accuracy after 2+ weeks of games"
echo "=============================================="
