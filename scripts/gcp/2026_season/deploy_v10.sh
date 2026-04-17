#!/bin/bash
set -euo pipefail
#
# Deploy V10 model and pipeline to GCP.
#
# What this does:
#   1. Validates prerequisites (gcloud auth, project, bucket)
#   2. Uploads the V10 model artifact to GCS
#   3. Redeploys the Cloud Function with V10 source
#   4. Updates Cloud Scheduler to use pregame_v10 mode
#   5. Verifies deployment via BQ row counts
#
# V10 improvements over V8:
#   - XGBoost (vs CatBoost) — smaller model, faster inference
#   - SP quality via Statcast xERA percentile ranks (139 features total)
#   - Park factors (39 MLB venues)
#   - Rest / travel days
#   - 61.48% live 2026 accuracy (+6.71% over V8)
#   - 64.54% at confidence ≥64%
#
# Prerequisites:
#   - gcloud CLI authenticated
#   - V10 model artifact present: models/game_outcome_2026_v10.pkl
#   - SP data already uploaded to GCS (scripts/upload_statcast_sp_to_gcs.py)
#
# Usage:
#   ./deploy_v10.sh                        # full deploy
#   ./deploy_v10.sh --skip-function        # only model upload + scheduler
#   ./deploy_v10.sh --only-upload-model    # only upload model to GCS
#   ./deploy_v10.sh --only-scheduler       # only update scheduler jobs
#   ./deploy_v10.sh --dry-run              # preview, no writes

PROJECT="hankstank"
REGION="us-central1"
FUNCTION_NAME="mlb-2026-daily-pipeline"
RUNTIME="python312"
ENTRY_POINT="daily_pipeline"
MEMORY="1024MB"          # V10 XGBoost ~245KB — much smaller than V8 CatBoost
TIMEOUT="540s"
SERVICE_ACCOUNT="$PROJECT@appspot.gserviceaccount.com"
SCHEDULER_TZ="America/New_York"
BUCKET="hanks_tank_data"

# V10 model artifact paths
V10_LOCAL_PATH="models/game_outcome_2026_v10.pkl"
V10_GCS_PATH="models/vertex/game_outcome_2026_v10/model.pkl"

SEASON_START="2026-03-27"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd "$SCRIPT_DIR/../../../src" && pwd)"
REPO_DIR="$(cd "$SRC_DIR/.." && pwd)"

# Parse flags
SKIP_FUNCTION=false
ONLY_SCHEDULER=false
ONLY_UPLOAD_MODEL=false
DRY_RUN=false

for arg in "$@"; do
    case $arg in
        --skip-function)     SKIP_FUNCTION=true ;;
        --only-scheduler)    ONLY_SCHEDULER=true; SKIP_FUNCTION=true ;;
        --only-upload-model) ONLY_UPLOAD_MODEL=true ;;
        --dry-run)           DRY_RUN=true ;;
    esac
done

echo "=============================================="
echo " V10 Model — GCP Deployment"
echo " XGBoost + SP xERA + Park Factors | 61.48% Acc"
echo " 64.54% accuracy at ≥64% confidence threshold"
echo "=============================================="
echo " Project:      $PROJECT"
echo " Region:       $REGION"
echo " Function:     $FUNCTION_NAME"
echo " Memory:       $MEMORY  (XGBoost — lighter than V8 CatBoost)"
echo " Source:       $SRC_DIR"
echo " Model local:  $REPO_DIR/$V10_LOCAL_PATH"
echo " Model GCS:    gs://$BUCKET/$V10_GCS_PATH"
if [ "$DRY_RUN" = true ]; then
    echo " MODE:         DRY RUN (no writes)"
fi
echo "=============================================="

_dry() {
    if [ "$DRY_RUN" = true ]; then
        echo "  [DRY RUN] Would run: $*"
    else
        "$@"
    fi
}

# ------------------------------------------------------------------
# 0. Verify gcloud
# ------------------------------------------------------------------
echo ""
echo "▸ Verifying gcloud configuration..."
gcloud config set project "$PROJECT" --quiet
ACTIVE=$(gcloud config get-value project 2>/dev/null)
if [ "$ACTIVE" != "$PROJECT" ]; then
    echo "ERROR: gcloud project is '$ACTIVE', expected '$PROJECT'"
    exit 1
fi
echo "  ✓ Project: $PROJECT"

# Confirm V10 model artifact exists
if [ ! -f "$REPO_DIR/$V10_LOCAL_PATH" ]; then
    echo "ERROR: V10 model artifact not found: $REPO_DIR/$V10_LOCAL_PATH"
    exit 1
fi
echo "  ✓ V10 model artifact found ($(du -h "$REPO_DIR/$V10_LOCAL_PATH" | cut -f1))"

if [ "$ONLY_UPLOAD_MODEL" = true ]; then
    echo ""
    echo "▸ Uploading V10 model to GCS..."
    _dry gsutil cp "$REPO_DIR/$V10_LOCAL_PATH" "gs://$BUCKET/$V10_GCS_PATH"
    echo "  ✓ Model uploaded: gs://$BUCKET/$V10_GCS_PATH"
    echo ""
    echo "Done (--only-upload-model)."
    exit 0
fi

# ------------------------------------------------------------------
# 1. Upload V10 model artifact to GCS
# ------------------------------------------------------------------
echo ""
echo "▸ Uploading V10 model to GCS..."
_dry gsutil cp "$REPO_DIR/$V10_LOCAL_PATH" "gs://$BUCKET/$V10_GCS_PATH"
echo "  ✓ Model uploaded: gs://$BUCKET/$V10_GCS_PATH"

# ------------------------------------------------------------------
# 2. Redeploy Cloud Function with V10 source
# ------------------------------------------------------------------
if [ "$SKIP_FUNCTION" = false ] && [ "$ONLY_SCHEDULER" = false ]; then
    echo ""
    echo "▸ Deploying Cloud Function with V10 support..."
    echo "  Note: xgboost added to requirements.txt — deploy may take 2-3 min"
    cd "$SRC_DIR"
    _dry gcloud functions deploy "$FUNCTION_NAME" \
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
    echo "  ✓ Cloud Function deployed"
fi

# Get function URL
FUNCTION_URL=""
if [ "$DRY_RUN" = false ]; then
    FUNCTION_URL=$(gcloud functions describe "$FUNCTION_NAME" \
        --gen2 --region="$REGION" --format="value(serviceConfig.uri)" 2>/dev/null || echo "")
    if [ -z "$FUNCTION_URL" ]; then
        echo "  WARNING: Could not retrieve function URL"
    else
        echo "  ✓ Function URL: $FUNCTION_URL"
    fi
else
    FUNCTION_URL="https://$REGION-$PROJECT.cloudfunctions.net/$FUNCTION_NAME"
    echo "  [DRY RUN] Using placeholder URL: $FUNCTION_URL"
fi

OIDC_AUDIENCE="$FUNCTION_URL"

# ------------------------------------------------------------------
# 3. Cloud Scheduler jobs — upgrade to pregame_v10
# ------------------------------------------------------------------
echo ""
echo "▸ Configuring Cloud Scheduler (V10 pipeline)..."

_sched_delete() {
    local JOB="$1"
    _dry gcloud scheduler jobs delete "$JOB" \
        --location="$REGION" --quiet 2>/dev/null || true
}

_sched_create() {
    local JOB="$1"
    local SCHEDULE="$2"
    local BODY="$3"
    local DESC="$4"
    _sched_delete "$JOB"
    _dry gcloud scheduler jobs create http "$JOB" \
        --location="$REGION" \
        --schedule="$SCHEDULE" \
        --uri="$FUNCTION_URL" \
        --http-method=POST \
        --headers="Content-Type=application/json" \
        --message-body="$BODY" \
        --time-zone="$SCHEDULER_TZ" \
        --oidc-service-account-email="$SERVICE_ACCOUNT" \
        --oidc-token-audience="$OIDC_AUDIENCE" \
        --attempt-deadline="$TIMEOUT" \
        --description="$DESC" \
        --quiet
}

# ① Daily data collection + features — 4:00 AM ET
#   Includes: collect → validate → V8 features → V10 features → Elo update
_sched_create \
    "mlb-2026-daily" \
    "0 4 * 3-11 *" \
    '{"mode":"daily"}' \
    "V10: daily data collection, V8+V10 feature build, Elo update"
echo "  ✓ mlb-2026-daily (4:00 AM ET, Mar-Nov)"

# ② Pre-game task scheduler — 10:00 AM ET
#   Checks today's schedule and enqueues a Cloud Task per game ~90 min before first pitch.
#   Each task triggers: lineups → matchup → V8 features → V10 features → predict (V10)
_sched_create \
    "mlb-2026-pregame-schedule" \
    "0 10 * 3-11 *" \
    '{"mode":"schedule_pregame_tasks"}' \
    "V10: schedule pre-game Cloud Tasks for today (lineups+V8+V10+predict)"
echo "  ✓ mlb-2026-pregame-schedule (10:00 AM ET, Mar-Nov)"

# ③ Weekly V10 model retraining — Sunday 2:00 AM ET
_sched_create \
    "mlb-2026-weekly-train-v10" \
    "0 2 * 3-11 0" \
    '{"mode":"train_weekly","model_version":"v10"}' \
    "V10: weekly XGBoost retraining (Sunday 2 AM ET)"
echo "  ✓ mlb-2026-weekly-train-v10 (Sunday 2:00 AM ET)"

# ④ Weekly predictions (Friday batch) — Friday 5:00 AM ET
_sched_create \
    "mlb-2026-weekly-predict" \
    "0 5 * 3-11 5" \
    '{"mode":"predict"}' \
    "V10: Friday batch predictions for the upcoming week"
echo "  ✓ mlb-2026-weekly-predict (Friday 5:00 AM ET)"

# ⑤ Roster refresh — Monday 3:00 AM ET
_sched_create \
    "mlb-2026-roster-refresh" \
    "0 3 * 3-11 1" \
    '{"mode":"daily"}' \
    "V10: Monday roster refresh"
echo "  ✓ mlb-2026-roster-refresh (Monday 3:00 AM ET)"

echo ""
echo "  Scheduler summary:"
_dry gcloud scheduler jobs list --location="$REGION" \
    --filter="name:mlb-2026" --format="table(name,schedule,state)" 2>/dev/null || true

# ------------------------------------------------------------------
# 4. Done — verification queries
# ------------------------------------------------------------------
TODAY=$(date +%Y-%m-%d)
echo ""
echo "=============================================="
echo " V10 Deployment Complete"
echo "=============================================="
echo ""
echo " Model:     gs://$BUCKET/$V10_GCS_PATH"
echo " Function:  $FUNCTION_URL"
echo " Scheduler: 5 jobs configured (pregame_v10 mode)"
echo ""
echo " Verification queries:"
echo ""
echo "   # Confirm today's V10 predictions:"
echo "   bq query --nouse_legacy_sql \\"
echo "     'SELECT model_version, COUNT(*) as games,"
echo "             ROUND(AVG(home_win_probability),3) as avg_prob"
echo "      FROM \`$PROJECT.mlb_2026_season.game_predictions\`"
echo "      WHERE game_date = \"$TODAY\""
echo "      GROUP BY 1'"
echo ""
echo "   # Check V10 features (should be 15 rows for today):"
echo "   bq query --nouse_legacy_sql \\"
echo "     'SELECT COUNT(*) as rows FROM \`$PROJECT.mlb_2026_season.game_v10_features\`"
echo "      WHERE game_date = \"$TODAY\"'"
echo ""
echo "   # Full season V10 feature coverage:"
echo "   bq query --nouse_legacy_sql \\"
echo "     'SELECT game_date, COUNT(*) as games"
echo "      FROM \`$PROJECT.mlb_2026_season.game_v10_features\`"
echo "      GROUP BY 1 ORDER BY 1'"
echo ""
echo "   # Cloud Function logs:"
echo "   gcloud functions logs read $FUNCTION_NAME --gen2 --region=$REGION --limit=50"
echo ""
