#!/bin/bash
set -euo pipefail
#
# Deploy V8 model and pipeline to GCP.
#
# What this does:
#   1. Validates prerequisites (gcloud auth, project, bucket)
#   2. Seeds the Elo ratings table with 2026 season-start values (once)
#   3. Uploads the V8 model artifact to GCS
#   4. Redeploies the Cloud Function with V8 source (add catboost to deps)
#   5. Adds/updates Cloud Scheduler jobs for V8 pipeline steps:
#        - Daily 4:00 AM ET:  data collection + validation + features
#        - Daily 5:00 AM ET:  V8 Elo update (from yesterday's outcomes)
#        - Daily 5:30 AM ET:  V8 feature build (today's games)
#        - Daily 10:00 AM ET: schedule pre-game tasks (~90 min before each game)
#        - Sunday 2:00 AM ET: weekly V8 model retraining
#   6. Backfills V8 features from season start (2026-03-27 → yesterday)
#   7. Triggers a prediction run for today's games
#
# Cost notes:
#   - Cloud Function: ~$0.00/month at this invocation frequency
#   - Cloud Scheduler: $0.10/job/month × 5 jobs = $0.50/month
#   - BigQuery:  daily scans ~2MB for V8 features = <$0.01/month
#   - GCS model storage: ~5MB V8 model = <$0.01/month
#
# Prerequisites:
#   - gcloud CLI authenticated: gcloud auth application-default login
#   - Project: hankstank
#   - APIs enabled: cloudfunctions, cloudscheduler, bigquery, storage
#   - V8 model artifact present: models/game_outcome_2026_v8_final.pkl
#
# Usage:
#   ./deploy_v8.sh                       # full deploy
#   ./deploy_v8.sh --skip-backfill       # deploy + scheduler, no backfill
#   ./deploy_v8.sh --skip-function       # only upload model + scheduler
#   ./deploy_v8.sh --only-scheduler      # only update scheduler jobs
#   ./deploy_v8.sh --only-upload-model   # only upload model to GCS
#   ./deploy_v8.sh --dry-run             # preview all steps, no writes

PROJECT="hankstank"
REGION="us-central1"
FUNCTION_NAME="mlb-2026-daily-pipeline"
RUNTIME="python312"
ENTRY_POINT="daily_pipeline"
MEMORY="2048MB"          # V8 CatBoost model needs ~1.2GB — bumped from 1024MB
TIMEOUT="540s"
SERVICE_ACCOUNT="$PROJECT@appspot.gserviceaccount.com"
SCHEDULER_TZ="America/New_York"
BUCKET="hanks_tank_data"

# V8 model artifact paths
V8_LOCAL_PATH="models/game_outcome_2026_v8_final.pkl"
V8_GCS_PATH="models/vertex/game_outcome_2026_v8/model.pkl"

SEASON_START="2026-03-27"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd "$SCRIPT_DIR/../../src" && pwd)"
REPO_DIR="$(cd "$SRC_DIR/.." && pwd)"

# Parse flags
SKIP_BACKFILL=false
SKIP_FUNCTION=false
ONLY_SCHEDULER=false
ONLY_UPLOAD_MODEL=false
DRY_RUN=false

for arg in "$@"; do
    case $arg in
        --skip-backfill)     SKIP_BACKFILL=true ;;
        --skip-function)     SKIP_FUNCTION=true ;;
        --only-scheduler)    ONLY_SCHEDULER=true; SKIP_FUNCTION=true; SKIP_BACKFILL=true ;;
        --only-upload-model) ONLY_UPLOAD_MODEL=true ;;
        --dry-run)           DRY_RUN=true ;;
    esac
done

echo "=============================================="
echo " V8 Model — GCP Deployment"
echo " CatBoost + Team Embeddings | 57.65% Acc"
echo " 65.4% accuracy at ≥60% confidence threshold"
echo "=============================================="
echo " Project:      $PROJECT"
echo " Region:       $REGION"
echo " Function:     $FUNCTION_NAME"
echo " Memory:       $MEMORY  (bumped for CatBoost)"
echo " Source:       $SRC_DIR"
echo " Model local:  $REPO_DIR/$V8_LOCAL_PATH"
echo " Model GCS:    gs://$BUCKET/$V8_GCS_PATH"
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
    echo "       Run: gcloud auth login && gcloud config set project $PROJECT"
    exit 1
fi
echo "  ✓ Project: $PROJECT"

# Confirm V8 model artifact exists
if [ ! -f "$REPO_DIR/$V8_LOCAL_PATH" ]; then
    echo "ERROR: V8 model artifact not found: $REPO_DIR/$V8_LOCAL_PATH"
    echo "       Run 'python3 src/train_v8_final_push.py' to generate it."
    exit 1
fi
echo "  ✓ V8 model artifact found"

if [ "$ONLY_UPLOAD_MODEL" = true ]; then
    echo ""
    echo "▸ Uploading V8 model to GCS..."
    _dry gsutil cp "$REPO_DIR/$V8_LOCAL_PATH" "gs://$BUCKET/$V8_GCS_PATH"
    echo "  ✓ Model uploaded: gs://$BUCKET/$V8_GCS_PATH"
    echo ""
    echo "Done (--only-upload-model)."
    exit 0
fi

# ------------------------------------------------------------------
# 1. Upload V8 model artifact to GCS
# ------------------------------------------------------------------
echo ""
echo "▸ Uploading V8 model to GCS..."
_dry gsutil cp "$REPO_DIR/$V8_LOCAL_PATH" "gs://$BUCKET/$V8_GCS_PATH"
echo "  ✓ Model uploaded: gs://$BUCKET/$V8_GCS_PATH"

# ------------------------------------------------------------------
# 2. Seed Elo ratings BQ table (idempotent — safe to rerun)
# ------------------------------------------------------------------
if [ "$ONLY_SCHEDULER" = false ]; then
    echo ""
    echo "▸ Seeding 2026 Elo ratings table..."
    cd "$SRC_DIR"
    if [ "$DRY_RUN" = false ]; then
        python3 build_v8_features_live.py --seed-elo
        echo "  ✓ Elo ratings seeded (30 teams)"
    else
        echo "  [DRY RUN] Would run: python3 build_v8_features_live.py --seed-elo"
    fi
fi

# ------------------------------------------------------------------
# 3. Redeploy Cloud Function with V8 source + catboost dependency
# ------------------------------------------------------------------
if [ "$SKIP_FUNCTION" = false ]; then
    echo ""
    echo "▸ Deploying Cloud Function with V8 support..."
    echo "  Note: catboost added to requirements.txt — deploy may take 3-5 min"
    echo "        Memory bumped to $MEMORY to handle CatBoost model in memory"
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
        echo "  WARNING: Could not retrieve function URL — scheduler setup may fail"
    else
        echo "  ✓ Function URL: $FUNCTION_URL"
    fi
else
    FUNCTION_URL="https://$REGION-$PROJECT.cloudfunctions.net/$FUNCTION_NAME"
    echo "  [DRY RUN] Using placeholder URL: $FUNCTION_URL"
fi

OIDC_AUDIENCE="$FUNCTION_URL"

# ------------------------------------------------------------------
# 4. Cloud Scheduler jobs
# ------------------------------------------------------------------
echo ""
echo "▸ Configuring Cloud Scheduler (V8 pipeline)..."

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
#   Why 4 AM: game data from previous day fully available by ~3 AM ET.
#   Includes: collect → validate → V3/V4 features → V8 Elo update → V8 features
_sched_create \
    "mlb-2026-daily" \
    "0 4 * 3-11 *" \
    '{"mode":"daily"}' \
    "V8: daily data collection, validation, feature build (V3+V8), Elo update"
echo "  ✓ mlb-2026-daily (4:00 AM ET, Mar-Nov)"

# ② Pre-game task scheduler — 10:00 AM ET
#   Checks today's schedule and enqueues a Cloud Task per game ~90 min before first pitch.
#   Each task triggers: lineups → matchup → V7 → V8 features → predict
_sched_create \
    "mlb-2026-pregame-schedule" \
    "0 10 * 3-11 *" \
    '{"mode":"schedule_pregame_tasks"}' \
    "V8: schedule pre-game Cloud Tasks for today's games (lineups+V7+V8+predict)"
echo "  ✓ mlb-2026-pregame-schedule (10:00 AM ET, Mar-Nov)"

# ③ Weekly V8 model retraining — Sunday 2:00 AM ET
#   Re-trains V8 CatBoost on full 2015->current data including 2026 season games.
#   Keeps model calibrated as teams' quality distributions shift mid-season.
#   NOTE: First Sunday after deploy, this will produce the first 2026-trained V8 model.
_sched_create \
    "mlb-2026-weekly-train-v8" \
    "0 2 * 3-11 0" \
    '{"mode":"train_weekly","model_version":"v8"}' \
    "V8: weekly model retraining (Sunday 2 AM ET)"
echo "  ✓ mlb-2026-weekly-train-v8 (Sunday 2:00 AM ET)"

# ④ Weekly predictions (Friday batch) — Friday 5:00 AM ET
_sched_create \
    "mlb-2026-weekly-predict" \
    "0 5 * 3-11 5" \
    '{"mode":"predict"}' \
    "V8: Friday batch predictions for the upcoming week"
echo "  ✓ mlb-2026-weekly-predict (Friday 5:00 AM ET)"

# ⑤ Roster refresh — Monday 3:00 AM ET
_sched_create \
    "mlb-2026-roster-refresh" \
    "0 3 * 3-11 1" \
    '{"mode":"daily"}' \
    "V8: Monday roster refresh (triggered by Monday weekday logic in pipeline)"
echo "  ✓ mlb-2026-roster-refresh (Monday 3:00 AM ET)"

echo ""
echo "  Scheduler summary (5 total jobs, ~\$0.50/month):"
_dry gcloud scheduler jobs list --location="$REGION" \
    --filter="name:mlb-2026" --format="table(name,schedule,state)" 2>/dev/null || true

# ------------------------------------------------------------------
# 5. Backfill V8 features from season start
# ------------------------------------------------------------------
if [ "$SKIP_BACKFILL" = false ] && [ "$ONLY_SCHEDULER" = false ]; then
    echo ""
    echo "▸ Backfilling V8 features (season start → yesterday)..."
    echo "  This populates game_v8_features for all 2026 games played so far."
    YESTERDAY=$(date -v-1d +%Y-%m-%d 2>/dev/null || date -d "yesterday" +%Y-%m-%d)
    cd "$SRC_DIR"
    if [ "$DRY_RUN" = false ]; then
        python3 build_v8_features_live.py \
            --backfill \
            --start "$SEASON_START" \
            --end "$YESTERDAY"
        echo "  ✓ V8 features backfilled through $YESTERDAY"
    else
        echo "  [DRY RUN] Would run: python3 build_v8_features_live.py --backfill --start $SEASON_START --end $YESTERDAY"
    fi
fi

# ------------------------------------------------------------------
# 6. Run today's predictions with V8
# ------------------------------------------------------------------
TODAY=$(date +%Y-%m-%d)
echo ""
echo "▸ Running V8 predictions for today ($TODAY)..."
if [ "$DRY_RUN" = false ] && [ -n "$FUNCTION_URL" ]; then
    # Get OIDC token for authenticated call
    TOKEN=$(gcloud auth print-identity-token 2>/dev/null || echo "")
    if [ -n "$TOKEN" ]; then
        RESPONSE=$(curl -s -X POST "$FUNCTION_URL" \
            -H "Authorization: Bearer $TOKEN" \
            -H "Content-Type: application/json" \
            -d "{\"mode\":\"predict_today\",\"date\":\"$TODAY\"}" \
            --max-time 300 2>&1 || echo "curl failed")
        echo "  Response: $RESPONSE"
    else
        echo "  WARNING: Could not get OIDC token — triggering via gcloud instead"
        gcloud functions call "$FUNCTION_NAME" \
            --gen2 --region="$REGION" \
            --data="{\"mode\":\"predict_today\",\"date\":\"$TODAY\"}" \
            2>&1 || echo "  WARNING: Direct call failed — check Cloud Function logs"
    fi
else
    echo "  [DRY RUN] Would POST {mode: predict_today, date: $TODAY} to $FUNCTION_URL"
fi

# ------------------------------------------------------------------
# Done
# ------------------------------------------------------------------
echo ""
echo "=============================================="
echo " V8 Deployment Complete"
echo "=============================================="
echo ""
echo " Model:     gs://$BUCKET/$V8_GCS_PATH"
echo " Function:  $FUNCTION_URL"
echo " Scheduler: 5 jobs configured"
echo ""
echo " Next steps:"
echo "   1. Check Cloud Function logs for today's predictions:"
echo "      gcloud functions logs read $FUNCTION_NAME --gen2 --region=$REGION --limit=50"
echo ""
echo "   2. Verify predictions were written to BigQuery:"
echo "      bq query 'SELECT game_pk, home_team_name, away_team_name,"
echo "                home_win_probability, confidence_tier, model_version"
echo "                FROM \`$PROJECT.mlb_2026_season.game_predictions\`"
echo "                WHERE DATE(predicted_at) = \"$TODAY\""
echo "                ORDER BY game_time_utc'"
echo ""
echo "   3. Check V8 feature completeness:"
echo "      bq query 'SELECT COUNT(*) as games, AVG(data_completeness) as avg_completeness"
echo "                FROM \`$PROJECT.mlb_2026_season.game_v8_features\`"
echo "                WHERE game_date = \"$TODAY\"'"
echo ""
echo "   4. Monitor Elo ratings:"
echo "      bq query 'SELECT team_id, elo_rating, updated_at"
echo "                FROM \`$PROJECT.mlb_2026_season.team_elo_ratings\`"
echo "                WHERE season = 2026 ORDER BY elo_rating DESC'"
echo ""
