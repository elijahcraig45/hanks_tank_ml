#!/bin/bash
set -euo pipefail
#
# Deploy the college football weekly pipeline.
#
# Third independent Cloud Function, alongside mlb-2026-daily-pipeline and
# nfl-weekly-pipeline. Each has its own source tree, dependencies and schedule, so a
# deploy of one can never break another.
#
# src/cfb imports two modules from src/nfl (features.py, train_nfl_models.py) — the
# shared football core. Rather than deploying all of src/ (which would drag in the MLB
# pipeline and pybaseball), this stages exactly those two files into a temp dir next to
# src/cfb and deploys that.
#
# Usage:
#   ./deploy_cfb.sh
#   ./deploy_cfb.sh --only-scheduler
#   ./deploy_cfb.sh --dry-run

PROJECT="hankstank"
REGION="us-central1"
FUNCTION_NAME="cfb-weekly-pipeline"
RUNTIME="python312"
ENTRY_POINT="cfb_pipeline"
MEMORY="2048MB"
TIMEOUT="540s"
SERVICE_ACCOUNT="$PROJECT@appspot.gserviceaccount.com"
SCHEDULER_TZ="America/New_York"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFB_DIR="$(cd "$SCRIPT_DIR/../../../src/cfb" && pwd)"
NFL_DIR="$(cd "$SCRIPT_DIR/../../../src/nfl" && pwd)"
SRC_DIR="$(cd "$SCRIPT_DIR/../../../src" && pwd)"

# CollegeFootballData key, mounted from Secret Manager as an env var rather than read
# through the SDK: it keeps the local and deployed code paths identical and adds no
# dependency to a cold start. The App Engine backend reads the same secret.
CFBD_SECRET="${CFBD_SECRET:-cfbd-api-key}"

DRY_RUN=false
ONLY_SCHEDULER=false
for arg in "$@"; do
    case $arg in
        --dry-run)        DRY_RUN=true ;;
        --only-scheduler) ONLY_SCHEDULER=true ;;
    esac
done

_dry() { if [ "$DRY_RUN" = true ]; then echo "  [DRY RUN] $*"; else "$@"; fi }

echo "=============================================="
echo " CFB Weekly Pipeline — deploy"
echo " Function: $FUNCTION_NAME  (FBS + FCS)"
[ "$DRY_RUN" = true ] && echo " MODE:     DRY RUN"
echo "=============================================="

gcloud config set project "$PROJECT" --quiet

STAGE=""
# Fail before deploying rather than at 6 AM on a Sunday: a missing secret or a missing
# IAM binding both surface as an unkeyed function that silently skips the cfbd mode.
if [ "$DRY_RUN" = false ]; then
    if ! gcloud secrets describe "$CFBD_SECRET" --project="$PROJECT" >/dev/null 2>&1; then
        cat >&2 <<EOF
ERROR: secret "$CFBD_SECRET" not found in project $PROJECT.

The cfbd mode needs it. Create it with:
  gcloud secrets create $CFBD_SECRET --project=$PROJECT --replication-policy=automatic
  printf '%s' '<key>' | gcloud secrets versions add $CFBD_SECRET --data-file=- --project=$PROJECT
EOF
        exit 1
    fi
    if ! gcloud secrets get-iam-policy "$CFBD_SECRET" --project="$PROJECT" \
            --format="value(bindings.members)" 2>/dev/null | grep -q "$SERVICE_ACCOUNT"; then
        cat >&2 <<EOF
ERROR: $SERVICE_ACCOUNT cannot read secret "$CFBD_SECRET".

  gcloud secrets add-iam-policy-binding $CFBD_SECRET --project=$PROJECT \\
    --member=serviceAccount:$SERVICE_ACCOUNT --role=roles/secretmanager.secretAccessor
EOF
        exit 1
    fi
fi

if [ "$ONLY_SCHEDULER" = false ]; then
    STAGE="$(mktemp -d)/cfb"
    mkdir -p "$STAGE"
    cp "$CFB_DIR"/*.py "$STAGE"/
    # The shared football core: the causal feature builder and the model/eval helpers.
    # models.py exists precisely so this does not have to drag in NFL's data loader.
    cp "$NFL_DIR/features.py" "$NFL_DIR/models.py" "$STAGE"/
    # features.py reads its default Elo constants from a module named `config`; in the
    # staged tree that name belongs to CFB, which defines the same constants. CFB
    # passes EloParams explicitly anyway, so the defaults are never what's used.
    cp "$CFB_DIR/cfb_config.py" "$STAGE/config.py"
    # Power rankings and stats stay real packages in the staged tree: they import each
    # other by package path (`from rankings import core`), which flattening would break.
    cp -R "$SRC_DIR/rankings" "$SRC_DIR/stats" "$STAGE"/
    # rankings.http is also reachable flat, for anything staged without the package.
    cp "$SRC_DIR/rankings/http.py" "$STAGE/http_transport.py"

    cat > "$STAGE/requirements.txt" <<'EOF'
functions-framework==3.*
google-cloud-bigquery==3.39.0
db-dtypes==1.5.0
pandas==2.3.3
numpy==2.4.0
pyarrow==22.0.0
scikit-learn==1.8.0
xgboost==3.1.3
requests==2.32.5
# Bradley-Terry power rankings build a sparse design matrix.
scipy==1.16.3
EOF

    echo ""
    echo "▸ Staged source: $STAGE"
    ls "$STAGE" | sed 's/^/    /'

    echo ""
    echo "▸ Deploying Cloud Function..."
    _dry gcloud functions deploy "$FUNCTION_NAME" \
        --gen2 --region="$REGION" --runtime="$RUNTIME" \
        --source="$STAGE" --entry-point="$ENTRY_POINT" \
        --trigger-http --no-allow-unauthenticated \
        --memory="$MEMORY" --timeout="$TIMEOUT" \
        --service-account="$SERVICE_ACCOUNT" \
        --set-env-vars="GCP_PROJECT=$PROJECT,CFB_DATASET=cfb_season,CFB_HIST_DATASET=cfb_historical" \
        --set-secrets="CFBD_API_KEY=$CFBD_SECRET:latest" \
        --quiet
fi

FUNCTION_URL="https://$REGION-$PROJECT.cloudfunctions.net/$FUNCTION_NAME"
if [ "$DRY_RUN" = false ] && [ "$ONLY_SCHEDULER" = false ]; then
    FUNCTION_URL=$(gcloud functions describe "$FUNCTION_NAME" --gen2 --region="$REGION" \
        --format="value(serviceConfig.uri)" 2>/dev/null || echo "$FUNCTION_URL")
fi
echo "  URL: $FUNCTION_URL"

_sched() {
    local JOB="$1" SCHEDULE="$2" BODY="$3" DESC="$4"
    _dry gcloud scheduler jobs delete "$JOB" --location="$REGION" --quiet 2>/dev/null || true
    _dry gcloud scheduler jobs create http "$JOB" \
        --location="$REGION" --schedule="$SCHEDULE" --uri="$FUNCTION_URL" \
        --http-method=POST --headers="Content-Type=application/json" \
        --message-body="$BODY" --time-zone="$SCHEDULER_TZ" \
        --oidc-service-account-email="$SERVICE_ACCOUNT" \
        --oidc-token-audience="$FUNCTION_URL" \
        --attempt-deadline="$TIMEOUT" --description="$DESC" --quiet
}

echo ""
echo "▸ Scheduler (Aug–Jan; CFB season through the playoff)..."

# Sunday 6 AM ET — Saturday slate is final, plus Thursday/Friday games.
_sched "cfb-weekly-ingest" "0 6 * 8-12,1 0" \
    '{"mode":"ingest"}' "CFB: refresh FBS+FCS games from ESPN"
echo "  ✓ cfb-weekly-ingest (Sun 6:00 AM ET)"

# Tuesday 6 AM ET — predict the coming Saturday, both divisions.
_sched "cfb-weekly-predict" "0 6 * 8-12,1 2" \
    '{"mode":"predict_next"}' "CFB: predict the next unplayed week, FBS and FCS"
echo "  ✓ cfb-weekly-predict (Tue 6:00 AM ET)"

# Sunday 7 AM ET — an hour after the ingest, so the games and rankings it produces are
# already on the record. Its own job rather than chained onto the ingest because the
# CollegeFootballData work is a dozen HTTP calls, an 85-column flatten and a 14,000-row
# pivot; a timeout inside the ingest invocation would cost the games load that
# everything else depends on. Same reasoning, and the same one-hour offset, as
# nfl-weekly-score.
_sched "cfb-weekly-cfbd" "0 7 * 8-12,1 0" \
    '{"mode":"cfbd"}' "CFB: CollegeFootballData advanced stats, players and lines"
echo "  ✓ cfb-weekly-cfbd (Sun 7:00 AM ET)"

# Rankings and ESPN stats are refreshed inside the Sunday ingest, not on their own jobs:
# they are derived from the games it loads, so chaining them makes the ordering
# structural instead of a race between two cron entries.
echo "  · rankings + ESPN stats refresh inside cfb-weekly-ingest"

echo ""
echo "=============================================="
echo " Done. Verify with:"
echo "   bq query --use_legacy_sql=false 'SELECT division, COUNT(*) FROM \`$PROJECT.cfb_season.game_predictions\` GROUP BY 1'"
echo "=============================================="
