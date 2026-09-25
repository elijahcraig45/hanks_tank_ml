#!/bin/bash
set -euo pipefail
#
# Deploy the plate-appearance Monte Carlo simulator (pa_sim_v1) in SHADOW mode.
#
# Shadow mode means: the simulator writes `mlb_2026_season.game_predictions_sim`
# and NEVER `game_predictions`. Deploying this therefore cannot change what
# hankstank.com serves. Promotion is a separate, deliberate step (see PROMOTION below).
#
# What this does:
#   1. Validates prerequisites (gcloud auth, project, dataset)
#   2. Creates the shadow table if absent (partitioned by game_date, clustered by game_pk)
#   3. Redeploys the Cloud Function with the pa_sim package included
#   4. Optionally adds a Scheduler job running mode=pa_sim after the V10 pregame job
#   5. Verifies by invoking mode=pa_sim with dry_run
#
# Backtest results this ships on (2026, both windows, see research/backtest_2026):
#   - beats V10 on log-loss in BOTH windows (+0.00428 search, +0.00329 holdout)
#   - does NOT beat the 3-feature logistic baseline in either window
#   - PA model gate: +0.0157 nats vs league baseline on 183k held-out PAs
#   - run model gate: -0.094 run bias, r=0.104 on 4,856 team-game run totals
#   - Monte Carlo verified against the exact analytic chain to within 0.2 SE
#
# Usage:
#   ./deploy_pa_sim.sh --dry-run          # print every action, change nothing
#   ./deploy_pa_sim.sh --only-table       # create the shadow table only
#   ./deploy_pa_sim.sh --skip-scheduler   # deploy function, no Scheduler change
#   ./deploy_pa_sim.sh                    # full shadow deploy
#
# PROMOTION (do NOT do this casually):
#   The simulator does not currently beat the 3-feature baseline. Promoting it means
#   setting PA_SIM_TABLE=game_predictions and accepting that its rows compete with
#   V10's in the table the backend reads. Re-run research/backtest_2026/29_ensemble.py
#   on fresh data and require a win in BOTH windows first.

PROJECT="${PROJECT:-hankstank}"
REGION="${REGION:-us-central1}"
FUNCTION="${FUNCTION:-mlb-2026-daily-pipeline}"
DATASET="${DATASET:-mlb_2026_season}"
TABLE="${TABLE:-game_predictions_sim}"
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)/src"

DRY_RUN=0; ONLY_TABLE=0; SKIP_SCHED=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --only-table) ONLY_TABLE=1 ;;
    --skip-scheduler) SKIP_SCHED=1 ;;
    *) echo "unknown flag: $arg" >&2; exit 2 ;;
  esac
done

run() {
  if [[ $DRY_RUN -eq 1 ]]; then echo "  [dry-run] $*"; else echo "  + $*"; "$@"; fi
}

echo "=== pa_sim shadow deploy ==="
echo "project=$PROJECT region=$REGION function=$FUNCTION"
echo "shadow table=$PROJECT.$DATASET.$TABLE"
echo "source=$SRC_DIR"
[[ -d "$SRC_DIR/pa_sim" ]] || { echo "ERROR: $SRC_DIR/pa_sim not found" >&2; exit 1; }

echo
echo "--- 1. prerequisites ---"
if ! gcloud auth list --filter=status:ACTIVE --format='value(account)' | grep -q .; then
  echo "ERROR: no active gcloud account. Run: gcloud auth login" >&2; exit 1
fi
echo "  active account: $(gcloud auth list --filter=status:ACTIVE --format='value(account)' | head -1)"

echo
echo "--- 2. shadow table ---"
DDL="CREATE TABLE IF NOT EXISTS \`$PROJECT.$DATASET.$TABLE\` (
  game_pk INT64, game_date DATE,
  home_team_id INT64, away_team_id INT64,
  home_team_name STRING, away_team_name STRING,
  home_starter_id INT64, away_starter_id INT64,
  home_win_probability FLOAT64, away_win_probability FLOAT64,
  predicted_winner STRING, confidence_tier STRING, model_version STRING,
  n_episodes INT64, mean_home_runs FLOAT64, mean_away_runs FLOAT64,
  p_extra_innings FLOAT64, park_alpha FLOAT64,
  game_time_utc TIMESTAMP, predicted_at TIMESTAMP
)
PARTITION BY game_date
CLUSTER BY game_pk"
if [[ $DRY_RUN -eq 1 ]]; then
  echo "  [dry-run] bq query --use_legacy_sql=false '<CREATE TABLE IF NOT EXISTS ...>'"
else
  echo "$DDL" | bq query --use_legacy_sql=false --project_id="$PROJECT" --format=none
  echo "  shadow table ready"
fi
[[ $ONLY_TABLE -eq 1 ]] && { echo "--only-table: done"; exit 0; }

echo
echo "--- 3. run the test suite before shipping ---"
if [[ $DRY_RUN -eq 1 ]]; then
  echo "  [dry-run] pytest tests/test_pa_sim.py -q"
else
  ( cd "$(dirname "$SRC_DIR")" && .venv/bin/python -m pytest tests/test_pa_sim.py -q ) \
    || { echo "ERROR: simulator tests failed; refusing to deploy" >&2; exit 1; }
fi

echo
echo "--- 4. redeploy Cloud Function ---"
run gcloud functions deploy "$FUNCTION" \
  --gen2 --region="$REGION" --project="$PROJECT" \
  --runtime=python312 --source="$SRC_DIR" \
  --entry-point=daily_pipeline --trigger-http \
  --memory=2Gi --cpu=2 --timeout=540s \
  --set-env-vars="PA_SIM_TABLE=$TABLE,PA_SIM_EPISODES=1000,PA_SIM_CACHE=/tmp/pa_sim"

if [[ $SKIP_SCHED -eq 0 ]]; then
  echo
  echo "--- 5. Scheduler job (shadow, 07:10 ET, months 3-11) ---"
  URI="$(gcloud functions describe "$FUNCTION" --gen2 --region="$REGION" \
          --project="$PROJECT" --format='value(serviceConfig.uri)' 2>/dev/null || echo '<function-uri>')"
  run gcloud scheduler jobs create http mlb-2026-pa-sim-shadow \
    --location="$REGION" --project="$PROJECT" \
    --schedule="10 7 * 3-11 *" --time-zone="America/New_York" \
    --uri="$URI" --http-method=POST \
    --message-body='{"mode":"pa_sim","n_episodes":1000}' \
    --oidc-service-account-email="$(gcloud config get-value account 2>/dev/null)" \
    || echo "  (job may already exist; use 'gcloud scheduler jobs update http' to change it)"
fi

echo
echo "--- 6. verify with a dry-run invocation ---"
if [[ $DRY_RUN -eq 1 ]]; then
  echo "  [dry-run] gcloud functions call $FUNCTION --data '{\"mode\":\"pa_sim\",\"dry_run\":true}'"
else
  gcloud functions call "$FUNCTION" --gen2 --region="$REGION" --project="$PROJECT" \
    --data '{"mode":"pa_sim","dry_run":true}' || echo "  invoke failed - check logs"
fi

echo
echo "=== done. Shadow only: game_predictions is untouched. ==="
echo "Compare the two models with:"
echo "  bq query --use_legacy_sql=false 'SELECT s.game_pk, s.home_win_probability AS sim,"
echo "    p.home_win_probability AS v10, IF(g.home_score>g.away_score,1,0) AS home_win"
echo "    FROM \`$PROJECT.$DATASET.$TABLE\` s"
echo "    JOIN \`$PROJECT.$DATASET.game_predictions\` p USING (game_pk)"
echo "    JOIN \`$PROJECT.$DATASET.games\` g USING (game_pk) WHERE g.status LIKE \"%Final%\"'"
