#!/bin/bash
# Cloud Scheduler Setup - Create daily update jobs
# Run this once to set up the schedule (modify project and region as needed)

PROJECT="hankstank"  # Your actual GCP project
REGION="us-central1"

echo "Creating Cloud Scheduler jobs for daily MLB data updates..."
echo "=================================================="

# Job 1: Fetch statcast data (6:00 AM UTC)
echo "Creating job: fetch-statcast-2026-daily"
gcloud scheduler jobs create http fetch-statcast-2026-daily \
  --schedule="0 6 * * *" \
  --timezone="UTC" \
  --uri="https://${REGION}-${PROJECT}.cloudfunctions.net/update_statcast_2026" \
  --http-method=POST \
  --oidc-service-account-email="cloud-functions@${PROJECT}.iam.gserviceaccount.com" \
  --project="${PROJECT}" \
  --message-body='{}' \
  || echo "Job may already exist"

# Job 2: Fetch pitcher stats (6:30 AM UTC)
echo "Creating job: fetch-pitcher-stats-2026-daily"
gcloud scheduler jobs create http fetch-pitcher-stats-2026-daily \
  --schedule="30 6 * * *" \
  --timezone="UTC" \
  --uri="https://${REGION}-${PROJECT}.cloudfunctions.net/update_pitcher_stats_2026" \
  --http-method=POST \
  --oidc-service-account-email="cloud-functions@${PROJECT}.iam.gserviceaccount.com" \
  --project="${PROJECT}" \
  --message-body='{}' \
  || echo "Job may already exist"

# Job 3: Rebuild V7 features (8:00 AM UTC - after all data loaded)
echo "Creating job: rebuild-v7-features-daily"
gcloud scheduler jobs create http rebuild-v7-features-daily \
  --schedule="0 8 * * *" \
  --timezone="UTC" \
  --uri="https://${REGION}-${PROJECT}.cloudfunctions.net/rebuild_v7_features" \
  --http-method=POST \
  --oidc-service-account-email="cloud-functions@${PROJECT}.iam.gserviceaccount.com" \
  --project="${PROJECT}" \
  --message-body='{}' \
  || echo "Job may already exist"

# Job 4: Generate predictions (12:00 PM UTC - after lineups confirmed)
echo "Creating job: predict-today-games-daily"
gcloud scheduler jobs create http predict-today-games-daily \
  --schedule="0 12 * * *" \
  --timezone="UTC" \
  --uri="https://${REGION}-${PROJECT}.cloudfunctions.net/predict_today_games" \
  --http-method=POST \
  --oidc-service-account-email="cloud-functions@${PROJECT}.iam.gserviceaccount.com" \
  --project="${PROJECT}" \
  --message-body='{}' \
  || echo "Job may already exist"

echo ""
echo "=================================================="
echo "Cloud Scheduler jobs configured!"
echo ""
echo "View jobs:"
echo "  gcloud scheduler jobs list --project=${PROJECT}"
echo ""
echo "Manually trigger a job:"
echo "  gcloud scheduler jobs run fetch-statcast-2026-daily --project=${PROJECT}"
echo ""
echo "View job logs:"
echo "  gcloud functions logs read update_statcast_2026 --project=${PROJECT} --limit 50"
