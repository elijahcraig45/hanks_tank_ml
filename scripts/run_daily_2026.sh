#!/bin/bash
#
# 2026 Season Daily Automation
#
# Schedule this with cron to run daily during the 2026 season:
# 0 4 * * * /path/to/hanks_tank_ml/run_daily_2026.sh
#
# Runs at 4 AM daily to:
# 1. Collect yesterday's completed games
# 2. Sync to BigQuery
# 3. Validate data quality
# 4. Send alert if issues found

set -e  # Exit on error

# Change to script directory
cd "$(dirname "$0")"

# Set up environment
export GOOGLE_APPLICATION_CREDENTIALS="${GOOGLE_APPLICATION_CREDENTIALS:-$HOME/.config/gcloud/application_default_credentials.json}"

# Logging
LOG_DIR="logs/2026"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/daily_$(date +%Y%m%d).log"

echo "=======================================" | tee -a "$LOG_FILE"
echo "2026 Season Daily Pipeline" | tee -a "$LOG_FILE"
echo "Started: $(date)" | tee -a "$LOG_FILE"
echo "=======================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Run pipeline
python3 ../src/season_2026_pipeline.py 2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

echo "" | tee -a "$LOG_FILE"
echo "=======================================" | tee -a "$LOG_FILE"
echo "Finished: $(date)" | tee -a "$LOG_FILE"
echo "Exit Code: $EXIT_CODE" | tee -a "$LOG_FILE"
echo "=======================================" | tee -a "$LOG_FILE"

# Alert on failure (optional - configure email or slack)
if [ $EXIT_CODE -ne 0 ]; then
    echo "❌ Pipeline failed with exit code $EXIT_CODE" | tee -a "$LOG_FILE"
    
    # Uncomment to send alert:
    # echo "2026 pipeline failed - see $LOG_FILE" | mail -s "MLB Pipeline Alert" your@email.com
fi

exit $EXIT_CODE
