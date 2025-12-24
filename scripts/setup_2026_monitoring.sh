#!/bin/bash
#
# Setup Automated 2026 Season Data Quality Monitoring
# 
# This script sets up daily automated validation for the 2026 MLB season.
# Run once at the beginning of the 2026 season to configure cron job.
#
# Usage:
#   ./setup_2026_monitoring.sh
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VALIDATE_SCRIPT="$SCRIPT_DIR/validate_2026.sh"
LOG_DIR="$SCRIPT_DIR/logs"

# Color output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  2026 MLB Season - Data Quality Monitoring Setup${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""

# Create logs directory
if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR"
    echo -e "${GREEN}✅ Created logs directory: $LOG_DIR${NC}"
fi

# Make validation script executable
if [ -f "$VALIDATE_SCRIPT" ]; then
    chmod +x "$VALIDATE_SCRIPT"
    echo -e "${GREEN}✅ Made validate_2026.sh executable${NC}"
else
    echo -e "${YELLOW}⚠️  Warning: validate_2026.sh not found at $VALIDATE_SCRIPT${NC}"
fi

# Create cron job entry
CRON_JOB="0 6 * * * cd $SCRIPT_DIR && $VALIDATE_SCRIPT >> $LOG_DIR/validation_2026.log 2>&1"

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Cron Job Configuration${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""
echo "The following cron job will run daily at 6:00 AM:"
echo ""
echo -e "${YELLOW}$CRON_JOB${NC}"
echo ""
echo "This will:"
echo "  • Validate 2026 data every morning at 6 AM"
echo "  • Append results to logs/validation_2026.log"
echo "  • Alert on critical data quality issues"
echo ""

# Ask user if they want to install the cron job
read -p "Do you want to install this cron job? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Check if cron job already exists
    if crontab -l 2>/dev/null | grep -q "validate_2026.sh"; then
        echo -e "${YELLOW}⚠️  Cron job already exists. Skipping installation.${NC}"
    else
        # Add to crontab
        (crontab -l 2>/dev/null; echo "$CRON_JOB") | crontab -
        echo -e "${GREEN}✅ Cron job installed successfully!${NC}"
    fi
    echo ""
    echo "View current cron jobs with:"
    echo "  crontab -l"
    echo ""
    echo "Remove this cron job with:"
    echo "  crontab -e  # Then delete the line containing 'validate_2026.sh'"
else
    echo ""
    echo -e "${YELLOW}Cron job NOT installed.${NC}"
    echo ""
    echo "To manually add later, run:"
    echo "  crontab -e"
    echo ""
    echo "Then add this line:"
    echo -e "${YELLOW}$CRON_JOB${NC}"
fi

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Manual Validation${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""
echo "You can always run validation manually:"
echo ""
echo "  # Validate 2026 data:"
echo "  ./validate_2026.sh"
echo ""
echo "  # Validate specific year:"
echo "  python3 data_validation.py --year 2026"
echo ""
echo "  # Validate all historical data:"
echo "  ./run_full_validation.sh"
echo ""

echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✅ 2026 monitoring setup complete!${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""
