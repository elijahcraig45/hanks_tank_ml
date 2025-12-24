#!/bin/bash
#
# Full Historical Data Quality Validation
# Runs comprehensive checks across all MLB historical data (2015-2025)
#
# Usage:
#   ./run_full_validation.sh
#   ./run_full_validation.sh --report-only  # Generate summary from last run
#

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  MLB Historical Data Quality Validation (2015-2025)${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""
echo "📅 Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo "📂 Location: $SCRIPT_DIR"
echo ""

# Check if Python script exists
if [ ! -f "../src/data_validation.py" ]; then
    echo -e "${RED}❌ Error: data_validation.py not found${NC}"
    exit 1
fi

# Run validation
echo -e "${YELLOW}🔍 Running comprehensive validation...${NC}"
echo ""

if python3 ../src/data_validation.py --year 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025; then
    EXIT_CODE=$?
else
    EXIT_CODE=$?
fi

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"

# Interpret results
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✅ VALIDATION PASSED${NC}"
    echo -e "${GREEN}   All data quality checks passed successfully!${NC}"
    echo -e "${GREEN}   Data is ready for ML model training.${NC}"
elif [ $EXIT_CODE -eq 2 ]; then
    echo -e "${YELLOW}⚠️  VALIDATION PASSED WITH WARNINGS${NC}"
    echo -e "${YELLOW}   Some non-critical issues detected.${NC}"
    echo -e "${YELLOW}   Review warnings but data can be used for training.${NC}"
else
    echo -e "${RED}❌ VALIDATION FAILED${NC}"
    echo -e "${RED}   Critical data quality issues detected.${NC}"
    echo -e "${RED}   Fix issues before using data for ML training.${NC}"
fi

echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""

# Show latest report file
LATEST_REPORT=$(ls -t validation_report_*.json 2>/dev/null | head -1)
if [ -n "$LATEST_REPORT" ]; then
    echo "📊 Detailed report: $LATEST_REPORT"
    echo ""
    
    # Extract summary if jq is available
    if command -v jq &> /dev/null; then
        echo -e "${BLUE}Summary:${NC}"
        jq -r '
            "  Total Checks: \(.summary.total_checks)",
            "  ✅ Passed:    \(.summary.passed)",
            "  ⚠️  Warnings:  \(.summary.warnings)",
            "  ❌ Critical:  \(.summary.critical)"
        ' "$LATEST_REPORT"
    fi
fi

echo ""
exit $EXIT_CODE
