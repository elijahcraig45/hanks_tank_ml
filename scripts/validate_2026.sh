#!/bin/bash
#
# Daily 2026 Season Data Quality Check
# Run this after your daily sync to ensure data quality
#
# Usage:
#   ./validate_2026.sh
#
# Exit codes:
#   0 = All checks passed
#   1 = Critical failures
#   2 = Warnings only
#

set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔍 MLB 2026 Season - Daily Data Quality Validation"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

cd "$(dirname "$0")"

# Run validation for 2026 data
echo "Running validation on 2026 season data..."
python3 ../src/data_validation.py --year 2026

EXIT_CODE=$?

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Validation PASSED - Data quality is good!"
elif [ $EXIT_CODE -eq 1 ]; then
    echo "❌ Validation FAILED - Critical issues found!"
    echo ""
    echo "To fix duplicates automatically:"
    echo "  python3 fix_data_issues.py --issue duplicates --year 2026 --execute"
elif [ $EXIT_CODE -eq 2 ]; then
    echo "⚠️  Validation PASSED with warnings"
    echo "Review warnings and determine if action needed"
fi

echo "Finished: $(date '+%Y-%m-%d %H:%M:%S')"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

exit $EXIT_CODE
