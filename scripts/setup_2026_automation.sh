#!/bin/bash
#
# Setup 2026 Season Automation
#
# Run this once to set up the complete 2026 season tracking system

set -e

echo "========================================"
echo "🚀 2026 Season Automation Setup"
echo "========================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required"
    exit 1
fi

echo "✓ Python 3 found: $(python3 --version)"
echo ""

# Install dependencies
echo "📦 Installing Python dependencies..."
pip3 install -r requirements.txt --quiet
echo "✓ Dependencies installed"
echo ""

# Create data directories
echo "📁 Creating data directories..."
mkdir -p data/2026/{games,stats,statcast}
mkdir -p logs/2026
echo "✓ Directories created"
echo ""

# Check GCP credentials
echo "🔐 Checking GCP credentials..."
if [ -z "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    DEFAULT_CREDS="$HOME/.config/gcloud/application_default_credentials.json"
    if [ -f "$DEFAULT_CREDS" ]; then
        export GOOGLE_APPLICATION_CREDENTIALS="$DEFAULT_CREDS"
        echo "✓ Using default credentials: $DEFAULT_CREDS"
    else
        echo "⚠️  No GCP credentials found"
        echo "   Run: gcloud auth application-default login"
        echo "   Or set: export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json"
    fi
else
    echo "✓ Credentials configured: $GOOGLE_APPLICATION_CREDENTIALS"
fi
echo ""

# Make scripts executable
echo "🔧 Making scripts executable..."
chmod +x scripts/run_daily_2026.sh
chmod +x scripts/run_full_validation.sh
chmod +x scripts/validate_2026.sh
echo "✓ Scripts ready"
echo ""

# Test connection to BigQuery
echo "🔍 Testing BigQuery connection..."
if python3 -c "from google.cloud import bigquery; client = bigquery.Client(project='hankstank'); list(client.list_datasets(max_results=1))" 2>/dev/null; then
    echo "✓ BigQuery connection successful"
else
    echo "⚠️  BigQuery connection failed - check credentials"
fi
echo ""

# Display next steps
echo "========================================"
echo "✅ Setup Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Test the pipeline:"
echo "   python3 src/season_2026_pipeline.py --dry-run"
echo ""
echo "2. Backfill season data (when season starts):"
echo "   python3 src/season_2026_pipeline.py --backfill --start 2026-03-27"
echo ""
echo "3. Run daily collection:"
echo "   ./scripts/run_daily_2026.sh"
echo ""
echo "4. Schedule daily automation (cron):"
echo "   crontab -e"
echo "   Add: 0 4 * * * cd $(pwd) && ./run_daily_2026.sh"
echo ""
echo "5. Monitor:"
echo "   tail -f logs/2026/daily_$(date +%Y%m%d).log"
echo ""
echo "========================================"
