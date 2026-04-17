#!/usr/bin/env python3
"""
Upload / Refresh Statcast SP Percentile Data to GCS

Uploads the local Baseball Savant SP percentile rank parquet files to GCS so
the Cloud Function can load them at runtime.

Files:  data/v9/raw/statcast_sp_pct_YEAR.parquet
   →    gs://hanks_tank_data/data/statcast_sp_pct_YEAR.parquet

The V10 live feature builder checks local paths first, then GCS as fallback.
For the Cloud Function (no local files), GCS is the only source.

GCS freshness:
  - 2018–2025 files are historical/stable — upload once.
  - 2026 file should be refreshed WEEKLY as pitchers accumulate stats.
    The Cloud Function's weekly_refresh_sp mode (Sundays) calls this with
    --year <current_year> --force automatically.

Run once for initial setup:
    python scripts/upload_statcast_sp_to_gcs.py

Refresh in-season (called by Cloud Scheduler or manually):
    python scripts/upload_statcast_sp_to_gcs.py --year 2026 --force

Usage:
    upload_statcast_sp_to_gcs.py              # upload all available years (skip existing)
    upload_statcast_sp_to_gcs.py --year 2026  # upload single year
    upload_statcast_sp_to_gcs.py --year 2026 --force   # re-upload even if already in GCS
    upload_statcast_sp_to_gcs.py --dry-run    # verify files exist, no upload
"""

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BUCKET = "hanks_tank_data"
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SP_DIRS = [
    _REPO_ROOT / "data" / "v9" / "raw",
    _REPO_ROOT / "data" / "v10" / "raw",
]


def upload_year(year: int, dry_run: bool = False) -> bool:
    """Upload statcast_sp_pct_{year}.parquet from local storage to GCS."""
    # Find the local file
    local_path = None
    for d in _SP_DIRS:
        fp = d / f"statcast_sp_pct_{year}.parquet"
        if fp.exists():
            local_path = fp
            break

    if local_path is None:
        logger.warning("  %d: not found locally (searched %s)", year,
                       [str(d) for d in _SP_DIRS])
        return False

    gcs_path = f"data/statcast_sp_pct_{year}.parquet"

    if dry_run:
        logger.info("  %d: DRY RUN — would upload %s → gs://%s/%s",
                    year, local_path, BUCKET, gcs_path)
        return True

    try:
        from google.cloud import storage
        client = storage.Client()
        bucket = client.bucket(BUCKET)
        blob = bucket.blob(gcs_path)
        blob.upload_from_filename(str(local_path))
        size_kb = local_path.stat().st_size // 1024
        logger.info("  %d: uploaded %s → gs://%s/%s (%d KB)",
                    year, local_path.name, BUCKET, gcs_path, size_kb)
        return True
    except Exception as e:
        logger.error("  %d: upload failed: %s", year, e)
        return False


def verify_gcs(year: int) -> bool:
    """Check if the file is already in GCS."""
    try:
        from google.cloud import storage
        client = storage.Client()
        bucket = client.bucket(BUCKET)
        blob = bucket.blob(f"data/statcast_sp_pct_{year}.parquet")
        return blob.exists()
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="Upload Statcast SP parquet files to GCS")
    parser.add_argument("--year", type=int, help="Single year to upload (default: all available)")
    parser.add_argument("--dry-run", action="store_true", help="Check files, don't upload")
    parser.add_argument("--force", action="store_true", help="Re-upload even if already in GCS")
    args = parser.parse_args()

    years = [args.year] if args.year else list(range(2018, 2027))

    logger.info("Uploading Statcast SP percentile data to GCS bucket: %s", BUCKET)
    logger.info("Years to process: %s", years)

    uploaded = skipped = missing = 0

    for year in years:
        # Check if already in GCS (skip unless --force)
        if not args.dry_run and not args.force and verify_gcs(year):
            logger.info("  %d: already in GCS (use --force to re-upload)", year)
            skipped += 1
            continue

        if upload_year(year, dry_run=args.dry_run):
            uploaded += 1
        else:
            missing += 1

    logger.info(
        "Done. Uploaded=%d, Skipped(already in GCS)=%d, Missing(no local file)=%d",
        uploaded, skipped, missing,
    )
    logger.info(
        "\nNext steps:\n"
        "  1. Deploy Cloud Function with updated build_v10_features_live.py\n"
        "  2. Run backfill to populate game_v10_features for 2026:\n"
        "     curl -X POST $CF_URL -H 'Content-Type: application/json' \\\\\n"
        "       -d '{\"mode\": \"backfill_v10\", \"start\": \"2026-03-27\"}'\n"
        "  3. Verify model upgrade in predictions:\n"
        "     SELECT model_version, COUNT(*) FROM mlb_2026_season.game_predictions\n"
        "     GROUP BY 1 ORDER BY 1 DESC LIMIT 5"
    )


if __name__ == "__main__":
    main()
