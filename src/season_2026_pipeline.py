#!/usr/bin/env python3
"""
2026 Season Data Pipeline Orchestrator

Runs the complete daily workflow:
1. Collect new 2026 data from MLB API
2. Sync to BigQuery
3. Run data quality validation
4. Generate summary report

Usage:
    python season_2026_pipeline.py              # Run daily pipeline
    python season_2026_pipeline.py --dry-run    # Preview without syncing
    python season_2026_pipeline.py --backfill --start 2026-03-27  # Backfill from date
"""

import argparse
import sys
from datetime import datetime
from typing import Dict, Any
import json

from season_2026_collector import Season2026Collector
from bigquery_sync import BigQuerySync
from data_validation import MLBDataValidator


class Season2026Pipeline:
    """Orchestrates complete 2026 season data pipeline"""
    
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.collector = Season2026Collector()
        self.sync = BigQuerySync()
        self.validator = MLBDataValidator()
        
    def run_daily(self, date: str = None) -> Dict[str, Any]:
        """
        Run daily data collection, sync, and validation pipeline
        
        Args:
            date: Specific date (YYYY-MM-DD) or None for today
            
        Returns:
            Pipeline execution summary
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        print("=" * 80)
        print(f"🚀 2026 MLB Season Data Pipeline")
        print(f"   Date: {date}")
        print(f"   Mode: {'DRY RUN' if self.dry_run else 'LIVE'}")
        print("=" * 80)
        print()
        
        pipeline_result = {
            'date': date,
            'timestamp': datetime.now().isoformat(),
            'dry_run': self.dry_run,
            'steps': {}
        }
        
        # Step 1: Collect data from MLB API
        print("📡 STEP 1: Collecting data from MLB Stats API")
        print("-" * 80)
        
        try:
            collection_result = self.collector.collect_daily_data(date)
            pipeline_result['steps']['collection'] = {
                'success': collection_result.success,
                'games_collected': collection_result.games_collected,
                'statcast_pitches': collection_result.statcast_pitches,
                'players_updated': collection_result.players_updated,
                'teams_updated': collection_result.teams_updated,
                'errors': collection_result.errors
            }
            
            if not collection_result.success:
                print(f"❌ Collection failed: {collection_result.errors}")
                return pipeline_result
                
        except Exception as e:
            print(f"❌ Collection failed: {e}")
            pipeline_result['steps']['collection'] = {'success': False, 'error': str(e)}
            return pipeline_result
        
        print()
        
        # Step 2: Sync to BigQuery
        print("☁️  STEP 2: Syncing to BigQuery")
        print("-" * 80)
        
        if self.dry_run:
            print("   ⏭️  Skipping (dry run mode)")
            pipeline_result['steps']['sync'] = {'skipped': True}
        else:
            try:
                sync_result = self.sync.sync_all(date)
                pipeline_result['steps']['sync'] = {
                    'success': sync_result.success,
                    'games_synced': sync_result.games_synced,
                    'statcast_synced': sync_result.statcast_synced,
                    'player_stats_synced': sync_result.player_stats_synced,
                    'team_stats_synced': sync_result.team_stats_synced,
                    'standings_synced': sync_result.standings_synced,
                    'errors': sync_result.errors
                }
                
                if not sync_result.success:
                    print(f"❌ Sync failed: {sync_result.errors}")
                    return pipeline_result
                    
            except Exception as e:
                print(f"❌ Sync failed: {e}")
                pipeline_result['steps']['sync'] = {'success': False, 'error': str(e)}
                return pipeline_result
        
        print()
        
        # Step 3: Validate data quality
        print("✓  STEP 3: Validating data quality")
        print("-" * 80)
        
        try:
            summary = self.validator.run_validation(years=[2026])
            validation_results = self.validator.results
            
            critical_count = sum(1 for r in validation_results if r.severity.value == 'CRITICAL')
            warning_count = sum(1 for r in validation_results if r.severity.value == 'WARNING')
            passed_count = sum(1 for r in validation_results if r.severity.value == 'PASS')
            
            pipeline_result['steps']['validation'] = {
                'success': critical_count == 0,
                'total_checks': len(validation_results),
                'passed': passed_count,
                'warnings': warning_count,
                'critical': critical_count
            }
            
            if critical_count > 0:
                print(f"\n⚠️  {critical_count} critical data quality issues detected!")
                for result in validation_results:
                    if result.severity.value == 'CRITICAL':
                        print(f"   ❌ {result.table} [{result.year}]: {result.message}")
            else:
                print(f"\n✅ All data quality checks passed!")
                
        except Exception as e:
            print(f"❌ Validation failed: {e}")
            pipeline_result['steps']['validation'] = {'success': False, 'error': str(e)}
        
        print()
        
        # Step 4: Summary
        print("=" * 80)
        self._print_summary(pipeline_result)
        print("=" * 80)
        
        # Save pipeline log
        self._save_pipeline_log(pipeline_result)
        
        return pipeline_result
    
    def run_backfill(self, start_date: str, end_date: str = None):
        """
        Backfill data from start_date to end_date
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD), default: today
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        print("=" * 80)
        print(f"🔄 2026 Season Backfill Pipeline")
        print(f"   Range: {start_date} to {end_date}")
        print("=" * 80)
        print()
        
        # Collect all data
        print("📡 Collecting data...")
        self.collector.backfill(start_date, end_date)
        
        print()
        
        # Sync to BigQuery
        if not self.dry_run:
            print("☁️  Syncing to BigQuery...")
            self.sync.sync_all()
            
            print()
            
            # Validate
            print("✓  Validating...")
            summary = self.validator.run_validation(years=[2026])
            print(f"\n✅ Validation: {summary.get('passed', 0)} passed, {summary.get('critical', 0)} critical")
        
        print()
        print("=" * 80)
        print("✅ Backfill complete")
        print("=" * 80)
    
    def _print_summary(self, result: Dict[str, Any]):
        """Print pipeline execution summary"""
        print("📊 PIPELINE SUMMARY")
        print()
        
        # Collection stats
        if 'collection' in result['steps']:
            coll = result['steps']['collection']
            if coll.get('success'):
                print(f"   📡 Data Collection:  ✅")
                print(f"      Games:            {coll.get('games_collected', 0)}")
                print(f"      Statcast Pitches: {coll.get('statcast_pitches', 0)}")
                print(f"      Player Records:   {coll.get('players_updated', 0)}")
                print(f"      Team Records:     {coll.get('teams_updated', 0)}")
            else:
                print(f"   📡 Data Collection:  ❌")
        
        print()
        
        # Sync stats
        if 'sync' in result['steps']:
            sync = result['steps']['sync']
            if sync.get('skipped'):
                print(f"   ☁️  BigQuery Sync:    ⏭️  (skipped)")
            elif sync.get('success'):
                print(f"   ☁️  BigQuery Sync:    ✅")
                print(f"      Games:            {sync.get('games_synced', 0)}")
                print(f"      Statcast Pitches: {sync.get('statcast_synced', 0)}")
                print(f"      Player Stats:     {sync.get('player_stats_synced', 0)}")
                print(f"      Team Stats:       {sync.get('team_stats_synced', 0)}")
            else:
                print(f"   ☁️  BigQuery Sync:    ❌")
        
        print()
        
        # Validation stats
        if 'validation' in result['steps']:
            val = result['steps']['validation']
            if val.get('success'):
                print(f"   ✓  Data Validation:   ✅")
                print(f"      Total Checks:     {val.get('total_checks', 0)}")
                print(f"      Passed:           {val.get('passed', 0)}")
                print(f"      Warnings:         {val.get('warnings', 0)}")
            else:
                print(f"   ✓  Data Validation:   ⚠️")
                print(f"      Critical Issues:  {val.get('critical', 0)}")
    
    def _save_pipeline_log(self, result: Dict[str, Any]):
        """Save pipeline execution log"""
        log_file = "data/2026/pipeline_log.jsonl"
        import os
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        with open(log_file, 'a') as f:
            f.write(json.dumps(result) + '\n')


def main():
    parser = argparse.ArgumentParser(description='2026 MLB season data pipeline')
    parser.add_argument('--date', help='Specific date (YYYY-MM-DD)')
    parser.add_argument('--dry-run', action='store_true', help='Preview without syncing to BigQuery')
    parser.add_argument('--backfill', action='store_true', help='Backfill mode')
    parser.add_argument('--start', default='2026-03-27', help='Start date for backfill')
    parser.add_argument('--end', help='End date for backfill (default: today)')
    
    args = parser.parse_args()
    
    pipeline = Season2026Pipeline(dry_run=args.dry_run)
    
    if args.backfill:
        pipeline.run_backfill(args.start, args.end)
    else:
        result = pipeline.run_daily(args.date)
        
        # Exit with error code if pipeline failed
        if not all(step.get('success', True) for step in result['steps'].values() if 'success' in step):
            sys.exit(1)


if __name__ == '__main__':
    main()
