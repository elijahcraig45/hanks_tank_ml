#!/usr/bin/env python3
"""
BigQuery Sync Module for 2026 Season Data

Syncs collected 2026 data to BigQuery tables:
- games_historical
- team_stats_historical
- player_stats_historical
- standings_historical
- statcast_pitches

Usage:
    python bigquery_sync.py              # Sync all new data
    python bigquery_sync.py --date 2026-04-15  # Sync specific date
    python bigquery_sync.py --validate   # Validate after sync
"""

from google.cloud import bigquery
from datetime import datetime
from typing import List, Dict, Any, Optional
import json
import os
import glob
from dataclasses import dataclass, asdict


@dataclass
class SyncResult:
    """Result of BigQuery sync operation"""
    timestamp: str
    games_synced: int
    player_stats_synced: int
    team_stats_synced: int
    statcast_synced: int
    standings_synced: bool
    success: bool
    errors: List[str]


class BigQuerySync:
    """Syncs 2026 season data to BigQuery"""
    
    def __init__(self, project_id: str = "hankstank", dataset: str = "mlb_historical_data"):
        self.project_id = project_id
        self.dataset = dataset
        self.client = bigquery.Client(project=project_id)
        self.data_dir = "data/2026"
        
    def sync_all(self, date: Optional[str] = None) -> SyncResult:
        """
        Sync all collected data to BigQuery
        
        Args:
            date: Specific date to sync (YYYY-MM-DD), or None for all new data
            
        Returns:
            SyncResult with summary of sync operation
        """
        print(f"🔄 Syncing 2026 data to BigQuery...")
        
        result = SyncResult(
            timestamp=datetime.now().isoformat(),
            games_synced=0,
            player_stats_synced=0,
            team_stats_synced=0,
            statcast_synced=0,
            standings_synced=False,
            success=True,
            errors=[]
        )
        
        try:
            # Sync games
            games_count = self._sync_games(date)
            result.games_synced = games_count
            
            # Sync statcast data
            statcast_count = self._sync_statcast(date)
            result.statcast_synced = statcast_count
            
            # Sync stats (always sync latest cumulative stats)
            player_count, team_count = self._sync_stats()
            result.player_stats_synced = player_count
            result.team_stats_synced = team_count
            
            # Sync standings
            standings_success = self._sync_standings()
            result.standings_synced = standings_success
            
            print(f"✅ Sync complete:")
            print(f"   Games: {result.games_synced}")
            print(f"   Statcast pitches: {result.statcast_synced}")
            print(f"   Player stats: {result.player_stats_synced}")
            print(f"   Team stats: {result.team_stats_synced}")
            print(f"   Standings: {'✅' if result.standings_synced else '❌'}")
            
        except Exception as e:
            result.success = False
            result.errors.append(str(e))
            print(f"❌ Sync failed: {e}")
        
        # Save sync log
        self._save_log(result)
        return result
    
    def _sync_games(self, date: Optional[str] = None) -> int:
        """Sync games to games_historical table"""
        games_dir = f"{self.data_dir}/games"
        
        if not os.path.exists(games_dir):
            print(f"   No games directory found")
            return 0
        
        # Get all game files
        game_files = glob.glob(f"{games_dir}/*.json")
        
        if not game_files:
            print(f"   No game files to sync")
            return 0
        
        rows_to_insert = []
        
        for game_file in game_files:
            try:
                with open(game_file, 'r') as f:
                    game = json.load(f)
                
                # Filter by date if specified
                if date and game.get('game_date') != date:
                    continue
                
                # Transform to BigQuery schema
                row = {
                    'game_id': str(game.get('game_id')),
                    'season': game.get('season'),
                    'game_date': game.get('game_date'),
                    'game_type': game.get('game_type'),
                    'home_team_id': game.get('home_team_id'),
                    'away_team_id': game.get('away_team_id'),
                    'home_team_name': game.get('home_team_name'),
                    'away_team_name': game.get('away_team_name'),
                    'home_score': game.get('home_score'),
                    'away_score': game.get('away_score'),
                    'venue_id': game.get('venue_id'),
                    'venue_name': game.get('venue_name'),
                    'synced_at': datetime.now().isoformat()
                }
                
                rows_to_insert.append(row)
                
            except Exception as e:
                print(f"   ⚠️  Error processing {game_file}: {e}")
        
        if rows_to_insert:
            # Insert or update (upsert) games
            table_id = f"{self.project_id}.{self.dataset}.games_historical"
            
            # Use MERGE to upsert
            self._upsert_rows(table_id, rows_to_insert, 'game_id')
            
            print(f"   Synced {len(rows_to_insert)} games")
            return len(rows_to_insert)
        
        return 0
    
    def _sync_statcast(self, date: Optional[str] = None) -> int:
        """Sync statcast pitch data"""
        statcast_dir = f"{self.data_dir}/statcast"
        
        if not os.path.exists(statcast_dir):
            return 0
        
        pitch_files = glob.glob(f"{statcast_dir}/*_pitches.json")
        
        if not pitch_files:
            return 0
        
        all_pitches = []
        
        for pitch_file in pitch_files:
            try:
                with open(pitch_file, 'r') as f:
                    pitches = json.load(f)
                
                for pitch in pitches:
                    row = {
                        'game_id': str(pitch.get('game_id')),
                        'play_id': pitch.get('play_id'),
                        'pitch_number': pitch.get('pitch_number'),
                        'pitch_type': pitch.get('pitch_type'),
                        'start_speed': pitch.get('start_speed'),
                        'end_speed': pitch.get('end_speed'),
                        'zone': pitch.get('zone'),
                        'pitcher_id': pitch.get('pitcher_id'),
                        'batter_id': pitch.get('batter_id'),
                        'result': pitch.get('result'),
                        'synced_at': datetime.now().isoformat()
                    }
                    all_pitches.append(row)
                
            except Exception as e:
                print(f"   ⚠️  Error processing {pitch_file}: {e}")
        
        if all_pitches:
            table_id = f"{self.project_id}.{self.dataset}.statcast_pitches"
            
            # Use composite key for upsert
            self._upsert_rows(table_id, all_pitches, ['game_id', 'play_id', 'pitch_number'])
            
            print(f"   Synced {len(all_pitches)} statcast pitches")
            return len(all_pitches)
        
        return 0
    
    def _sync_stats(self) -> tuple[int, int]:
        """Sync player and team stats"""
        player_count = 0
        team_count = 0
        
        # Sync team stats
        team_stats_file = f"{self.data_dir}/stats/team_stats.json"
        if os.path.exists(team_stats_file):
            try:
                with open(team_stats_file, 'r') as f:
                    team_stats = json.load(f)
                
                rows = []
                for stat in team_stats:
                    # Parse stats from API response
                    stats_data = stat.get('stats', '')
                    
                    row = {
                        'team_id': stat.get('team_id'),
                        'team_name': stat.get('team_name'),
                        'season': stat.get('season'),
                        'stat_type': stat.get('stat_type'),
                        'stats_json': json.dumps(stats_data) if isinstance(stats_data, dict) else str(stats_data),
                        'updated_at': stat.get('updated_at')
                    }
                    rows.append(row)
                
                if rows:
                    table_id = f"{self.project_id}.{self.dataset}.team_stats_historical"
                    self._upsert_rows(table_id, rows, ['team_id', 'season', 'stat_type'])
                    team_count = len(rows)
                    
            except Exception as e:
                print(f"   ⚠️  Error syncing team stats: {e}")
        
        # Sync player stats
        player_stats_file = f"{self.data_dir}/stats/player_stats.json"
        if os.path.exists(player_stats_file):
            try:
                with open(player_stats_file, 'r') as f:
                    player_stats = json.load(f)
                
                rows = []
                for stat in player_stats:
                    stats_data = stat.get('stats', '')
                    
                    row = {
                        'player_id': stat.get('player_id'),
                        'player_name': stat.get('player_name'),
                        'team_id': stat.get('team_id'),
                        'season': stat.get('season'),
                        'stat_group': stat.get('stat_group'),
                        'stats_json': json.dumps(stats_data) if isinstance(stats_data, dict) else str(stats_data),
                        'updated_at': stat.get('updated_at')
                    }
                    rows.append(row)
                
                if rows:
                    table_id = f"{self.project_id}.{self.dataset}.player_stats_historical"
                    self._upsert_rows(table_id, rows, ['player_id', 'season', 'stat_group'])
                    player_count = len(rows)
                    
            except Exception as e:
                print(f"   ⚠️  Error syncing player stats: {e}")
        
        return player_count, team_count
    
    def _sync_standings(self) -> bool:
        """Sync standings data"""
        standings_file = f"{self.data_dir}/stats/standings.json"
        
        if not os.path.exists(standings_file):
            return False
        
        try:
            with open(standings_file, 'r') as f:
                standings_data = json.load(f)
            
            # Store standings as JSON for now
            # Can be parsed further if needed
            row = {
                'season': standings_data.get('season'),
                'standings_data': json.dumps(standings_data.get('standings')),
                'updated_at': standings_data.get('updated_at')
            }
            
            table_id = f"{self.project_id}.{self.dataset}.standings_historical"
            self._upsert_rows(table_id, [row], 'season')
            
            return True
            
        except Exception as e:
            print(f"   ⚠️  Error syncing standings: {e}")
            return False
    
    def _upsert_rows(self, table_id: str, rows: List[Dict[str, Any]], key_fields: List[str] | str):
        """
        Insert or update rows in BigQuery table
        
        Args:
            table_id: Full table ID (project.dataset.table)
            rows: List of row dictionaries
            key_fields: Field(s) to use as unique key for upsert
        """
        if isinstance(key_fields, str):
            key_fields = [key_fields]
        
        # For now, use simple insert (could enhance with MERGE later)
        # Delete existing rows with same keys, then insert
        try:
            # Build delete conditions
            if len(rows) > 0:
                # Delete rows that will be replaced
                delete_conditions = []
                for row in rows:
                    conditions = []
                    for key in key_fields:
                        value = row.get(key)
                        if isinstance(value, str):
                            conditions.append(f"{key} = '{value}'")
                        else:
                            conditions.append(f"{key} = {value}")
                    delete_conditions.append(f"({' AND '.join(conditions)})")
                
                if delete_conditions:
                    delete_query = f"""
                    DELETE FROM `{table_id}`
                    WHERE {' OR '.join(delete_conditions)}
                    """
                    self.client.query(delete_query).result()
            
            # Insert new rows
            errors = self.client.insert_rows_json(table_id, rows)
            
            if errors:
                print(f"   ⚠️  Insert errors: {errors}")
                
        except Exception as e:
            print(f"   ⚠️  Upsert error: {e}")
            # If delete fails, try simple insert
            errors = self.client.insert_rows_json(table_id, rows)
            if errors:
                print(f"   ⚠️  Insert errors: {errors}")
    
    def _save_log(self, result: SyncResult):
        """Save sync log"""
        log_file = f"{self.data_dir}/sync_log.jsonl"
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, 'a') as f:
            f.write(json.dumps(asdict(result)) + '\n')


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Sync 2026 data to BigQuery')
    parser.add_argument('--date', help='Specific date to sync (YYYY-MM-DD)')
    parser.add_argument('--validate', action='store_true', help='Run validation after sync')
    parser.add_argument('--project', default='hankstank', help='GCP project ID')
    parser.add_argument('--dataset', default='mlb_historical_data', help='BigQuery dataset')
    
    args = parser.parse_args()
    
    sync = BigQuerySync(project_id=args.project, dataset=args.dataset)
    result = sync.sync_all(date=args.date)
    
    if args.validate and result.success:
        print("\n🔍 Running data validation...")
        os.system("python src/data_validation.py --year 2026")


if __name__ == '__main__':
    main()
