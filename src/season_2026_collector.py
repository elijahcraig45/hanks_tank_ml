#!/usr/bin/env python3
"""
2026 Season Daily Data Collector

Fetches live 2026 season data from MLB Stats API:
- New games played (with statcast data)
- Updated player stats
- Updated team stats
- Updated standings

Usage:
    python season_2026_collector.py              # Collect today's data
    python season_2026_collector.py --date 2026-04-15  # Specific date
    python season_2026_collector.py --backfill    # Fill gaps since season start
"""

import statsapi
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import json
import argparse
from dataclasses import dataclass, asdict
import os


@dataclass
class CollectionResult:
    """Result of data collection operation"""
    date: str
    games_collected: int
    players_updated: int
    teams_updated: int
    statcast_pitches: int
    success: bool
    errors: List[str]


class Season2026Collector:
    """Collects live 2026 season data from MLB Stats API"""
    
    def __init__(self, output_dir: str = "data/2026"):
        self.output_dir = output_dir
        self.season = 2026
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/games", exist_ok=True)
        os.makedirs(f"{output_dir}/stats", exist_ok=True)
        os.makedirs(f"{output_dir}/statcast", exist_ok=True)
        
    def collect_daily_data(self, date: Optional[str] = None) -> CollectionResult:
        """
        Collect all data for a specific date
        
        Args:
            date: Date string in YYYY-MM-DD format (default: today)
            
        Returns:
            CollectionResult with summary of collected data
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        print(f"📅 Collecting 2026 season data for {date}")
        
        result = CollectionResult(
            date=date,
            games_collected=0,
            players_updated=0,
            teams_updated=0,
            statcast_pitches=0,
            success=True,
            errors=[]
        )
        
        try:
            # Collect games for this date
            games = self._collect_games(date)
            result.games_collected = len(games)
            
            # Collect statcast data for completed games
            if games:
                pitches = self._collect_statcast(games)
                result.statcast_pitches = pitches
            
            # Update cumulative stats
            player_count, team_count = self._update_season_stats()
            result.players_updated = player_count
            result.teams_updated = team_count
            
            # Update standings
            self._update_standings()
            
            print(f"✅ Collection complete:")
            print(f"   Games: {result.games_collected}")
            print(f"   Statcast pitches: {result.statcast_pitches}")
            print(f"   Players updated: {result.players_updated}")
            print(f"   Teams updated: {result.teams_updated}")
            
        except Exception as e:
            result.success = False
            result.errors.append(str(e))
            print(f"❌ Collection failed: {e}")
        
        # Save collection log
        self._save_log(result)
        return result
    
    def _collect_games(self, date: str) -> List[Dict[str, Any]]:
        """Collect games for a specific date"""
        games = []
        
        try:
            # Get schedule for date
            schedule = statsapi.schedule(date=date)
            
            for game_info in schedule:
                game_id = game_info.get('game_id')
                
                # Only collect completed games
                if game_info.get('status') != 'Final':
                    continue
                
                # Get detailed game data
                game_data = statsapi.get('game', {'gamePk': game_id})
                
                # Extract relevant data
                game = {
                    'game_id': game_id,
                    'game_date': date,
                    'season': self.season,
                    'game_type': game_info.get('game_type', 'R'),
                    'home_team_id': game_info.get('home_id'),
                    'away_team_id': game_info.get('away_id'),
                    'home_team_name': game_info.get('home_name'),
                    'away_team_name': game_info.get('away_name'),
                    'home_score': game_info.get('home_score'),
                    'away_score': game_info.get('away_score'),
                    'venue_id': game_data.get('gameData', {}).get('venue', {}).get('id'),
                    'venue_name': game_data.get('gameData', {}).get('venue', {}).get('name'),
                    'game_data': game_data,
                    'collected_at': datetime.now().isoformat()
                }
                
                games.append(game)
                
                # Save individual game
                filename = f"{self.output_dir}/games/{game_id}.json"
                with open(filename, 'w') as f:
                    json.dump(game, f, indent=2)
            
            print(f"   Collected {len(games)} completed games")
            
        except Exception as e:
            print(f"   ⚠️  Error collecting games: {e}")
        
        return games
    
    def _collect_statcast(self, games: List[Dict[str, Any]]) -> int:
        """Collect statcast pitch-by-pitch data for games"""
        total_pitches = 0
        
        for game in games:
            try:
                game_id = game['game_id']
                
                # Get play-by-play data
                playbyplay = statsapi.get('game_playByPlay', {'gamePk': game_id})
                
                pitches = []
                for play in playbyplay.get('allPlays', []):
                    for event in play.get('playEvents', []):
                        if 'pitchData' in event:
                            pitch = {
                                'game_id': game_id,
                                'play_id': play.get('atBatIndex'),
                                'pitch_number': event.get('pitchNumber'),
                                'pitch_type': event.get('details', {}).get('type', {}).get('code'),
                                'start_speed': event.get('pitchData', {}).get('startSpeed'),
                                'end_speed': event.get('pitchData', {}).get('endSpeed'),
                                'zone': event.get('pitchData', {}).get('zone'),
                                'coordinates': event.get('pitchData', {}).get('coordinates'),
                                'pitcher_id': play.get('matchup', {}).get('pitcher', {}).get('id'),
                                'batter_id': play.get('matchup', {}).get('batter', {}).get('id'),
                                'result': event.get('details', {}).get('description'),
                                'collected_at': datetime.now().isoformat()
                            }
                            pitches.append(pitch)
                
                if pitches:
                    filename = f"{self.output_dir}/statcast/{game_id}_pitches.json"
                    with open(filename, 'w') as f:
                        json.dump(pitches, f, indent=2)
                    total_pitches += len(pitches)
                
            except Exception as e:
                print(f"   ⚠️  Error collecting statcast for game {game['game_id']}: {e}")
        
        return total_pitches
    
    def _update_season_stats(self) -> tuple[int, int]:
        """Update cumulative season stats for players and teams"""
        player_count = 0
        team_count = 0
        
        try:
            # Get all teams
            teams_data = statsapi.get('teams', {'sportId': 1, 'season': self.season})
            teams = teams_data.get('teams', [])
            
            all_team_stats = []
            all_player_stats = []
            
            for team in teams:
                team_id = team.get('id')
                
                # Get team batting stats
                batting_stats = statsapi.team_stats(team_id, 'hitting', self.season)
                team_batting = {
                    'team_id': team_id,
                    'team_name': team.get('name'),
                    'season': self.season,
                    'stat_type': 'batting',
                    'stats': batting_stats,
                    'updated_at': datetime.now().isoformat()
                }
                all_team_stats.append(team_batting)
                
                # Get team pitching stats
                pitching_stats = statsapi.team_stats(team_id, 'pitching', self.season)
                team_pitching = {
                    'team_id': team_id,
                    'team_name': team.get('name'),
                    'season': self.season,
                    'stat_type': 'pitching',
                    'stats': pitching_stats,
                    'updated_at': datetime.now().isoformat()
                }
                all_team_stats.append(team_pitching)
                
                # Get roster and player stats
                roster = statsapi.roster(team_id, season=self.season)
                for player_name in roster.split('\n'):
                    if not player_name.strip():
                        continue
                    
                    try:
                        # Parse player info (format varies, handle carefully)
                        player_info = statsapi.lookup_player(player_name.strip())
                        if player_info:
                            player_id = player_info[0]['id']
                            
                            # Get player stats
                            player_data = statsapi.player_stats(player_id, 'hitting', self.season)
                            all_player_stats.append({
                                'player_id': player_id,
                                'player_name': player_name.strip(),
                                'team_id': team_id,
                                'season': self.season,
                                'stat_group': 'hitting',
                                'stats': player_data,
                                'updated_at': datetime.now().isoformat()
                            })
                            
                            player_data = statsapi.player_stats(player_id, 'pitching', self.season)
                            all_player_stats.append({
                                'player_id': player_id,
                                'player_name': player_name.strip(),
                                'team_id': team_id,
                                'season': self.season,
                                'stat_group': 'pitching',
                                'stats': player_data,
                                'updated_at': datetime.now().isoformat()
                            })
                            
                    except Exception as e:
                        # Some players might not have stats yet
                        pass
                
                team_count += 1
            
            # Save stats
            with open(f"{self.output_dir}/stats/team_stats.json", 'w') as f:
                json.dump(all_team_stats, f, indent=2)
            
            with open(f"{self.output_dir}/stats/player_stats.json", 'w') as f:
                json.dump(all_player_stats, f, indent=2)
            
            player_count = len(all_player_stats)
            print(f"   Updated stats for {team_count} teams, {player_count} player records")
            
        except Exception as e:
            print(f"   ⚠️  Error updating stats: {e}")
        
        return player_count, team_count
    
    def _update_standings(self):
        """Update division standings"""
        try:
            standings = statsapi.standings(date=None)  # Current standings
            
            standings_data = {
                'season': self.season,
                'standings': standings,
                'updated_at': datetime.now().isoformat()
            }
            
            with open(f"{self.output_dir}/stats/standings.json", 'w') as f:
                json.dump(standings_data, f, indent=2)
            
            print(f"   Updated standings")
            
        except Exception as e:
            print(f"   ⚠️  Error updating standings: {e}")
    
    def _save_log(self, result: CollectionResult):
        """Save collection log"""
        log_file = f"{self.output_dir}/collection_log.jsonl"
        with open(log_file, 'a') as f:
            f.write(json.dumps(asdict(result)) + '\n')
    
    def backfill(self, start_date: str, end_date: Optional[str] = None):
        """
        Backfill data from start_date to end_date
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format (default: today)
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        print(f"🔄 Backfilling data from {start_date} to {end_date}")
        
        current = start
        while current <= end:
            date_str = current.strftime('%Y-%m-%d')
            self.collect_daily_data(date_str)
            current += timedelta(days=1)
        
        print(f"✅ Backfill complete")


def main():
    parser = argparse.ArgumentParser(description='Collect 2026 MLB season data')
    parser.add_argument('--date', help='Specific date to collect (YYYY-MM-DD)')
    parser.add_argument('--backfill', action='store_true', help='Backfill from season start')
    parser.add_argument('--start-date', default='2026-03-27', help='Start date for backfill')
    parser.add_argument('--output-dir', default='data/2026', help='Output directory')
    
    args = parser.parse_args()
    
    collector = Season2026Collector(output_dir=args.output_dir)
    
    if args.backfill:
        collector.backfill(args.start_date)
    else:
        collector.collect_daily_data(args.date)


if __name__ == '__main__':
    main()
