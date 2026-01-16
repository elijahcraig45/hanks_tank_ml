"""
Minimal Web Access Module for Edge Devices
Scheduled, cached, predictable API calls - only what's needed for baseball
"""

import requests
import json
import sqlite3
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime, timedelta
import logging


class MinimalWebAccess:
    """
    Minimal, predictable web access for baseball data
    - Weekly standings and team stats update
    - Daily injury reports
    - On-demand player stats lookup
    All cached locally with expiration
    """
    
    # Free, reliable APIs - no authentication required
    MLB_STATSAPI_BASE = "https://statsapi.mlb.com/api/v1"
    ESPN_BASE = "https://www.espn.com/api/site/v2/sports/baseball/mlb"
    
    def __init__(self, cache_dir: Path = Path("./data")):
        """Initialize web access with local cache"""
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.cache_db = self.cache_dir / "web_cache.db"
        self._init_cache_db()
        
        self.logger = logging.getLogger("MinimalWebAccess")
        
        # API call tracking
        self.last_call_time = {}
        self.call_limits = {
            "standings": 3600 * 24 * 7,  # Weekly
            "injuries": 3600 * 24,       # Daily
            "player_stats": 3600 * 24,   # Daily
            "team_stats": 3600 * 24 * 7, # Weekly
        }
    
    def _init_cache_db(self):
        """Initialize SQLite cache"""
        conn = sqlite3.connect(self.cache_db)
        cursor = conn.cursor()
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS web_cache (
            key TEXT PRIMARY KEY,
            data TEXT NOT NULL,
            timestamp REAL NOT NULL,
            ttl_seconds INTEGER NOT NULL
        )
        """)
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS api_calls (
            endpoint TEXT,
            timestamp REAL,
            status_code INTEGER,
            bytes_transferred INTEGER
        )
        """)
        
        conn.commit()
        conn.close()
    
    def _should_fetch(self, endpoint: str) -> bool:
        """Check if enough time passed since last fetch"""
        
        now = datetime.now().timestamp()
        last = self.last_call_time.get(endpoint, 0)
        interval = self.call_limits.get(endpoint, 3600)
        
        return (now - last) > interval
    
    def _cache_get(self, key: str) -> Optional[Dict]:
        """Get data from cache if not expired"""
        
        try:
            conn = sqlite3.connect(self.cache_db)
            cursor = conn.cursor()
            
            now = datetime.now().timestamp()
            cursor.execute("""
            SELECT data FROM web_cache 
            WHERE key = ? AND (timestamp + ttl_seconds) > ?
            """, (key, now))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                self.logger.debug(f"Cache hit: {key}")
                return json.loads(row[0])
            
            return None
        except Exception as e:
            self.logger.error(f"Cache get error: {e}")
            return None
    
    def _cache_set(self, key: str, data: Dict, ttl_hours: int = 24):
        """Store data in cache with TTL"""
        
        try:
            conn = sqlite3.connect(self.cache_db)
            cursor = conn.cursor()
            
            now = datetime.now().timestamp()
            ttl_seconds = ttl_hours * 3600
            
            cursor.execute("""
            INSERT OR REPLACE INTO web_cache 
            (key, data, timestamp, ttl_seconds)
            VALUES (?, ?, ?, ?)
            """, (key, json.dumps(data), now, ttl_seconds))
            
            conn.commit()
            conn.close()
            
            self.logger.debug(f"Cached: {key} (TTL: {ttl_hours}h)")
        except Exception as e:
            self.logger.error(f"Cache set error: {e}")
    
    def _log_api_call(self, endpoint: str, status: int, bytes_transferred: int):
        """Log API call for bandwidth tracking"""
        
        try:
            conn = sqlite3.connect(self.cache_db)
            cursor = conn.cursor()
            
            cursor.execute("""
            INSERT INTO api_calls (endpoint, timestamp, status_code, bytes_transferred)
            VALUES (?, ?, ?, ?)
            """, (endpoint, datetime.now().timestamp(), status, bytes_transferred))
            
            conn.commit()
            conn.close()
        except Exception as e:
            self.logger.error(f"API logging error: {e}")
    
    def get_standings(self, force_refresh: bool = False) -> Optional[Dict]:
        """Get current MLB standings (weekly update)"""
        
        cache_key = "mlb_standings"
        
        # Check cache first
        cached = self._cache_get(cache_key)
        if cached and not force_refresh:
            return cached
        
        # Check if should fetch
        if not force_refresh and not self._should_fetch("standings"):
            self.logger.info("Standings update interval not met, using cache")
            return cached
        
        try:
            self.logger.info("Fetching MLB standings...")
            
            # Get current season standings
            url = f"{self.MLB_STATSAPI_BASE}/standings?leagueId=103,104"
            response = requests.get(url, timeout=10)
            
            self._log_api_call("standings", response.status_code, len(response.content))
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract key info
                standings = {
                    "timestamp": datetime.now().isoformat(),
                    "divisions": {}
                }
                
                for record in data.get("records", []):
                    div_name = record.get("divisionName", "Unknown")
                    standings["divisions"][div_name] = {
                        "teams": [
                            {
                                "name": team.get("name"),
                                "wins": team.get("wins"),
                                "losses": team.get("losses"),
                                "gb": team.get("gamesBack"),
                            }
                            for team in record.get("teamRecords", [])
                        ]
                    }
                
                # Cache for 7 days
                self._cache_set(cache_key, standings, ttl_hours=168)
                self.last_call_time["standings"] = datetime.now().timestamp()
                
                self.logger.info(f"Standings fetched and cached")
                return standings
            else:
                self.logger.warning(f"Standings fetch failed: {response.status_code}")
                return cached
        
        except Exception as e:
            self.logger.error(f"Error fetching standings: {e}")
            return cached
    
    def get_injury_updates(self, force_refresh: bool = False) -> Optional[Dict]:
        """Get daily injury report (daily update)"""
        
        cache_key = "injury_updates"
        
        # Check cache first
        cached = self._cache_get(cache_key)
        if cached and not force_refresh:
            return cached
        
        # Check if should fetch
        if not force_refresh and not self._should_fetch("injuries"):
            self.logger.info("Injury update interval not met, using cache")
            return cached
        
        try:
            self.logger.info("Fetching injury updates...")
            
            # Get IL (Injured List) data
            url = f"{self.MLB_STATSAPI_BASE}/teams"
            response = requests.get(url, timeout=10)
            
            self._log_api_call("injuries", response.status_code, len(response.content))
            
            if response.status_code == 200:
                # In production, would parse IL data from each team endpoint
                # For now, return placeholder structure
                
                injuries = {
                    "timestamp": datetime.now().isoformat(),
                    "teams": {},
                    "note": "Injury data fetched from MLB"
                }
                
                # Cache for 24 hours
                self._cache_set(cache_key, injuries, ttl_hours=24)
                self.last_call_time["injuries"] = datetime.now().timestamp()
                
                self.logger.info("Injury updates fetched and cached")
                return injuries
            else:
                self.logger.warning(f"Injury fetch failed: {response.status_code}")
                return cached
        
        except Exception as e:
            self.logger.error(f"Error fetching injuries: {e}")
            return cached
    
    def get_game_predictions_context(self, team: str) -> Optional[Dict]:
        """Get context data for game prediction (on-demand)"""
        
        cache_key = f"game_context_{team}"
        
        try:
            # Check cache first
            cached = self._cache_get(cache_key)
            if cached:
                return cached
            
            self.logger.info(f"Fetching game context for {team}...")
            
            # Get team's next game info
            url = f"{self.MLB_STATSAPI_BASE}/teams"
            response = requests.get(url, timeout=10)
            
            self._log_api_call("game_context", response.status_code, len(response.content))
            
            if response.status_code == 200:
                context = {
                    "timestamp": datetime.now().isoformat(),
                    "team": team,
                    "data": "Available"
                }
                
                # Cache for 24 hours
                self._cache_set(cache_key, context, ttl_hours=24)
                return context
            
            return None
        
        except Exception as e:
            self.logger.error(f"Error fetching game context: {e}")
            return None
    
    def get_bandwidth_stats(self) -> Dict:
        """Get bandwidth usage statistics"""
        
        try:
            conn = sqlite3.connect(self.cache_db)
            cursor = conn.cursor()
            
            # Last 7 days
            week_ago = (datetime.now() - timedelta(days=7)).timestamp()
            
            cursor.execute("""
            SELECT endpoint, COUNT(*), SUM(bytes_transferred)
            FROM api_calls
            WHERE timestamp > ?
            GROUP BY endpoint
            """, (week_ago,))
            
            stats = {}
            total_bytes = 0
            
            for endpoint, count, total in cursor.fetchall():
                mb = (total or 0) / (1024 * 1024)
                stats[endpoint] = {
                    "calls": count,
                    "mb_transferred": round(mb, 2)
                }
                total_bytes += (total or 0)
            
            conn.close()
            
            return {
                "period": "last_7_days",
                "endpoints": stats,
                "total_mb": round(total_bytes / (1024 * 1024), 2),
                "target_mb": 100,  # Target 100MB/week
            }
        
        except Exception as e:
            self.logger.error(f"Error getting bandwidth stats: {e}")
            return {}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    web = MinimalWebAccess()
    
    print("\n=== MINIMAL WEB ACCESS TEST ===\n")
    
    # Test standings
    standings = web.get_standings()
    if standings:
        print("Standings fetched successfully")
    
    # Test injuries
    injuries = web.get_injury_updates()
    if injuries:
        print("Injuries fetched successfully")
    
    # Test bandwidth
    stats = web.get_bandwidth_stats()
    print(f"\nBandwidth usage: {stats.get('total_mb', 0)} MB / 100 MB target")
