# Data Collection System Design
**Project:** Hank's Tank MLB Data Platform  
**Component:** Automated Data Collector  
**Version:** 1.0  
**Last Updated:** December 30, 2025

---

## 1. System Overview

The Data Collection System is a serverless, event-driven pipeline responsible for fetching, validating, and storing MLB data from multiple sources. It is designed to be **idempotent**, **self-healing**, and **comprehensive** (collecting all available data points).

### Architecture Diagram
```mermaid
graph TD
    Scheduler[Cloud Scheduler] -->|Trigger (Daily 2AM)| Collector[Data Collector Function]
    Collector -->|1. Fetch Schedule| MLB_API[MLB Stats API]
    Collector -->|2. Check Existing| BQ[BigQuery]
    
    subgraph "Collection Strategy"
        Collector -->|3. Fetch Games| GameData[Game Results & Boxscores]
        Collector -->|4. Fetch Statcast| Statcast[Statcast / Baseball Savant]
        Collector -->|5. Fetch Stats| PlayerStats[Player & Team Stats]
        Collector -->|6. Fetch Meta| MetaData[Transactions & Rosters]
    end
    
    Collector -->|7. Validate| Validator[Validation Service]
    Validator -->|Pass| Loader[BigQuery Loader]
    Validator -->|Fail| DLQ[Dead Letter Queue]
    Loader -->|Upsert| BQ
```

---

## 2. API Endpoints & Data Sources

### 2.1 Game Schedule (Discovery)
**Purpose:** Identify all games played on a specific date (Spring, Regular, Postseason).

- **Endpoint:** `GET https://statsapi.mlb.com/api/v1/schedule`
- **Parameters:**
  - `sportId`: `1` (MLB)
  - `startDate`: `YYYY-MM-DD`
  - `endDate`: `YYYY-MM-DD`
  - `hydrate`: `team,linescore,flags,venue,decisions` (Pre-fetches key metadata)
- **Key Response Fields:**
  ```json
  {
    "dates": [{
      "date": "2026-04-15",
      "games": [{
        "gamePk": 123456,
        "gameType": "R",  // R=Regular, S=Spring, F=Postseason
        "status": { "detailedState": "Final" },
        "teams": {
          "away": { "team": { "id": 147, "name": "New York Yankees" }, "score": 5 },
          "home": { "team": { "id": 111, "name": "Boston Red Sox" }, "score": 3 }
        },
        "venue": { "id": 3, "name": "Fenway Park" }
      }]
    }]
  }
  ```

### 2.2 Full Game Data (The "Firehose")
**Purpose:** Get complete box scores, line scores, and play-by-play for a single game.

- **Endpoint:** `GET https://statsapi.mlb.com/api/v1.1/game/{gamePk}/feed/live`
- **Parameters:** None (Game PK is in URL)
- **Key Response Fields:**
  - `gameData`: Static info (teams, venue, players, weather, datetime).
  - `liveData.linescore`: Inning-by-inning scores, hits, errors.
  - `liveData.boxscore`: Detailed stats for every player in the game.
  - `liveData.plays`: Play-by-play events (used as Statcast fallback).

### 2.3 Player Stats (Daily Batch)
**Purpose:** Get daily stats for ALL players (qualified and non-qualified).

- **Endpoint:** `GET https://statsapi.mlb.com/api/v1/stats`
- **Parameters:**
  - `stats`: `gameLog` (specific date) or `season` (cumulative)
  - `group`: `hitting,pitching,fielding`
  - `gameType`: `R` (or S, F, D, L, W based on season phase)
  - `date`: `YYYY-MM-DD`
  - `limit`: `10000` (Ensure we get everyone)
  - `qualified`: `false` (**CRITICAL**: Gets bench players/relievers)
- **Key Response Fields:**
  ```json
  {
    "stats": [{
      "splits": [{
        "date": "2026-04-15",
        "team": { "id": 147 },
        "player": { "id": 999999, "fullName": "Aaron Judge" },
        "stat": {
          "atBats": 4, "hits": 2, "homeRuns": 1, "rbi": 3,
          "exitVelocity": 110.5, "launchAngle": 25.0
        }
      }]
    }]
  }
  ```

### 2.4 Statcast (Pitch-by-Pitch)
**Purpose:** Detailed physics data for every pitch.

- **Source:** Baseball Savant (via `pybaseball` or CSV export)
- **Fallback:** MLB API `playByPlay`
- **Key Data Points (80+ columns):**
  - `pitch_type` (FF, SL, CU, etc.)
  - `release_speed` (mph)
  - `release_spin_rate` (rpm)
  - `effective_speed` (perceived velocity)
  - `zone` (1-9 strike zone location)
  - `pfx_x`, `pfx_z` (horizontal/vertical break)
  - `plate_x`, `plate_z` (location at plate)
  - `launch_speed` (exit velocity)
  - `launch_angle` (degrees)
  - `hit_distance_sc` (projected distance)

### 2.5 Transactions
**Purpose:** Track roster moves, injuries, and trades.

- **Endpoint:** `GET https://statsapi.mlb.com/api/v1/transactions`
- **Parameters:**
  - `sportId`: `1`
  - `startDate`: `YYYY-MM-DD`
  - `endDate`: `YYYY-MM-DD`
- **Key Response Fields:**
  - `typeCode`: (TR=Trade, IL=Injury, SU=Suspension, etc.)
  - `person`: Player ID and Name.
  - `fromTeam`, `toTeam`: Movement details.
  - `effectiveDate`: When the move takes effect.

### 2.6 Rosters
**Purpose:** Daily snapshot of active rosters (who is available to play).

- **Endpoint:** `GET https://statsapi.mlb.com/api/v1/teams/{teamId}/roster`
- **Parameters:**
  - `rosterType`: `active` (26-man), `40Man` (Expanded)
  - `date`: `YYYY-MM-DD`
- **Key Response Fields:**
  - `person`: Player ID.
  - `position`: Code (1=P, 2=C, etc.) and Name.
  - `status`: Active, Injured, Minors.

---

## 3. Data Collection Logic

### 3.1 Smart Collection Algorithm
The collector does not blindly fetch data. It uses a "Check-Then-Fetch" strategy to minimize API calls and duplicates.

```python
def run_daily_collection(target_date):
    # 1. Get Schedule
    games = mlb_api.get_schedule(date=target_date)
    
    for game in games:
        # 2. Check BigQuery
        if is_game_complete_in_bq(game.pk):
            continue  # Skip if we have full data
            
        # 3. Fetch & Process
        try:
            # Parallel fetch for speed
            game_data = fetch_game_feed(game.pk)
            statcast_data = fetch_statcast(game.pk)
            
            # 4. Validate
            if validate_game(game_data, statcast_data):
                # 5. Stage for Load
                stage_data(game_data, statcast_data)
            else:
                log_error(f"Validation failed for {game.pk}")
                
        except Exception as e:
            retry_queue.add(game.pk)
            
    # 6. Batch Fetch Stats (More efficient than per-game)
    fetch_and_store_daily_stats(target_date)
    fetch_and_store_transactions(target_date)
```

### 3.2 Handling Doubleheaders & Postponements
- **Doubleheaders:** The schedule endpoint returns multiple games for the same team/date. The system uses `gamePk` (unique ID) as the primary key, so doubleheaders are handled naturally.
- **Postponements:** Games with `status="Postponed"` are logged but skipped for stats collection. They are picked up when rescheduled (new `gamePk` or updated date).

---

## 4. BigQuery Schema Mapping

### 4.1 `games_historical`
| Field | Type | Source | Description |
|-------|------|--------|-------------|
| `game_pk` | INTEGER | Schedule | Unique Game ID |
| `game_date` | DATE | Schedule | Date of game |
| `game_type` | STRING | Schedule | R, S, F, D, L, W |
| `home_team_id` | INTEGER | Schedule | Home Team ID |
| `away_team_id` | INTEGER | Schedule | Away Team ID |
| `home_score` | INTEGER | Linescore | Final runs |
| `away_score` | INTEGER | Linescore | Final runs |
| `venue_id` | INTEGER | GameData | Stadium ID |
| `weather_temp` | INTEGER | GameData | Temperature at first pitch |
| `wind_speed` | INTEGER | GameData | Wind speed (mph) |
| `wind_direction` | STRING | GameData | "Out to CF", etc. |

### 4.2 `statcast_pitches`
| Field | Type | Source | Description |
|-------|------|--------|-------------|
| `game_pk` | INTEGER | Statcast | Foreign Key to Games |
| `play_id` | STRING | Statcast | Unique Play ID |
| `pitch_type` | STRING | Statcast | FF, SL, CH, etc. |
| `release_speed` | FLOAT | Statcast | Velocity (mph) |
| `batter_id` | INTEGER | Statcast | Batter MLB ID |
| `pitcher_id` | INTEGER | Statcast | Pitcher MLB ID |
| `events` | STRING | Statcast | "single", "strikeout", etc. |
| `launch_speed` | FLOAT | Statcast | Exit Velocity |
| `launch_angle` | FLOAT | Statcast | Launch Angle |
| `hit_distance` | FLOAT | Statcast | Projected distance |

### 4.3 `player_stats_historical`
| Field | Type | Source | Description |
|-------|------|--------|-------------|
| `player_id` | INTEGER | Stats API | Player MLB ID |
| `game_date` | DATE | Stats API | Date of stats |
| `team_id` | INTEGER | Stats API | Team ID |
| `ab` | INTEGER | Stats API | At Bats |
| `h` | INTEGER | Stats API | Hits |
| `hr` | INTEGER | Stats API | Home Runs |
| `rbi` | INTEGER | Stats API | RBIs |
| `ip` | FLOAT | Stats API | Innings Pitched |
| `er` | INTEGER | Stats API | Earned Runs |
| `so` | INTEGER | Stats API | Strikeouts |

---

## 5. Error Handling & Retry Strategy

### 5.1 HTTP Status Codes
- **200 OK:** Success. Process data.
- **404 Not Found:** Log warning. (Game might be cancelled/deleted).
- **429 Too Many Requests:** **CRITICAL**. Pause execution for 60s, then retry with exponential backoff.
- **500/502/503 Server Error:** Retry up to 3 times (1s, 2s, 5s delay).

### 5.2 Data Quality Errors
- **Missing Statcast:** If Statcast returns 0 pitches for a completed game:
  1. Retry Statcast fetch (sometimes data is delayed 1-2 hours).
  2. Fallback to MLB API `playByPlay` (less precise, but has pitch types/velocities).
  3. Flag game as `partial_data` in BigQuery.

- **Incomplete Boxscore:** If player count < 9 per team:
  1. Flag as `incomplete`.
  2. Trigger alert for manual review.

---

## 6. Implementation Details

### 6.1 Python Libraries
- `requests`: For raw API calls.
- `pybaseball`: Wrapper for Statcast/Savant (simplifies scraping).
- `google-cloud-bigquery`: For database operations.
- `pandas`: For data manipulation and schema alignment.

### 6.2 Rate Limiting
The MLB API is generally permissive but can rate limit aggressive scraping.
- **Limit:** ~100 requests/minute is safe.
- **Strategy:** Use `time.sleep(0.5)` between game detail fetches. Batch stats requests where possible.

### 6.3 Idempotency
All BigQuery loads use `MERGE` statements.
```sql
MERGE `hankstank.mlb_historical_data.games_historical` T
USING staging_table S
ON T.game_pk = S.game_pk
WHEN MATCHED THEN UPDATE SET ...
WHEN NOT MATCHED THEN INSERT ...
```
This ensures that re-running the collector for the same date corrects data without creating duplicates.
