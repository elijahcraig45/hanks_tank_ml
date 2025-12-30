# Production ML Architecture Plan
**Project:** Hank's Tank MLB Prediction System  
**Version:** 1.0  
**Created:** December 30, 2025  
**Status:** Planning Phase

---

## Table of Contents
1. [System Overview](#system-overview)
2. [Architecture Principles](#architecture-principles)
3. [Component Design](#component-design)
4. [Data Pipeline Architecture](#data-pipeline-architecture)
5. [Feature Engineering Pipeline](#feature-engineering-pipeline)
6. [Model Training & Serving](#model-training--serving)
7. [Infrastructure & Deployment](#infrastructure--deployment)
8. [Cost Optimization](#cost-optimization)
9. [Error Handling & Monitoring](#error-handling--monitoring)
10. [Security & Access Control](#security--access-control)
11. [Implementation Roadmap](#implementation-roadmap)

---

## System Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         EXTERNAL DATA SOURCES                        │
│  MLB Stats API  │  Statcast  │  Weather APIs  │  Park Factors       │
└────────────┬────────────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     DATA COLLECTION LAYER                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │
│  │  MLB Fetcher │  │ Statcast     │  │  External    │             │
│  │  (Primary)   │  │ Fetcher      │  │  Data Fetch  │             │
│  │              │  │ (Fallback)   │  │              │             │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘             │
│         │                 │                  │                      │
│         └─────────────────┴──────────────────┘                      │
│                           │                                          │
│                  ┌────────▼────────┐                                │
│                  │ Data Validator  │ (Retry Logic, Multi-Source)    │
│                  │ & Enricher      │                                │
│                  └────────┬────────┘                                │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       STORAGE LAYER (GCP)                            │
│  ┌──────────────────────────────────────────────────────┐           │
│  │  BigQuery - Raw Data Tables                          │           │
│  │  - games_historical    - team_stats_historical       │           │
│  │  - player_stats_historical  - statcast_pitches       │           │
│  │  - transactions_historical  - rosters_historical     │           │
│  │  (Partitioned by year, Clustered by date/team)       │           │
│  └──────────────────────────────────────────────────────┘           │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   FEATURE ENGINEERING LAYER                          │
│  ┌────────────────────────────────────────────────────┐             │
│  │  Cloud Functions / Cloud Run                       │             │
│  │  - Feature Calculator                              │             │
│  │  - Rolling Window Aggregator                       │             │
│  │  - Advanced Metrics Generator                      │             │
│  └──────────────────────┬─────────────────────────────┘             │
│                         │                                            │
│                         ▼                                            │
│  ┌────────────────────────────────────────────────────┐             │
│  │  BigQuery - Feature Tables                         │             │
│  │  - game_features (daily)                           │             │
│  │  - player_features (daily)                         │             │
│  │  - team_rolling_features (windows: 7/14/30/60d)    │             │
│  │  (Materialized views for efficient queries)        │             │
│  └────────────────────────────────────────────────────┘             │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    ML TRAINING & SERVING LAYER                       │
│  ┌────────────────────────────────────────────────────┐             │
│  │  Vertex AI / Cloud Run                             │             │
│  │                                                     │             │
│  │  Training Pipeline (Triggered by new features):    │             │
│  │  - Pull features from BigQuery                     │             │
│  │  - Train/Update model (incremental learning)       │             │
│  │  - Evaluate performance                            │             │
│  │  - A/B test vs current model                       │             │
│  │  - Deploy if improved                              │             │
│  │                                                     │             │
│  │  Model Artifacts Storage:                          │             │
│  │  - Cloud Storage (models, versioning)              │             │
│  └────────────────────────────────────────────────────┘             │
│                                                                       │
│  ┌────────────────────────────────────────────────────┐             │
│  │  Prediction API (Cloud Run)                        │             │
│  │  - Load latest model                               │             │
│  │  - Serve predictions                               │             │
│  │  - Cache recent predictions                        │             │
│  └────────────────────────────────────────────────────┘             │
└───────────────────────────────────────────────────────────────────┬─┘
                                                                     │
                                                                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   ORCHESTRATION & MONITORING                         │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐ │
│  │  Cloud Scheduler │  │  Cloud Logging   │  │  Cloud           │ │
│  │  (Cron Jobs)     │  │  & Monitoring    │  │  Monitoring      │ │
│  │  - Daily at 2AM  │  │  - Error alerts  │  │  - Dashboards    │ │
│  │  - Retry logic   │  │  - Data quality  │  │  - Cost tracking │ │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

### Design Goals
1. ✅ **Automated** - Minimal human intervention
2. ✅ **Resilient** - Smart retry logic, multi-source fallbacks
3. ✅ **Cost-Effective** - Serverless, pay-per-use, optimized queries
4. ✅ **Scalable** - Future-proof beyond 2026
5. ✅ **Secure** - IAM, service accounts, least privilege
6. ✅ **Observable** - Comprehensive logging and monitoring

---

## Architecture Principles

### 1. Serverless-First
- **Why:** No server management, automatic scaling, pay only for usage
- **Services:** Cloud Functions, Cloud Run, Cloud Scheduler
- **Benefit:** Lowest operational overhead

### 2. Event-Driven Pipeline
- **Data Collection** → Triggers → **Validation** → Triggers → **Feature Creation** → Triggers → **Model Update**
- **Implementation:** Cloud Pub/Sub for decoupled components
- **Benefit:** Reliable, retryable, auditable

### 3. Incremental Processing
- **Data:** Only process new/changed data (upserts, not full refreshes)
- **Features:** Update only affected features when new data arrives
- **Models:** Incremental learning when possible, periodic full retraining
- **Benefit:** Faster processing, lower costs

### 4. Idempotent Operations
- **Every step can be safely retried** without creating duplicates or errors
- **Implementation:** Use MERGE (upsert) not INSERT, check existence before processing
- **Benefit:** Reliable automation

### 5. Defense in Depth
- **Data validation** at multiple stages
- **Schema enforcement** in BigQuery
- **Type checking** in code
- **Unit tests** for critical logic
- **Integration tests** for pipelines

---

## Component Design

### 1. Data Collection Service

**Technology:** Cloud Function (Python 3.11)  
**Trigger:** Cloud Scheduler (daily at 2 AM ET)  
**Runtime:** 540s timeout, 2GB memory

#### Responsibilities
- Fetch daily game schedules from MLB API
- Collect **full game results** (box scores, line scores, final stats)
- Retrieve **all Statcast pitch-by-pitch data** for every game
- Pull **player stats** (batting and pitching for all players who appeared)
- Pull **team stats** (daily team-level aggregates)
- Fetch **transactions** (trades, waivers, IL placements, call-ups, etc.)
- Fetch **roster updates** (active rosters, 40-man rosters, position changes)
- Fetch external data (weather, umpires, etc.)
- Query BigQuery to determine what data already exists (avoid re-collecting)
- Use date range parameters to fetch only new/missing data

#### MLB API Date Range Support

**Yes! The MLB Stats API supports date filtering:**

```python
# MLB Stats API - Date range parameters
mlb_api.schedule(
    start_date='2026-04-15',  # Start date (inclusive)
    end_date='2026-04-17',    # End date (inclusive)
    sportId=1,                 # MLB (1 = MLB, use for all game types)
    gameType='R'               # R=Regular, S=Spring, P=Postseason, etc.
)

mlb_api.stats(
    personId=660271,
    stats='gameLog',
    season=2026,
    startDate='2026-04-01',   # Filter game log by date range
    endDate='2026-04-30',
    gameType='R'               # Optional: filter by game type
)

mlb_api.game_data(
    gamePk=12345,
    timecode='20260415_143000'  # Specific timestamp
)

# Statcast API - Baseball Savant (works for all game types)
statcast_params = {
    'start_dt': '2026-04-15',
    'end_dt': '2026-04-17',
    'player_type': 'batter'  # or 'pitcher'
}
```

#### Season Phase Configuration

**The system handles ALL game types automatically:**

```python
# MLB Game Types
GAME_TYPES = {
    'S': 'Spring Training',      # February - March
    'R': 'Regular Season',       # March/April - September
    'F': 'Wild Card',            # October
    'D': 'Division Series',      # October
    'L': 'League Championship',  # October
    'W': 'World Series',         # October/November
    'E': 'Exhibition',           # Various
    'A': 'All-Star Game',        # July
}

# Season calendar for 2026 (approximate dates)
SEASON_CALENDAR = {
    'spring_training': {
        'start': '2026-02-20',
        'end': '2026-03-25',
        'game_types': ['S', 'E'],
        'collection_frequency': 'daily',
        'avg_games_per_day': 20,  # More games during spring
    },
    'regular_season': {
        'start': '2026-03-26',
        'end': '2026-09-27',
        'game_types': ['R'],
        'collection_frequency': 'daily',
        'avg_games_per_day': 15,
    },
    'postseason': {
        'start': '2026-10-01',
        'end': '2026-11-05',
        'game_types': ['F', 'D', 'L', 'W'],
        'collection_frequency': 'daily',
        'avg_games_per_day': 3,  # Fewer games, but critical!
        'priority': 'high',  # Extra validation for playoff games
    },
    'all_star': {
        'date': '2026-07-14',  # Approximate
        'game_types': ['A'],
        'collection_frequency': 'one-time',
    },
}

def get_current_season_phase(date: str = None) -> str:
    """
    Determine which phase of the season we're in
    """
    if date is None:
        date = datetime.now().strftime('%Y-%m-%d')
    
    date_obj = datetime.strptime(date, '%Y-%m-%d')
    
    for phase, config in SEASON_CALENDAR.items():
        if phase == 'all_star':
            continue
        
        start = datetime.strptime(config['start'], '%Y-%m-%d')
        end = datetime.strptime(config['end'], '%Y-%m-%d')
        
        if start <= date_obj <= end:
            return phase
    
    return 'off_season'

def get_game_types_for_date(date: str) -> List[str]:
    """
    Get expected game types for a date
    """
    phase = get_current_season_phase(date)
    
    if phase == 'off_season':
        return []  # No games expected
    
    return SEASON_CALENDAR[phase]['game_types']
```

#### Complete Daily Data Collection

```python
def collect_all_data_for_date(date: str) -> Dict:
    """
    Fetch ALL data types for a given date
    Returns summary of what was collected
    """
    collected = {
        'date': date,
        'games': [],
        'statcast_pitches': 0,
        'player_stats': 0,
        'team_stats': 0,
        'transactions': 0,
        'rosters': 0
    }
    
    # 1. GAMES - Get schedule and full game results
    logger.info(f"Fetching games for {date}")
    games = fetch_games(date)
    collected['games'] = games
    
    # 2. STATCAST - All pitch-by-pitch data for each game
    logger.info(f"Fetching Statcast data for {len(games)} games")
    for game in games:
        if game['status'] == 'Final':  # Only fetch completed games
            pitches = fetch_statcast_pitches(game['game_pk'], date)
            collected['statcast_pitches'] += len(pitches)
    
    # 3. PLAYER STATS - All players (qualified and non-qualified)
    logger.info(f"Fetching player stats for {date}")
    player_stats = fetch_all_player_stats(date)
    collected['player_stats'] = len(player_stats)
    
    # 4. TEAM STATS - Daily team aggregates
    logger.info(f"Fetching team stats for {date}")
    team_stats = fetch_team_stats(date)
    collected['team_stats'] = len(team_stats)
    
    # 5. TRANSACTIONS - All player movement
    logger.info(f"Fetching transactions for {date}")
    transactions = fetch_transactions(date)
    collected['transactions'] = len(transactions)
    
    # 6. ROSTERS - Active rosters for all teams
    logger.info(f"Fetching rosters for {date}")
    rosters = fetch_all_rosters(date)
    collected['rosters'] = len(rosters)
    
    logger.info(f"Collection complete: {collected}")
    return collected

def fetch_games(date: str) -> List[Dict]:
    """
    Fetch complete game data including:
    - Game metadata (teams, date, venue)
    - Final scores
    - Box scores (all players)
    - Line scores (inning-by-inning)
    - Game logs
    
    Works for ALL game types:
    - Spring Training (gameType='S')
    - Regular Season (gameType='R')
    - Postseason (gameType='F','D','L','W')
    - Exhibitions, All-Star Game
    """
    # Get games scheduled for this date (all game types)
    # Not specifying gameType gets ALL games for the date
    schedule = mlb_api.schedule(
        date=date, 
        sportId=1,  # MLB games only
        hydrate='team,linescore,decisions'
    )
    
    games = []
    for game_entry in schedule['dates'][0]['games']:
        game_pk = game_entry['gamePk']
        
        # Check if we need to fetch this game
        if not should_fetch_game(game_pk, date):
            continue
        
        # Fetch full game data
        game_data = mlb_api.get_game(
            game_pk,
            hydrate='linescore,boxscore,game,decisions,probablePitcher'
        )
        
        games.append({
            'game_pk': game_pk,
            'game_date': date,
            'status': game_data['gameData']['status']['detailedState'],
            'home_team': game_data['gameData']['teams']['home']['id'],
            'away_team': game_data['gameData']['teams']['away']['id'],
            'home_score': game_data.get('liveData', {}).get('linescore', {}).get('teams', {}).get('home', {}).get('runs'),
            'away_score': game_data.get('liveData', {}).get('linescore', {}).get('teams', {}).get('away', {}).get('runs'),
            'venue_id': game_data['gameData']['venue']['id'],
            'box_score': game_data.get('liveData', {}).get('boxscore'),
            'line_score': game_data.get('liveData', {}).get('linescore'),
        })
    
    logger.info(f"Fetched {len(games)} games for {date}")
    return games

def fetch_statcast_pitches(game_pk: int, date: str) -> List[Dict]:
    """
    Fetch ALL pitch-by-pitch Statcast data for a game
    Includes: pitch type, velocity, spin rate, exit velo, launch angle, etc.
    """
    # Option 1: Baseball Savant (pybaseball)
    try:
        from pybaseball import statcast_single_game
        pitches = statcast_single_game(game_pk)
        logger.info(f"Fetched {len(pitches)} pitches for game {game_pk}")
        return pitches.to_dict('records')
    except Exception as e:
        logger.warning(f"Statcast fetch failed for game {game_pk}: {e}")
    
    # Option 2: MLB API playByPlay (less detailed but reliable)
    try:
        pbp = mlb_api.get_game_play_by_play(game_pk)
        pitches = []
        for play in pbp.get('allPlays', []):
            for pitch in play.get('playEvents', []):
                if pitch.get('isPitch'):
                    pitches.append({
                        'game_pk': game_pk,
                        'game_date': date,
                        'at_bat_index': play.get('atBatIndex'),
                        'pitch_number': pitch.get('pitchNumber'),
                        'pitch_type': pitch.get('details', {}).get('type', {}).get('code'),
                        'pitch_speed': pitch.get('pitchData', {}).get('startSpeed'),
                        'zone': pitch.get('pitchData', {}).get('zone'),
                        **pitch.get('pitchData', {})
                    })
        return pitches
    except Exception as e:
        logger.error(f"All Statcast sources failed for game {game_pk}: {e}")
        return []

def fetch_all_player_stats(date: str) -> List[Dict]:
    """
    Fetch stats for ALL players who appeared on this date
    (Already implemented above with multi-strategy approach)
    """
    # Use the multi-strategy approach from earlier
    return fetch_player_stats(date)

def fetch_team_stats(date: str) -> List[Dict]:
    """
    Fetch team-level statistics for all 30 teams
    Includes: team batting, pitching, fielding aggregates
    """
    team_stats = []
    
    # Get all teams
    teams = mlb_api.get_teams(sportId=1, season=2026)
    
    for team in teams['teams']:
        team_id = team['id']
        
        # Check if we already have stats for this team/date
        if not should_fetch_team_stats(team_id, date):
            continue
        
        # Fetch team stats
        stats = mlb_api.get_team_stats(
            team_id,
            stats='season',
            season=2026,
            date=date,
            group='hitting,pitching,fielding'
        )
        
        team_stats.append({
            'team_id': team_id,
            'date': date,
            'season': 2026,
            **parse_team_stats(stats)
        })
    
    logger.info(f"Fetched stats for {len(team_stats)} teams")
    return team_stats

def fetch_transactions(date: str) -> List[Dict]:
    """
    Fetch ALL transactions for the date:
    - Trades
    - Waivers
    - IL placements/activations
    - Call-ups/send-downs (minors)
    - Signings
    - Releases
    - Retirements
    """
    # MLB API transactions endpoint
    transactions = mlb_api.get_transactions(
        startDate=date,
        endDate=date,
        sportId=1
    )
    
    processed = []
    for txn in transactions.get('transactions', []):
        processed.append({
            'transaction_id': txn.get('id'),
            'date': txn.get('date'),
            'type': txn.get('typeCode'),
            'description': txn.get('description'),
            'from_team': txn.get('fromTeam', {}).get('id'),
            'to_team': txn.get('toTeam', {}).get('id'),
            'player_id': txn.get('person', {}).get('id'),
            'player_name': txn.get('person', {}).get('fullName'),
            'effective_date': txn.get('effectiveDate'),
            'resolution_date': txn.get('resolutionDate'),
        })
    
    logger.info(f"Fetched {len(processed)} transactions for {date}")
    return processed

def fetch_all_rosters(date: str) -> List[Dict]:
    """
    Fetch active rosters for all 30 MLB teams
    Includes: player positions, jersey numbers, status
    """
    rosters = []
    
    # Get all teams
    teams = mlb_api.get_teams(sportId=1, season=2026)
    
    for team in teams['teams']:
        team_id = team['id']
        
        # Fetch active roster
        roster = mlb_api.get_team_roster(
            team_id,
            rosterType='active',  # or 'fullSeason', '40Man'
            date=date,
            hydrate='person'
        )
        
        for player in roster.get('roster', []):
            rosters.append({
                'team_id': team_id,
                'player_id': player['person']['id'],
                'player_name': player['person']['fullName'],
                'jersey_number': player.get('jerseyNumber'),
                'position_code': player['position']['code'],
                'position_name': player['position']['name'],
                'position_type': player['position']['type'],
                'status': player.get('status', {}).get('code'),
                'roster_date': date,
                'parent_team_id': player.get('parentTeamId'),
            })
    
    logger.info(f"Fetched {len(rosters)} roster entries across all teams")
    return rosters

def should_fetch_team_stats(team_id: int, date: str) -> bool:
    """Check if team stats already exist for this date"""
    query = f"""
    SELECT team_id FROM `hankstank.mlb_historical_data.team_stats_historical`
    WHERE team_id = {team_id} AND date = '{date}'
    """
    result = bq_client.query(query).to_dataframe()
    return result.empty
```

#### Data Volume Estimates

**Regular Season Day (15 games):**

| Data Type | Records | Storage Size |
|-----------|---------|-------------|
| Games | 15 games | ~500 KB |
| Statcast Pitches | ~45,000 pitches (3000/game) | ~25 MB |
| Player Stats | ~500-700 players | ~200 KB |
| Team Stats | 30 teams | ~30 KB |
| Transactions | 5-20 per day | ~10 KB |
| Rosters | 30 teams × 26 players = 780 | ~100 KB |
| **TOTAL/DAY** | ~46,000 records | **~26 MB/day** |

**Full Year Estimates (All Phases):**

| Season Phase | Days | Games/Day | Total Games | Statcast Pitches | Storage |
|--------------|------|-----------|-------------|------------------|----------|
| Spring Training | 35 | 20 | 700 | 2.1M | 900 MB |
| Regular Season | 182 | 15 | 2,730 | 8.2M | 4.7 GB |
| Postseason | 30 | 3 | 90 | 270K | 780 MB |
| **TOTAL** | **247** | - | **3,520** | **~10.6M** | **~6.4 GB** |

**Notes:**
- Spring Training: More games but often incomplete stats/Statcast
- Regular Season: Full data collection for everything
- Postseason: Fewer games but CRITICAL accuracy (high priority validation)
- All-Star break: Minimal data collection (~1 week in July)

**Cost Impact (Full Year):**
- BigQuery storage: $0.02/GB/month → $0.13/month average
- BigQuery streaming inserts: $0.01/200MB → $1.00/year
- Still very affordable!

#### Intelligent Collection Strategy (Avoid Duplicates)

```python
def collect_data_for_date_range(start_date: str, end_date: str):
    """
    Smart collection that only fetches what's missing from BigQuery
    """
    # Step 1: Check what we already have in BigQuery
    existing_data = get_existing_data_summary(start_date, end_date)
    
    # Step 2: Determine what's missing
    missing_dates = find_missing_dates(start_date, end_date, existing_data)
    incomplete_games = find_incomplete_games(existing_data)
    
    # Step 3: Fetch only what's needed
    if not missing_dates and not incomplete_games:
        logger.info(f"All data for {start_date} to {end_date} already collected. Skipping.")
        return {'status': 'skipped', 'reason': 'data_exists'}
    
    logger.info(f"Collecting data for {len(missing_dates)} dates and {len(incomplete_games)} incomplete games")
    
    # Fetch missing data using date ranges
    for date in missing_dates:
        fetch_all_data_for_date(date)
    
    for game_pk in incomplete_games:
        fetch_game_data(game_pk, retry=True)
    
    return {'status': 'collected', 'dates': missing_dates, 'games': incomplete_games}

def get_existing_data_summary(start_date: str, end_date: str) -> Dict:
    """
    Query BigQuery to see what data we already have
    Returns summary of existing games, stats, etc.
    """
    query = f"""
    WITH date_range AS (
        SELECT DATE('{start_date}') as start_dt, DATE('{end_date}') as end_dt
    ),
    existing_games AS (
        SELECT 
            game_date,
            COUNT(DISTINCT game_pk) as game_count,
            COUNTIF(home_score IS NULL OR away_score IS NULL) as incomplete_games
        FROM `hankstank.mlb_historical_data.games_historical`
        WHERE game_date BETWEEN (SELECT start_dt FROM date_range) 
                           AND (SELECT end_dt FROM date_range)
        GROUP BY game_date
    ),
    existing_stats AS (
        SELECT
            date,
            COUNT(DISTINCT player_id) as player_count
        FROM `hankstank.mlb_historical_data.player_stats_historical`
        WHERE date BETWEEN (SELECT start_dt FROM date_range) 
                      AND (SELECT end_dt FROM date_range)
        GROUP BY date
    )
    SELECT 
        eg.game_date,
        COALESCE(eg.game_count, 0) as games_collected,
        COALESCE(eg.incomplete_games, 0) as incomplete_count,
        COALESCE(es.player_count, 0) as players_collected
    FROM existing_games eg
    FULL OUTER JOIN existing_stats es ON eg.game_date = es.date
    ORDER BY game_date
    """
    
    results = bq_client.query(query).to_dataframe()
    return results.to_dict('records')

def find_missing_dates(start_date: str, end_date: str, existing_data: List[Dict]) -> List[str]:
    """
    Find dates in range that have no data or incomplete data
    """
    from datetime import datetime, timedelta
    
    # Generate all dates in range
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    all_dates = []
    current = start
    while current <= end:
        all_dates.append(current.strftime('%Y-%m-%d'))
        current += timedelta(days=1)
    
    # Check which dates are missing or incomplete
    existing_dates = {d['game_date'].strftime('%Y-%m-%d') for d in existing_data 
                     if d.get('games_collected', 0) > 0}
    
    missing = [d for d in all_dates if d not in existing_dates]
    
    # Also include dates with incomplete games (final scores missing)
    incomplete = [d['game_date'].strftime('%Y-%m-%d') for d in existing_data 
                 if d.get('incomplete_count', 0) > 0]
    
    return sorted(set(missing + incomplete))

def find_incomplete_games(existing_data: List[Dict]) -> List[int]:
    """
    Find specific games that need re-collection (missing data)
    """
    query = """
    SELECT DISTINCT game_pk
    FROM `hankstank.mlb_historical_data.games_historical`
    WHERE 
        -- Missing final scores (game not finished when collected)
        (home_score IS NULL OR away_score IS NULL)
        -- Or missing key metadata
        OR home_team IS NULL
        OR away_team IS NULL
        -- Or collected more than 7 days ago but still incomplete (stale)
        OR (updated_at < TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
            AND (home_score IS NULL OR away_score IS NULL))
    """
    
    results = bq_client.query(query).to_dataframe()
    return results['game_pk'].tolist()
```

#### Smart Multi-Source Strategy
```python
def fetch_game_data(game_id: str, date: str) -> Dict:
    """
    Attempts multiple sources in priority order:
    1. MLB Stats API (primary)
    2. Statcast Baseball Savant (secondary)
    3. Cached data from previous fetch (fallback)
    """
    sources = [
        MLBStatsAPI(),
        StatcastAPI(), 
        CachedDataSource()
    ]
    
    for source in sources:
        try:
            data = source.fetch_game(game_id, date)
            if validate_completeness(data):
                return data
        except Exception as e:
            log_error(f"{source}: {e}")
            continue
    
    raise DataUnavailableError(f"All sources failed for {game_id}")

def fetch_player_stats(date: str, team_id: int = None) -> List[Dict]:
    """
    Fetch ALL players, not just qualified players.
    
    Key insight: APIs often have 'qualified' filters that exclude players
    who don't meet minimum PA (plate appearances) or IP (innings pitched) thresholds.
    We want EVERYONE who appeared in a game.
    """
    all_players = []
    
    # Strategy 1: Get from box scores (most complete)
    # Box scores include every player who appeared, even for 1 AB
    games = fetch_games_for_date(date)
    for game in games:
        box_score = mlb_api.get_box_score(game['game_pk'])
        
        # Extract all batters
        for team_side in ['home', 'away']:
            batters = box_score['teams'][team_side]['batters']
            for player_id in batters:
                player_stats = box_score['teams'][team_side]['players'][f'ID{player_id}']['stats']['batting']
                if player_stats:  # Has stats for this game
                    all_players.append({
                        'player_id': player_id,
                        'date': date,
                        'team_id': box_score['teams'][team_side]['team']['id'],
                        **player_stats
                    })
            
            # Extract all pitchers
            pitchers = box_score['teams'][team_side]['pitchers']
            for player_id in pitchers:
                player_stats = box_score['teams'][team_side]['players'][f'ID{player_id}']['stats']['pitching']
                if player_stats:
                    all_players.append({
                        'player_id': player_id,
                        'date': date,
                        'team_id': box_score['teams'][team_side]['team']['id'],
                        **player_stats
                    })
    
    # Strategy 2: Cross-reference with roster (catch any missed)
    # Sometimes box scores are incomplete, rosters show who was available
    if team_id:
        roster = mlb_api.get_team_roster(team_id, date=date)
        roster_player_ids = {p['person']['id'] for p in roster['roster']}
        collected_player_ids = {p['player_id'] for p in all_players}
        
        missing_from_stats = roster_player_ids - collected_player_ids
        if missing_from_stats:
            logger.warning(f"Found {len(missing_from_stats)} rostered players missing from stats")
            # Fetch individual player stats for these
            for player_id in missing_from_stats:
                try:
                    stats = mlb_api.get_player_stats(
                        player_id, 
                        date=date,
                        hydrate='stats'  # Forces return even if 0 stats
                    )
                    if stats:
                        all_players.append(stats)
                except Exception as e:
                    logger.error(f"Failed to fetch stats for player {player_id}: {e}")
    
    # Strategy 3: Use season stats endpoint with NO qualification filter
    # Fallback if box scores are incomplete
    try:
        season_stats = mlb_api.get_season_stats(
            season=2026,
            date=date,
            qualified=False,  # CRITICAL: Don't filter by qualification
            limit=10000  # High limit to ensure we get everyone
        )
        # Merge with box score data (prefer box score for accuracy)
        for player in season_stats:
            if player['player_id'] not in {p['player_id'] for p in all_players}:
                all_players.append(player)
    except Exception as e:
        logger.warning(f"Season stats fallback failed: {e}")
    
    logger.info(f"Collected stats for {len(all_players)} total players (including non-qualified)")
    return all_players
```

#### Retry Logic
- **Exponential backoff:** 1s, 2s, 4s, 8s, 16s
- **Max retries:** 5 per source
- **Circuit breaker:** Skip source if 3 consecutive failures
- **Dead letter queue:** Store failed requests for manual review
- **Deduplication:** Check BigQuery before fetching to avoid re-collecting existing data

#### Deduplication Strategy

```python
def should_fetch_game(game_pk: int, game_date: str) -> bool:
    """
    Check if game data already exists in BigQuery
    Returns True if we need to fetch, False if we can skip
    """
    query = f"""
    SELECT 
        game_pk,
        home_score,
        away_score,
        updated_at,
        TIMESTAMP_DIFF(CURRENT_TIMESTAMP(), updated_at, HOUR) as hours_since_update
    FROM `hankstank.mlb_historical_data.games_historical`
    WHERE game_pk = {game_pk}
    """
    
    result = bq_client.query(query).to_dataframe()
    
    if result.empty:
        # Game doesn't exist, fetch it
        return True
    
    game = result.iloc[0]
    
    # Re-fetch if incomplete (missing scores)
    if pd.isna(game['home_score']) or pd.isna(game['away_score']):
        logger.info(f"Game {game_pk} incomplete, re-fetching")
        return True
    
    # Re-fetch if data is stale (>24 hours old for in-progress games)
    if game['hours_since_update'] > 24:
        logger.info(f"Game {game_pk} data is {game['hours_since_update']}h old, refreshing")
        return True
    
    # Data exists and is complete/fresh, skip
    logger.info(f"Game {game_pk} already collected, skipping")
    return False

def should_fetch_player_stats(player_id: int, date: str) -> bool:
    """
    Check if player stats already exist for this date
    """
    query = f"""
    SELECT player_id, date, updated_at
    FROM `hankstank.mlb_historical_data.player_stats_historical`
    WHERE player_id = {player_id} AND date = '{date}'
    """
    
    result = bq_client.query(query).to_dataframe()
    
    if result.empty:
        return True  # Player stats don't exist, fetch
    
    # Stats exist, skip (unless you want to refresh daily)
    return False
```

#### API Configuration for Complete Data

**Critical API Parameters:**
```python
# MLB Stats API - Key parameters to avoid missing players
API_CONFIGS = {
    "player_stats": {
        "qualified": False,  # ⚠️ CRITICAL: Include non-qualified players
        "limit": 10000,      # High limit to avoid pagination issues
        "hydrate": "stats,person",  # Get full details even for players with minimal stats
    },
    "box_score": {
        "hydrate": "stats,players",  # Include all player details
    },
    "roster": {
        "rosterType": "active",  # Active roster (40-man has everyone)
        "hydrate": "person,stats",
    },
    "team_stats": {
        "group": "hitting,pitching",  # Get both batters and pitchers
        "qualified": False,  # Include everyone who appeared
    }
}

# Baseball Savant / Statcast - Use game-level exports
STATCAST_CONFIG = {
    "source": "game",  # Per-game data includes all players who batted/pitched
    "player_type": "all",  # Not just qualified
}
```

**Schema Definition Files:**
```python
# schemas/player_stats.py
PLAYER_STATS_SCHEMA = {
    # Required fields (must exist, can be null)
    "player_id": "INTEGER",
    "team_id": "INTEGER",
    "game_date": "DATE",
    "player_name": "STRING",
    
    # Batting stats (nullable for pitchers)
    "at_bats": "INTEGER",
    "hits": "INTEGER",
    "home_runs": "INTEGER",
    "rbis": "INTEGER",
    "batting_avg": "FLOAT",
    
    # Pitching stats (nullable for batters)
    "innings_pitched": "FLOAT",
    "earned_runs": "INTEGER",
    "strikeouts": "INTEGER",
    "era": "FLOAT",
    
    # Metadata
    "is_qualified_batter": "BOOLEAN",  # Track qualification status
    "is_qualified_pitcher": "BOOLEAN",
    "games_played": "INTEGER",
    "created_at": "TIMESTAMP",
}

# Load schema from BigQuery table directly
def get_bq_schema(table_name: str) -> Dict:
    """Fetch actual BigQuery table schema to ensure perfect match"""
    table_ref = bq_client.dataset('mlb_historical_data').table(table_name)
    table = bq_client.get_table(table_ref)
    
    return {
        field.name: field.field_type 
        for field in table.schema
    }
```

#### Output
- JSON files to Cloud Storage (temporary)
- Publishes `data.collected` event to Pub/Sub
- Metadata includes player count (qualified vs non-qualified)

---

### 2. Data Validation Service

**Technology:** Cloud Function (Python 3.11)  
**Trigger:** Pub/Sub topic `data.collected`  
**Runtime:** 300s timeout, 1GB memory

#### Responsibilities
- Validate data completeness (required fields present)
- Check data quality (reasonable ranges, no nulls where expected)
- Detect anomalies (outliers, suspicious patterns)
- Enrich with metadata (timestamps, source info)
- Log validation reports

#### Validation Checks
```python
validation_rules = {
    "games": [
        RequiredFields(["game_id", "game_date", "home_team", "away_team"]),
        DateRange("game_date", min="2026-01-01", max="today"),
        ScoreRange("home_score", min=0, max=50),
        ScoreRange("away_score", min=0, max=50),
        BoxScoreCompleteness()  # All players in box score
    ],
    "statcast": [
        RequiredFields(["pitch_type", "release_speed", "game_pk"]),
        ValueRange("release_speed", min=50, max=110),
        ValueRange("launch_speed", min=0, max=130),
        EnumCheck("pitch_type", allowed=["FF", "SL", "CH", "CU", ...]),
        PitchCountCheck()  # Expected ~250-350 pitches per game
    ],
    "player_stats": [
        RequiredFields(["player_id", "team_id", "date"]),
        SchemaMatch(expected_schema=PLAYER_STATS_SCHEMA),
        PlayerCompletenessCheck()  # Ensures ALL players, not just qualified
    ],
    "team_stats": [
        RequiredFields(["team_id", "date", "season"]),
        SchemaMatch(expected_schema=TEAM_STATS_SCHEMA),
        TeamCountCheck(expected=30)  # Should have all 30 teams
    ],
    "transactions": [
        RequiredFields(["transaction_id", "date", "type"]),
        SchemaMatch(expected_schema=TRANSACTIONS_SCHEMA),
        DateMatch("date", expected_date="{collection_date}")
    ],
    "rosters": [
        RequiredFields(["team_id", "player_id", "position_code"]),
        SchemaMatch(expected_schema=ROSTERS_SCHEMA),
        TeamCountCheck(expected=30),  # All teams
        RosterSizeCheck(min=24, max=28)  # Active roster size
    ]
}

class PitchCountCheck:
    """Validate that Statcast data is complete for a game"""
    def validate(self, pitches: List[Dict], game_pk: int) -> ValidationResult:
        pitch_count = len(pitches)
        
        # Typical game: 250-350 pitches
        # Low-scoring: 200-250
        # High-scoring/extra innings: 350-500+
        
        if pitch_count < 150:
            return ValidationResult(
                status="ERROR",
                message=f"Only {pitch_count} pitches for game {game_pk}, likely incomplete",
                action="RETRY_STATCAST"
            )
        elif pitch_count < 200:
            return ValidationResult(
                status="WARNING",
                message=f"{pitch_count} pitches (low but possible for short game)"
            )
        
        return ValidationResult(status="PASS", message=f"{pitch_count} pitches collected")

class TeamCountCheck:
    """Ensure we have data for all teams"""
    def __init__(self, expected: int = 30):
        self.expected = expected
    
    def validate(self, data: List[Dict]) -> ValidationResult:
        unique_teams = len(set(d['team_id'] for d in data))
        
        if unique_teams < self.expected:
            return ValidationResult(
                status="WARNING",
                message=f"Only {unique_teams}/{self.expected} teams found",
                missing_teams=self.expected - unique_teams
            )
        
        return ValidationResult(status="PASS", message=f"All {unique_teams} teams present")

class PlayerCompletenessCheck:
    """
    Validates that we have ALL players who appeared in games,
    not just 'qualified' players (min PA/IP thresholds)
    """
    def validate(self, player_data: List[Dict], game_data: List[Dict]) -> ValidationResult:
        # Extract all unique player IDs from game box scores
        players_in_games = set()
        for game in game_data:
            players_in_games.update(game.get('home_batters', []))
            players_in_games.update(game.get('away_batters', []))
            players_in_games.update(game.get('home_pitchers', []))
            players_in_games.update(game.get('away_pitchers', []))
        
        # Check if we have stats for all players who appeared
        players_with_stats = {p['player_id'] for p in player_data}
        missing_players = players_in_games - players_with_stats
        
        if missing_players:
            return ValidationResult(
                status="WARNING",
                message=f"Missing {len(missing_players)} players from stats",
                missing_player_ids=list(missing_players),
                action="FETCH_MISSING_PLAYERS"
            )
        
        return ValidationResult(status="PASS", message="All players accounted for")

class SchemaMatch:
    """
    Validates data matches expected BigQuery schema exactly
    """
    def __init__(self, expected_schema: Dict):
        self.expected_schema = expected_schema
    
    def validate(self, data: List[Dict]) -> ValidationResult:
        if not data:
            return ValidationResult(status="PASS", message="No data to validate")
        
        sample = data[0]
        
        # Check for missing required fields
        missing_fields = set(self.expected_schema.keys()) - set(sample.keys())
        if missing_fields:
            return ValidationResult(
                status="ERROR",
                message=f"Missing required fields: {missing_fields}",
                action="SCHEMA_MISMATCH"
            )
        
        # Check for extra fields (warning only)
        extra_fields = set(sample.keys()) - set(self.expected_schema.keys())
        if extra_fields:
            logger.warning(f"Extra fields found (will be ignored): {extra_fields}")
        
        # Validate field types
        type_mismatches = []
        for field, expected_type in self.expected_schema.items():
            if field in sample:
                actual_type = type(sample[field]).__name__
                if not self._types_compatible(actual_type, expected_type):
                    type_mismatches.append(f"{field}: expected {expected_type}, got {actual_type}")
        
        if type_mismatches:
            return ValidationResult(
                status="ERROR",
                message=f"Type mismatches: {type_mismatches}",
                action="SCHEMA_MISMATCH"
            )
        
        return ValidationResult(status="PASS", message="Schema matches expected")
```

#### Quality Scoring
```python
def calculate_quality_score(validation_results: Dict) -> str:
    """
    Multi-dimensional quality grading
    """
    scores = {
        'schema_match': 0,      # Fields match expected schema
        'data_completeness': 0, # Required fields populated
        'player_completeness': 0, # All players from games accounted for
        'value_ranges': 0,      # Stats in reasonable ranges
    }
    
    # Schema matching (binary: pass/fail)
    if validation_results['schema_check'].status == 'PASS':
        scores['schema_match'] = 1.0
    
    # Data completeness (% of non-null required fields)
    scores['data_completeness'] = validation_results['null_check'].completion_rate
    
    # Player completeness (critical!)
    player_check = validation_results['player_completeness']
    if player_check.missing_players:
        # Penalize based on how many players are missing
        total_players = len(player_check.expected_players)
        missing_pct = len(player_check.missing_players) / total_players
        scores['player_completeness'] = 1.0 - missing_pct
    else:
        scores['player_completeness'] = 1.0
    
    # Value range checks
    scores['value_ranges'] = validation_results['range_check'].pass_rate
    
    # Weighted overall score
    overall = (
        scores['schema_match'] * 0.3 +        # Schema is critical
        scores['data_completeness'] * 0.2 +   
        scores['player_completeness'] * 0.4 + # Player completeness is MOST critical
        scores['value_ranges'] * 0.1
    )
    
    # Grade assignment
    if overall >= 0.99 and scores['schema_match'] == 1.0:
        return 'A'  # Perfect or near-perfect
    elif overall >= 0.95:
        return 'B'  # Minor issues acceptable
    elif overall >= 0.80 and scores['player_completeness'] >= 0.90:
        return 'C'  # Usable, but needs attention (MUST have most players)
    else:
        return 'F'  # Too incomplete, retry needed

grading_actions = {
    'A': {'proceed': True, 'alert': False, 'retry': False},
    'B': {'proceed': True, 'alert': False, 'retry': False},
    'C': {'proceed': True, 'alert': True, 'retry': False},  # Log warning
    'F': {'proceed': False, 'alert': True, 'retry': True},  # Trigger re-collection
}
```

**Grade Criteria:**
- **Grade A:** Schema match + 99%+ complete + all players accounted for → Proceed immediately
- **Grade B:** Schema match + 95%+ complete + >95% players → Proceed
- **Grade C:** Schema match + >80% complete + >90% players → Proceed with warnings (usable)
- **Grade F:** Schema mismatch OR <80% complete OR <90% players → Retry collection

**Player Completeness is Critical:** Even if other metrics are good, missing >10% of players = Grade F

**Season-Specific Considerations:**

```python
def adjust_quality_thresholds(season_phase: str) -> Dict:
    """
    Different quality expectations for different season phases
    """
    if season_phase == 'spring_training':
        return {
            'statcast_required': False,  # Statcast often incomplete for spring
            'player_completeness_threshold': 0.80,  # More lenient (roster churn)
            'allow_missing_fields': ['exit_velocity', 'launch_angle'],
        }
    
    elif season_phase == 'postseason':
        return {
            'statcast_required': True,  # MUST have complete Statcast
            'player_completeness_threshold': 1.0,  # MUST have ALL players
            'retry_on_warning': True,  # Even Grade B triggers retry
            'alert_on_any_issue': True,  # Alert humans immediately
        }
    
    else:  # regular_season
        return {
            'statcast_required': True,
            'player_completeness_threshold': 0.90,
            'retry_on_warning': False,
        }
```

#### Output
- Validation report to Cloud Storage
- Publishes `data.validated` event (if passing)
- Publishes `data.retry` event (if failed, triggers re-collection)

---

### 3. BigQuery Data Loader

**Technology:** Cloud Function (Python 3.11)  
**Trigger:** Pub/Sub topic `data.validated`  
**Runtime:** 540s timeout, 2GB memory

#### Responsibilities
- Load validated data into BigQuery
- Use MERGE for upsert operations (no duplicates)
- Handle schema evolution
- Maintain data lineage

#### Upsert Strategy
```sql
MERGE `hankstank.mlb_historical_data.games_historical` AS target
USING (SELECT * FROM UNNEST(@new_games)) AS source
ON target.game_pk = source.game_pk AND target.game_date = source.game_date
WHEN MATCHED THEN
  UPDATE SET 
    home_score = source.home_score,
    away_score = source.away_score,
    updated_at = CURRENT_TIMESTAMP()
WHEN NOT MATCHED THEN
  INSERT (game_pk, game_date, home_team, away_team, home_score, away_score, created_at)
  VALUES (source.game_pk, source.game_date, source.home_team, source.away_team, 
          source.home_score, source.away_score, CURRENT_TIMESTAMP())
```

#### Optimization
- Batch inserts (up to 10,000 rows per API call)
- Partitioned tables (by year/month)
- Clustered by date and team_id
- Streaming inserts only for real-time needs

#### Output
- Publishes `data.loaded` event with row counts

---

### 4. Feature Engineering Service

**Technology:** Cloud Run (Python 3.11 + BigQuery)  
**Trigger:** Pub/Sub topic `data.loaded`  
**Scaling:** 0-10 instances, CPU allocated  
**Memory:** 2GB per instance

#### Responsibilities
- Calculate game-level features
- Generate player rolling statistics (7/14/30/60 day windows)
- Compute team aggregates
- Create advanced metrics (park factors, strength of schedule, etc.)
- Store features in BigQuery feature tables

#### Feature Categories

**1. Basic Features (Direct from raw data)**
```sql
-- Example: Team batting average last 30 days
CREATE OR REPLACE TABLE mlb_historical_data.team_rolling_features AS
SELECT 
  team_id,
  date,
  AVG(batting_avg) OVER (
    PARTITION BY team_id 
    ORDER BY date 
    ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
  ) as batting_avg_30d,
  ...
FROM team_stats_historical
WHERE date >= DATE_SUB(CURRENT_DATE(), INTERVAL 60 DAY)
```

**2. Advanced Features (Engineered)**
- Rolling window aggregations
- Opponent-adjusted metrics
- Home/away splits
- Rest days calculations
- Pitcher fatigue indicators
- Temporal features (day of week, month, etc.)

**3. Derived Features (ML-ready)**
- Feature interactions (team_strength * opponent_weakness)
- Normalized features (z-scores)
- Encoded categoricals (one-hot, target encoding)

#### Feature Storage Strategy

**Option A: BigQuery Tables (Recommended for cost)**
```
mlb_features/
  ├── game_features_daily        (Updated daily, ~200 games/day)
  ├── player_features_daily      (Updated daily, ~1000 players)
  ├── team_rolling_features      (Materialized view, refreshed daily)
  └── advanced_features          (Computed on-demand for training)
```

**Option B: BigQuery + Cloud Storage (Hybrid for flexibility)**
- BigQuery: Structured features for querying
- Cloud Storage: Parquet files for fast ML training

#### Materialized Views
```sql
-- Auto-refresh daily to keep features current
CREATE MATERIALIZED VIEW mlb_features.team_rolling_features AS
SELECT 
  team_id,
  date,
  -- 7-day rolling
  AVG(runs_scored) OVER w7 as runs_scored_7d,
  AVG(runs_allowed) OVER w7 as runs_allowed_7d,
  -- 30-day rolling  
  AVG(runs_scored) OVER w30 as runs_scored_30d,
  AVG(runs_allowed) OVER w30 as runs_allowed_30d
FROM mlb_historical_data.team_stats_historical
WINDOW 
  w7 AS (PARTITION BY team_id ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW),
  w30 AS (PARTITION BY team_id ORDER BY date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW)
```

#### Output
- Feature tables updated in BigQuery
- Publishes `features.created` event with stats (rows updated, tables affected)

---

### 5. Model Training Service

**Technology:** Cloud Run or Vertex AI Custom Training  
**Trigger:** Pub/Sub topic `features.created` OR scheduled weekly  
**Compute:** Configurable (CPU for simple models, GPU for deep learning)

#### Training Strategy

**Incremental vs Full Retraining Decision Tree:**
```
New data available?
├─ < 7 days worth → Incremental update (fast, cheap)
│  └─ Update model weights with new features
│  └─ Quick validation
│  └─ Deploy if performance >= current
│
└─ >= 7 days worth OR weekly trigger → Full retraining
   └─ Pull all historical + new features
   └─ Hyperparameter tuning (if monthly)
   └─ Full cross-validation
   └─ A/B test candidate vs production model
   └─ Deploy champion model

Season Phase Considerations:
├─ Spring Training → Train on exhibition data (lower weight)
│  └─ Use for early season predictions
│  └─ Focus on roster changes, player availability
│
├─ Regular Season → Primary training data
│  └─ Full weight in model training
│  └─ Continuous model improvement
│
└─ Postseason → High-stakes predictions!
   └─ Model frozen (no updates during playoffs)
   └─ Use best validated model from regular season
   └─ Extra features: playoff experience, momentum
```

#### Model Types

**1. Game Outcome Prediction**
- **Algorithm:** LightGBM or XGBoost (gradient boosting)
- **Why:** Fast, accurate, handles tabular data well
- **Features:** ~100-200 engineered features
- **Target:** Binary (win/loss) + probability
- **Training data:** Last 3 seasons (rolling window)

**2. Player Performance Prediction**
- **Algorithm:** Random Forest or LightGBM
- **Features:** Player history, opponent strength, park factors
- **Targets:** Multiple (hits, HRs, RBIs, etc.)
- **Training:** Separate model per stat or multi-output

**3. Advanced Models (Future)**
- **LSTM/GRU:** For time-series patterns
- **Neural Networks:** For complex interactions
- **Ensemble:** Combine multiple models

#### Training Pipeline
```python
def train_model(trigger_type: str):
    """
    trigger_type: 'daily', 'weekly', 'manual'
    """
    # 1. Pull features from BigQuery
    features = bq_client.query("""
        SELECT * FROM mlb_features.game_features_daily
        WHERE date >= DATE_SUB(CURRENT_DATE(), INTERVAL 1095 DAY)
    """).to_dataframe()
    
    # 2. Feature preprocessing
    X, y = prepare_features(features)
    
    # 3. Train/update model
    if trigger_type == 'daily':
        model = load_latest_model()
        model = incremental_fit(model, X, y)
    else:
        model = full_training(X, y, tune_hyperparams=(trigger_type=='weekly'))
    
    # 4. Evaluate
    metrics = evaluate_model(model, test_data)
    
    # 5. Compare to production
    if metrics['accuracy'] > production_metrics['accuracy']:
        deploy_model(model)
        log_deployment(metrics)
    else:
        log_inferior_model(metrics)
```

#### Model Versioning
- **Storage:** Cloud Storage bucket `gs://hankstank-models/`
- **Path structure:** `{model_type}/{version}/{model_file}`
  - `game_outcome/v1.2.3/model.pkl`
  - `player_performance/v2.0.1/model.pkl`
- **Metadata:** JSON with training date, features used, performance metrics
- **Retention:** Keep last 10 versions

#### Output
- Model artifact to Cloud Storage
- Metrics logged to Cloud Monitoring
- Publishes `model.deployed` event (if deployed)

---

### 6. Prediction API

**Technology:** Cloud Run (Python 3.11)  
**Scaling:** 1-50 instances (min 1 for fast cold start)  
**Endpoint:** `https://predictions-{hash}.run.app/predict`

#### Endpoints

**POST /predict/game**
```json
{
  "home_team": "NYY",
  "away_team": "BOS",
  "game_date": "2026-04-15",
  "starting_pitchers": {
    "home": 543037,
    "away": 605483
  }
}

Response:
{
  "game_id": "2026-04-15-NYY-BOS",
  "predictions": {
    "home_win_probability": 0.62,
    "away_win_probability": 0.38,
    "predicted_score": {"home": 5, "away": 3}
  },
  "confidence": 0.75,
  "model_version": "v1.2.3",
  "feature_importance": [...]
}
```

**POST /predict/player**
```json
{
  "player_id": 660271,
  "game_date": "2026-04-15",
  "opponent": "NYY",
  "stats": ["hits", "home_runs", "rbis"]
}
```

#### Caching Strategy
- **Redis Cloud Memorystore:** Cache predictions for upcoming games
- **TTL:** 12 hours (predictions updated twice daily)
- **Cache key:** `{model_version}:{team1}:{team2}:{date}`

---

## Data Pipeline Architecture

### Daily Workflow

```
Time (ET)  | Action                           | Service
-----------|----------------------------------|-------------------
02:00 AM   | Trigger data collection          | Cloud Scheduler
02:01 AM   | Check BigQuery for missing dates | Cloud Function
02:02 AM   | Determine date range to fetch    | Cloud Function
           | (yesterday + any gaps found)     |
02:05 AM   | Fetch MLB data (smart range)     | Cloud Function
           | - Games (box scores, line scores)|
           | - Player stats (all players)     |
           | - Team stats (all 30 teams)      |
           | - Transactions (daily)           |
           | - Rosters (all teams)            |
02:10 AM   | Fetch Statcast pitch data        | Cloud Function
           | - All pitches from completed games|
           | - ~45k pitches for 15 games      |
02:20 AM   | Validate all collected data      | Cloud Function
           | - Schema validation              |
           | - Completeness checks            |
           | - Player count verification      |
02:30 AM   | Load to BigQuery (MERGE upsert)  | Cloud Function
           | - No duplicates created          |
           | - Updates existing records       |
02:40 AM   | Generate features                | Cloud Run
02:55 AM   | Update model (incremental)       | Cloud Run
03:15 AM   | Generate daily predictions       | Cloud Run
03:30 AM   | Send summary report              | Cloud Function
           | - Data collected stats           |
           | - Quality grades                 |
           | - Errors (if any)                |
```

**Smart Collection Logic (Works for All Season Phases):**
```python
def daily_collection_trigger():
    """
    Daily job that determines what to collect
    Automatically handles: Spring Training, Regular Season, Postseason
    """
    # Default: collect yesterday's data
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    # Check what season phase we're in
    season_phase = get_current_season_phase(yesterday)
    
    if season_phase == 'off_season':
        logger.info("Off-season, no games expected. Skipping collection.")
        return {'status': 'skipped', 'reason': 'off_season'}
    
    logger.info(f"Current season phase: {season_phase}")
    
    # Check for gaps in last 7 days (in case we missed any)
    seven_days_ago = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    
    missing_dates = find_missing_dates(seven_days_ago, yesterday, 
                                       get_existing_data_summary(seven_days_ago, yesterday))
    
    if not missing_dates:
        logger.info("No missing data, nothing to collect")
        return {'status': 'skipped', 'reason': 'all_data_exists'}
    
    logger.info(f"Collecting data for dates: {missing_dates} (phase: {season_phase})")
    
    # Use date range API to fetch all missing dates efficiently
    # Works for all game types (spring, regular, postseason)
    if len(missing_dates) > 1:
        # Fetch range (more efficient for consecutive dates)
        start = min(missing_dates)
        end = max(missing_dates)
        fetch_date_range(start, end, season_phase=season_phase)
    else:
        # Fetch single date
        fetch_all_data_for_date(missing_dates[0], season_phase=season_phase)
    
    return {
        'status': 'collected', 
        'dates': missing_dates,
        'season_phase': season_phase
    }
```

### Weekly Workflow (Sundays)

```
Time (ET)  | Action                           | Service
-----------|----------------------------------|-------------------
03:00 AM   | Full model retraining            | Vertex AI
04:00 AM   | Model evaluation & A/B test      | Vertex AI
04:30 AM   | Deploy champion model            | Cloud Run
05:00 AM   | Update monitoring dashboards     | Cloud Monitoring
```

### Retry & Error Handling

```
Data Collection Fails
  ├─ Retry: 3 times with exponential backoff
  ├─ Fallback to secondary source
  ├─ If all fail → Dead letter queue
  └─ Alert via email + Slack

Validation Fails
  ├─ Grade F → Trigger re-collection
  ├─ Grade C → Proceed with warning
  └─ Log quality metrics

BigQuery Load Fails
  ├─ Retry: 2 times
  ├─ If schema error → Alert engineer
  └─ Store data in Cloud Storage for manual fix

Feature Creation Fails
  ├─ Retry: 1 time
  ├─ Use cached features from previous day
  └─ Alert if critical features missing

Model Training Fails
  ├─ Keep using previous model
  ├─ Alert engineer
  └─ Retry tomorrow
```

---

## Feature Engineering Pipeline

### Feature Types & Computation

#### 1. Real-Time Features (Computed on-demand)
- Current roster
- Latest player stats (today)
- Live weather data

**Storage:** Cloud Firestore or Redis  
**Why:** Low latency reads, frequent updates

#### 2. Daily Batch Features (Computed overnight)
- Rolling window statistics
- Team aggregates
- Player trend indicators

**Storage:** BigQuery tables  
**Why:** Cost-effective, SQL-based, handles large volumes

#### 3. Historical Features (Pre-computed, rarely change)
- Career statistics
- Park factors (by year)
- Historical matchups

**Storage:** BigQuery (partitioned by year)  
**Why:** Efficient storage, fast joins

### Feature Calculation Flow

```
BigQuery Raw Data
  │
  ├─ SQL Transformations (in BigQuery)
  │   ├─ Window functions for rolling aggregates
  │   ├─ JOINs for enrichment
  │   └─ GROUP BY for team/player summaries
  │
  ├─ Store to Feature Tables
  │   ├─ game_features_daily (partitioned by date)
  │   ├─ player_features_daily (partitioned by date) 
  │   └─ team_rolling_features (materialized view)
  │
  └─ Trigger model update
```

### Example: Complex Feature Calculation

```sql
-- Calculate team strength adjusted for opponent quality
CREATE OR REPLACE TABLE mlb_features.team_strength_adjusted AS
WITH team_stats AS (
  SELECT 
    team_id,
    date,
    runs_scored,
    runs_allowed
  FROM mlb_historical_data.team_stats_historical
),
opponent_quality AS (
  SELECT
    g.game_date,
    g.home_team,
    g.away_team,
    AVG(ts.runs_allowed) OVER (
      PARTITION BY ts.team_id 
      ORDER BY ts.date 
      ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
    ) as opponent_runs_allowed_30d
  FROM mlb_historical_data.games_historical g
  JOIN team_stats ts ON g.away_team = ts.team_id
)
SELECT 
  t.team_id,
  t.date,
  t.runs_scored / NULLIF(oq.opponent_runs_allowed_30d, 0) as strength_adj_offense,
  oq.opponent_runs_allowed_30d / NULLIF(t.runs_allowed, 0) as strength_adj_defense
FROM team_stats t
JOIN opponent_quality oq ON t.team_id = oq.home_team AND t.date = oq.game_date
```

---

## Model Training & Serving

### Training Infrastructure

**Option 1: Cloud Run (Recommended for simple models)**
- **Cost:** ~$0.10-0.50 per training job
- **Speed:** 5-15 minutes
- **Best for:** LightGBM, XGBoost, scikit-learn

**Option 2: Vertex AI Custom Training (For complex models)**
- **Cost:** ~$0.50-5.00 per training job (depends on machine type)
- **Speed:** 10-60 minutes
- **Best for:** Neural networks, hyperparameter tuning, AutoML

**Option 3: Vertex AI AutoML (Lowest code, highest cost)**
- **Cost:** ~$20 per training job
- **Speed:** 1-4 hours
- **Best for:** Initial baseline, experimenting

### Serving Infrastructure

**Cloud Run Prediction API:**
- **Latency:** <100ms (p99)
- **Throughput:** 1000+ requests/second
- **Cost:** ~$10/month for low traffic
- **Scaling:** Automatic (0 to 100+ instances)

**Model Loading Strategy:**
```python
# Global scope - loaded once per instance
model_cache = {}

def load_model(model_version: str):
    """Load model from Cloud Storage, cache in memory"""
    if model_version not in model_cache:
        blob = storage_client.bucket('hankstank-models').blob(
            f'game_outcome/{model_version}/model.pkl'
        )
        model_cache[model_version] = pickle.loads(blob.download_as_bytes())
    return model_cache[model_version]

@app.route('/predict', methods=['POST'])
def predict():
    model = load_model('latest')  # Cached after first call
    features = extract_features(request.json)
    prediction = model.predict(features)
    return jsonify(prediction)
```

### A/B Testing Framework

```python
def route_prediction_request(request):
    """
    Route 95% to production model, 5% to candidate model
    Track performance for both
    """
    if random.random() < 0.05:
        model_version = 'candidate'
    else:
        model_version = 'production'
    
    prediction = predict_with_model(request, model_version)
    
    # Log prediction for later evaluation
    log_prediction(
        model_version=model_version,
        request=request,
        prediction=prediction,
        timestamp=datetime.now()
    )
    
    return prediction

# Daily job: Compare actual outcomes to predictions
def evaluate_ab_test():
    """
    After games complete, compare prediction accuracy
    If candidate > production (statistically significant):
        Promote candidate to production
    """
    results = query_predictions_and_outcomes()
    candidate_acc = calculate_accuracy(results, 'candidate')
    production_acc = calculate_accuracy(results, 'production')
    
    if candidate_acc > production_acc + 0.02:  # 2% improvement threshold
        promote_candidate_to_production()
```

---

## Infrastructure & Deployment

### GCP Project Structure

```
hankstank (GCP Project)
├── BigQuery Datasets
│   ├── mlb_historical_data (raw data tables)
│   │   ├── games_historical (partitioned by game_date)
│   │   ├── statcast_pitches (partitioned by game_date)
│   │   ├── player_stats_historical (partitioned by date)
│   │   ├── team_stats_historical (partitioned by date)
│   │   ├── transactions_historical (partitioned by date)
│   │   ├── rosters_historical (partitioned by roster_date)
│   │   ├── standings_historical
│   │   └── teams_historical
│   ├── mlb_features (feature tables)
│   └── mlb_predictions (prediction logs)
│
├── Cloud Storage Buckets
│   ├── hankstank-models (model artifacts)
│   ├── hankstank-data-staging (temporary data files)
│   └── hankstank-logs (archived logs)
│
├── Cloud Functions
│   ├── data-collector (gen 2)
│   ├── data-validator (gen 2)
│   └── bigquery-loader (gen 2)
│
├── Cloud Run Services
│   ├── feature-engine (CPU, 2GB RAM)
│   ├── model-trainer (CPU/GPU, 8GB RAM)
│   └── prediction-api (CPU, 2GB RAM, autoscaling)
│
├── Cloud Scheduler Jobs
│   ├── daily-collection (cron: 0 2 * * *)
│   │   # Runs year-round, auto-detects season phase
│   │   # Skips if off-season (no games scheduled)
│   ├── weekly-training (cron: 0 3 * * 0)
│   └── monthly-full-retrain (cron: 0 4 1 * *)
│   │   # Pauses during off-season, resumes for spring training
│
├── Pub/Sub Topics
│   ├── data.collected
│   ├── data.validated
│   ├── data.loaded
│   ├── features.created
│   └── model.deployed
│
└── Secret Manager (API keys, credentials)
    ├── mlb-api-key
    ├── weather-api-key
    └── monitoring-webhook
```

### Infrastructure as Code

**Use Terraform for reproducibility:**

```hcl
# terraform/main.tf
resource "google_bigquery_dataset" "mlb_features" {
  dataset_id  = "mlb_features"
  project     = "hankstank"
  location    = "US"
  description = "ML feature tables"
  
  access {
    role          = "OWNER"
    user_by_email = google_service_account.feature_engine.email
  }
}

resource "google_cloud_run_service" "prediction_api" {
  name     = "prediction-api"
  location = "us-central1"
  
  template {
    spec {
      containers {
        image = "gcr.io/hankstank/prediction-api:latest"
        resources {
          limits = {
            memory = "2Gi"
            cpu    = "2"
          }
        }
      }
      service_account_name = google_service_account.prediction_api.email
    }
    
    metadata {
      annotations = {
        "autoscaling.knative.dev/minScale" = "1"
        "autoscaling.knative.dev/maxScale" = "50"
      }
    }
  }
}
```

### Deployment Strategy

**CI/CD Pipeline (GitHub Actions):**

```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run tests
        run: |
          pip install -r requirements.txt
          pytest tests/
          
  deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - name: Deploy Cloud Functions
        run: |
          gcloud functions deploy data-collector \
            --gen2 \
            --runtime python311 \
            --entry-point main \
            --trigger-topic data-collection-trigger
            
      - name: Deploy Cloud Run
        run: |
          gcloud run deploy prediction-api \
            --image gcr.io/hankstank/prediction-api:${{ github.sha }} \
            --region us-central1 \
            --allow-unauthenticated
```

---

## Cost Optimization

### Monthly Cost Estimate

| Service | Usage | Monthly Cost |
|---------|-------|--------------|
| **BigQuery Storage** | 150 GB (season data + features) | $3.00 |
| **BigQuery Queries** | 200 GB processed/day | $18.00 |
| **BigQuery Streaming** | 26 MB/day × 30 days | $0.10 |
| **Cloud Functions** | 30 invocations/day × 60s | $1.00 |
| **Cloud Run (Feature Engine)** | 30 runs/day × 5min | $2.00 |
| **Cloud Run (Training)** | 7 runs/week × 15min | $1.00 |
| **Cloud Run (Prediction API)** | 1000 req/day, 1 instance min | $10.00 |
| **Cloud Storage** | 20 GB models + staging | $0.40 |
| **Cloud Scheduler** | 3 jobs | $0.30 |
| **Pub/Sub** | 2M messages/month | $0.80 |
| **Cloud Monitoring** | Basic metrics | $0.00 (free tier) |
| **Networking** | Egress (API calls, minimal) | $2.00 |
| **TOTAL** | | **~$38/month** |

### Cost Optimization Strategies

#### 1. BigQuery Optimization
```sql
-- ❌ BAD: Scans entire table (expensive)
SELECT * FROM mlb_historical_data.games_historical
WHERE game_date = '2026-04-15'

-- ✅ GOOD: Uses partition pruning (cheap)
SELECT * FROM mlb_historical_data.games_historical
WHERE game_date = '2026-04-15'  -- Partition column
AND home_team = 'NYY'           -- Cluster column
```

**Best Practices:**
- Partition tables by date/year
- Cluster by commonly filtered columns (team_id, player_id)
- Use `_TABLE_SUFFIX` for date-partitioned tables
- Avoid `SELECT *`, specify needed columns
- Use materialized views for repeated queries

#### 2. Cloud Run Cost Reduction
- **Min instances: 1** for prediction API (fast response, worth $7/month)
- **Min instances: 0** for training jobs (only run when needed)
- **CPU allocated:** Only during request (not idle)
- **Right-size memory:** 2GB is usually sufficient

#### 3. Storage Tiering
```python
# Lifecycle policy: Move old data to cheaper storage
lifecycle_policy = {
    "rule": [
        {
            "action": {"type": "SetStorageClass", "storageClass": "NEARLINE"},
            "condition": {"age": 90}  # 90 days
        },
        {
            "action": {"type": "SetStorageClass", "storageClass": "COLDLINE"},
            "condition": {"age": 365}  # 1 year
        }
    ]
}
```

#### 4. Batch Processing
- Collect data once daily (not per-game)
- Batch BigQuery inserts (10K rows at a time)
- Use BigQuery batch queries (not streaming) when possible

#### 5. Caching
- Cache predictions in Redis (12h TTL)
- Cache model in Cloud Run memory (avoid reloading)
- Cache feature computations when possible

### Free Tier Benefits (Always-Free)
- **Cloud Functions:** 2M invocations/month
- **Cloud Run:** 2M requests/month
- **BigQuery:** 1 TB queries/month, 10 GB storage
- **Cloud Storage:** 5 GB/month
- **Pub/Sub:** 10 GB/month

**With careful optimization, could run under $20/month during off-season!**

---

## Error Handling & Monitoring

### Logging Strategy

**Cloud Logging (Structured Logs):**
```python
import logging
from google.cloud import logging as cloud_logging

# Setup
client = cloud_logging.Client()
client.setup_logging()
logger = logging.getLogger(__name__)

# Log structured data
logger.info("Data collection started", extra={
    "date": "2026-04-15",
    "games_scheduled": 12,
    "component": "data-collector"
})

logger.error("API request failed", extra={
    "error": str(e),
    "source": "MLB Stats API",
    "retry_count": 3,
    "severity": "ERROR"
})
```

### Monitoring Dashboards

**Key Metrics to Track:**

1. **Data Pipeline Health**
   - Games collected per day (expected vs actual)
   - Data quality score (A/B/C/F grades)
   - API success rate
   - Processing latency (end-to-end)

2. **Feature Engineering**
   - Features computed (count, timing)
   - Null/missing value rates
   - Feature staleness (last update time)

3. **Model Performance**
   - Prediction accuracy (rolling 7-day)
   - Prediction latency (p50, p95, p99)
   - Model drift (feature distribution shifts)
   - A/B test results

4. **System Health**
   - Cloud Run instances (scaling behavior)
   - BigQuery slot usage
   - Error rates by component
   - Cost per day

### Alerting Rules

```yaml
# monitoring/alerts.yaml
alerts:
  - name: "Data Collection Failed"
    condition: "error_count > 3 in 1 hour"
    severity: "critical"
    notification: "email + slack"
    
  - name: "Model Accuracy Drop"
    condition: "accuracy < 0.50 for 3 days"
    severity: "high"
    notification: "email"
    
  - name: "BigQuery Cost Spike"
    condition: "daily_cost > $5"
    severity: "medium"
    notification: "email"
    
  - name: "Prediction API Down"
    condition: "uptime < 99% in 1 hour"
    severity: "critical"
    notification: "email + slack"
```

### Dead Letter Queue (DLQ)

**For failed events that need manual review:**

```python
def handle_failed_event(event, error):
    """
    Store failed events in DLQ for later processing
    """
    dlq_table = bq_client.dataset('mlb_historical_data').table('dead_letter_queue')
    
    row = {
        'event_id': event['id'],
        'event_type': event['type'],
        'payload': json.dumps(event['data']),
        'error_message': str(error),
        'failed_at': datetime.now().isoformat(),
        'retry_count': event.get('retry_count', 0)
    }
    
    bq_client.insert_rows_json(dlq_table, [row])
    
    # Alert if DLQ is filling up
    if get_dlq_size() > 100:
        send_alert("DLQ has 100+ events, manual review needed")
```

### Self-Healing Mechanisms

```python
def self_heal_pipeline():
    """
    Automated recovery for common issues
    """
    # 1. Check for stale data
    last_update = get_last_data_update()
    if (datetime.now() - last_update) > timedelta(days=2):
        trigger_backfill(start_date=last_update, end_date='today')
    
    # 2. Check for missing features
    missing_features = detect_missing_features()
    if missing_features:
        regenerate_features(missing_features)
    
    # 3. Check for model staleness
    if model_age() > timedelta(days=14):
        trigger_model_retraining()
    
    # 4. Validate data quality
    quality_report = run_validation()
    if quality_report['grade'] == 'F':
        retry_data_collection()
```

---

## Security & Access Control

### IAM Roles & Service Accounts

**Principle of Least Privilege:**

```
Service Account: data-collector@hankstank.iam
  Roles:
    ✅ Secret Manager Secret Accessor (read API keys)
    ✅ Storage Object Creator (write to staging bucket)
    ✅ Pub/Sub Publisher (trigger next step)
    ❌ BigQuery Data Editor (doesn't need direct BQ access)

Service Account: bigquery-loader@hankstank.iam
  Roles:
    ✅ BigQuery Data Editor (insert/update data)
    ✅ Storage Object Viewer (read from staging)
    ✅ Pub/Sub Publisher (trigger feature engineering)
    ❌ Storage Admin (doesn't need delete)

Service Account: feature-engine@hankstank.iam
  Roles:
    ✅ BigQuery Data Editor (write features)
    ✅ BigQuery Job User (run queries)
    ✅ Pub/Sub Publisher (trigger training)

Service Account: model-trainer@hankstank.iam
  Roles:
    ✅ BigQuery Data Viewer (read features)
    ✅ Storage Object Admin (save models)
    ✅ Vertex AI User (if using Vertex AI)
    ✅ Pub/Sub Publisher (notify deployment)

Service Account: prediction-api@hankstank.iam
  Roles:
    ✅ Storage Object Viewer (load models)
    ✅ BigQuery Data Viewer (fetch features if needed)
    ❌ No write permissions
```

### API Security

**Prediction API Authentication:**

```python
# Option 1: API Key (for public use)
@app.before_request
def validate_api_key():
    api_key = request.headers.get('X-API-Key')
    if api_key != os.environ['EXPECTED_API_KEY']:
        abort(401, "Invalid API key")

# Option 2: Google Cloud IAM (for internal use)
# Cloud Run automatically validates service account tokens
# No code needed, configure IAM permissions

# Option 3: Rate Limiting (prevent abuse)
from flask_limiter import Limiter

limiter = Limiter(
    app,
    key_func=lambda: request.remote_addr,
    default_limits=["100 per hour", "10 per minute"]
)

@app.route('/predict')
@limiter.limit("10 per minute")
def predict():
    ...
```

### Secret Management

**Never hardcode credentials:**

```python
from google.cloud import secretmanager

def get_secret(secret_id: str) -> str:
    """Fetch secret from Secret Manager"""
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/hankstank/secrets/{secret_id}/versions/latest"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode('UTF-8')

# Usage
MLB_API_KEY = get_secret("mlb-api-key")
```

### Data Privacy & Compliance

**PII Handling (if collecting user data in future):**
- ❌ Don't store PII in logs
- ✅ Encrypt sensitive data at rest (GCS encryption enabled by default)
- ✅ Use VPC Service Controls for network isolation (if needed)
- ✅ Audit access with Cloud Audit Logs

---

## Implementation Roadmap

### Phase 1: MVP (Weeks 1-2)
**Goal:** Automated daily data collection + BigQuery sync

- [x] Current: Manual collection scripts
- [ ] Migrate to Cloud Functions (data-collector)
- [ ] Setup Cloud Scheduler (daily trigger)
- [ ] Implement retry logic
- [ ] Setup BigQuery MERGE upserts
- [ ] Basic validation checks
- [ ] Pub/Sub event flow
- [ ] Error notifications (email)

**Deliverables:**
- Fully automated daily data pipeline
- No manual intervention needed for data collection
- Data quality reports

---

### Phase 2: Feature Engineering (Weeks 3-4)
**Goal:** Automated feature generation from raw data

- [ ] Design feature tables schema
- [ ] Implement SQL-based feature calculations
- [ ] Create Cloud Run service for feature engine
- [ ] Setup materialized views for rolling windows
- [ ] Trigger feature creation on data.loaded event
- [ ] Validate feature quality
- [ ] Document feature catalog

**Deliverables:**
- ML-ready feature tables in BigQuery
- Daily feature updates
- Feature lineage tracking

---

### Phase 3: Model Training Automation (Weeks 5-6)
**Goal:** Automated model training and deployment

- [ ] Baseline model (LightGBM game outcome prediction)
- [ ] Training pipeline (Cloud Run or Vertex AI)
- [ ] Model versioning system
- [ ] A/B testing framework
- [ ] Automated model evaluation
- [ ] Model registry (Cloud Storage)
- [ ] Rollback capability

**Deliverables:**
- Weekly automated retraining
- Model performance tracking
- Champion/challenger framework

---

### Phase 4: Prediction API (Weeks 7-8)
**Goal:** Serve predictions via REST API

- [ ] Cloud Run prediction service
- [ ] API endpoint design
- [ ] Model loading optimization
- [ ] Caching layer (Redis)
- [ ] Rate limiting
- [ ] API documentation
- [ ] Basic frontend (optional)

**Deliverables:**
- Production prediction API
- Sub-100ms latency
- 99.9% uptime

---

### Phase 5: Monitoring & Optimization (Weeks 9-10)
**Goal:** Production-ready observability

- [ ] Comprehensive logging (structured)
- [ ] Monitoring dashboards (Cloud Monitoring)
- [ ] Alerting rules
- [ ] Cost tracking dashboard
- [ ] Performance optimization
- [ ] Self-healing mechanisms
- [ ] Runbook documentation

**Deliverables:**
- Full observability stack
- Automated alerting
- Cost < $50/month

---

### Phase 6: Advanced Features (Weeks 11-12)
**Goal:** Enhanced predictions and external data

- [ ] Weather data integration
- [ ] Park factors
- [ ] Player injury tracking
- [ ] Advanced sabermetrics features
- [ ] Ensemble models
- [ ] Player performance predictions
- [ ] Prediction explanations (SHAP values)

**Deliverables:**
- Multi-model prediction system
- External data pipelines
- Improved accuracy

---

### Future Enhancements (Post-2026)

**Season-Agnostic Design:**
- Configuration-driven (seasons defined in YAML/JSON)
- Automatic schema detection for new fields
- Backward-compatible feature engineering
- Multi-season model training

**Advanced ML:**
- Deep learning models (LSTM for time series)
- AutoML for hyperparameter tuning
- Online learning (real-time model updates)
- Prediction intervals (uncertainty quantification)

**Scaling:**
- Multi-region deployment (latency optimization)
- Real-time predictions (streaming features)
- Mobile app integration
- Public API (monetization?)

---

## Technology Choices Rationale

### Why GCP?
- ✅ Already using BigQuery
- ✅ Serverless offerings (Cloud Functions, Cloud Run)
- ✅ Excellent ML tools (Vertex AI)
- ✅ Cost-effective for low-traffic use cases
- ✅ Free tier generous

### Why BigQuery for Features?
- ✅ Already have data there
- ✅ SQL-based feature engineering (easy to maintain)
- ✅ Handles large datasets efficiently
- ✅ Materialized views for auto-refresh
- ✅ Direct integration with ML tools

**Alternative:** Cloud Storage + Parquet
- Better for: Very large feature sets, frequent full-table scans
- Worse for: Ad-hoc queries, feature exploration

### Why Cloud Run?
- ✅ Serverless (no ops)
- ✅ Scales to zero (cost-effective)
- ✅ Fast cold starts (<1s)
- ✅ Run any container (flexible)
- ✅ Easy CI/CD integration

**Alternative:** Kubernetes (GKE)
- Better for: Complex multi-service deployments, fine-grained control
- Worse for: Cost (always-on cluster), ops overhead

### Why LightGBM/XGBoost?
- ✅ Fast training and inference
- ✅ Excellent for tabular data
- ✅ Built-in feature importance
- ✅ Handles missing values well
- ✅ Low resource requirements

**Alternative:** Neural Networks
- Better for: Complex non-linear patterns, very large datasets
- Worse for: Interpretability, training time, resource needs

---

## Success Metrics

### System Health
- **Uptime:** >99.5% (max 3.6 hours downtime/month)
- **Data freshness:** <24 hours lag
- **Feature completeness:** >95% of expected features
- **Pipeline success rate:** >98%

### Model Performance
- **Game outcome accuracy:** >55% (better than coin flip!)
- **Calibration:** Brier score <0.25
- **Prediction latency:** <100ms (p99)

### Operational
- **Monthly cost:** <$50
- **Manual interventions:** <1 per week
- **Incident response time:** <2 hours
- **Data quality:** >90% Grade A

---

## Conclusion

This architecture provides a **scalable, cost-effective, and fully automated** system for MLB data collection, feature engineering, and prediction serving. Key benefits:

1. **Automation:** Zero-touch daily operations
2. **Resilience:** Multi-source fallbacks, retry logic, self-healing
3. **Cost:** ~$35/month in production
4. **Scalability:** Handles 2026 and beyond with no changes
5. **Security:** IAM-based access control, secret management
6. **Observability:** Comprehensive logging and monitoring

**Next Steps:**
1. Review this architecture plan
2. Prioritize phases based on immediate needs
3. Setup GCP infrastructure (Terraform)
4. Begin Phase 1 implementation

Let's build this! 🚀⚾
