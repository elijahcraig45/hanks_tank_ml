# BigQuery Data Schema Documentation
**Project:** Hank's Tank MLB Data Platform  
**Dataset:** `hankstank.mlb_historical_data`  
**Last Updated:** December 24, 2025

---

## Overview

This document provides a comprehensive schema reference for all MLB data stored in BigQuery. The dataset contains **8 tables** with historical data spanning **2015-2025**, including team stats, player stats, game data, transactions, and detailed Statcast pitch-by-pitch analytics.

### Quick Stats
- **Total Tables:** 8
- **Total Records:** ~8.4 million rows
- **Year Coverage:** 2015-2025 (varies by table)
- **Dataset Name:** `mlb_historical_data`
- **Project ID:** `hankstank`

---

## Table of Contents
1. [teams_historical](#1-teams_historical)
2. [team_stats_historical](#2-team_stats_historical)
3. [player_stats_historical](#3-player_stats_historical)
4. [standings_historical](#4-standings_historical)
5. [games_historical](#5-games_historical)
6. [rosters_historical](#6-rosters_historical)
7. [transactions_historical](#7-transactions_historical)
8. [statcast_pitches](#8-statcast_pitches)
9. [How to Access Data](#how-to-access-data)

---

## 1. teams_historical

**Purpose:** Stores basic team information for each MLB season.

### Data Coverage
- **Total Records:** 330
- **Years:** 2015-2025 (11 complete seasons)
- **Coverage:** ✅ **COMPLETE** - All 30 teams × 11 years
- **Status:** ✅ **INCLUDES 2025** - Backfilled December 24, 2025

### Schema

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `year` | INTEGER | Season year | `2024` |
| `team_id` | INTEGER | Unique MLB team ID | `144` |
| `team_name` | STRING | Full team name | `Atlanta Braves` |
| `team_code` | STRING | Three-letter abbreviation | `ATL` |
| `location_name` | STRING | City/region name | `Atlanta` |
| `team_name_full` | STRING | Team nickname | `Braves` |
| `league_id` | INTEGER | League ID (103=AL, 104=NL) | `104` |
| `league_name` | STRING | League name | `National League` |
| `division_id` | INTEGER | Division ID | `204` |
| `division_name` | STRING | Division name | `National League East` |
| `venue_id` | INTEGER | Home stadium ID | `4705` |
| `venue_name` | STRING | Stadium name | `Truist Park` |
| `first_year_of_play` | INTEGER | Franchise inception year | `1966` |
| `active` | BOOLEAN | Whether team is active | `true` |

### Sample Query
```sql
-- Get all Braves team records
SELECT * 
FROM `hankstank.mlb_historical_data.teams_historical`
WHERE team_id = 144
ORDER BY year DESC;
```

### Access Pattern
- Used for team lookups, franchise history
- Join key: `team_id` + `year`
- Historical seasons only (2015-2024)
- Current season (2025) fetched from live MLB API

---

## 2. team_stats_historical

**Purpose:** Comprehensive team batting and pitching statistics for each season.

### Data Coverage
- **Total Records:** 700
- **Years:** 2015-2025 (11 complete seasons)
- **Coverage:** ✅ **COMPLETE** - All teams × 2 stat types × 11 years
- **Stat Types:** `batting` (350 records), `pitching` (350 records)
- **Status:** ✅ **INCLUDES 2025** - Backfilled December 24, 2025

### Schema

#### Core Fields
| Field | Type | Description |
|-------|------|-------------|
| `year` | INTEGER | Season year |
| `team_id` | INTEGER | Team identifier |
| `team_name` | STRING | Team name |
| `stat_type` | STRING | `batting` or `pitching` |
| `games_played` | INTEGER | Games played |

#### Batting Stats (when stat_type = 'batting')
| Field | Type | Description |
|-------|------|-------------|
| `at_bats` | FLOAT | At bats |
| `runs` | FLOAT | Runs scored |
| `hits` | FLOAT | Hits |
| `doubles` | FLOAT | Doubles |
| `triples` | FLOAT | Triples |
| `home_runs` | FLOAT | Home runs |
| `rbi` | FLOAT | Runs batted in |
| `stolen_bases` | FLOAT | Stolen bases |
| `caught_stealing` | FLOAT | Caught stealing |
| `walks` | FLOAT | Walks (BB) |
| `strikeouts` | INTEGER | Strikeouts |
| `batting_avg` | FLOAT | Batting average |
| `obp` | FLOAT | On-base percentage |
| `slg` | FLOAT | Slugging percentage |
| `ops` | FLOAT | On-base + slugging |
| `total_bases` | FLOAT | Total bases |
| `hit_by_pitch` | FLOAT | Hit by pitch |
| `sac_flies` | FLOAT | Sacrifice flies |
| `sac_bunts` | FLOAT | Sacrifice bunts |
| `left_on_base` | FLOAT | Runners left on base |

#### Pitching Stats (when stat_type = 'pitching')
| Field | Type | Description |
|-------|------|-------------|
| `wins` | FLOAT | Wins |
| `losses` | FLOAT | Losses |
| `win_percentage` | FLOAT | Win percentage |
| `era` | FLOAT | Earned run average |
| `games_started` | FLOAT | Games started |
| `games_finished` | FLOAT | Games finished |
| `complete_games` | FLOAT | Complete games |
| `shutouts` | FLOAT | Shutouts |
| `saves` | FLOAT | Saves |
| `save_opportunities` | FLOAT | Save opportunities |
| `holds` | FLOAT | Holds |
| `blown_saves` | FLOAT | Blown saves |
| `innings_pitched` | FLOAT | Innings pitched |
| `hits_allowed` | FLOAT | Hits allowed |
| `runs_allowed` | FLOAT | Runs allowed |
| `earned_runs` | FLOAT | Earned runs |
| `home_runs_allowed` | FLOAT | Home runs allowed |
| `walks_allowed` | FLOAT | Walks allowed |
| `strikeouts` | FLOAT | Strikeouts |
| `whip` | FLOAT | Walks + hits per IP |
| `batters_faced` | FLOAT | Batters faced |
| `wild_pitches` | FLOAT | Wild pitches |
| `hit_batsmen` | FLOAT | Hit batsmen |
| `balks` | FLOAT | Balks |

### Sample Queries
```sql
-- Get Braves batting stats for 2024
SELECT * 
FROM `hankstank.mlb_historical_data.team_stats_historical`
WHERE team_id = 144 
  AND year = 2024 
  AND stat_type = 'batting';

-- Compare team ERAs across seasons
SELECT year, team_name, era
FROM `hankstank.mlb_historical_data.team_stats_historical`
WHERE stat_type = 'pitching'
  AND year >= 2020
ORDER BY era ASC;
```

---

## 3. player_stats_historical

**Purpose:** Individual player batting and pitching statistics.

### Data Coverage
- **Total Records:** 1,706
- **Years:** 2015-2025 (11 seasons)
- **Unique Players:** 614+
- **Stat Types:** `batting`, `pitching`
- **Coverage:** ⚠️ **PARTIAL** - Top ~50-100 players per season (not comprehensive)
- **Status:** ✅ **INCLUDES 2025** - Backfilled December 24, 2025 (152 player records added)

### Schema
**Note:** Schema mirrors `team_stats_historical` but with player-specific fields:

| Field | Type | Description |
|-------|------|-------------|
| `year` | INTEGER | Season year |
| `player_id` | INTEGER | Unique player identifier |
| `player_name` | STRING | Player full name |
| `team_id` | INTEGER | Player's team |
| `team_name` | STRING | Team name |
| `stat_type` | STRING | `batting` or `pitching` |
| ... | ... | (Same stat fields as team_stats) |

### Sample Query
```sql
-- Get all seasons for a specific player
SELECT year, team_name, stat_type, 
       batting_avg, home_runs, rbi
FROM `hankstank.mlb_historical_data.player_stats_historical`
WHERE player_id = 660271  -- Ronald Acuña Jr.
  AND stat_type = 'batting'
ORDER BY year DESC;
```

### Limitations
- **Not comprehensive** - Only stores top performers
- For complete player rosters, use `rosters_historical`
- For exhaustive stats, query MLB API directly or use Statcast data

---

## 4. standings_historical

**Purpose:** End-of-season standings for all teams.

### Data Coverage
- **Total Records:** 330
- **Years:** 2015-2025 (11 complete seasons)
- **Coverage:** ✅ **COMPLETE** - All 30 teams × 11 years
- **Status:** ✅ **INCLUDES 2025** - Backfilled December 24, 2025

### Schema

| Field | Type | Description |
|-------|------|-------------|
| `year` | INTEGER | Season year |
| `team_id` | INTEGER | Team identifier |
| `team_name` | STRING | Team name |
| `league_id` | INTEGER | League ID |
| `league_name` | STRING | League name |
| `division_id` | INTEGER | Division ID |
| `division_name` | STRING | Division name |
| `wins` | INTEGER | Wins |
| `losses` | INTEGER | Losses |
| `win_percentage` | FLOAT | Winning percentage |
| `games_back` | FLOAT | Games behind division leader |
| `wildcard_games_back` | FLOAT | Games behind wildcard |
| `division_rank` | STRING | Rank in division |
| `league_rank` | STRING | Rank in league |
| `runs_scored` | INTEGER | Total runs scored |
| `runs_allowed` | INTEGER | Total runs allowed |
| `run_differential` | INTEGER | Run differential |
| `home_wins` | INTEGER | Home record wins |
| `home_losses` | INTEGER | Home record losses |
| `away_wins` | INTEGER | Away record wins |
| `away_losses` | INTEGER | Away record losses |

### Sample Query
```sql
-- Get NL East standings for 2024
SELECT team_name, wins, losses, win_percentage, 
       games_back, division_rank
FROM `hankstank.mlb_historical_data.standings_historical`
WHERE year = 2024 
  AND division_name = 'National League East'
ORDER BY CAST(division_rank AS INT64);
```

---

## 5. games_historical

**Purpose:** Individual game results and metadata.

### Data Coverage
- **Total Records:** 27,703
- **Years:** 2015-2025 (11 complete seasons)
- **Coverage:** ✅ **COMPLETE** - All regular season + playoff games
- **Average:** ~2,500 games per season
- **Status:** ✅ **INCLUDES 2025** - 2,511 games backfilled December 24, 2025

### Schema

| Field | Type | Description |
|-------|------|-------------|
| `year` | INTEGER | Season year |
| `game_pk` | INTEGER | Unique game identifier |
| `game_date` | DATE/STRING | Game date |
| `game_type` | STRING | R=Regular, P=Playoff, etc. |
| `season` | INTEGER | Season year |
| `home_team_id` | INTEGER | Home team ID |
| `home_team_name` | STRING | Home team name |
| `away_team_id` | INTEGER | Away team ID |
| `away_team_name` | STRING | Away team name |
| `home_score` | INTEGER | Home team final score |
| `away_score` | INTEGER | Away team final score |
| `venue_id` | INTEGER | Venue/stadium ID |
| `venue_name` | STRING | Stadium name |
| `status` | STRING | Game status (Final, In Progress, etc.) |
| `innings` | INTEGER | Number of innings played |
| `winning_pitcher_id` | INTEGER | Winning pitcher ID |
| `winning_pitcher_name` | STRING | Winning pitcher name |
| `losing_pitcher_id` | INTEGER | Losing pitcher ID |
| `losing_pitcher_name` | STRING | Losing pitcher name |
| `save_pitcher_id` | INTEGER | Save pitcher ID |
| `save_pitcher_name` | STRING | Save pitcher name |

### Sample Query
```sql
-- Get all Braves home games in 2024
SELECT game_date, away_team_name, 
       home_score, away_score
FROM `hankstank.mlb_historical_data.games_historical`
WHERE home_team_id = 144 
  AND year = 2024
ORDER BY game_date;
```

---

## 6. rosters_historical

**Purpose:** Team rosters with player positions.

### Data Coverage
- **Total Records:** 16,733
- **Years:** 2015-2025 (11 complete seasons)
- **Unique Players:** ~8,000+
- **Coverage:** ✅ **COMPLETE** - All teams × all seasons
- **Status:** ✅ **INCLUDES 2025** - Fully backfilled December 24, 2025

### Schema

| Field | Type | Description |
|-------|------|-------------|
| `year` | INTEGER | Season year |
| `team_id` | INTEGER | Team identifier |
| `team_name` | STRING | Team name |
| `player_id` | INTEGER | Player identifier |
| `player_name` | STRING | Player full name |
| `jersey_number` | STRING | Jersey number |
| `position_code` | STRING | Position code (P, C, 1B, etc.) |
| `position_name` | STRING | Position full name |
| `position_type` | STRING | Pitcher/Infielder/Outfielder |
| `status` | STRING | Roster status code |

### Sample Query
```sql
-- Get 2020 Braves roster
SELECT player_name, position_code, jersey_number
FROM `hankstank.mlb_historical_data.rosters_historical`
WHERE team_id = 144 
  AND year = 2020
ORDER BY position_code, player_name;
```

### Notes
- ✅ **Complete coverage** through 2025
- Includes detailed player info (birth date, height, weight, bat/throw, debut date)
- Average ~1,500 players per season across all 30 teams

---

## 7. transactions_historical

**Purpose:** All MLB transactions (trades, signings, releases, roster moves, IL, etc.)

### Data Coverage
- **Total Records:** 490,520
- **Years:** 2015-2025 (11 years, **includes current season**)
- **Unique Players:** 58,108
- **Transaction Types:** 27 different types
- **Coverage:** ✅ **COMPLETE & CURRENT**
- **Partitioning:** RANGE_BUCKET by year
- **Clustering:** year, date, person_id

### Schema

| Field | Type | Mode | Description |
|-------|------|------|-------------|
| `year` | INT64 | REQUIRED | Transaction year |
| `transaction_id` | INT64 | NULLABLE | Unique transaction ID |
| `date` | STRING | REQUIRED | Transaction date (YYYY-MM-DD) |
| `type_code` | STRING | NULLABLE | Transaction type code |
| `type_desc` | STRING | NULLABLE | Transaction description |
| `description` | STRING | NULLABLE | Full transaction text |
| `from_team_id` | INT64 | NULLABLE | Source team ID |
| `from_team_name` | STRING | NULLABLE | Source team name |
| `from_team_abbreviation` | STRING | NULLABLE | Source team abbreviation |
| `to_team_id` | INT64 | NULLABLE | Destination team ID |
| `to_team_name` | STRING | NULLABLE | Destination team name |
| `to_team_abbreviation` | STRING | NULLABLE | Dest. team abbreviation |
| `person_id` | INT64 | REQUIRED | Player ID |
| `person_full_name` | STRING | REQUIRED | Player name |
| `person_link` | STRING | NULLABLE | API link to player |
| `resolution` | STRING | NULLABLE | Transaction resolution |
| `notes` | STRING | NULLABLE | Additional notes |
| `synced_at` | TIMESTAMP | NULLABLE | When record was synced |

### Top Transaction Types

| Type | Count | % of Total |
|------|-------|------------|
| Assigned | 231,046 | 47.1% |
| Status Change | 142,763 | 29.1% |
| Signed as Free Agent | 26,344 | 5.4% |
| Released | 19,816 | 4.0% |
| Optioned | 14,381 | 2.9% |
| Recalled | 13,032 | 2.7% |
| Declared Free Agency | 9,276 | 1.9% |
| Signed | 7,744 | 1.6% |
| Selected | 5,488 | 1.1% |
| Trade | 5,473 | 1.1% |

### Sample Queries
```sql
-- Get all Braves transactions in 2025
SELECT date, type_desc, person_full_name, description
FROM `hankstank.mlb_historical_data.transactions_historical`
WHERE (from_team_id = 144 OR to_team_id = 144)
  AND year = 2025
ORDER BY date DESC;

-- Track a player's transaction history
SELECT date, type_desc, description, 
       from_team_name, to_team_name
FROM `hankstank.mlb_historical_data.transactions_historical`
WHERE person_id = 660271  -- Ronald Acuña Jr.
ORDER BY date DESC;

-- Get all trades in 2024
SELECT date, person_full_name, 
       from_team_name, to_team_name, description
FROM `hankstank.mlb_historical_data.transactions_historical`
WHERE type_desc = 'Trade' 
  AND year = 2024
ORDER BY date DESC;
```

### Special Features
- **Current data**: Updated for 2025 season
- **Optimized queries**: Partitioned and clustered
- **Labels**: `data_type:transactions`, `sport:mlb`

---

## 8. statcast_pitches

**Purpose:** Pitch-by-pitch Statcast data with advanced metrics (exit velocity, launch angle, spin rate, etc.)

### Data Coverage
- **Total Records:** 7,813,531 pitches
- **Years:** 2015-2025 (11 seasons, **includes current season**)
- **Date Range:** March 20, 2015 → September 28, 2025
- **Partitioning:** DAY (field: game_date)
- **Clustering:** year, game_pk, pitcher, batter
- **Coverage:** ✅ **COMPLETE & CURRENT**

### Pitch Counts by Year

| Year | Pitches | Notes |
|------|---------|-------|
| 2015 | 751,698 | First Statcast year |
| 2016 | 760,070 | Full season |
| 2017 | 765,857 | Full season |
| 2018 | 751,991 | Full season |
| 2019 | 758,283 | Full season |
| 2020 | 289,279 | **COVID-shortened** |
| 2021 | 752,574 | Full season |
| 2022 | 768,161 | Full season |
| 2023 | 758,372 | Full season |
| 2024 | 745,349 | Full season |
| 2025 | 711,897 | **Season in progress** |

### Schema Highlights

#### Core Pitch Data
| Field | Type | Description |
|-------|------|-------------|
| `pitch_type` | STRING | Pitch type (FF, SL, CH, CU, etc.) |
| `game_date` | DATE | Game date |
| `game_year` | INTEGER | Season year |
| `year` | INTEGER | Year field for partitioning |
| `game_pk` | INTEGER | Unique game identifier |
| `pitcher` | INTEGER | Pitcher ID |
| `batter` | INTEGER | Batter ID |
| `player_name` | STRING | Player name |

#### Pitch Characteristics
| Field | Type | Description |
|-------|------|-------------|
| `release_speed` | FLOAT | Velocity at release (mph) |
| `release_pos_x` | FLOAT | Horizontal release point |
| `release_pos_z` | FLOAT | Vertical release point |
| `release_spin_rate` | FLOAT | Spin rate (rpm) |
| `spin_axis` | FLOAT | Spin axis (degrees) |
| `release_extension` | FLOAT | Release extension (ft) |
| `effective_speed` | FLOAT | Effective velocity |

#### Ball Flight & Location
| Field | Type | Description |
|-------|------|-------------|
| `plate_x` | FLOAT | Horizontal location at plate |
| `plate_z` | FLOAT | Vertical location at plate |
| `pfx_x` | FLOAT | Horizontal movement (in) |
| `pfx_z` | FLOAT | Vertical movement (in) |
| `zone` | INTEGER | Strike zone location (1-14) |
| `vx0`, `vy0`, `vz0` | FLOAT | Initial velocity components |
| `ax`, `ay`, `az` | FLOAT | Acceleration components |

#### Batted Ball Data
| Field | Type | Description |
|-------|------|-------------|
| `launch_speed` | FLOAT | Exit velocity (mph) |
| `launch_angle` | FLOAT | Launch angle (degrees) |
| `hit_distance_sc` | FLOAT | Projected hit distance |
| `bat_speed` | FLOAT | Bat speed (mph) |
| `swing_length` | FLOAT | Swing length |
| `attack_angle` | FLOAT | Attack angle |
| `bb_type` | STRING | Batted ball type |

#### Expected Stats
| Field | Type | Description |
|-------|------|-------------|
| `estimated_ba_using_speedangle` | FLOAT | xBA |
| `estimated_woba_using_speedangle` | FLOAT | xwOBA |
| `estimated_slg_using_speedangle` | FLOAT | xSLG |
| `woba_value` | FLOAT | wOBA value |
| `woba_denom` | FLOAT | wOBA denominator |

#### Game Context
| Field | Type | Description |
|-------|------|-------------|
| `game_type` | STRING | R, P, S, etc. |
| `home_team` | STRING | Home team abbreviation |
| `away_team` | STRING | Away team abbreviation |
| `inning` | INTEGER | Inning number |
| `inning_topbot` | STRING | Top or Bot |
| `outs_when_up` | INTEGER | Outs when PA started |
| `balls` | INTEGER | Ball count |
| `strikes` | INTEGER | Strike count |
| `home_score` | INTEGER | Home score |
| `away_score` | INTEGER | Away score |
| `stand` | STRING | Batter stance (L/R) |
| `p_throws` | STRING | Pitcher handedness (L/R) |
| `events` | STRING | Result of PA |
| `description` | STRING | Pitch result description |

### Sample Queries
```sql
-- Get all pitches from a specific game
SELECT pitch_type, release_speed, launch_speed, 
       launch_angle, events, description
FROM `hankstank.mlb_historical_data.statcast_pitches`
WHERE game_pk = 717667
ORDER BY inning, pitcher, batter;

-- Average fastball velocity by pitcher (2024)
SELECT pitcher, player_name, 
       AVG(release_speed) as avg_velo,
       COUNT(*) as pitches
FROM `hankstank.mlb_historical_data.statcast_pitches`
WHERE game_year = 2024
  AND pitch_type IN ('FF', 'SI')  -- 4-seam, sinker
GROUP BY pitcher, player_name
HAVING pitches > 100
ORDER BY avg_velo DESC
LIMIT 20;

-- Hardest hit balls in 2025
SELECT game_date, player_name, launch_speed, 
       launch_angle, hit_distance_sc, events
FROM `hankstank.mlb_historical_data.statcast_pitches`
WHERE game_year = 2025
  AND launch_speed IS NOT NULL
ORDER BY launch_speed DESC
LIMIT 10;

-- Spin rate distribution by pitch type (2024)
SELECT pitch_type, 
       AVG(release_spin_rate) as avg_spin,
       MIN(release_spin_rate) as min_spin,
       MAX(release_spin_rate) as max_spin
FROM `hankstank.mlb_historical_data.statcast_pitches`
WHERE game_year = 2024
  AND release_spin_rate IS NOT NULL
GROUP BY pitch_type
ORDER BY avg_spin DESC;
```

### Special Features
- **Day-partitioned**: Optimized for date-range queries
- **Clustered**: Fast lookups by year, game, pitcher, batter
- **Comprehensive**: 7.8M+ pitches with 80+ attributes
- **Current data**: Includes 2025 season (711K+ pitches)
- **COVID note**: 2020 has ~60% fewer pitches (shortened season)

---

## How to Access Data

### Prerequisites
```bash
# Install Google Cloud SDK
brew install google-cloud-sdk  # macOS
# or download from https://cloud.google.com/sdk/docs/install

# Authenticate
gcloud auth login
gcloud config set project hankstank
```

### Command Line Access

#### List all tables
```bash
bq ls --project_id=hankstank mlb_historical_data
```

#### Query from terminal
```bash
bq query --project_id=hankstank --use_legacy_sql=false \
  "SELECT * FROM \`hankstank.mlb_historical_data.teams_historical\` LIMIT 10"
```

#### Export to CSV
```bash
bq extract --destination_format=CSV \
  hankstank:mlb_historical_data.teams_historical \
  gs://your-bucket/teams.csv
```

### Python Access

```python
from google.cloud import bigquery

# Initialize client
client = bigquery.Client(project='hankstank')

# Run query
query = """
SELECT team_name, wins, losses, win_percentage
FROM `hankstank.mlb_historical_data.standings_historical`
WHERE year = 2024 AND division_name = 'National League East'
ORDER BY wins DESC
"""
df = client.query(query).to_dataframe()
print(df)
```

### Node.js / TypeScript Access

```typescript
import { BigQuery } from '@google-cloud/bigquery';

const bigquery = new BigQuery({ projectId: 'hankstank' });

async function queryData() {
  const query = `
    SELECT * FROM \`hankstank.mlb_historical_data.teams_historical\`
    WHERE team_id = 144 AND year = 2024
  `;
  
  const [rows] = await bigquery.query({ query });
  console.log(rows);
}
```

### BigQuery Console
Access via web UI: https://console.cloud.google.com/bigquery?project=hankstank

---

## Data Completeness Summary

| Table | Years | Records | Status |
|-------|-------|---------|--------|
| `teams_historical` | 2015-2025 (11) | 330 | ✅ Complete & Current |
| `team_stats_historical` | 2015-2025 (11) | 700 | ✅ Complete & Current |
| `player_stats_historical` | 2015-2025 (11) | 1,706 | ⚠️ Partial (top players only) |
| `standings_historical` | 2015-2025 (11) | 330 | ✅ COMPLETE - Final 2025 standings |
| `games_historical` | 2015-2025 (11) | 27,703 | ✅ Complete & Current |
| `rosters_historical` | 2015-2025 (11) | 16,733 | ✅ Complete & Current |
| `transactions_historical` | 2015-2025 (11) | 490,520 | ✅ Complete & Current |
| `statcast_pitches` | 2015-2025 (11) | 7,813,531 | ✅ Complete & Current |

### Notes on Current Season (2025)
- **Successfully Backfilled** (December 24, 2025):
  - ✅ `teams_historical` - All 30 teams for 2025
  - ✅ `team_stats_historical` - Complete batting & pitching stats (100 records)
  - ✅ `player_stats_historical` - Top 152 players added
  - ✅ `games_historical` - All 2,511 games from 2025 season
  - ✅ `rosters_historical` - Complete 2025 rosters (1,175 players) + backfilled 2021-2024
  - ✅ `transactions_historical` - Updated through Dec 2025
  - ✅ `statcast_pitches` - 711K+ pitches through Sept 28, 2025

- **Needs Attention**:
  - ⚠️ `standings_historical` - 2025 standings not available yet (season not officially complete)

---

## Common Query Patterns

### Team Performance Analysis
```sql
-- Team win-loss trends over time
SELECT year, team_name, wins, losses, win_percentage
FROM `hankstank.mlb_historical_data.standings_historical`
WHERE team_id = 144  -- Braves
ORDER BY year;
```

### Player Transaction Timeline
```sql
-- Track player movement between teams
SELECT date, type_desc, description,
       from_team_name, to_team_name
FROM `hankstank.mlb_historical_data.transactions_historical`
WHERE person_full_name LIKE '%Acuña%'
ORDER BY date;
```

### Advanced Statcast Analysis
```sql
-- Pitcher arsenal breakdown
SELECT pitch_type, 
       COUNT(*) as pitches,
       AVG(release_speed) as avg_velo,
       AVG(release_spin_rate) as avg_spin
FROM `hankstank.mlb_historical_data.statcast_pitches`
WHERE pitcher = 592789  -- Spencer Strider
  AND game_year = 2024
GROUP BY pitch_type
ORDER BY pitches DESC;
```

### Game Results
```sql
-- Braves home game results
SELECT game_date, away_team_name,
       home_score, away_score,
       CASE WHEN home_score > away_score THEN 'W' ELSE 'L' END as result
FROM `hankstank.mlb_historical_data.games_historical`
WHERE home_team_id = 144 AND year = 2024
ORDER BY game_date;
```

---

## Performance Tips

1. **Use partitioning/clustering** - Both Statcast and transactions tables are optimized
2. **Filter early** - Add WHERE clauses on year/date before other filters
3. **Limit large queries** - Use LIMIT during development
4. **Avoid SELECT *** - Specify only needed columns for large tables
5. **Use APPROX functions** - For exploratory analysis (APPROX_COUNT_DISTINCT, etc.)
6. **Cache results** - Store frequently used aggregations in new tables

---

## Maintenance & Updates

### Historical Tables (2015-2024)
- **Sync Method:** Manual sync via backend API endpoints
- **Update Frequency:** Once per year after season ends
- **API:** `/api/sync/missing` endpoint

### Current Season Tables
- **Statcast:** Updated periodically during season
- **Transactions:** Near real-time updates
- **Games/Stats:** Fetched live from MLB API (not stored in BQ for current season)

### Known Issues
1. **Rosters incomplete** - Missing 2021-2025, needs backfill
2. **Player stats partial** - Only top performers, not exhaustive
3. **2020 data** - COVID-shortened season (reflected in all tables)

---

## Contact & Support

For questions about data access, schema changes, or issues:
- Check backend documentation in `hanks_tank_backend/`
- Review sync guides: `BIGQUERY_SYNC_GUIDE.md`
- Review transaction docs: `TRANSACTIONS_README.md`

**Last Schema Review:** December 24, 2025
