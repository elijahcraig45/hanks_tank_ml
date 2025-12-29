# Lesson 2: BigQuery Deep Dive

**Prerequisites:** Lesson 1 (Data Pipelines), Basic SQL  
**Duration:** 2-3 hours  
**Focus:** Optimize BigQuery for ML feature engineering and analytics

---

## Learning Objectives

By the end of this lesson, you will:
- Understand BigQuery architecture and how it differs from traditional databases
- Write efficient queries that minimize costs and maximize performance
- Use partitioning and clustering to optimize table design
- Master window functions and CTEs for complex analytics
- Create materialized views for feature engineering

---

## 1. BigQuery Architecture

### How BigQuery Works

BigQuery is a **serverless, columnar data warehouse** designed for analytics at scale.

**Key Differences from Traditional Databases:**

| Traditional DB | BigQuery |
|---------------|----------|
| Row-oriented storage | Columnar storage |
| Indexes for fast lookups | Full table scans optimized |
| Designed for OLTP | Designed for OLAP |
| Manual scaling | Auto-scaling |
| Fixed capacity | On-demand resources |

**Columnar Storage Benefits:**
```sql
-- Traditional row storage reads ALL columns
SELECT pitcher_id, pitch_speed 
FROM pitches 
WHERE game_date = '2026-04-15';

-- BigQuery only reads pitcher_id and pitch_speed columns
-- Ignores 20+ other columns = massive performance gain
```

### BigQuery Pricing Model

**Two main costs:**
1. **Storage:** $0.02 per GB/month (active), $0.01 per GB/month (long-term storage)
2. **Query processing:** $5 per TB scanned

**💡 Key Insight:** Query costs based on data scanned, NOT rows returned!

```sql
-- BAD: Scans entire table (expensive)
SELECT * FROM `project.dataset.games_historical`
WHERE game_date = '2026-04-15';

-- GOOD: Only scans needed columns (cheap)
SELECT game_id, home_team, away_team, home_score, away_score
FROM `project.dataset.games_historical`
WHERE game_date = '2026-04-15';
```

---

## 2. Query Optimization Techniques

### Optimization #1: Avoid SELECT *

**Never use `SELECT *` in production queries!**

```sql
-- ❌ BAD: Scans all 50 columns
SELECT * FROM games_historical WHERE season = 2026;

-- ✅ GOOD: Only scan needed columns
SELECT game_id, game_date, home_team, away_team, 
       home_score, away_score, winning_team
FROM games_historical 
WHERE season = 2026;
```

**Estimate query cost before running:**
- Click "Query Validator" in BigQuery console
- Shows "This query will process X GB when run"
- Helps catch expensive queries before execution

### Optimization #2: Filter Early with WHERE

Push filters as early as possible in the query:

```sql
-- ❌ BAD: Joins first, then filters (processes all data)
SELECT g.*, s.*
FROM games_historical g
JOIN team_stats s ON g.game_id = s.game_id
WHERE g.season = 2026 AND g.home_team = 'ATL';

-- ✅ GOOD: Filter before join (processes only needed data)
SELECT g.*, s.*
FROM (
  SELECT * FROM games_historical 
  WHERE season = 2026 AND home_team = 'ATL'
) g
JOIN team_stats s ON g.game_id = s.game_id;
```

### Optimization #3: Use Approximate Aggregations

For exploratory analysis, approximate is often good enough:

```sql
-- Exact count (scans all data)
SELECT COUNT(DISTINCT player_id) FROM statcast_pitches;

-- Approximate count (much faster, 1% error)
SELECT APPROX_COUNT_DISTINCT(player_id) FROM statcast_pitches;
```

### Optimization #4: Denormalize for Analytics

**Joins are expensive in BigQuery!** Denormalize when possible.

```sql
-- ❌ EXPENSIVE: Multiple joins per query
SELECT 
  g.game_date,
  t1.team_name,
  t1.batting_avg,
  t2.team_name,
  t2.batting_avg
FROM games g
JOIN teams t1 ON g.home_team = t1.team_id
JOIN teams t2 ON g.away_team = t2.team_id;

-- ✅ CHEAPER: Denormalized table (join once, query many times)
SELECT 
  game_date,
  home_team_name,
  home_batting_avg,
  away_team_name,
  away_batting_avg
FROM games_denormalized;
```

---

## 3. Partitioning and Clustering

### Partitioning: Speed Up Date-Based Queries

Partitioning splits a table into segments, allowing BigQuery to skip irrelevant data.

**Create a partitioned table:**

```sql
CREATE OR REPLACE TABLE `your-project.mlb_data.games_historical`
PARTITION BY game_date
AS
SELECT * FROM `your-project.mlb_data.games_historical_old`;
```

**Query with partition pruning:**

```sql
-- Only scans April 2026 partition
SELECT * FROM games_historical
WHERE game_date BETWEEN '2026-04-01' AND '2026-04-30';
```

**Check your current tables:**

```sql
-- See if tables are partitioned
SELECT 
  table_name,
  is_partitioning_column,
  column_name
FROM `your-project.mlb_data.INFORMATION_SCHEMA.COLUMNS`
WHERE is_partitioning_column = 'YES';
```

### Clustering: Speed Up Filter Queries

Clustering organizes data by specific columns for faster filtering.

**Best practice:** Partition by date, cluster by frequently filtered columns.

```sql
CREATE OR REPLACE TABLE `your-project.mlb_data.statcast_pitches`
PARTITION BY DATE(game_date)
CLUSTER BY pitcher_id, pitch_type
AS
SELECT * FROM `your-project.mlb_data.statcast_pitches_old`;
```

Now queries filtering by `pitcher_id` or `pitch_type` are much faster!

```sql
-- Fast: Uses clustering
SELECT * FROM statcast_pitches
WHERE game_date = '2026-04-15'
  AND pitcher_id = 592789
  AND pitch_type = 'FF';
```

**Clustering guidelines:**
- Choose columns you filter/aggregate on frequently
- Order matters: most selective column first
- Maximum 4 clustering columns

---

## 4. Window Functions

Window functions perform calculations across rows related to the current row.

### Basic Window Function Syntax

```sql
SELECT 
  column1,
  column2,
  AGGREGATE_FUNCTION(column3) OVER (
    PARTITION BY column4
    ORDER BY column5
    ROWS BETWEEN start AND end
  ) AS window_result
FROM table;
```

### Example 1: Rolling Averages

Calculate a team's rolling 10-game batting average:

```sql
SELECT 
  team_id,
  game_date,
  batting_avg,
  AVG(batting_avg) OVER (
    PARTITION BY team_id
    ORDER BY game_date
    ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
  ) AS rolling_10_game_avg
FROM team_batting_by_game
ORDER BY team_id, game_date;
```

### Example 2: Rank and Row Number

Find each team's best pitching performances:

```sql
SELECT 
  team_id,
  game_date,
  era,
  ROW_NUMBER() OVER (
    PARTITION BY team_id 
    ORDER BY era ASC
  ) AS rank_in_team
FROM team_pitching_by_game
QUALIFY rank_in_team <= 10;  -- Top 10 games per team
```

**Note:** `QUALIFY` filters on window function results (BigQuery-specific!)

### Example 3: Lag and Lead

Compare team performance to previous game:

```sql
SELECT 
  team_id,
  game_date,
  runs_scored,
  LAG(runs_scored, 1) OVER (
    PARTITION BY team_id 
    ORDER BY game_date
  ) AS prev_game_runs,
  runs_scored - LAG(runs_scored, 1) OVER (
    PARTITION BY team_id 
    ORDER BY game_date
  ) AS runs_change
FROM team_batting_by_game;
```

### Example 4: Cumulative Stats

Track season-to-date wins:

```sql
SELECT 
  team_id,
  game_date,
  won,
  SUM(won) OVER (
    PARTITION BY team_id, season
    ORDER BY game_date
    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
  ) AS season_wins
FROM games_historical;
```

---

## 5. Common Table Expressions (CTEs)

CTEs make complex queries more readable and maintainable.

### Basic CTE Syntax

```sql
WITH cte_name AS (
  SELECT ...
  FROM ...
  WHERE ...
)
SELECT * FROM cte_name;
```

### Example: Multi-Step Feature Engineering

```sql
-- Step 1: Calculate team stats per game
WITH team_game_stats AS (
  SELECT 
    game_id,
    team_id,
    SUM(hits) AS hits,
    SUM(at_bats) AS at_bats,
    SUM(runs) AS runs
  FROM batting_plays
  GROUP BY game_id, team_id
),

-- Step 2: Calculate rolling averages
team_rolling_stats AS (
  SELECT 
    team_id,
    game_id,
    hits,
    at_bats,
    AVG(hits / at_bats) OVER (
      PARTITION BY team_id
      ORDER BY game_id
      ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
    ) AS rolling_10_avg
  FROM team_game_stats
),

-- Step 3: Join with game results
final_features AS (
  SELECT 
    g.game_id,
    g.game_date,
    g.home_team,
    g.away_team,
    h.rolling_10_avg AS home_rolling_avg,
    a.rolling_10_avg AS away_rolling_avg,
    g.winning_team
  FROM games_historical g
  JOIN team_rolling_stats h 
    ON g.game_id = h.game_id AND g.home_team = h.team_id
  JOIN team_rolling_stats a 
    ON g.game_id = a.game_id AND g.away_team = a.team_id
)

SELECT * FROM final_features
ORDER BY game_date DESC;
```

---

## 6. Materialized Views

Materialized views pre-compute expensive queries and auto-refresh.

### When to Use Materialized Views

- Query runs frequently (daily feature refresh)
- Query is expensive (joins, aggregations, window functions)
- Data doesn't change constantly
- You're willing to accept slightly stale data

### Create a Materialized View

```sql
CREATE MATERIALIZED VIEW `your-project.mlb_data.team_rolling_features`
PARTITION BY game_date
CLUSTER BY team_id
AS
SELECT 
  team_id,
  game_date,
  game_id,
  AVG(batting_avg) OVER (
    PARTITION BY team_id
    ORDER BY game_date
    ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
  ) AS rolling_10_batting_avg,
  AVG(era) OVER (
    PARTITION BY team_id
    ORDER BY game_date
    ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
  ) AS rolling_10_era,
  SUM(won) OVER (
    PARTITION BY team_id
    ORDER BY game_date
    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
  ) AS season_wins
FROM team_game_aggregates;
```

**Benefits:**
- Query the materialized view instead of base tables
- BigQuery automatically refreshes when base data changes
- Much faster queries (data pre-computed)
- Lower costs (less data scanned)

**Limitations:**
- Not real-time (refresh can take minutes)
- Refresh uses query slots
- Additional storage costs

---

## Hands-On Exercise: Optimize Your MLB Queries

### Task 1: Analyze Current Query Performance

Run this query to see your most expensive tables:

```sql
-- Find which tables are scanned most frequently
SELECT 
  referenced_table.table_id,
  SUM(total_bytes_processed) / 1e12 AS total_tb_processed,
  COUNT(*) AS query_count,
  SUM(total_bytes_processed) / 1e12 * 5 AS estimated_cost_usd
FROM `region-us`.INFORMATION_SCHEMA.JOBS_BY_PROJECT
WHERE DATE(creation_time) >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)
  AND referenced_table.table_id IS NOT NULL
GROUP BY referenced_table.table_id
ORDER BY total_tb_processed DESC
LIMIT 20;
```

### Task 2: Partition and Cluster Your Tables

Check if your main tables are partitioned:

```sql
SELECT 
  table_name,
  partition_by,
  clustering_fields
FROM `your-project.mlb_data.__TABLES__`;
```

If not partitioned, create partitioned versions:

```sql
-- Partition games_historical
CREATE OR REPLACE TABLE `your-project.mlb_data.games_historical_new`
PARTITION BY game_date
CLUSTER BY home_team, away_team
AS
SELECT * FROM `your-project.mlb_data.games_historical`;

-- Partition statcast_pitches
CREATE OR REPLACE TABLE `your-project.mlb_data.statcast_pitches_new`
PARTITION BY DATE(game_date)
CLUSTER BY pitcher_id, pitch_type
AS
SELECT * FROM `your-project.mlb_data.statcast_pitches`;
```

### Task 3: Create a Feature Engineering Materialized View

Build a materialized view with common ML features:

```sql
CREATE MATERIALIZED VIEW `your-project.mlb_data.game_prediction_features`
PARTITION BY game_date
CLUSTER BY home_team, away_team
AS
WITH home_stats AS (
  SELECT 
    team_id,
    game_date,
    game_id,
    AVG(batting_avg) OVER w10 AS home_l10_batting,
    AVG(era) OVER w10 AS home_l10_era,
    AVG(runs_scored) OVER w10 AS home_l10_runs,
    SUM(won) OVER season AS home_season_wins
  FROM team_game_stats
  WINDOW 
    w10 AS (PARTITION BY team_id ORDER BY game_date ROWS BETWEEN 9 PRECEDING AND CURRENT ROW),
    season AS (PARTITION BY team_id, EXTRACT(YEAR FROM game_date) ORDER BY game_date)
),
away_stats AS (
  SELECT 
    team_id,
    game_date,
    game_id,
    AVG(batting_avg) OVER w10 AS away_l10_batting,
    AVG(era) OVER w10 AS away_l10_era,
    AVG(runs_scored) OVER w10 AS away_l10_runs,
    SUM(won) OVER season AS away_season_wins
  FROM team_game_stats
  WINDOW 
    w10 AS (PARTITION BY team_id ORDER BY game_date ROWS BETWEEN 9 PRECEDING AND CURRENT ROW),
    season AS (PARTITION BY team_id, EXTRACT(YEAR FROM game_date) ORDER BY game_date)
)
SELECT 
  g.game_id,
  g.game_date,
  g.home_team,
  g.away_team,
  h.home_l10_batting,
  h.home_l10_era,
  h.home_l10_runs,
  h.home_season_wins,
  a.away_l10_batting,
  a.away_l10_era,
  a.away_l10_runs,
  a.away_season_wins,
  g.home_score,
  g.away_score,
  g.winning_team
FROM games_historical g
JOIN home_stats h ON g.game_id = h.game_id AND g.home_team = h.team_id
JOIN away_stats a ON g.game_id = a.game_id AND g.away_team = a.team_id;
```

### Task 4: Benchmark Performance

Compare query performance before and after optimization:

```sql
-- Before: Query base tables with window functions
-- (Run this and note the bytes processed)
SELECT 
  g.game_id,
  AVG(t.batting_avg) OVER (PARTITION BY t.team_id ORDER BY g.game_date ROWS BETWEEN 9 PRECEDING AND CURRENT ROW) AS rolling_avg
FROM games_historical g
JOIN team_stats t ON g.game_id = t.game_id;

-- After: Query materialized view
-- (Run this and compare bytes processed)
SELECT 
  game_id,
  home_l10_batting
FROM game_prediction_features;
```

**Expected results:**
- Materialized view should scan 10-100x less data
- Query should run 5-20x faster
- Estimated cost should be much lower

---

## Practice Queries

### Query 1: Team Momentum Score

Create a momentum score based on recent performance:

```sql
SELECT 
  team_id,
  game_date,
  -- Win rate in last 10 games
  AVG(CAST(won AS FLOAT64)) OVER (
    PARTITION BY team_id
    ORDER BY game_date
    ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
  ) AS win_rate_l10,
  -- Run differential in last 10 games
  SUM(runs_scored - runs_allowed) OVER (
    PARTITION BY team_id
    ORDER BY game_date
    ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
  ) AS run_diff_l10,
  -- Momentum score (weighted average)
  0.7 * AVG(CAST(won AS FLOAT64)) OVER w10 + 
  0.3 * (SUM(runs_scored - runs_allowed) OVER w10 / 40.0) AS momentum_score
FROM team_game_results
WINDOW w10 AS (
  PARTITION BY team_id 
  ORDER BY game_date 
  ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
)
ORDER BY game_date DESC, momentum_score DESC;
```

### Query 2: Home Field Advantage Analysis

Analyze home vs away performance:

```sql
WITH team_splits AS (
  SELECT 
    team_id,
    season,
    SUM(CASE WHEN is_home THEN won ELSE 0 END) AS home_wins,
    COUNT(CASE WHEN is_home THEN 1 END) AS home_games,
    SUM(CASE WHEN NOT is_home THEN won ELSE 0 END) AS away_wins,
    COUNT(CASE WHEN NOT is_home THEN 1 END) AS away_games
  FROM team_games
  GROUP BY team_id, season
)
SELECT 
  team_id,
  season,
  home_wins * 1.0 / home_games AS home_win_pct,
  away_wins * 1.0 / away_games AS away_win_pct,
  (home_wins * 1.0 / home_games) - (away_wins * 1.0 / away_games) AS home_advantage,
  RANK() OVER (PARTITION BY season ORDER BY (home_wins * 1.0 / home_games) - (away_wins * 1.0 / away_games) DESC) AS advantage_rank
FROM team_splits
ORDER BY season DESC, home_advantage DESC;
```

### Query 3: Pitcher Usage Patterns

Analyze pitcher workload and rest days:

```sql
SELECT 
  pitcher_id,
  pitcher_name,
  game_date,
  pitches_thrown,
  LAG(game_date) OVER (PARTITION BY pitcher_id ORDER BY game_date) AS prev_appearance,
  DATE_DIFF(game_date, LAG(game_date) OVER (PARTITION BY pitcher_id ORDER BY game_date), DAY) AS days_rest,
  AVG(pitches_thrown) OVER (
    PARTITION BY pitcher_id
    ORDER BY game_date
    ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
  ) AS avg_pitches_last_5
FROM pitcher_appearances
WHERE pitcher_type = 'starter'
ORDER BY game_date DESC;
```

---

## Key Takeaways

✅ **Architecture:** BigQuery is columnar, serverless, and optimized for analytics  
✅ **Cost:** Pay for data scanned, not rows returned - avoid `SELECT *`  
✅ **Partitioning:** Partition by date for time-based queries  
✅ **Clustering:** Cluster by frequently filtered columns  
✅ **Window Functions:** Essential for rolling stats and time-series features  
✅ **CTEs:** Make complex queries readable and maintainable  
✅ **Materialized Views:** Pre-compute expensive feature engineering queries  

---

## Next Steps

1. **Audit your current queries** - Find the most expensive ones
2. **Implement partitioning and clustering** on your main tables
3. **Create materialized views** for your most common feature engineering queries
4. **Practice window functions** - They're critical for ML feature engineering
5. **Move to Lesson 3:** Data Modeling & Schema Design

---

## Additional Resources

- [BigQuery Best Practices](https://cloud.google.com/bigquery/docs/best-practices-performance-overview)
- [Window Functions Guide](https://cloud.google.com/bigquery/docs/reference/standard-sql/window-functions)
- [Partitioning and Clustering](https://cloud.google.com/bigquery/docs/partitioned-tables)
- [Materialized Views](https://cloud.google.com/bigquery/docs/materialized-views-intro)
- [Query Optimization Tips](https://cloud.google.com/bigquery/docs/best-practices-costs)

---

**Ready to optimize your BigQuery queries and reduce costs by 10-100x? Start with the hands-on exercises above!** 🚀
