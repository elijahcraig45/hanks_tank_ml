# Lesson 1: Introduction to Data Pipelines & ETL

**Module:** Data Engineering Foundations  
**Duration:** 1 week  
**Prerequisites:** Python, SQL  
**Learning Objectives:**
- Understand data pipeline architecture
- Master ETL vs ELT patterns
- Implement data quality checks
- Build your first pipeline improvement

---

## Table of Contents
1. [What is a Data Pipeline?](#what-is-a-data-pipeline)
2. [ETL vs ELT](#etl-vs-elt)
3. [Batch vs Streaming](#batch-vs-streaming)
4. [Data Quality & Validation](#data-quality--validation)
5. [Your MLB Pipeline Analysis](#your-mlb-pipeline-analysis)
6. [Hands-On Exercises](#hands-on-exercises)
7. [Further Reading](#further-reading)

---

## What is a Data Pipeline?

A **data pipeline** is an automated workflow that moves data from source(s) to destination(s), often transforming it along the way.

### Core Components

```
┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
│ Source  │ --> │ Extract │ --> │Transform│ --> │  Load   │ --> ┌──────────┐
│  (API)  │     │         │     │         │     │         │     │ Warehouse│
└─────────┘     └─────────┘     └─────────┘     └─────────┘     └──────────┘
```

**1. Source:** Where data originates
- APIs (MLB Stats API in your case)
- Databases
- Files (CSV, JSON, Parquet)
- Streaming platforms (Kafka, Pub/Sub)

**2. Extract:** Pull data from source
- Handle authentication
- Manage rate limits
- Retry failed requests
- Parse responses

**3. Transform:** Clean and reshape data
- Data type conversion
- Filtering and aggregation
- Business logic application
- Enrichment (joining multiple sources)

**4. Load:** Write to destination
- Batch inserts
- Upserts (update or insert)
- Schema validation
- Error handling

---

## ETL vs ELT

### ETL (Extract, Transform, Load)

**Traditional approach:** Transform data *before* loading into warehouse

```python
# ETL Pattern
def etl_pipeline():
    # 1. Extract
    raw_data = extract_from_api()
    
    # 2. Transform (happens BEFORE loading)
    transformed_data = transform_player_stats(raw_data)
    cleaned_data = validate_and_clean(transformed_data)
    enriched_data = add_derived_metrics(cleaned_data)
    
    # 3. Load (already transformed)
    load_to_warehouse(enriched_data)
```

**Pros:**
- ✅ Less storage needed (don't store raw data)
- ✅ Data quality enforced early
- ✅ Privacy/compliance (can filter sensitive data)

**Cons:**
- ❌ Can't re-transform later without re-extracting
- ❌ Slower for large datasets
- ❌ Less flexible for ad-hoc analysis

### ELT (Extract, Load, Transform)

**Modern approach:** Load raw data first, transform *inside* warehouse

```python
# ELT Pattern
def elt_pipeline():
    # 1. Extract
    raw_data = extract_from_api()
    
    # 2. Load (raw, untransformed)
    load_to_warehouse(raw_data, table='raw_player_stats')
    
    # 3. Transform (inside BigQuery)
    bigquery.query('''
        CREATE OR REPLACE TABLE processed_player_stats AS
        SELECT 
            player_id,
            CAST(batting_avg AS FLOAT64) as batting_avg,
            -- More transformations...
        FROM raw_player_stats
        WHERE season = 2025
    ''')
```

**Pros:**
- ✅ Keep raw data for reprocessing
- ✅ Leverage warehouse computing power (BigQuery is fast!)
- ✅ Flexible transformations
- ✅ Multiple derived tables from same raw data

**Cons:**
- ❌ More storage costs
- ❌ Bad data enters warehouse
- ❌ Transformation logic in SQL (can be complex)

### Which Should You Use?

**For MLB Project:** **ELT** is better because:
1. BigQuery is powerful and cheap for storage
2. You may want to re-engineer features later
3. Raw data is valuable (Statcast pitch data!)
4. SQL transformations are faster than Python for aggregations

**Your Current Approach:** Hybrid
- You do *some* transformation in `bigquery-sync.service.ts` (mapping fields)
- But you could load even rawer data and transform more in BigQuery

---

## Batch vs Streaming

### Batch Processing

**Process data in chunks at scheduled intervals**

```javascript
// Your current approach (batch)
async function dailySync() {
    // Run once per day at 2 AM
    const yesterday = new Date();
    yesterday.setDate(yesterday.getDate() - 1);
    
    // Process ALL of yesterday's games
    const games = await mlbApi.getGames(yesterday);
    await bigQuery.insertBatch(games); // Insert all at once
}

// Cron: 0 2 * * *
```

**When to use batch:**
- ✅ Data doesn't change frequently (historical stats)
- ✅ Don't need real-time updates
- ✅ Can process large volumes efficiently
- ✅ Lower cost (fewer compute cycles)

### Streaming Processing

**Process data in real-time as it arrives**

```javascript
// Streaming approach (not your current setup)
async function streamGameUpdates() {
    // Listen for game events continuously
    mlbApi.subscribeToGameFeed((gameUpdate) => {
        // Process each event immediately
        processGameEvent(gameUpdate);
        bigQuery.insertRow(gameUpdate);
    });
}
```

**When to use streaming:**
- ✅ Need real-time insights (live game predictions)
- ✅ Data arrives continuously (Statcast pitch data during games)
- ✅ Time-sensitive actions (betting odds updates)

**For MLB Project:**
- **Batch** for historical data (what you have now) ✅
- **Streaming** for live game predictions (future enhancement)
- **Hybrid:** Batch daily, stream during game days

---

## Data Quality & Validation

**"Garbage in, garbage out"** - Bad data = bad models

### Common Data Quality Issues

#### 1. Missing Values
```python
# Check for missing data
query = """
SELECT 
    COUNT(*) as total_games,
    COUNTIF(home_score IS NULL) as missing_home_score,
    COUNTIF(away_score IS NULL) as missing_away_score,
    COUNTIF(winning_team_id IS NULL) as missing_winner
FROM `hankstank.mlb_historical_data.games_historical`
"""

# Result interpretation:
# - If missing_winner > 0: Games in progress or data errors
# - Action: Filter out incomplete games
```

#### 2. Duplicates
```python
# Find duplicate games
query = """
SELECT 
    game_pk, 
    COUNT(*) as occurrences
FROM `hankstank.mlb_historical_data.games_historical`
GROUP BY game_pk
HAVING COUNT(*) > 1
"""

# If duplicates exist:
# - Deduplicate using ROW_NUMBER()
# - Investigate why duplicates entered
```

#### 3. Outliers
```python
# Detect impossible values
query = """
SELECT 
    game_pk,
    home_score,
    away_score
FROM `hankstank.mlb_historical_data.games_historical`
WHERE home_score > 30 OR away_score > 30  -- Suspiciously high
   OR home_score < 0 OR away_score < 0     -- Impossible
"""

# Baseball records: Highest score ~30 runs
# Anything above is likely data error
```

#### 4. Schema Violations
```python
# Type mismatches, NULL constraints
# Example: game_date should be DATE, not STRING

# Check data types
query = """
SELECT 
    game_pk,
    game_date,
    TYPEOF(game_date) as date_type  -- Should be DATE
FROM `hankstank.mlb_historical_data.games_historical`
LIMIT 5
"""
```

### Data Quality Framework

```python
class DataQualityChecker:
    """Automated data quality validation"""
    
    def __init__(self, bigquery_client):
        self.bq = bigquery_client
        self.quality_report = {
            'passed': [],
            'failed': [],
            'warnings': []
        }
    
    def check_completeness(self, table, required_columns):
        """Check for NULL values in required columns"""
        for col in required_columns:
            query = f"""
            SELECT COUNTIF({col} IS NULL) as null_count
            FROM `{table}`
            """
            result = self.bq.query(query).result()
            null_count = list(result)[0]['null_count']
            
            if null_count > 0:
                self.quality_report['failed'].append({
                    'check': 'completeness',
                    'column': col,
                    'issue': f'{null_count} NULL values found'
                })
            else:
                self.quality_report['passed'].append(f'{col}: complete')
    
    def check_uniqueness(self, table, unique_column):
        """Check for duplicate values in unique column"""
        query = f"""
        SELECT COUNT(*) - COUNT(DISTINCT {unique_column}) as duplicates
        FROM `{table}`
        """
        result = self.bq.query(query).result()
        duplicates = list(result)[0]['duplicates']
        
        if duplicates > 0:
            self.quality_report['failed'].append({
                'check': 'uniqueness',
                'column': unique_column,
                'issue': f'{duplicates} duplicate values'
            })
        else:
            self.quality_report['passed'].append(f'{unique_column}: unique')
    
    def check_range(self, table, column, min_val, max_val):
        """Check values are within expected range"""
        query = f"""
        SELECT COUNTIF({column} < {min_val} OR {column} > {max_val}) as out_of_range
        FROM `{table}`
        """
        result = self.bq.query(query).result()
        out_of_range = list(result)[0]['out_of_range']
        
        if out_of_range > 0:
            self.quality_report['warnings'].append({
                'check': 'range',
                'column': column,
                'issue': f'{out_of_range} values outside [{min_val}, {max_val}]'
            })
    
    def check_freshness(self, table, date_column, max_age_days=7):
        """Check if data is recent enough"""
        query = f"""
        SELECT DATE_DIFF(CURRENT_DATE(), MAX({date_column}), DAY) as days_old
        FROM `{table}`
        """
        result = self.bq.query(query).result()
        days_old = list(result)[0]['days_old']
        
        if days_old > max_age_days:
            self.quality_report['warnings'].append({
                'check': 'freshness',
                'issue': f'Data is {days_old} days old (threshold: {max_age_days})'
            })
        else:
            self.quality_report['passed'].append(f'Data freshness: OK ({days_old} days)')
    
    def generate_report(self):
        """Print quality report"""
        print("=" * 60)
        print("DATA QUALITY REPORT")
        print("=" * 60)
        
        print(f"\n✅ PASSED ({len(self.quality_report['passed'])} checks)")
        for check in self.quality_report['passed']:
            print(f"  • {check}")
        
        print(f"\n⚠️  WARNINGS ({len(self.quality_report['warnings'])} issues)")
        for warning in self.quality_report['warnings']:
            print(f"  • {warning['check']}: {warning['issue']}")
        
        print(f"\n❌ FAILED ({len(self.quality_report['failed'])} checks)")
        for failure in self.quality_report['failed']:
            print(f"  • {failure['check']} on {failure['column']}: {failure['issue']}")
        
        return self.quality_report

# Usage
checker = DataQualityChecker(bigquery_client)
checker.check_completeness('hankstank.mlb_historical_data.games_historical', 
                          ['game_pk', 'game_date', 'home_team_id', 'away_team_id'])
checker.check_uniqueness('hankstank.mlb_historical_data.games_historical', 'game_pk')
checker.check_range('hankstank.mlb_historical_data.games_historical', 'home_score', 0, 30)
checker.check_freshness('hankstank.mlb_historical_data.games_historical', 'game_date', max_age_days=7)
report = checker.generate_report()
```

---

## Your MLB Pipeline Analysis

Let's analyze your current data pipeline in `hanks_tank_backend`:

### Current Architecture

```
┌──────────────┐
│  MLB Stats   │
│     API      │ (Source)
└──────┬───────┘
       │
       │ HTTP requests
       │ (mlb-api.service.ts)
       ▼
┌──────────────────┐
│  Node.js Backend │ (Extract & Transform)
│  ---------------  │
│  • mlb-api.service.ts    : API calls
│  • bigquery-sync.service: Field mapping
│  • backfill scripts      : Historical loads
└──────┬───────────┘
       │
       │ BigQuery API
       │ (insertRows)
       ▼
┌──────────────────┐
│    BigQuery      │ (Load & Store)
│  --------------  │
│  • games_historical
│  • rosters_historical
│  • standings_historical
│  • team_stats
│  • player_stats
│  • statcast_pitches
│  • transactions_historical
│  • teams
└──────────────────┘
```

### Pipeline Type: **Hybrid ETL/ELT**

**ETL Parts:**
1. `mlb-api.service.ts`: Extracts from API
2. `bigquery-sync.service.ts`: Transforms (field mapping, type conversion)
3. BigQuery: Loads transformed data

**ELT Potential:**
- Could load raw API responses to `raw_*` tables
- Transform in BigQuery SQL for more flexibility

### Strengths ✅
1. **Batch processing:** Efficient for historical data
2. **Error handling:** SSL cert handling, retries in backfill scripts
3. **Incremental loads:** Can process specific years/tables
4. **Batching:** Respects BigQuery limits (5000 rows/chunk)

### Weaknesses ❌
1. **No data quality checks:** Data goes straight to BigQuery
2. **Limited monitoring:** No alerts if pipeline fails
3. **Manual scheduling:** Have to run backfill scripts manually
4. **No raw data preservation:** Can't reprocess if transform logic changes
5. **Tight coupling:** Transform logic in app code (hard to change)

### Improvement Opportunities 🚀

**1. Add Data Quality Layer**
```typescript
// Before inserting to BigQuery
async syncGames(year: number) {
    const games = await this.mlbApi.getGames(year);
    
    // NEW: Validate before loading
    const validation = this.validateGames(games);
    if (!validation.passed) {
        logger.error('Data quality check failed', validation.errors);
        throw new Error('Invalid game data');
    }
    
    await this.bigQuery.insertGames(games);
}

private validateGames(games: any[]): ValidationResult {
    return {
        passed: games.every(g => 
            g.game_pk != null &&
            g.home_score >= 0 &&
            g.away_score >= 0 &&
            g.game_date != null
        ),
        errors: games.filter(g => g.game_pk == null).map(g => ({
            game: g,
            issue: 'Missing game_pk'
        }))
    };
}
```

**2. Separate Raw and Processed Tables**
```sql
-- Raw layer (ELT approach)
CREATE TABLE raw_games_api_response (
    ingested_at TIMESTAMP,
    api_response JSON,  -- Store full API response
    year INT64
);

-- Processed layer (transformed via SQL)
CREATE TABLE games_historical AS
SELECT 
    JSON_VALUE(api_response, '$.gamePk') as game_pk,
    PARSE_DATE('%Y-%m-%d', JSON_VALUE(api_response, '$.gameDate')) as game_date,
    -- ... more transformations
FROM raw_games_api_response;
```

**3. Add Orchestration (Airflow/Cron)**
```python
# Airflow DAG example
from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'mlb_pipeline',
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'mlb_daily_sync',
    default_args=default_args,
    schedule_interval='0 2 * * *',  # 2 AM daily
    start_date=datetime(2025, 1, 1),
)

# Task 1: Extract games from yesterday
extract_games = BashOperator(
    task_id='extract_games',
    bash_command='cd /path/to/backend && npm run sync:games',
    dag=dag,
)

# Task 2: Validate data quality
validate_data = BashOperator(
    task_id='validate_data',
    bash_command='python /path/to/quality_checker.py',
    dag=dag,
)

# Task 3: Load to BigQuery (already done by sync script)

# Dependencies
extract_games >> validate_data
```

**4. Monitoring & Alerting**
```typescript
// Add to bigquery-sync.service.ts
async syncWithMonitoring(syncFunction: Function, tableName: string) {
    const startTime = Date.now();
    
    try {
        const result = await syncFunction();
        const duration = Date.now() - startTime;
        
        // Log success metrics
        logger.info('Sync completed', {
            table: tableName,
            rows: result.rowsInserted,
            duration: duration,
            timestamp: new Date().toISOString()
        });
        
        // Could send to monitoring service (Cloud Monitoring, Datadog)
        await this.sendMetric('sync_success', tableName, duration);
        
        return result;
        
    } catch (error) {
        // Alert on failure
        logger.error('Sync failed', { table: tableName, error });
        await this.sendAlert(`Pipeline failed for ${tableName}`, error);
        throw error;
    }
}
```

---

## Hands-On Exercises

### Exercise 1: Build a Data Quality Checker (30 min)

**Task:** Create a script to validate your `games_historical` table

**Create:** `hanks_tank_backend/scripts/validate_games_data.ts`

```typescript
import { BigQuery } from '@google-cloud/bigquery';

const bigquery = new BigQuery({ projectId: 'hankstank' });

interface QualityCheck {
    name: string;
    query: string;
    threshold: number;
    comparison: 'equals' | 'lessThan' | 'greaterThan';
}

async function runQualityChecks() {
    const checks: QualityCheck[] = [
        {
            name: 'No NULL game_pks',
            query: `SELECT COUNTIF(game_pk IS NULL) as null_count 
                    FROM hankstank.mlb_historical_data.games_historical`,
            threshold: 0,
            comparison: 'equals'
        },
        {
            name: 'No duplicate game_pks',
            query: `SELECT COUNT(*) - COUNT(DISTINCT game_pk) as duplicates
                    FROM hankstank.mlb_historical_data.games_historical`,
            threshold: 0,
            comparison: 'equals'
        },
        {
            name: 'Scores in valid range',
            query: `SELECT COUNTIF(home_score < 0 OR home_score > 30 OR 
                                   away_score < 0 OR away_score > 30) as invalid
                    FROM hankstank.mlb_historical_data.games_historical`,
            threshold: 0,
            comparison: 'equals'
        },
        {
            name: 'Data freshness (within 7 days)',
            query: `SELECT DATE_DIFF(CURRENT_DATE(), MAX(game_date), DAY) as days_old
                    FROM hankstank.mlb_historical_data.games_historical`,
            threshold: 7,
            comparison: 'lessThan'
        },
        {
            name: 'Minimum data volume',
            query: `SELECT COUNT(*) as total_games
                    FROM hankstank.mlb_historical_data.games_historical
                    WHERE year = 2025`,
            threshold: 2000,  // Expect at least 2000 games in 2025
            comparison: 'greaterThan'
        }
    ];
    
    const results = {
        passed: [] as string[],
        failed: [] as Array<{ name: string, actual: number, expected: number }>
    };
    
    for (const check of checks) {
        const [rows] = await bigquery.query({ query: check.query });
        const actual = Object.values(rows[0])[0] as number;
        
        let passed = false;
        switch (check.comparison) {
            case 'equals':
                passed = actual === check.threshold;
                break;
            case 'lessThan':
                passed = actual < check.threshold;
                break;
            case 'greaterThan':
                passed = actual > check.threshold;
                break;
        }
        
        if (passed) {
            results.passed.push(check.name);
            console.log(`✅ ${check.name}`);
        } else {
            results.failed.push({
                name: check.name,
                actual: actual,
                expected: check.threshold
            });
            console.log(`❌ ${check.name}: ${actual} (expected ${check.comparison} ${check.threshold})`);
        }
    }
    
    console.log(`\n📊 Summary: ${results.passed.length} passed, ${results.failed.length} failed`);
    
    if (results.failed.length > 0) {
        console.error('❌ Data quality checks failed!');
        process.exit(1);
    } else {
        console.log('✅ All quality checks passed!');
    }
}

runQualityChecks().catch(console.error);
```

**Run it:**
```bash
cd hanks_tank_backend
npx ts-node scripts/validate_games_data.ts
```

**Expected output:**
```
✅ No NULL game_pks
✅ No duplicate game_pks
✅ Scores in valid range
✅ Data freshness (within 7 days)
✅ Minimum data volume

📊 Summary: 5 passed, 0 failed
✅ All quality checks passed!
```

---

### Exercise 2: Analyze Pipeline Performance (20 min)

**Task:** Measure how long each table sync takes

**Create:** `hanks_tank_backend/scripts/benchmark_pipeline.ts`

```typescript
import { MLBApiService } from '../src/services/mlb-api.service';
import { BigQuerySyncService } from '../src/services/bigquery-sync.service';

async function benchmarkSync() {
    const mlbApi = new MLBApiService();
    const syncService = new BigQuerySyncService(mlbApi);
    
    const testYear = 2025;
    const benchmarks: Array<{ table: string, duration: number, rows: number }> = [];
    
    // Benchmark each sync operation
    const operations = [
        { name: 'teams', fn: () => syncService.syncTeams(testYear) },
        { name: 'team_stats', fn: () => syncService.syncTeamStats(testYear) },
        { name: 'standings', fn: () => syncService.syncStandings(testYear) },
    ];
    
    for (const op of operations) {
        console.log(`\nBenchmarking ${op.name}...`);
        const start = Date.now();
        
        try {
            const result = await op.fn();
            const duration = Date.now() - start;
            
            benchmarks.push({
                table: op.name,
                duration: duration,
                rows: result.rowsInserted || 0
            });
            
            console.log(`  ⏱️  Duration: ${duration}ms`);
            console.log(`  📊 Rows: ${result.rowsInserted || 0}`);
            console.log(`  ⚡ Rate: ${((result.rowsInserted || 0) / (duration / 1000)).toFixed(2)} rows/sec`);
            
        } catch (error) {
            console.error(`  ❌ Failed: ${error.message}`);
        }
    }
    
    // Summary
    console.log('\n' + '='.repeat(60));
    console.log('BENCHMARK SUMMARY');
    console.log('='.repeat(60));
    
    benchmarks.forEach(b => {
        console.log(`${b.table.padEnd(20)} ${b.duration}ms\t${b.rows} rows`);
    });
    
    const totalDuration = benchmarks.reduce((sum, b) => sum + b.duration, 0);
    const totalRows = benchmarks.reduce((sum, b) => sum + b.rows, 0);
    
    console.log('='.repeat(60));
    console.log(`TOTAL: ${totalDuration}ms (${(totalDuration / 1000).toFixed(2)}s)`);
    console.log(`TOTAL ROWS: ${totalRows}`);
}

benchmarkSync().catch(console.error);
```

---

### Exercise 3: Design a Better Pipeline (45 min)

**Task:** Draw out an improved architecture for your MLB pipeline

**What to include:**
1. **Raw data layer:** Where you store unprocessed API responses
2. **Data quality layer:** Validation before loading
3. **Processed data layer:** Your current tables
4. **Feature layer:** Pre-computed features for ML
5. **Orchestration:** How/when each step runs
6. **Monitoring:** What metrics to track

**Tool:** Use draw.io, Excalidraw, or pen & paper

**Key questions to answer:**
- Where should data validation happen?
- Should you keep raw API responses?
- How often should each table update?
- What happens if a sync fails?
- How do you know if data is stale?

**Example improved architecture:**

```
┌─────────────────────────────────────────────────────┐
│                   MLB Stats API                      │
└─────────────────┬───────────────────────────────────┘
                  │
                  │ (1) Extract
                  ▼
┌─────────────────────────────────────────────────────┐
│              Extraction Service                      │
│  • Rate limiting                                     │
│  • Error handling                                    │
│  • Raw response storage                              │
└─────────────────┬───────────────────────────────────┘
                  │
                  │ (2) Store Raw
                  ▼
┌─────────────────────────────────────────────────────┐
│         BigQuery: raw_* tables                       │
│  • raw_games_api_response                            │
│  • raw_rosters_api_response                          │
└─────────────────┬───────────────────────────────────┘
                  │
                  │ (3) Data Quality Check
                  ▼
┌─────────────────────────────────────────────────────┐
│           Quality Validation Service                 │
│  • Schema validation                                 │
│  • Range checks                                      │
│  • Completeness checks                               │
│  • Alert on failures                                 │
└─────────────────┬───────────────────────────────────┘
                  │
                  │ (4) Transform (BigQuery SQL)
                  ▼
┌─────────────────────────────────────────────────────┐
│      BigQuery: Processed Tables (current)            │
│  • games_historical                                  │
│  • rosters_historical                                │
│  • standings_historical                              │
└─────────────────┬───────────────────────────────────┘
                  │
                  │ (5) Feature Engineering (BigQuery SQL)
                  ▼
┌─────────────────────────────────────────────────────┐
│       BigQuery: Feature Tables (new!)                │
│  • game_features (team strength, recent form)        │
│  • player_features (rolling stats)                   │
└─────────────────────────────────────────────────────┘
                  │
                  │ (6) ML Training
                  ▼
┌─────────────────────────────────────────────────────┐
│              ML Training Pipeline                    │
│  • Read from feature tables                          │
│  • Train models                                      │
│  • Deploy to prediction API                          │
└─────────────────────────────────────────────────────┘

        (Orchestrated by Airflow/Cloud Scheduler)
        (Monitored by Cloud Monitoring/Datadog)
```

---

## Key Takeaways

✅ **Data pipelines automate the flow of data from source to destination**

✅ **ELT is better for modern data warehouses** (load raw, transform in SQL)

✅ **Batch processing is efficient for historical data**, streaming for real-time

✅ **Data quality is critical** - validate early and often

✅ **Your MLB pipeline is solid but can be improved** with:
- Data quality checks
- Raw data preservation
- Better orchestration
- Monitoring & alerting

✅ **Think in layers**: Raw → Validated → Processed → Features → Models

---

## Further Reading

### Documentation
- [BigQuery Best Practices](https://cloud.google.com/bigquery/docs/best-practices)
- [Google Cloud Composer (Airflow)](https://cloud.google.com/composer/docs)
- [Data Pipeline Design Patterns](https://www.oreilly.com/library/view/data-pipelines-pocket/9781492087823/)

### Articles
- [The Rise of ELT](https://www.fivetran.com/blog/elt-vs-etl)
- [Data Quality at Scale](https://www.montecarlodata.com/blog-data-quality-at-scale/)

### Next Lesson
👉 **[Lesson 2: BigQuery Deep Dive](LESSON_02_BIGQUERY.md)**
- Query optimization
- Partitioning and clustering
- Window functions for feature engineering
- Cost optimization

---

## Quiz (Test Your Understanding)

1. **What's the main difference between ETL and ELT?**
   <details>
   <summary>Answer</summary>
   ETL transforms data before loading (outside warehouse), ELT loads raw data first and transforms inside the warehouse using SQL.
   </details>

2. **When should you use batch processing vs streaming?**
   <details>
   <summary>Answer</summary>
   Batch: When data changes infrequently and you don't need real-time updates. Streaming: When you need real-time insights or data arrives continuously.
   </details>

3. **Name 3 data quality checks you should perform on your games_historical table**
   <details>
   <summary>Answer</summary>
   (1) Check for NULL game_pks (completeness), (2) Check for duplicate game_pks (uniqueness), (3) Check scores are in valid range 0-30 (validity)
   </details>

4. **Why is it valuable to store raw API responses?**
   <details>
   <summary>Answer</summary>
   Allows you to re-process data later if transformation logic changes, provides full audit trail, enables debugging data issues.
   </details>

---

**Ready for Lesson 2? Let's dive into BigQuery optimization! 🚀**
