# Duplicate Games Fix Summary

## Problem Discovered
Found **531 duplicate game_pk entries** across all historical years (2015-2025).

### Root Cause
MLB reuses the same `game_pk` when games are postponed and rescheduled:
- **First record**: `status_description="Postponed"`, scores=0-0, original scheduled date
- **Second record**: `status_description="Final"`, actual scores, rescheduled date (usually next day)

### Duplicate Breakdown by Year
| Year | Duplicates |
|------|-----------|
| 2015 | 41        |
| 2016 | 28        |
| 2017 | 40        |
| 2018 | 52        |
| 2019 | 42        |
| 2020 | 74        |
| 2021 | 83        |
| 2022 | 52        |
| 2023 | 46        |
| 2024 | 39        |
| 2025 | 34        |
| **Total** | **531** |

## Solution Implemented

### Intelligent Deduplication Strategy
Instead of naive date-based deduplication, implemented **status-aware** logic:

1. **Prioritize by status code**:
   - Keep: `F` (Final), `FR` (Final/Resume), `FO` (Final/Other) → Priority 1
   - Consider: Other statuses → Priority 2
   - Remove: `DR` (Delayed/Rain), `DI` (Delayed/Other), `DS` (Delayed/Suspension) → Priority 3

2. **Preserve game types**: Keep all different game types (R=Regular, D/F/L/W=Special games)

3. **Secondary sorting**: Most recent date, highest total score (most complete data)

### SQL Logic
```sql
ROW_NUMBER() OVER (
    PARTITION BY game_pk, game_type 
    ORDER BY 
        CASE 
            WHEN status_code IN ('F', 'FR', 'FO') THEN 1  -- Final first
            WHEN status_code IN ('DR', 'DI', 'DS') THEN 3  -- Postponed last
            ELSE 2  -- In progress middle
        END,
        game_date DESC,  -- Most recent
        (COALESCE(home_score, 0) + COALESCE(away_score, 0)) DESC  -- Most complete
) as rn
```

## Execution

### Date: December 24, 2025
### Command:
```bash
python3 fix_data_issues.py --issue duplicates --year 2025 --execute
```

### Results:
- **Removed**: 531 total duplicate rows across all years
- **Postponed versions removed**: 30 (for 2025 alone)
- **Exact duplicates removed**: 4 (for 2025 alone)
- **Final/Completed games preserved**: 2,477 (for 2025)
- **Game types preserved**: 5 types (R, D, F, L, W)
- **Backup created**: `hankstank.mlb_historical_data.games_historical_backup_20251224_112327`

### Verification (Post-Fix):
```
Year | Total Games | Unique game_pks | Duplicates
-----|-------------|-----------------|----------
2025 |     2,477   |      2,477      |    0
2024 |     2,857   |      2,857      |    0
2023 |     2,895   |      2,895      |    0
2022 |     2,751   |      2,751      |    0
2021 |     2,871   |      2,871      |    0
2020 |       985   |        985      |    0
2019 |     2,414   |      2,414      |    0
2018 |     2,428   |      2,428      |    0
2017 |     2,477   |      2,477      |    0
2016 |     2,492   |      2,492      |    0
2015 |     2,525   |      2,525      |    0
```

✅ **Zero duplicates across all years!**

## Game Types Preserved

The fix carefully preserved all game types:

- **R**: Regular Season (majority of games)
- **D**: Division Series
- **F**: World Series
- **L**: League Championship Series  
- **W**: Wild Card Game
- **S**: Spring Training (if present)

All postponed/rescheduled games now have only their **final played version**, not the postponed placeholder.

## Safety Measures

1. **Backup table created** before any changes
2. **Dry-run mode** available for preview
3. **Detailed logging** of what was removed
4. **Verification queries** to confirm game type preservation
5. **Rollback command** provided:
   ```bash
   bq cp hankstank.mlb_historical_data.games_historical_backup_20251224_112327 \
         hankstank.mlb_historical_data.games_historical
   ```

## Impact on Data Quality

### Before:
- ❌ 531 duplicate game_pk entries
- ❌ Postponed games (0-0 scores) mixed with final games
- ❌ Data validation failing

### After:
- ✅ Zero duplicates
- ✅ Only final/completed games retained
- ✅ All spring training and postseason games preserved
- ✅ Data validation passing for duplicates check
- ✅ Clean dataset ready for ML feature engineering

## Future Prevention

To prevent duplicates in future data syncs:

1. **Add status check** in sync pipeline: Only insert/update games with final status codes
2. **Upsert logic**: Use `game_pk + game_type + status_code` as compound key
3. **Validation**: Run `data_validation.py` after each sync
4. **Monitoring**: Check for status='Postponed' records daily

## Files Modified

1. `fix_data_issues.py` - Enhanced with status-aware deduplication (350 lines)
2. `data_validation.py` - Already had duplicate detection (used to discover issue)
3. `DUPLICATE_FIX_SUMMARY.md` - This documentation

## Code Location

All cleanup code available at:
- `/Users/VTNX82W/Documents/personalDev/hanks_tank_ml/fix_data_issues.py`
- Reusable for future years with `--year YYYY --execute`
