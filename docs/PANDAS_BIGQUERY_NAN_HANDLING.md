# Pandas BigQuery NaN Handling Guide

## Overview

This document details a critical type-checking bug that can occur when converting BigQuery NULL values to Python objects via pandas DataFrames. The issue caused pitcher velocity data to silently fail to populate despite valid queries returning correct values.

## The Problem

### Symptom
When querying BigQuery and converting results to pandas DataFrames, NULL values may appear to be populated but actually contain `NaN` (Not a Number) values. Using Python's `is not None` operator to check these values produces unexpected results, causing silent failures in data conversion.

### Root Cause
When pandas loads NULL values from BigQuery into a DataFrame:
- Numeric columns become `numpy.nan` (not Python's `None`)
- String/object columns become pandas `NA` type
- The Python `is not None` operator returns `True` for both `numpy.nan` and pandas `NA`

This means code like this appears to work but silently fails:
```python
# WRONG - NaN is not None returns True!
value = float(row["field"]) if row["field"] is not None else None
# If row["field"] is NaN, the condition is True
# float(NaN) creates a silent NaN that breaks downstream code
```

### Example Failure Scenario

In `src/build_v7_features.py`, the pitcher arsenal conversion code was checking for None incorrectly:

```python
# BEFORE (broken)
mean_velo = float(r["mean_velo"]) if r["mean_velo"] is not None else None

# If r["mean_velo"] is NaN from BigQuery NULL:
# - Condition evaluates to True (NaN is not None)
# - float(NaN) executes → creates NaN float
# - Downstream code sees NaN as "missing data"
# - Result: pitcher velo appears NULL in database
```

### Why This Affects Pitcher Data
The V7 feature builder queries pitcher_game_stats table. When that table doesn't exist (2026 season), it falls back to aggregating statcast data:

```sql
SELECT
    AVG(mean_velo) AS mean_velo,
    NULL as fastball_pct,
    ...
```

The explicit `NULL` values become pandas `NA`, triggering the `is not None` bug.

## The Solution

### Use `pd.notna()` Instead of `is not None`

```python
import pandas as pd

# CORRECT
mean_velo = float(r["mean_velo"]) if pd.notna(r["mean_velo"]) else None

# pd.notna(NaN) returns False
# pd.notna(real_value) returns True
# No more silent failures!
```

### When to Apply This Fix

**Always use `pd.notna()` when:**
1. Converting BigQuery results to Python objects
2. Checking if a pandas DataFrame value is not missing
3. Filtering or validating data from BigQuery queries
4. Working with numeric columns that may contain NULL

**Equivalent checks:**
```python
pd.notna(value)           # Best: explicit pandas check
pd.notnull(value)         # Also works: pandas alias
not pd.isna(value)        # Works: inverse of isna()
```

**DO NOT use:**
```python
value is not None         # WRONG: returns True for NaN
value != None             # WRONG: may have undefined behavior
```

## Implementation in Hank's Tank

### Files Modified
- `src/build_v7_features.py` (lines 562-570, 744-751)

### Methods Affected
1. **`_pitcher_arsenal_2026_statcast()`** - Lines 562-570
   ```python
   # Convert statcast aggregation results with proper NaN handling
   mean_velo = float(r["mean_velo"]) if pd.notna(r["mean_velo"]) else None
   fb_pct = float(r["fastball_pct"]) if pd.notna(r["fastball_pct"]) else None
   break_pct = float(r["breaking_pct"]) if pd.notna(r["breaking_pct"]) else None
   off_pct = float(r["offspeed_pct"]) if pd.notna(r["offspeed_pct"]) else None
   ```

2. **`_pitcher_arsenal_from_game_stats()`** - Lines 744-751
   ```python
   # Convert pitcher_game_stats query results with proper NaN handling
   mean_velo = float(r["mean_velo"]) if pd.notna(r["mean_velo"]) else None
   k_bb_pct = float(r["k_bb_pct"]) if pd.notna(r["k_bb_pct"]) else None
   xwoba_allowed = float(r["xwoba_allowed"]) if pd.notna(r["xwoba_allowed"]) else None
   # ... etc for all float fields
   ```

## Testing & Validation

### Debug Test Script
```python
# Verify the fix with this pattern:
from google.cloud import bigquery
import pandas as pd

client = bigquery.Client(project='hankstank')
row = client.query(your_sql).to_dataframe()

# Check raw types
print(f"Type of field: {type(row['field'].iloc[0])}")
print(f"Value: {row['field'].iloc[0]}")

# Test the check
if pd.notna(row['field'].iloc[0]):
    print("✓ Correctly detected non-null value")
else:
    print("✓ Correctly detected null/NaN value")
```

### Verification Steps
1. Run the feature builder for a test date
2. Check that pitcher velo values appear in BigQuery table (not NULL)
3. Verify velo_norm is calculated correctly
4. Confirm pitcher velocity is not 0 or -999 (would indicate failed conversion)

## Prevention Strategies

### Code Review Checklist
When reviewing BigQuery-to-Python conversion code:
- [ ] Are we using `pd.notna()` for type checks? (not `is not None`)
- [ ] Are we handling numeric columns from BigQuery?
- [ ] Are there explicit NULL values in SELECT clauses that might trigger this?
- [ ] Do we have unit tests for the conversion logic?

### Testing Template
```python
def test_bigquery_nan_handling():
    """Test that BigQuery NULL → pandas NaN conversion works."""
    df = client.query("""
        SELECT 
            1 as id,
            NULL as nullable_field,
            95.5 as numeric_field
    """).to_dataframe()
    
    row = df.iloc[0]
    
    # These should all pass with proper NaN handling
    assert pd.isna(row['nullable_field'])
    assert pd.notna(row['numeric_field'])
    
    # This conversion should work correctly
    value = float(row['numeric_field']) if pd.notna(row['numeric_field']) else None
    assert value == 95.5
```

## Related Issues & Edge Cases

### Multi-type DataFrames
When a DataFrame mixes numeric and string columns:
```python
df = client.query("""
    SELECT
        pitcher,
        CAST(NULL AS FLOAT64) as mean_velo,  # → pandas NaN (float type)
        CAST(NULL AS STRING) as pitch_type    # → pandas NA (object type)
""").to_dataframe()

# pd.notna() works for both!
for col in df.columns:
    for val in df[col]:
        if pd.notna(val):
            print(f"Non-null: {val}")
```

### DataFrame Indexing
Both of these work correctly with pd.notna():
```python
# Single value
if pd.notna(df.iloc[0]['column']):
    process(df.iloc[0]['column'])

# Series operations
df[pd.notna(df['column'])].process()
```

## Performance Considerations

- `pd.notna()` is optimized for DataFrames and is actually faster than `is not None` for bulk operations
- No performance penalty for switching from `is not None` to `pd.notna()`
- Use `pd.notna()` for all validity checks on DataFrame values

## References

- [Pandas isna() Documentation](https://pandas.pydata.org/docs/reference/api/pandas.isna.html)
- [BigQuery and Pandas Null Handling](https://cloud.google.com/python/docs/reference/bigquery/latest/pandas)
- [NumPy NaN vs Python None](https://numpy.org/doc/stable/reference/constants.html#numpy.nan)

## Quick Fix Checklist

If you encounter NULL values appearing in your data despite valid queries:

1. [ ] Identify the conversion code (look for `is not None`)
2. [ ] Add `import pandas as pd` at the top of the file
3. [ ] Replace all `value is not None` with `pd.notna(value)` in BigQuery conversion code
4. [ ] Test with `debug_pitcher_conversion.py` pattern
5. [ ] Run feature builder and verify data appears in BigQuery
6. [ ] Commit with message: "Fix: Use pd.notna() for BigQuery NULL handling"

---

**Last Updated:** 2026-04-07
**Hank's Tank ML Project**
