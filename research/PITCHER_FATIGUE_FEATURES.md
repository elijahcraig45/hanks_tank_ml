# Pitcher Fatigue & Rest Features

Based on the research in `research/pitcher_rest_cycles.md`, this document outlines specific features to engineer for the predictive models. These features aim to quantify "rest," "fatigue," and "managerial usage patterns."

## 1. Starting Pitcher Features

### A. Rest Days
*   **`days_since_last_start`**: Integer. The primary metric.
*   **`is_short_rest`**: Boolean. True if `days_since_last_start` < 4.
    *   *Hypothesis:* Significant negative coefficient for performance; increased variance.
*   **`is_extra_rest`**: Boolean. True if `days_since_last_start` > 5.
    *   *Hypothesis:* Slight positive impact on velocity, potential negative impact on command (walk rate).

### B. Recent Workload (The "Dead Arm" Proxies)
*   **`pitches_last_start`**: Integer. Raw count.
*   **`pitches_last_3_starts`**: Integer. Rolling sum. High values indicate accumulating fatigue.
*   **`innings_pitched_season`**: Float. Cumulative innings.
*   **`season_workload_ratio`**: Float. `innings_pitched_season` / `previous_season_total_innings`.
    *   *Hypothesis:* If ratio > 1.2 (Verducci Effect), injury risk and performance regression increase.

### C. Velocity & Stuff Trends
*   **`fastball_velo_trend`**: Float. (Average fastball velocity in last start) - (Season average fastball velocity).
    *   *Signal:* A drop of > 1.5 mph is a strong "dead arm" or injury indicator.
*   **`spin_rate_trend`**: Float. (Average spin rate last start) - (Season average).
    *   *Signal:* Drop in spin often precedes velocity drop and injury.

### D. The "Third Time Through" (TTO) Penalty
*   **`opp_batting_avg_3rd_time`**: Float. Historical opponent BA vs. this pitcher the 3rd time through the order.
*   **`avg_innings_per_start`**: Float. Does the manager typically pull them early (Kevin Cash style) or let them ride (Dusty Baker style)?

## 2. Relief Pitcher Features

### A. Immediate Availability
*   **`days_consecutive_pitched`**: Integer. 0, 1, 2, 3+.
    *   *Hypothesis:* If 2, performance degrades. If 3, high risk of "blowup" or injury.
*   **`pitches_last_24h`**: Integer.
*   **`pitches_last_72h`**: Integer. The standard "3-day window" for reliever availability.

### B. "Abuse" Metrics
*   **`high_leverage_innings_count`**: Integer. Count of innings pitched with Leverage Index (LI) > 1.5 in last 7 days.
    *   *Reasoning:* 20 pitches in a blowout is different than 20 pitches with bases loaded.
*   **`warmup_penalty`**: (Requires detailed log data) If available, count of times warmed up without entering. If not, proxy with `games_active_roster` vs `games_pitched`.

## 3. Team/Manager Context Features

### A. Bullpen Management Style
*   **`manager_hook_speed`**: Float. Average pitch count at which starting pitchers are pulled.
*   **`bullpen_usage_rate`**: Float. Average bullpen innings per game for the team.
    *   *Impact:* Teams with high bullpen usage (e.g., Rays) might have fresher relievers due to frequent rotation, or overworked ones if depth is poor.

## 4. Implementation Logic (Python/Pandas)

```python
# Example Logic for 'days_since_last_start'
df['game_date'] = pd.to_datetime(df['game_date'])
df = df.sort_values(['pitcher_id', 'game_date'])
df['prev_game_date'] = df.groupby('pitcher_id')['game_date'].shift(1)
df['days_rest'] = (df['game_date'] - df['prev_game_date']).dt.days - 1

# Example Logic for 'is_short_rest'
df['is_short_rest'] = (df['days_rest'] < 4).astype(int)

# Example Logic for Velocity Trend (requires Statcast data)
# Calculate rolling average of last 3 games vs season average
df['rolling_velo'] = df.groupby('pitcher_id')['release_speed'].transform(lambda x: x.rolling(3).mean())
df['season_velo'] = df.groupby('pitcher_id')['release_speed'].transform('mean')
df['velo_drop'] = df['rolling_velo'] - df['season_velo']
```

## 5. Predictive Signals to Look For

*   **Starter Velo Drop + Short Rest** -> High probability of `runs_allowed` > projected.
*   **Reliever 3rd Day Consecutive** -> High probability of `blown_save` or `walks`.
*   **"Verducci Effect" (Year-over-Year Innings Jump)** -> Long-term fade for second half of season performance.
