# Moneyball Principles & Feature Engineering for MLB ML

This document breaks down core principles from *Moneyball* (and modern Sabermetrics) and translates them into actionable features for machine learning models predicting game results and player performance.

## 1. Core Moneyball Principles

### A. Market Inefficiency (Undervalued Assets)
*   **Concept:** The market (and public perception) often overvalues "flashy" stats (Batting Average, RBI, Pitcher Wins) and undervalues stats that actually contribute to winning (On-Base Percentage, Slugging).
*   **ML Application:**
    *   **Feature Idea:** Instead of raw `AVG`, prioritize `OBP` (On-Base Percentage) and `ISO` (Isolated Power).
    *   **Feature Idea:** Create "Value" features. Compare a player's recent performance metrics against their salary or "public" projection to find undervalued players in daily fantasy or betting contexts.

### B. Process Over Outcome
*   **Concept:** A batter hitting a line drive right at a fielder is a "good process" with a "bad outcome." Over a long season, the process matters more.
*   **ML Application:**
    *   **Feature Idea:** **Expected Stats (xStats)**. Use `xBA` (Expected Batting Average), `xwOBA` (Expected Weighted On-Base Average) based on quality of contact (Exit Velocity, Launch Angle) rather than actual results.
    *   **Feature Idea:** **Luck Quantification**. Calculate `BABIP` (Batting Average on Balls In Play).
        *   *High BABIP (> .350)* suggests a player is "lucky" and due for regression (sell/fade).
        *   *Low BABIP (< .260)* suggests a player is "unlucky" and due for a bounce-back (buy/target).

### C. Pythagorean Expectation
*   **Concept:** A team's win-loss record is often deceptive. Their Run Differential (Runs Scored vs. Runs Allowed) is a better predictor of *future* success than their current Win %.
*   **ML Application:**
    *   **Feature Idea:** `Pythagorean Win Expectancy`. Formula: $ \frac{RunsScored^2}{RunsScored^2 + RunsAllowed^2} $.
    *   **Feature Idea:** `Record Luck`. Difference between `Actual Win %` and `Pythagorean Win %`. Teams winning more than they "should" are prime candidates to lose upcoming games against strong opponents.

---

## 2. Key Statistics & Features

### Batting Features
| Stat | Moneyball Rationale | ML Feature Name |
| :--- | :--- | :--- |
| **OBP** | Getting on base prevents outs, the most finite resource. | `team_rolling_obp_15d` |
| **OPS+** | On-Base + Slugging, adjusted for park factors. | `batter_ops_plus_season` |
| **BB/K** | Walk-to-Strikeout ratio indicates plate discipline. | `batter_bb_k_ratio_30d` |
| **wRC+** | Weighted Runs Created Plus. The "gold standard" of total offensive value. | `lineup_avg_wrc_plus` |

### Pitching Features
| Stat | Moneyball Rationale | ML Feature Name |
| :--- | :--- | :--- |
| **FIP** | Fielding Independent Pitching. Measures what a pitcher controls (K, BB, HR). | `starter_fip_season` |
| **WHIP** | Walks + Hits per Inning Pitched. Prevention of baserunners. | `bullpen_whip_14d` |
| **K% - BB%** | Strikeout rate minus Walk rate. Dominance indicator. | `starter_k_minus_bb_diff` |

---

## 3. Time-Series & Trend Features (Days, Weeks, Months)

Baseball is a game of streaks and adjustments. Static season-long averages often miss the signal.

### A. The "Hot Hand" (Short-Term)
*   **Window:** Last 7-14 Days.
*   **Features:**
    *   `batter_rolling_ops_7d`: Is the batter seeing the ball well *now*?
    *   `bullpen_rolling_era_14d`: Is the bullpen overworked or struggling recently?

### B. Consistency & Fatigue (Medium-Term)
*   **Window:** Last 30 Days / Rolling Month.
*   **Features:**
    *   `pitcher_pitch_count_30d`: Heavy workload often leads to velocity dips or "dead arm."
    *   `team_wrc_plus_30d`: How is the offense performing this month?

### C. True Talent (Long-Term)
*   **Window:** Season-to-Date (or Career if early in season).
*   **Features:**
    *   `career_platoon_splits`: How does this batter perform against Left-Handed Pitching (LHP) vs RHP historically? (Crucial Moneyball tactic: Platoon Advantage).

---

## 4. Modern Adaptations (Statcast Era)

The "New Moneyball" is physical data.

*   **Barrel %:** The percentage of batted balls with perfect exit velocity and launch angle.
    *   *Feature:* `lineup_barrel_rate_avg` vs `starter_barrel_rate_allowed`.
*   **Spin Rate:** High spin rate fastballs stay up; high spin curveballs drop more.
    *   *Feature:* `starter_fastball_spin_deviation` (Is their stuff "flat" today?).
*   **Sprint Speed:**
    *   *Feature:* `team_avg_sprint_speed` (Can they score from first on a double? Can they beat out infield hits?).

## 5. Feature Engineering Strategy for `hanks_tank_ml`

1.  **Create "Matchup" Features:** Never use a stat in isolation.
    *   *Bad:* `Judge_HomeRun_Rate`
    *   *Good:* `Judge_ISO` **minus** `OpposingPitcher_HR_per_9`
2.  **Park Factors:** Adjust all raw stats for the stadium.
    *   A 1.000 OPS at Coors Field (Denver) is different than at Oracle Park (SF).
    *   *Feature:* `park_factor_runs` (Multiplier for the specific venue).
3.  **The "Bullpen Game":**
    *   Moneyball emphasized undervalued relievers.
    *   *Feature:* `bullpen_rest_score` (Sum of pitches thrown by top 3 relievers in last 2 days).
