# Moneyball Principles & MLB Feature Engineering Research

## 1. Core Moneyball Concepts
"Moneyball" (based on Michael Lewis's book about the 2002 Oakland Athletics) is fundamentally about **exploiting market inefficiencies**. In baseball terms, this meant identifying skills that correlated with winning but were undervalued by the market (scouts and GMs).

### Key Concepts for ML Models:
*   **Process over Outcome:** Traditional stats (Wins, RBI, Batting Average) are often "outcome" based and dependent on teammates. Moneyball focuses on "process" stats (Walks, On-Base Percentage, Slugging) that are more within an individual's control and stable over time.
*   **Undervalued Assets:** The A's found that **On-Base Percentage (OBP)** was 3x more correlated with scoring runs than Batting Average (AVG), yet cost significantly less on the free-agent market.
    *   *ML Feature Idea:* Create "Value" features by comparing a player's production (WAR, wRC+) to their salary or roster slot expectation.
*   **The Pythagorean Expectation:** A formula derived by Bill James to estimate how many games a team *should* have won based on runs scored and runs allowed.
    *   Formula: $Win\% = \frac{RunsScored^2}{RunsScored^2 + RunsAllowed^2}$
    *   *ML Feature Idea:* Use the difference between `Actual Win %` and `Pythagorean Win %` to identify teams that are "lucky" (likely to regress) or "unlucky" (likely to improve).

## 2. Key Statistics (Sabermetrics)
These are the building blocks for any predictive baseball model.

### Batting
*   **OBP (On-Base Percentage):** Measures how often a batter reaches base. More valuable than AVG because it includes walks (BB).
*   **SLG (Slugging Percentage):** Measures power (total bases / at bats).
*   **OPS (On-Base Plus Slugging):** $OBP + SLG$. A quick, dirty, but highly effective measure of offensive production.
*   **wOBA (Weighted On-Base Average):** The "gold standard" rate stat. It weights each outcome (BB, 1B, 2B, HR) by its actual run value.
    *   *Why it matters:* A double is worth more than a single, but OPS treats OBP and SLG as equal. wOBA is mathematically precise.
*   **wRC+ (Weighted Runs Created Plus):** wOBA adjusted for park factors and league averages. 100 is average. 150 is 50% better than league average.

### Pitching
*   **WHIP (Walks + Hits per Inning Pitched):** Measures how many baserunners a pitcher allows. Lower is better.
*   **FIP (Fielding Independent Pitching):** Estimates what a pitcher's ERA *should* be based only on strikeouts, walks, hit-by-pitches, and home runs (outcomes the defense doesn't touch).
    *   *ML Feature Idea:* `ERA - FIP` differential. A positive value suggests the pitcher has been unlucky or has bad defense; a negative value suggests they are lucky.
*   **K/BB (Strikeout-to-Walk Ratio):** A pure measure of command and dominance.

### Comprehensive
*   **WAR (Wins Above Replacement):** An attempt to summarize a player's total contribution (batting, fielding, baserunning, pitching) into a single number representing wins added vs. a replacement-level player (AAAA call-up).
*   **VORP (Value Over Replacement Player):** An older metric (popularized by Baseball Prospectus) similar to WAR but focused on offense.

## 3. Translating Principles to Machine Learning Features
To build a model that "thinks" like a Moneyball GM, we need features that capture these dynamics.

### Differential Features
Instead of just raw stats, feed the model the *difference* between opponents.
*   **Starting Pitcher vs. Opposing Lineup:**
    *   `SP_FIP` vs `Opp_Lineup_wRC+`
    *   `SP_K_Rate` vs `Opp_Lineup_K_Rate`
*   **Bullpen Mismatches:**
    *   `Bullpen_xFIP_Last30Days` vs `Opp_LateInning_OPS`

### Luck/Regression Features
Identify when performance is unsustainable.
*   **BABIP (Batting Average on Balls In Play):**
    *   League average is usually ~.300.
    *   *Feature:* `Player_Current_BABIP - Player_Career_BABIP`. If high positive, expect regression (slump coming).
*   **Pythagorean Differential:** `Team_Actual_Win_Pct - Team_Pythagorean_Win_Pct`.

### Platoon Splits
Moneyball heavily utilized "platooning" (lefty batters vs. righty pitchers).
*   *Feature:* Do not use generic season stats. Use specific splits:
    *   `Batter_wOBA_vs_LHP` (when facing a lefty starter).
    *   `Pitcher_FIP_vs_LHB`.

## 4. Time-Series Feature Engineering
Baseball is a long season (162 games). "Recent form" often matters more than season averages, but sample size is tricky.

### Rolling Averages
Create multiple windows to capture "Form" vs. "Talent".
*   **Short-term (Hot/Cold):** Last 7 days / Last 25 PA (Plate Appearances).
*   **Medium-term:** Last 30 days / Last 100 PA.
*   **Long-term (True Talent):** Season-to-date or Last 365 days.
*   *Feature:* `Rolling_30_OPS / Rolling_365_OPS` (Ratio > 1 indicates a hot streak).

### Decay Functions
Instead of hard cutoffs (last 30 days), use **Exponential Weighted Moving Averages (EWMA)**.
*   Give more weight to yesterday's game than a game in April.
*   *Formula:* $S_t = \alpha Y_t + (1 - \alpha) S_{t-1}$

### Fatigue & Schedule
*   **Rest Days:** Days since last game (critical for Starting Pitchers and Catchers).
*   **Travel:** Distance traveled, time zone changes (East Coast to West Coast).
*   **Streakiness:** Number of consecutive games with a hit/run.

## 5. Modern Adaptations: Statcast (The "New" Moneyball)
Since 2015, Statcast (radar + optical tracking) has revolutionized evaluation. It measures *physics*, not just box scores.

### Quality of Contact
*   **Exit Velocity (EV):** How fast the ball leaves the bat.
*   **Launch Angle (LA):** The vertical angle of the ball.
*   **Barrel %:** The perfect combination of EV and LA (usually leads to HRs).
*   *ML Application:* A batter with low AVG but high `Hard_Hit_Rate` and `Expected_wOBA` (xwOBA) is a prime "buy low" candidate—they are hitting the ball hard but right at fielders.

### Pitching Physics
*   **Spin Rate:** Higher spin on 4-seam fastballs leads to "rising" action (more swinging strikes).
*   **Extension:** How far down the mound the pitcher releases the ball (perceived velocity).
*   **Velocity Differential:** Difference between Fastball and Changeup speed.

### Defense
*   **OAA (Outs Above Average):** Range-based defensive metric. Far superior to Errors or Fielding %.
*   *Feature:* `Team_Defense_OAA` (A pitcher with high FIP but low ERA might be bailed out by a high OAA defense).

## Summary of Recommended Features for 2026 Model

| Category | Feature Name | Description |
| :--- | :--- | :--- |
| **Talent (Base)** | `Season_wRC+`, `Season_FIP` | Baseline quality of team/player. |
| **Form (Time)** | `Rolling_14_wOBA`, `Rolling_30_ERA` | Recent performance trends. |
| **Matchup** | `SP_vs_Lineup_wOBA_Exp` | Expected production based on specific batter/pitcher handedness matchups. |
| **Luck/Regression** | `Pythag_Diff`, `BABIP_Diff` | Is the team overperforming their underlying metrics? |
| **Statcast** | `Hard_Hit_Rate_Diff` | Team A Hard Hit % - Team B Hard Hit Allowed %. |
| **Context** | `Park_Factor_Runs` | Multiplier for the specific stadium (e.g., Coors Field = 1.3x). |
