# Advanced Sabermetrics & Predictive Modeling in the Modern Era

## 1. Introduction
Post-Moneyball, the focus of sabermetrics has shifted from exploiting market inefficiencies in OBP to understanding the underlying quality of contact and player skill independent of defense and luck. This document outlines key metrics for predictive modeling in MLB.

## 2. Advanced Batting Metrics

### wOBA (Weighted On-Base Average)
*   **Definition:** A version of On-Base Percentage that accounts for how a player reached base. Unlike OPS, which treats OBP and SLG as equal, wOBA weights them based on their actual run value.
*   **Formula Concept:** `(0.69×uBB + 0.72×HBP + 0.89×1B + 1.27×2B + 1.62×3B + 2.10×HR) / (AB + BB – IBB + SF + HBP)` (Weights change yearly).
*   **Predictive Value:** The single best correlation to run scoring. A better target variable than AVG or OPS.

### wRC+ (Weighted Runs Created Plus)
*   **Definition:** wOBA adjusted for park factors and league averages. Scaled so 100 is league average.
*   **Formula Concept:** `(((wRAA/PA + League R/PA) + (League R/PA - Park Factor* League R/PA))/ (AL or NL wRC/PA excluding pitchers))*100`
*   **Predictive Value:** Best metric for comparing true offensive talent across different teams and environments.

### ISO (Isolated Power)
*   **Definition:** Measures raw power by removing singles from Slugging Percentage.
*   **Formula:** `SLG - AVG` or `(2B + 2*3B + 3*HR) / AB`
*   **Predictive Value:** Stabilizes slower than K%, but indicates a player's ability to hit for extra bases independent of their BABIP luck.

### BABIP (Batting Average on Balls In Play)
*   **Definition:** The rate at which a ball hit into play becomes a hit. Excludes HRs and Ks.
*   **Formula:** `(H - HR) / (AB - K - HR + SF)`
*   **Predictive Value:** Used primarily to identify regression candidates.
    *   **Batters:** High BABIP (> .350) often suggests luck unless the player is elite (fast/hard hitter). Low BABIP (< .260) suggests bad luck.
    *   **Pitchers:** Pitchers have little control over BABIP. High allowed BABIP suggests bad defense or bad luck; expect regression to mean (~.300).

## 3. Advanced Pitching Metrics

### FIP (Fielding Independent Pitching)
*   **Definition:** Estimates what a pitcher's ERA *should* be based only on outcomes they control: Strikeouts, Walks, Hit By Pitches, and Home Runs.
*   **Formula:** `((13*HR + 3*(BB+HBP) - 2*K) / IP) + Constant`
*   **Predictive Value:** FIP is a better predictor of *future* ERA than *current* ERA is. It strips out defense and sequencing luck.

### xFIP (Expected FIP)
*   **Definition:** Replaces a pitcher's actual HR total with a league-average HR/FB rate.
*   **Predictive Value:** Better than FIP for predicting future performance for pitchers with extreme HR/FB rates (which tend to be unstable/lucky).

### SIERA (Skill-Interactive ERA)
*   **Definition:** The most complex and accurate estimator. Accounts for balls in play complexity (ground balls vs fly balls) and the fact that high-K pitchers induce weaker contact.
*   **Predictive Value:** The gold standard for pitching projections.

### K-BB% (Strikeout minus Walk Percentage)
*   **Definition:** Simply `K% - BB%`.
*   **Predictive Value:** Extremely stable and quickly reliable. A pitcher with a high K-BB% dominates the game regardless of ball-in-play luck.

## 4. Predictive Superiority & Feature Engineering

### Why Advanced Metrics Win
*   **Stability:** Metrics like K% and Exit Velocity stabilize in small sample sizes (50-100 PA), whereas ERA and AVG require huge samples to be meaningful.
*   **Context-Neutral:** wRC+ and FIP remove external noise (ballpark dimensions, bad defense), isolating the player's true skill level.

### Feature Engineering Strategies

#### A. "Luck" & Regression Features
Create features that highlight the gap between results and expected performance. Large gaps predict regression.
*   `luck_era_diff`: **ERA - FIP**. (Positive = Unlucky/Bad Defense -> Bet on improvement. Negative = Lucky -> Bet on decline).
*   `luck_woba_diff`: **wOBA - xwOBA**. (xwOBA is derived from Statcast Exit Vel & Launch Angle).
*   `babip_diff`: **Player BABIP - Career Average BABIP**.

#### B. Matchup Features
Instead of raw "Batter vs Pitcher" history (small sample size), use aggregate attribute matchups.
*   **Power vs Flyball:** `Batter ISO` vs `Pitcher FB%`.
*   **Discipline vs Control:** `Batter BB%` vs `Pitcher BB%`.
*   **Platoon Advantage:** Create a binary or weighted feature for L vs R matchups, adjusting projections based on wOBA splits.
*   **Skill Interaction:** `Pitcher SIERA` - `Batter wRC+` (normalized).

## 5. Data Sources & Implementation

### Primary Sources
1.  **FanGraphs:** The source of truth for wOBA, wRC+, FIP, xFIP, SIERA.
    *   *Access:* CSV exports or `pybaseball` library.
2.  **Baseball Savant (Statcast):** Source for "Expected" stats (xBA, xwOBA, xERA) based on ball tracking data.
    *   *Access:* `pybaseball.statcast()`.

### Calculation Notes
*   **wOBA Weights:** These change every year. Ensure the pipeline fetches the current year's "wOBA Scale" and weights from FanGraphs Guts page.
*   **FIP Constant:** Also changes yearly based on league run environment.
