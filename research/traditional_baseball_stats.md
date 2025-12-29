# Traditional Baseball Statistics in Predictive Modeling

This document outlines the role of traditional (pre-Sabermetric) baseball statistics in modern predictive modeling. While advanced metrics often provide better isolation of skill, traditional stats represent the actual outcomes of games and remain valuable for baseline models, feature engineering, and capturing public sentiment/betting market behavior.

## 1. Key Traditional Statistics

These are the foundational metrics used in baseball for over a century.

### Batting
*   **Batting Average (AVG):** Hits divided by At-Bats.
    *   *Relevance:* Measures contact consistency. High AVG implies a player puts the ball in play often and finds holes in the defense.
*   **Home Runs (HR):** Total home runs hit.
    *   *Relevance:* Measures raw power and instant scoring potential.
*   **Runs Batted In (RBI):** Runs scored as a result of a batter's action.
    *   *Relevance:* Context-dependent measure of "clutch" performance and lineup opportunity.
*   **Runs (R):** Times a player crosses home plate.
    *   *Relevance:* Indicates a player's ability to get on base and run the bases effectively.
*   **Stolen Bases (SB):** Bases stolen.
    *   *Relevance:* Measures speed and aggressiveness; disrupts pitcher timing.

### Pitching
*   **Earned Run Average (ERA):** Earned runs allowed per 9 innings.
    *   *Relevance:* The standard measure of run prevention, though heavily influenced by defense and park factors.
*   **Wins/Losses (W-L):** Pitcher decisions.
    *   *Relevance:* Historically used to value pitchers, though highly dependent on team offense and bullpen support.
*   **Saves (SV):** Successful preservation of a lead by a relief pitcher.
    *   *Relevance:* Identifies high-leverage relievers and bullpen hierarchy.
*   **Strikeouts (K) & Walks (BB):**
    *   *Relevance:* K's measure dominance/missing bats; BB's measure control. These are the most "independent" of the traditional stats.

### Fielding
*   **Fielding Percentage (FPCT):** (Putouts + Assists) / (Putouts + Assists + Errors).
    *   *Relevance:* Basic measure of defensive reliability (avoiding mistakes), though it ignores range.
*   **Errors (E):** Mistakes allowing runners to advance.
    *   *Relevance:* Direct measure of defensive lapses.

## 2. Feature Engineering with Basic Stats

Raw totals are rarely predictive on their own. They must be transformed into rates, trends, or context-specific features.

### Temporal Transformations (Recency Bias vs. True Talent)
*   **Rolling Averages:** "Batting Average Last 7/14/30 Days". Captures "hot" or "cold" streaks.
*   **Season-to-Date vs. Career:** Weighting current season performance against career norms to stabilize predictions early in the year.
*   **Lag Features:** "Runs Scored in Previous Game". Testing for momentum effects.

### Contextual Splits (Matchups)
*   **Platoon Splits (L/R):**
    *   *Batter AVG vs. LHP / vs. RHP:* Crucial for predicting lineup optimization and pinch-hitting risks.
    *   *Pitcher ERA vs. LHB / vs. RHB:* Identifies vulnerabilities in specific matchups.
*   **Home/Away Splits:**
    *   *Home ERA vs. Road ERA:* Captures park factors and travel fatigue without complex adjustments.
    *   *Home HR Totals:* Some players are built specifically for their home stadium dimensions.

### Team-Level Aggregates
*   **Team Batting Average:** General measure of offensive consistency.
*   **Bullpen ERA:** Aggregate ERA of all relief pitchers. Critical for modeling late-game variance.
*   **Starter vs. Bullpen Split:** separating the first 6 innings from the last 3.
*   **Run Differential (R - RA):** The "Pythagorean" expectation foundation; often a better predictor of future W-L than current W-L.

## 3. Common Heuristics (Old-School Analysis)

These heuristics often represent non-linear interactions that simple linear models might miss.

*   **"Never Bet Against a Streak":** Teams winning 5+ games in a row are presumed to have high morale and momentum.
*   **"Ace vs. Slump":** A top-tier pitcher (low ERA, high Wins) facing an offense with a low rolling AVG is seen as a "lock."
*   **"The Sunday Lineup":** Expecting lower offensive output on day games following night games due to resting starters (can be modeled by checking lineup cards for non-regulars).
*   **"Good Pitching Beats Good Hitting":** The belief that in a specific matchup, a dominant pitcher neutralizes a high-AVG offense.
*   **"Lefty-Killer":** Managers will specifically deploy lineups heavy on right-handed batters against LHP starters.

## 4. Limitations & Baseline Value

### Why They Can Be Misleading
*   **Context Dependency:** RBI and W-L are heavily dependent on teammates. A great pitcher on a bad team might have a poor W-L record.
*   **Ignoring Defense:** ERA treats all balls in play equally, punishing pitchers for bad defense.
*   **Sample Size Noise:** Batting Average takes a long time to stabilize; a "hot streak" (e.g., 5 for 10) is often just variance.
*   **Lack of Granularity:** FPCT ignores range; a slow fielder who doesn't reach a ball gets no error, while a fast fielder who almost makes a play might get an error.

### The "Baseline" Value
Despite flaws, traditional stats are essential for:
1.  **Market Calibration:** Betting lines often overreact to traditional stats (e.g., W-L records), creating value for models that identify the discrepancy.
2.  **Descriptive Reality:** They describe *what actually happened*. To predict runs, you must understand how runs were scored historically.
3.  **Simplicity:** They are robust. Complex metrics can overfit; AVG and ERA rarely "break" as concepts.
4.  **Proxy Variables:** High Strikeout totals (traditional) are a strong proxy for Stuff+ (advanced).
