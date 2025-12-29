# Basic Player & Team Stats (Traditional)

This document focuses on "Old School" baseball statistics—the numbers found on the back of a baseball card. While modern Sabermetrics (WAR, FIP, Statcast) are often more predictive, traditional stats still drive public perception, betting markets, and player confidence.

## 1. The "Back of the Baseball Card" Stats

These are the fundamental metrics that have defined the game for over a century.

### A. Batting
*   **Batting Average (AVG):** Hits / At-Bats.
    *   *Relevance:* Measures contact skills and consistency. High AVG teams rarely go into prolonged slumps.
*   **Home Runs (HR) & RBI:** Power and production.
    *   *Relevance:* HRs are the quickest way to score. RBI measures "clutch" opportunity (though heavily dependent on teammates).
*   **Runs Scored (R):** The ultimate objective.
    *   *Relevance:* High correlation with winning (obviously).
*   **Stolen Bases (SB):** Speed and aggression.
    *   *Relevance:* Disrupts pitchers and creates scoring opportunities without hits.

### B. Pitching
*   **Earned Run Average (ERA):** (Earned Runs / IP) * 9.
    *   *Relevance:* The standard for preventing runs.
*   **Wins (W) & Losses (L):**
    *   *Relevance:* While flawed for individual talent, "Winners win." It captures run support and bullpen reliability.
*   **Strikeouts (K) & Walks (BB):**
    *   *Relevance:* K's are safe outs (no defense needed). Walks are free passes (disastrous).
*   **Saves (SV):**
    *   *Relevance:* Measures bullpen reliability in high-leverage spots.

### C. Fielding
*   **Errors (E) & Fielding Percentage (FPCT):**
    *   *Relevance:* Measures defensive stability. High-error teams give opponents "extra outs."

---

## 2. Feature Engineering with Basic Stats

Raw season totals (e.g., "25 HRs") are useless for predicting *tomorrow's* game. We must transform them.

### A. Rolling Averages (The "Hot Hand")
Baseball is a game of streaks.
*   **Features:**
    *   `batter_avg_last_7d`: Is the batter seeing the ball well *this week*?
    *   `team_runs_per_game_last_10`: Is the offense clicking?
    *   `starter_era_last_3_starts`: Recent form often trumps season-long ERA.

### B. Splits (Context Matters)
*   **Home vs. Away:**
    *   *Feature:* `team_home_avg` vs. `team_away_avg`. Some teams are built for their home park.
*   **Lefty vs. Righty (Platoon Splits):**
    *   *Feature:* `batter_avg_vs_LHP` / `batter_avg_vs_RHP`.
    *   *Usage:* If the opponent is starting a Lefty, use the `vs_LHP` stats, not the season total.

### C. Differentials (Matchups)
*   **Starter vs. Offense:**
    *   *Feature:* `starter_era` **minus** `opp_team_runs_per_game`.
    *   *Logic:* A 3.00 ERA pitcher vs. a team scoring 5.0 runs/game -> Who has the edge?

---

## 3. Common Heuristics & "Old School" Rules

These are simple rules of thumb used by bettors and managers for decades.

*   **"The Ace Stopper":** An Ace pitcher (low ERA, high Wins) starting after a team loss.
    *   *Feature:* `is_ace_starting` AND `team_lost_last_game`.
*   **"The Sunday Lineup":** Day games after night games often feature "B-teams" (backup catchers, resting stars).
    *   *Feature:* `is_day_after_night` -> Expect lower offensive output (or check lineup explicitly).
*   **"Good Pitching Beats Good Hitting":**
    *   *Feature:* Weight `starter_era` higher than `team_batting_avg` in the model.

---

## 4. Why Use Basic Stats? (The "Baseline")

If Sabermetrics are better, why bother with AVG and ERA?

1.  **Market Sentiment:** Betting lines often overreact to basic stats (e.g., a pitcher with a 2.50 ERA but 4.50 xFIP is "overvalued" by the public). To find value, you must know what the public sees.
2.  **Player Psychology:** Players care about AVG. A player hitting .190 might press and perform worse than their underlying metrics suggest.
3.  **Simplicity:** They are robust. They don't rely on complex optical tracking systems that might break or have missing data.

## 5. Implementation Checklist

1.  [ ] **Data Collection:** Ensure `statsapi` fetches basic box score stats daily.
2.  [ ] **Transformation:** Create a pipeline to calculate `Last_N_Games` rolling averages for all players.
3.  [ ] **Splits:** Store separate accumulators for Home/Away and vs LHP/RHP.
