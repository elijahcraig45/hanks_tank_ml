# WAR Deep Dive: bWAR vs. fWAR for Machine Learning

This document details the differences between the two major Wins Above Replacement (WAR) models—**bWAR** (Baseball-Reference) and **fWAR** (FanGraphs)—and how to utilize them effectively in predictive machine learning models.

## 1. Calculation Differences

While both metrics attempt to answer "how many wins is this player worth over a replacement-level player?", they use fundamentally different inputs, particularly for pitchers and defense.

| Feature | **bWAR (Baseball-Reference)** | **fWAR (FanGraphs)** | **ML Implication** |
| :--- | :--- | :--- | :--- |
| **Pitching Base** | **RA9 (Runs Allowed per 9)**. Based on actual runs allowed (earned and unearned). Adjusts for team defense behind the pitcher. | **FIP (Fielding Independent Pitching)**. Based on strikeouts, walks, and home runs. Assumes average luck on balls in play. | **bWAR** measures *what happened* (value). **fWAR** measures *talent/process* (predictive). |
| **Defense Metric** | **DRS (Defensive Runs Saved)**. A play-by-play system that credits/debits players for plays made vs. average. | **UZR (Ultimate Zone Rating)** or **OAA (Outs Above Average)**. Generally considered slightly more stable than DRS, though both are volatile in small samples. | Defensive metrics are noisy. For daily predictions, defensive WAR is less useful than specific matchup data (e.g., Catcher Framing). |
| **Park Factors** | Uses a dynamic park factor that changes based on actual run scoring environments. | Uses a modeled park factor (generally more static). | Minor difference, but bWAR captures "weird" stadium effects in a specific season better. |

## 2. Predictive vs. Descriptive Value

For machine learning models, the choice of WAR depends on the target variable.

*   **fWAR (FanGraphs) is more PREDICTIVE.**
    *   Because it uses FIP for pitchers, it strips out the noise of defense and luck (BABIP).
    *   *Best for:* Predicting future game outcomes, season win totals, or "rest of season" performance.
    *   *Why:* A pitcher with a 3.00 ERA but 4.50 FIP will likely regress. fWAR "knows" this; bWAR thinks he's an ace.

*   **bWAR (Baseball-Reference) is more DESCRIPTIVE.**
    *   It tells you exactly how valuable a player's results were to their team's actual standing.
    *   *Best for:* Awards prediction (MVP/Cy Young), historical analysis, or analyzing "value generated."
    *   *Why:* If a pitcher induces weak contact that isn't captured by FIP, bWAR gives them credit.

## 3. Aggregating WAR for Team Success

A team of "replacement level" players is theoretically expected to win ~47-48 games in a 162-game season (a .294 winning percentage).

**The Formula:**
$$ \text{Expected Wins} = \text{Replacement Wins} + \sum (\text{Player WAR}) $$

*   **Replacement Level:** ~48 Wins
*   **Example:** A team with 40 total WAR.
    *   $48 + 40 = 88 \text{ Wins}$ (Playoff Contender)
*   **Example:** A team with 20 total WAR.
    *   $48 + 20 = 68 \text{ Wins}$ (Rebuilding)

**ML Feature Engineering Note:**
Do not sum the *entire roster's* WAR. For a daily game prediction, you must sum the WAR of the **active lineup** and the **starting pitcher**, plus a weighted average of the **bullpen**.

## 4. Pitfalls of WAR in Daily Models

Using raw season WAR as a feature for a single game prediction is dangerous.

1.  **Cumulative Nature:** WAR is a counting stat. A player with 6.0 WAR in September is "better" than a player with 1.0 WAR in April, but the April player might be just as talented (just hasn't played enough).
    *   *Fix:* Use **WAR/162** or **WAR per Plate Appearance** to normalize.
2.  **Context Neutral:** WAR assumes a "neutral" opponent. It does not account for:
    *   Platoon splits (Left vs. Right).
    *   Batter vs. Pitcher history.
    *   Hot/Cold streaks.
3.  **Defensive Noise:** Defensive WAR takes ~3 years to stabilize. Using "0.1 dWAR" from 20 games of data is adding noise, not signal.

## 5. Feature Engineering Guide: "WAR-based" Features

Here is how to properly engineer WAR features for a daily predictive model (e.g., XGBoost, LightGBM).

### A. The "Talent Mismatch" Feature (fWAR Differential)
Use fWAR to capture the "true talent" gap between the starting pitchers.
*   **Feature:** `sp_fwar_diff`
*   **Calculation:** `(Home_SP_Projected_fWAR_per_IP) - (Away_SP_Projected_fWAR_per_IP)`
*   *Note:* Use "Projected" (from systems like Steamer/ZiPS) rather than current season accumulated WAR early in the season.

### B. Lineup Strength (WAR/PA)
Calculate the aggregate strength of the specific 9 batters in the lineup that day.
*   **Feature:** `home_lineup_avg_war_per_pa`
*   **Calculation:** $\frac{\sum (\text{Batter WAR})}{\sum (\text{Batter PA})}$ for the confirmed starting 9.
*   *Insight:* This captures when a star player is resting (load management), which drastically drops the team's win probability.

### C. Bullpen WAR Weighting
Bullpens are volatile. Aggregate the WAR of the "High Leverage" relievers (Closer + Setup men).
*   **Feature:** `bullpen_top3_war_sum`
*   **Calculation:** Sum of fWAR for the top 3 relievers by "Leverage Index" or innings pitched.

### D. The "Luck" Feature (bWAR - fWAR)
Since bWAR is results-based and fWAR is process-based, the difference can indicate regression candidates.
*   **Feature:** `sp_war_luck_diff`
*   **Calculation:** `Normalized_bWAR - Normalized_fWAR`
*   *Interpretation:*
    *   **Positive Value:** Pitcher has results better than peripherals (Lucky/Good Defense) -> **Fade**.
    *   **Negative Value:** Pitcher has peripherals better than results (Unlucky/Bad Defense) -> **Buy**.
