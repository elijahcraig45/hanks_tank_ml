# Circadian Rhythms and Athletic Performance in Baseball

## Executive Summary
This document summarizes research regarding the impact of circadian rhythms, sleep, and travel on athletic performance, with a specific focus on Major League Baseball (MLB). The findings suggest that circadian biology significantly influences reaction time, physical strength, and cognitive decision-making—all critical components of baseball performance.

## 1. The Peak Performance Window
Human physiological function follows a distinct circadian rhythm, regulated by the suprachiasmatic nucleus (SCN). Research consistently identifies a "Peak Performance Window" for athletic endeavors.

*   **Body Temperature & Performance**: Athletic performance metrics (strength, power, flexibility, reaction time) tend to peak in the **late afternoon and early evening (approx. 16:00 - 20:00)**. This coincides with the daily maximum of core body temperature.
*   **Mechanism**: Higher body temperature facilitates actin-myosin cross-bridging (muscle contraction), increases nerve conduction velocity, and improves metabolic efficiency.
*   **Relevance to Baseball**: Most MLB games are played in the evening (19:00 start). For a player whose body clock is aligned with local time, this is slightly past the absolute physiological peak but still within a high-performance zone. However, misalignment (jet lag) can shift this window significantly.

## 2. Jet Lag and Directionality: The "West Coast Advantage"
One of the most cited phenomena in sports circadian research is the impact of travel direction on winning percentages.

### The Winter et al. Findings
Dr. W. Christopher Winter has conducted extensive research on MLB travel. Key findings often cited include:
*   **West Coast Teams Traveling East**: Teams from the Pacific Time Zone (PT) traveling to the Eastern Time Zone (ET) often exhibit a statistical **advantage** in night games.
    *   *Reasoning*: A 7:00 PM game in New York is 4:00 PM "body time" for a Los Angeles player. This 4:00 PM slot aligns perfectly with the player's circadian peak (maximum alertness and body temperature).
*   **East Coast Teams Traveling West**: Teams from the ET traveling to the PT often face a **disadvantage**.
    *   *Reasoning*: A 7:00 PM game in Los Angeles is 10:00 PM "body time" for a New York player. At this time, the body is beginning to wind down, melatonin secretion may be initiating, and reaction times begin to degrade.
*   **Magnitude**: Studies have suggested winning percentages for advantaged teams can be significantly higher (e.g., winning ~60% of games in the "advantaged" context vs. ~50% normally), though exact statistics vary by specific study year and sample size.

### Circadian Phase Response
*   **Eastward Travel (Phase Advance)**: Harder for the body to adjust to. It is more difficult to fall asleep earlier and wake up earlier.
*   **Westward Travel (Phase Delay)**: Easier for the body to adjust to. Staying up later is generally easier than forcing sleep.

## 3. Chronotypes: Larks vs. Owls
Individual differences in circadian preference ("chronotype") play a role.
*   **Evening Types ("Owls")**: Tend to perform better in evening games. In the general population, young adults (the age of most MLB players) skew towards eveningness.
*   **Morning Types ("Larks")**: May struggle more with late-night games or travel that pushes game times later into their biological night.
*   **Pitchers vs. Batters**: Some research suggests that starting pitchers (who need sustained focus over hours) and batters (who need split-second reaction time) may be affected differently by sleep debt and circadian misalignment.

## 4. Sleep Extension and Recovery
Research by Cheri Mah (Stanford) and others highlights the critical role of sleep duration.
*   **Sleep Extension**: Extending sleep to ~10 hours/night has been shown to improve sprinting speed, hitting accuracy, and reaction time in collegiate athletes (tennis, basketball).
*   **Sleep Debt**: Chronic sleep restriction (common in MLB due to travel schedules) leads to cumulative deficits in vigilance and cognitive processing.

## 5. Implications for Hanks Tank ML Model (2026 Season)
To incorporate these findings into the `hanks_tank_ml` predictive model, we should consider the following feature engineering strategies:

### Proposed Features
1.  **`circadian_offset`**:
    *   Calculate the difference between the *local game time* and the *body clock time* of the starting pitcher and lineup.
    *   *Logic*: If `game_time_local` is 19:00, but `team_home_timezone` is PST (-3 hours), `body_time` is 16:00.
    *   Feature Value: `body_time_hour` (e.g., 16.0, 22.0).

2.  **`travel_direction`**:
    *   Categorical: `Eastward`, `Westward`, `None`.
    *   Or Continuous: `hours_shifted` (positive for East, negative for West).

3.  **`days_since_travel`**:
    *   Circadian adjustment takes time (approx. 1 day per hour of shift).
    *   Feature: Number of days the team has been in the current time zone.
    *   *Interaction*: `circadian_offset` * decay_factor based on `days_since_travel`.

4.  **`game_time_body_clock`**:
    *   Specific feature capturing the "West Coast Advantage".
    *   Boolean: `is_west_coast_team_in_east_night_game`.

### Hypothesis for Model
*   **Positive Correlation**: `body_time` between 15:00 and 19:00 should correlate with higher offensive production (wOBA, HardHit%).
*   **Negative Correlation**: `body_time` > 22:00 should correlate with increased error rates and decreased plate discipline (higher Chase%).

## References
*   **Winter, W. C.** (Various presentations/abstracts). *Impact of circadian rhythm on MLB performance.*
*   **Teo, W., et al. (2011)**. *Circadian Rhythms in Exercise Performance: Implications for Hormonal and Muscular Adaptation.* Journal of Sports Science & Medicine.
*   **Mah, C. D., et al. (2011)**. *The effects of sleep extension on the athletic performance of collegiate basketball players.* Sleep.
*   **Smith, R. S., et al. (2013)**. *Circadian advantage for West Coast teams traveling East.*
