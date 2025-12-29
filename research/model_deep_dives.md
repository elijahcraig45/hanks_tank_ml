# Deep Dive: Top MLB Projection Models

This document provides a detailed breakdown of the industry-standard projection systems. Understanding *how* these models work allows us to reverse-engineer their best features for `hanks_tank_ml`.

## 1. PECOTA (Baseball Prospectus)
**"The Comparator"**

*   **Core Philosophy:** **Nearest Neighbor Analysis.**
    *   Instead of just looking at a player's past stats, PECOTA asks: *"Who else in history looked like this player at this age?"*
    *   It finds "doppelgangers" based on body type, minor league performance, and skills (e.g., "Short, high-walk, low-power 2nd baseman").
*   **Methodology:**
    1.  **Cluster:** Identify historical comparables.
    2.  **Project:** Use the *career paths* of those historical players to predict the current player's future.
    3.  **Probabilistic:** Unlike other systems that give one number (e.g., "25 HR"), PECOTA gives a range (10th, 50th, 90th percentile outcomes).
*   **Key Feature for Us:** **Probabilistic Ranges.** Don't just predict "Dodgers win." Predict "Dodgers win 60% of the time with a standard deviation of X runs."
*   **Best For:** Breakout candidates and prospects with little MLB history.

## 2. ZiPS (Dan Szymborski / FanGraphs)
**"The Historian"**

*   **Core Philosophy:** **Weighted History & Aging Curves.**
    *   ZiPS believes the best predictor of the future is a weighted average of the past, adjusted for how players typically age.
*   **Methodology:**
    1.  **Baseline:** Calculates a baseline using 4 years of data (weighted 8/5/4/3 for most recent years).
    2.  **Aging:** Applies a generic aging curve (e.g., "Speed declines by X% at age 29") but adjusts it based on player type (e.g., "Fast players age differently than slow sluggers").
    3.  **Similarity:** Like PECOTA, it uses Mahalanobis distance to find similar players to refine the aging curve.
*   **Key Feature for Us:** **Decay Functions.** We must weight recent games (last 14 days) heavily for "form," but use long-term data (3 years) for "talent."
*   **Best For:** Long-term career projections and season totals.

## 3. Steamer (Jared Cross et al.)
**"The Regressor"**

*   **Core Philosophy:** **Regression to the Mean.**
    *   Steamer assumes that extreme performances (good or bad) are largely luck and will revert to the average.
*   **Methodology:**
    1.  **Components:** It projects individual components (K%, BB%, ISO) rather than results (ERA, AVG).
    2.  **Heavy Regression:** It aggressively regresses rookies and small-sample players to the league average.
    3.  **Split-Halves:** It uses "split-half" reliability to determine how much to trust a stat. (e.g., K% stabilizes quickly, so trust it. BABIP stabilizes slowly, so regress it heavily).
*   **Key Feature for Us:** **Component Modeling.** Don't predict "Runs." Predict "Walks," "Singles," and "Homers," then combine them.
*   **Best For:** Daily fantasy and safe, conservative projections.

## 4. THE BAT X (Derek Carty)
**"The Modernist"**

*   **Core Philosophy:** **Statcast & Context.**
    *   THE BAT X was the first major system to integrate Statcast data (Exit Velocity, Launch Angle) directly into projections.
*   **Methodology:**
    1.  **Statcast Baseline:** Instead of "Did he hit a HR?", it asks "Did he hit the ball hard enough to *deserve* a HR?" (xStats).
    2.  **Contextual Adjustments:** It applies massive adjustments for:
        *   **Ballpark Dimensions** (not just "Coors", but "Wind blowing out at Wrigley").
        *   **Umpire Tendencies** (Small zone vs Large zone).
        *   **Catcher Framing.**
*   **Key Feature for Us:** **Granular Context.** This validates our "Park & Weather" research. We *must* adjust for the environment.
*   **Best For:** Daily Fantasy Sports (DFS) and Betting.

## 5. ATC (Average Total Cost - Ariel Cohen)
**"The Aggregator"**

*   **Core Philosophy:** **Wisdom of the Crowds.**
    *   ATC does not have its own physics engine. It aggregates other models (Steamer, ZiPS, THE BAT, etc.).
*   **Methodology:**
    *   It assigns "weights" to other models based on their historical accuracy for specific stats.
    *   *Example:* "Steamer is best at Strikeouts, so weight Steamer 60% for Ks. ZiPS is best at HRs, so weight ZiPS 50% for HRs."
*   **Key Feature for Us:** **Ensembling.** If we build multiple models (e.g., a "Recent Form" model and a "Long Term Talent" model), averaging them will likely beat either one individually.
*   **Best For:** Draft drafts (most accurate overall).

---

## Summary: Lessons for `hanks_tank_ml`

| Model | Lesson to Steal |
| :--- | :--- |
| **PECOTA** | Use **probabilistic outputs** (confidence intervals), not just binary wins. |
| **ZiPS** | Use **weighted decay** for historical data (recent > distant). |
| **Steamer** | Predict **components** (K, BB, HR), not results (ERA, Runs). |
| **THE BAT X** | **Context is King.** Adjust for Weather, Park, and Umpires. |
| **ATC** | **Ensemble** your models. Average the predictions of a Random Forest and a Neural Net. |
