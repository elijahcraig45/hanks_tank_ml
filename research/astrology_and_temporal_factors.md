# Research: Astrology, Moon Phases, and Temporal Factors in MLB

This document outlines a research plan to investigate the potential impact of "non-traditional" factors—specifically astrology, moon phases, and calendar events—on MLB game outcomes and player performance. While this is a "fun" analysis, the data collection and feature engineering will be approached with rigor.

## 1. Moon Phases

### Hypothesis
Anecdotal evidence suggests "weird" things happen during a Full Moon (e.g., more errors, higher scoring, unpredictable plays). We aim to quantify this.

### Research Questions
*   Do games played under a **Full Moon** have higher total runs scored?
*   Is there a statistically significant increase in **errors** committed during Full Moon games?
*   Do specific players perform better or worse during specific moon phases?

### Data Collection & Tools
*   **Library:** `ephem` or `skyfield` (Python).
*   **Method:** Calculate the moon phase (0.0 to 1.0) for the date and time of each game.
*   **Granularity:** Daily/Game-level.

### Feature Engineering
*   `moon_phase_float`: Continuous variable from 0 (New) to 0.5 (Full) to 1.0 (New).
*   `is_full_moon`: Boolean (True if `moon_phase_float` is within a small threshold of 0.5, e.g., +/- 2 days).
*   `is_new_moon`: Boolean (True if `moon_phase_float` is close to 0.0 or 1.0).
*   `moon_illumination`: Percentage of the moon illuminated.

## 2. Astrology & Zodiac

### Hypothesis
Players may have inherent traits associated with their Zodiac signs that influence performance, or they may perform better during their "season" (e.g., Leos in August).

### Research Questions
*   **Zodiac Performance:** Do Fire signs (Aries, Leo, Sagittarius) hit more home runs? Do Earth signs (Taurus, Virgo, Capricorn) have lower ERA?
*   **Birthday Boost:** Do players perform better on or near their birthday?
*   **Biorhythms:** Do players peak when their physical, emotional, and intellectual cycles align?

### Data Collection & Tools
*   **Source:** MLB Stats API (`people` endpoint) provides `birthDate`.
*   **Library:** Standard Python `datetime` for Zodiac calculation. Custom function for Biorhythms.

### Feature Engineering
*   `player_zodiac_sign`: Categorical (Aries, Taurus, etc.).
*   `player_element`: Categorical (Fire, Earth, Air, Water).
*   `is_player_birthday`: Boolean.
*   `days_since_birthday`: Integer (0-364).
*   **Biorhythms:**
    *   `bio_physical`: $\sin(2\pi t / 23)$
    *   `bio_emotional`: $\sin(2\pi t / 28)$
    *   `bio_intellectual`: $\sin(2\pi t / 33)$
    *   *Where $t$ is days lived.*

## 3. Calendar & Temporal Factors

### Hypothesis
Time-based factors like day of the week, month of the season, and circadian disruptions affect player fatigue and focus.

### Research Questions
*   **Sunday Funday:** Are Sunday day games (often following Saturday night games) higher variance due to fatigue?
*   **Month Effects:** Do certain players historically perform better in specific months ("Mr. October", "June Swoon")?
*   **Circadian Rhythm:** Do West Coast teams struggle more when playing early games on the East Coast (body clock is 3 hours behind)?

### Data Collection & Tools
*   **Source:** Game schedule data (date, time, venue).
*   **Time Zones:** Map stadium locations to time zones.

### Feature Engineering
*   `day_of_week`: Categorical (Mon-Sun).
*   `month`: Categorical (Apr-Oct).
*   `is_day_game_after_night`: Boolean (Fatigue proxy).
*   `circadian_diff`: Integer (Hours difference between team's home time zone and game time zone).
    *   *Example:* LA Dodgers (PT) playing in New York (ET) at 1 PM ET feels like 10 AM PT (`-3` disadvantage).

## 4. Implementation Plan

### Phase 1: Data Enrichment
1.  **Player Metadata:** Create a reference table of all active players with `birthDate` and calculated `zodiac_sign`.
2.  **Game Metadata:** Enhance the game log dataset with `moon_phase` and `moon_illumination`.

### Phase 2: Feature Generation
1.  Implement `calculate_biorhythm(birth_date, game_date)` function.
2.  Implement `calculate_zodiac(birth_date)` function.
3.  Implement `calculate_circadian_impact(home_team_tz, away_team_tz, game_time)` function.

### Phase 3: Analysis
1.  **Correlation Matrix:** Check correlations between these "fun" features and target variables (Runs, Hits, Errors, Win/Loss).
2.  **Segmented Analysis:** Compare "Full Moon" vs. "Non-Full Moon" game stats using t-tests.
3.  **Predictive Modeling:** Add these features to the main ML model to see if they improve accuracy (even marginally).

## 5. Action Items
*   [ ] Verify `ephem` or `skyfield` installation in the environment.
*   [ ] Write script to fetch and cache player birth dates from Stats API.
*   [ ] Create a mapping of MLB Stadiums to Time Zones.
