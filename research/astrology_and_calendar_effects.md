# Astrology, Moon Phases, & Calendar Effects in MLB

This document explores "fringe" and temporal features—Astrology, Moon Phases, Biorhythms, and Circadian Rhythms—to determine if they have any predictive power or correlation with game outcomes. While often considered pseudoscience, these factors provide a fun and unique angle for analysis.

## 1. Moon Phases & Lunar Cycles

The "Full Moon Effect" is a common superstition in emergency rooms and police stations, suggesting chaos and high energy. Does it apply to baseball?

### A. Hypotheses
*   **The "Chaos" Theory:** Full Moons lead to higher variance games (more errors, wild pitches, extra innings).
*   **Visibility:** Historically, brighter nights might have aided visibility, though modern stadium lights negate this.
*   **Gravitational Pull:** Does the "tide" affect air density or player physiology? (Highly unlikely, but testable).

### B. Data Collection & Engineering
*   **Library:** `ephem` or `skyfield` (Python).
*   **Features:**
    *   `moon_phase_pct`: 0.0 (New) to 1.0 (Full).
    *   `is_full_moon`: Boolean (if phase > 0.95).
    *   `is_waxing` / `is_waning`.

---

## 2. Astrology & Zodiac Signs

Do players perform better during their "season"? Do Leos hit more home runs?

### A. Player Zodiac
*   **Data Source:** `statsapi` provides `birthDate`.
*   **Features:**
    *   `batter_zodiac_sign`: Categorical (Aries, Taurus, etc.).
    *   `is_player_birthday`: The "Birthday Home Run" narrative.
    *   `zodiac_compatibility`: Does a Scorpio pitcher dominate a Gemini batter? (Matrix of Sign vs. Sign performance).

### B. Biorhythms
A pseudo-scientific theory claiming humans operate on fixed mathematical cycles starting from birth.
*   **Cycles:**
    *   **Physical (23 days):** Strength, coordination, stamina.
    *   **Emotional (28 days):** Mood, nerves, creativity.
    *   **Intellectual (33 days):** Strategy, focus.
*   **Feature Engineering:**
    *   Calculate `sin(2 * pi * days_lived / cycle_length)`.
    *   Range: -1.0 (Low/Critical) to +1.0 (High/Peak).
    *   *Hypothesis:* A pitcher with a "Low Physical" score might have lower velocity.

---

## 3. Calendar & Temporal Factors (The "Real" Science)

These factors have grounded physiological backing and are supported by sports science research.

### A. Circadian Science & Performance
Research confirms that circadian rhythms significantly impact athletic performance, primarily driven by core body temperature and reaction times.

*   **Peak Performance Window:**
    *   **Science:** Athletic performance (strength, flexibility, reaction time) peaks in the **late afternoon/early evening (approx. 4:00 PM – 8:00 PM)**, coinciding with maximum body temperature.
    *   **Implication:** Players are biologically primed for night games (7:00 PM starts).

*   **The "West Coast Advantage" (Eastward Travel):**
    *   **Scenario:** A West Coast team (e.g., Dodgers) plays a night game on the East Coast (e.g., vs. Mets at 7:00 PM ET).
    *   **Body Clock:** The Dodgers' players feel like it is **4:00 PM PT**.
    *   **Advantage:** They are playing exactly during their biological peak window.
    *   **Stat:** Historical data suggests West Coast teams have a higher winning percentage in East Coast night games than expected.

*   **The "East Coast Disadvantage" (Westward Travel):**
    *   **Scenario:** An East Coast team (e.g., Yankees) plays a night game on the West Coast (e.g., vs. Angels at 7:00 PM PT).
    *   **Body Clock:** The Yankees' players feel like it is **10:00 PM ET**.
    *   **Disadvantage:** They are playing past their peak, as their bodies begin to wind down for sleep. Reaction times degrade.

*   **Feature Engineering:**
    *   `body_clock_time`: Estimate the "internal time" of the starting pitcher based on their home time zone.
    *   `circadian_offset`: Difference between `Game_Time` and `Optimal_Peak_Time` (e.g., 6:00 PM).

### B. Day of Week & "Sunday Funday"
*   **Sunday Games:** Often day games following Saturday night games.
*   **Hypothesis:** "Getaway Day" lineups are often weaker (resting stars), and players might be fatigued or hungover.
*   **Feature:** `is_day_game_after_night_game`.

### C. Seasonality
*   **"June Swoon" vs. "Mr. October":**
*   **Feature:** `month_of_season` (April = Cold/Pitchers, July/August = Hot/Hitters).

---

## 4. Implementation Plan

### Step 1: Install Libraries
```bash
pip install ephem skyfield
```

### Step 2: Create `AstrologyFeatureGenerator`
A Python class to handle the "weird" math.

```python
import ephem
import math

def get_moon_phase(date):
    observer = ephem.Observer()
    observer.date = date
    moon = ephem.Moon(observer)
    return moon.phase # 0 to 100

def get_biorhythm(birth_date, game_date, cycle=23):
    delta = (game_date - birth_date).days
    return math.sin(2 * math.pi * delta / cycle)
```

### Step 3: Analysis Goals
1.  **Correlation Matrix:** Check `moon_phase` vs. `total_runs_scored`.
2.  **Split Stats:** Compare `LeBron James` (Capricorn) stats in Capricorn season vs. others.
3.  **Model Feature Importance:** Feed `biorhythm_physical` into the model and see if it gains any weight (likely low, but worth checking).
