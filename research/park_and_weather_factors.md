# Research: Park Factors and Weather Impact on MLB

## 1. Park Factors

Park factors quantify how a specific stadium influences game events compared to a league-average neutral environment.

### Run Factors vs. Component Factors
*   **Run Factors:** A high-level metric indicating if a park favors hitters (> 100) or pitchers (< 100).
    *   *Example:* Coors Field typically has a Run Factor > 110 (10% more runs than average).
    *   *Example:* T-Mobile Park (Seattle) often has a Run Factor < 95.
*   **Component Factors:** More granular multipliers for specific events (1B, 2B, 3B, HR, BB, SO).
    *   **Home Runs:** Great American Ball Park (Cincinnati) and Yankee Stadium are known for high HR factors.
    *   **Triples:** Oracle Park (SF) and Coors Field (Denver) have large outfields, inflating 3B factors.
    *   **BABIP:** Coors Field has a massive outfield, leading to more hits on balls in play (high BABIP factor).

### Handedness Effects
Stadium geometry creates asymmetric effects for Left-Handed Hitters (LHH) vs. Right-Handed Hitters (RHH).
*   **Yankee Stadium:** Short right-field porch significantly boosts HRs for LHH.
*   **Fenway Park:** The "Green Monster" in left field turns potential LHH fly-outs into doubles (high 2B factor for RHH/LHH pulling) but can suppress line-drive HRs.
*   **Oracle Park:** High wall in right field historically suppressed LHH power (though recent renovations changed this slightly).

### Specific Park Examples
*   **Coors Field (Denver):** The extreme outlier due to elevation (5,280 ft). Ball travels ~5-10% farther. Large outfield to compensate leads to cheap hits.
*   **Dodger Stadium:** Historically a pitcher's park at night due to the "marine layer" (dense cool air) settling in.

## 2. Weather Factors

Weather conditions significantly alter the physics of baseball flight and player performance.

### Temperature
*   **Impact:** Warmer air is less dense, allowing the ball to travel farther.
*   **Rule of Thumb:** ~3-5 feet of added distance for every 10°F increase in temperature.
*   **Pitching:** Cold weather can reduce grip and sensation, leading to lower spin rates and velocity.

### Wind
*   **Direction:**
    *   **Blowing Out:** Increases HR probability and run scoring (e.g., Wrigley Field when wind blows out to center).
    *   **Blowing In:** Suppresses offense, turning HRs into flyouts.
    *   **Crosswinds:** Can affect the movement of breaking balls and knuckleballs.
*   **Speed:** 10+ mph is usually the threshold for significant impact.
*   **Stadium Architecture:** Open stadiums (Wrigley, Kauffman) are more susceptible than enclosed or semi-enclosed ones.

### Humidity & Dew Point
*   **Air Density:** Counter-intuitively, humid air is *less* dense than dry air (water vapor is lighter than N2/O2), theoretically aiding flight. However, the effect is negligible compared to temperature/elevation.
*   **Ball Properties:** High humidity can make the ball "heavy" or "soggy" if not stored properly.
*   **Humidors:** Most MLB parks now use humidors to standardize ball moisture, dampening the extreme effects of dry air (Coors Field, Chase Field).

### Air Density (The Combined Metric)
*   **Definition:** A function of Temperature, Barometric Pressure (Elevation), and Humidity.
*   **AD Index:** A derived feature representing the resistance the ball faces.
    *   *Low Density:* High Elevation, High Temp -> High Offense.
    *   *High Density:* Low Elevation, Low Temp -> Low Offense.

## 3. Data Sources

### Park Factors
*   **FanGraphs:** "Guts!" page provides annual and multi-year park factors (Basic & 5-year regressed).
*   **Baseball Savant (Statcast):** Provides "Park Factor" based on observed vs. expected outcomes.
*   **ESPN:** Daily updated park factors.
*   **Swish Analytics:** Good for betting-related park adjustments.

### Weather Data
*   **MLB Stats API (`statsapi`):**
    *   **Confirmed:** The API provides game-time weather in the `gameData` object.
    *   **Fields:** `condition` (e.g., "Partly Cloudy"), `temp` (Fahrenheit), `wind` (Speed & Direction).
    *   *Note:* This is a snapshot at game start.
*   **VisualCrossing / OpenWeatherMap:** For historical hourly data (to track changes during the game) and forecasts.
*   **NCDC (NOAA):** Authoritative historical weather data, but requires mapping stadium lat/long.

## 4. Feature Engineering

### Proposed Features

1.  **`air_density_index`**
    *   *Formula:* Calculate air density ($\rho$) using Temp ($T$), Pressure ($P$), and Humidity ($RH$).
    *   *Usage:* Normalize to a league average (e.g., 1.0).
    *   *Code:*
        ```python
        def calculate_air_density(temp_f, elevation_ft, humidity_pct):
            # Convert to metric for standard formula or use approximation
            # ...
            return density_value
        ```

2.  **`park_adjusted_ops`**
    *   *Concept:* Normalize a player's OPS based on the parks they played in.
    *   *Formula:* $OPS_{adj} = \frac{OPS}{ParkFactor_{run}}$ (Simplified) or component-based adjustment.

3.  **`wind_adjusted_hr_prob`**
    *   *Concept:* Adjust the probability of a fly ball becoming a HR based on wind vector.
    *   *Logic:*
        *   Map wind direction (e.g., "R to L", "Out to CF") to stadium azimuth.
        *   Calculate tailwind/headwind component.
        *   Apply coefficient to expected distance.

4.  **`stadium_elevation_category`**
    *   *Categorical:* High (>2000ft), Medium, Low (Sea Level).
    *   *Usage:* Interaction term for curveball movement (breaking balls break less at high altitude due to less drag/Magnus effect).

5.  **`roof_status`**
    *   *Binary/Categorical:* Open, Closed, Retractable-Open, Retractable-Closed.
    *   *Source:* Often available in game metadata or scraped. Closed roofs negate wind/temp effects (controlled environment).
