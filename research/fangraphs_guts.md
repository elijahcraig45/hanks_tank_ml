# FanGraphs "Guts" Breakdown

The "Guts" page on FanGraphs is the Rosetta Stone of modern sabermetrics. It contains the **yearly constants and linear weights** required to calculate advanced metrics like wOBA, FIP, and wRC+.

Because the run-scoring environment changes every year (e.g., the "Juiced Ball" era of 2019 vs. the "Dead Ball" era of 1968), the value of a Walk or a Home Run changes. The "Guts" table provides these changing values.

## 1. Why Do We Need It?

You cannot calculate **wOBA** or **FIP** correctly without these numbers.
*   **Standard Formula:** `wOBA = (wBB*BB + wHBP*HBP + w1B*1B + ... ) / PA`
*   **The Problem:** `wBB` (Weight of a Walk) is not 0.69 every year. In 2000, it might be 0.72. In 2015, it might be 0.68.
*   **The Solution:** The Guts table provides the exact `wBB` for every season.

## 2. Key Columns Explained

### A. wOBA Weights (The Run Values)
These represent the average number of runs added by each specific event in that specific season.

*   **`wBB` (Weighted Walk):** Value of a base on balls. (~0.69)
*   **`wHBP` (Weighted Hit By Pitch):** Usually slightly higher than a walk (~0.72) because it keeps the play alive differently or reflects toughness.
*   **`w1B` (Weighted Single):** (~0.88)
*   **`w2B` (Weighted Double):** (~1.25)
*   **`w3B` (Weighted Triple):** (~1.60)
*   **`wHR` (Weighted Home Run):** (~2.00 - 2.10)

### B. Scaling Constants
These are used to make the advanced math "look" like normal stats we are used to.

*   **`wOBA Scale`:**
    *   *Purpose:* wOBA is calculated using run values, which results in a number like 0.250. But baseball fans are used to OBP (where .320 is average).
    *   *Function:* The raw wOBA is multiplied by this scale (~1.15 to 1.25) to make the league average wOBA match the league average OBP.
*   **`cFIP` (FIP Constant):**
    *   *Purpose:* FIP (Fielding Independent Pitching) is calculated using K, BB, and HR.
    *   *Function:* This constant (usually ~3.10 to 3.20) is added to the result to make the league average FIP match the league average ERA.

### C. Run Environment Context
*   **`R/PA` (Runs per Plate Appearance):** The overall offensive level of the league.
*   **`R/W` (Runs per Win):** How many runs does it take to generate 1 Win? (Usually ~9-10). Used in WAR calculations.

## 3. How to Use "Guts" in Machine Learning

If you are building a model that spans multiple seasons (e.g., training on 2020-2025), you **cannot** use static weights.

### Implementation Strategy
1.  **Create a Lookup Table:**
    ```python
    guts_table = {
        2024: {'wBB': 0.693, 'wHR': 2.05, 'cFIP': 3.15},
        2025: {'wBB': 0.689, 'wHR': 2.01, 'cFIP': 3.12},
        # ...
    }
    ```
2.  **Dynamic Calculation:**
    When engineering features for a game on `2024-05-12`:
    *   Fetch the player's raw stats (BB, HR, etc.).
    *   Look up the `2024` weights.
    *   Calculate `wOBA`.

### Data Source
*   **Web:** [FanGraphs Guts Page](https://www.fangraphs.com/guts.aspx?type=cn)
*   **Python:** The `pybaseball` library has a built-in function:
    ```python
    from pybaseball import fangraphs
    guts_df = fangraphs.get_guts()
    ```

## 4. Summary
The "Guts" page is the configuration file for the MLB season. It ensures that your advanced metrics are context-aware and mathematically accurate for the specific era you are analyzing.
