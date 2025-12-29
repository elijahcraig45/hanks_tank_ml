# Competitor Analysis & Benchmarks

This document outlines the landscape of existing MLB predictive modeling, providing resources for inspiration and benchmarks to measure `hanks_tank_ml` against.

## 1. Benchmarks: What is "Good"?

Baseball is a high-variance sport. Unlike the NBA (where the better team wins ~70% of the time), the worst MLB team wins ~40% of their games.

### The "Magic Numbers"
*   **50.0%:** Coin flip.
*   **54.0%:** **The Baseline.** Historically, the Home Team wins ~54% of games. If your model doesn't beat 54%, it's worse than just betting "Home" every time.
*   **55.0% - 57.0%:** **Profitable/Elite.** This is the "State of the Art" for public models. Sustaining 56% over a full season is incredibly difficult.
*   **> 60.0%:** **Suspicious.** Likely overfitting, data leakage (using future data), or too small a sample size.

### Metrics to Track
*   **Accuracy:** (Correct Picks / Total Games). Simple, but ignores confidence.
*   **Log Loss:** The gold standard. It penalizes being *confident and wrong*.
    *   *Goal:* Log Loss < 0.685 (approx).
*   **ROI (Return on Investment):** If you bet $100 on every game based on your model's edge, do you make money?

---

## 2. Open Source Resources & Repos

### Data Libraries
*   **[pybaseball](https://github.com/jldbc/pybaseball):** The absolute standard. Wraps Statcast, FanGraphs, and B-Ref.
    *   *Use for:* Fetching raw data, Statcast metrics, and "Guts" constants.
*   **[Lahman Database](http://www.seanlahman.com/baseball-archive/statistics/):** The definitive historical database (SQL/CSV).
    *   *Use for:* Historical seasonal data back to 1871.
*   **[Retrosheet](https://www.retrosheet.org/):** Play-by-play event files.
    *   *Use for:* Building custom "Run Expectancy" matrices or game state models.

### Modeling Inspiration
*   **[OpenWorm / OpenBaseball](https://github.com/topics/baseball-analytics):** Search GitHub topics for "baseball-analytics". Many students publish their thesis projects here.
*   **Kaggle Competitions:** Look at the "MLB Player Digital Engagement" competition (2021). While focused on fan engagement, the *feature engineering* kernels are gold mines for handling MLB time-series data.

---

## 3. Famous Systems (The "Competition")

While their code is proprietary, their *philosophies* are public and can be mimicked.

### A. PECOTA (Baseball Prospectus)
*   **Core Philosophy:** **Comparables (Doppelgangers).**
*   **Method:** "Who is the next Dustin Pedroia?" It finds historical players with similar body types, minor league stats, and age, then uses *their* career paths to predict the current player.
*   **Lesson for Us:** Cluster players by "type" (e.g., "High K, Low BB Power Hitter") rather than just using raw regression.

### B. ZiPS (Dan Szymborski / FanGraphs)
*   **Core Philosophy:** **Weighted History.**
*   **Method:** Uses 4 years of data, weighted heavily towards the most recent.
    *   *Year N (Projection) = f(Year N-1 * 50%, Year N-2 * 30%, ...)*
*   **Lesson for Us:** Exponentially decay older data. A game from 2023 matters less than a game from last week.

### C. THE BAT (Derek Carty)
*   **Core Philosophy:** **Context is King.**
*   **Method:** Heavily weights Park Factors, Weather, Umpire tendencies, and Catcher Framing.
*   **Lesson for Us:** This validates our "Park & Weather" research. A generic projection fails because it ignores that the game is at Coors Field.

---

## 4. Community Hubs

*   **[The Prediction Tracker](http://www.thepredictiontracker.com/baseball.php):** Tracks the daily performance of 50+ public computer models.
    *   *Use this:* To see how `hanks_tank_ml` compares to the field in real-time.
*   **[FanGraphs Community Research](https://community.fangraphs.com/):** User-submitted articles. Often contain cutting-edge ideas before they become mainstream.
*   **[r/Sabermetrics](https://www.reddit.com/r/Sabermetrics/):** The active developer community. Good for "How do I calculate X?" questions.

---

## 5. Summary Checklist

1.  [ ] **Beat the Baseline:** First goal is > 54% accuracy.
2.  [ ] **Track Log Loss:** Don't just count wins; measure probability quality.
3.  [ ] **Mimic the Greats:**
    *   Use **ZiPS-style** weighted history for player baselines.
    *   Use **THE BAT-style** context adjustments (Park/Weather) for daily variance.
