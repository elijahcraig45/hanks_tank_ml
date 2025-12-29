# External Resources, Benchmarks, and Prior Art

This document serves as a guide to the existing landscape of MLB predictive modeling. It covers open-source tools, performance benchmarks to aim for, and lessons from famous proprietary systems.

## 1. Open Source Repositories & Tools

Leveraging existing tools can save hundreds of hours of scraping and data cleaning code.

### **pybaseball** (Python)
*   **Repo:** [jldbc/pybaseball](https://github.com/jldbc/pybaseball)
*   **Description:** The absolute gold standard for Python-based baseball analytics. It wraps scraping logic for:
    *   **Statcast:** Pitch-level data (velocity, spin, exit velo).
    *   **FanGraphs:** Advanced stats (wRC+, WAR, projections).
    *   **Baseball Reference:** Traditional stats and standings.
*   **Use Case:** Use this to fetch training data or daily stats instead of writing your own scrapers.

### **baseball-scraper** (Python)
*   **Repo:** [darenw/baseball-scraper](https://github.com/darenw/baseball-scraper)
*   **Description:** A robust alternative to pybaseball, often used for pulling Statcast and Baseball Reference data.

### **Retrosheet**
*   **Website:** [retrosheet.org](https://www.retrosheet.org/)
*   **Description:** The holy grail of historical play-by-play data. While not a "repo" in the traditional sense, the Chadwick Bureau tools (often used to parse Retrosheet) are open source.
*   **Use Case:** If you need to build a model based on specific game states (e.g., "probability of scoring from 1st with 1 out"), this is the source.

### **Lahman Database**
*   **Description:** A standardized database of baseball statistics going back to 1871. Available as a CSV bundle or SQL dump. Good for seasonal trends, less useful for daily game prediction.

---

## 2. Benchmarks: What is "Good"?

Predicting baseball games is notoriously difficult due to the high variance (luck) involved in the sport. Unlike the NBA, where the better team wins ~70% of the time, the best MLB teams rarely win more than 60-65% of their games over a season.

### **The Metrics**
*   **Accuracy:** The percentage of games correctly predicted.
*   **Log Loss (Cross-Entropy):** A measure of confidence. A model that predicts 51% confidence and wins is "less right" than one that predicts 90% confidence and wins. **This is the preferred metric for betting models.**

### **The Standards**
*   **Random Guessing:** 50.0%
*   **Home Team Blind Bet:** ~54.0% (Home field advantage is real but small).
*   **Vegas / Closing Line:** The betting market is extremely efficient. Beating the closing line consistently is the "Holy Grail."
*   **State of the Art (Public):** **55% - 57%**.
    *   If your model hits **55%** over a full season, you are doing very well.
    *   If your model hits **58%+**, you are likely overfitting, leaking data, or have discovered something revolutionary.
    *   **The Prediction Tracker:** A site that tracks public computer ratings. Top models often hover around 55-56% accuracy.

### **Key Takeaway**
Do not be discouraged by 54% accuracy. In baseball, a 55% win rate with proper money management (Kelly Criterion) is profitable.

---

## 3. Community Resources

### **FanGraphs Community Research**
*   **Focus:** Blogs and articles written by the community. Often details specific modeling approaches (e.g., "Stabilization points for K%", "Aging curves").
*   **Value:** Great for finding specific feature engineering ideas (e.g., "Does catcher framing impact game totals?").

### **SABR (Society for American Baseball Research)**
*   **Focus:** Academic-style research.
*   **Key Resource:** *The Sabermetric Review* and the *Guide to Sabermetric Research*.
*   **Value:** Deep dives into historical context and mathematical rigor.

### **r/Sabermetrics (Reddit)**
*   **Focus:** Q&A, methodology discussions, and sharing open-source projects.
*   **Value:** The best place to ask "How do I calculate xFIP programmatically?" or "Why is my Statcast query failing?".

### **Tom Tango's Blog (Tangotiger)**
*   **Focus:** The author of "The Book: Playing the Percentages in Baseball".
*   **Value:** Discussions on run expectancy, leverage index, and the math behind the game.

---

## 4. Famous Proprietary Systems (For Inspiration)

You cannot copy their code, but you can copy their *philosophy*.

### **PECOTA (Baseball Prospectus)**
*   **Core Concept:** **Comparables**. It looks at a player's physical stats, minor league performance, and age, then finds historical "doppelgangers" to predict their future development.
*   **Lesson:** Context matters. A 22-year-old in AA hitting .300 is different than a 28-year-old in AA hitting .300.

### **ZiPS (Dan Szymborski / FanGraphs)**
*   **Core Concept:** **Growth & Decline Curves**. It uses weighted averages of past performance (most recent years weighted heavier) and applies generic aging curves based on player type.
*   **Lesson:** The most recent data is the most important, but you need a large sample size (3-4 years) to stabilize the baseline.

### **Steamer (Projections)**
*   **Core Concept:** **Regression to the Mean**. Steamer is famous for being conservative. It assumes extreme performances (good or bad) are likely luck and regresses them heavily towards the league average.
*   **Lesson:** If a player hits .400 in April, don't predict he'll hit .400 in May. Predict he'll hit .270 (or whatever his true talent level is).

### **THE BAT (Derek Carty)**
*   **Core Concept:** **Contextual Factors**. Heavily weighs park factors, weather, umpire tendencies, and catcher framing.
*   **Lesson:** Talent isn't everything. A fly-ball pitcher in Coors Field is a different pitcher than in Petco Park.
