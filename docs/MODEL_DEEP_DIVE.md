# Model Deep Dive: MLB Projection Systems

## 1. PECOTA (Baseball Prospectus)
**Creator:** Nate Silver (originally), now managed by the Baseball Prospectus team.

### Core Algorithm: Nearest Neighbor Analysis & Probabilistic Outcomes
PECOTA (Player Empirical Comparison and Optimization Test Algorithm) is distinct because it relies heavily on **comparables**.
*   **Nearest Neighbor Analysis:** Instead of just looking at a player's past performance and aging him, PECOTA scans a massive database of historical player seasons (major and minor leagues) to find the most similar players ("comps") based on performance, usage, and physical attributes (height, weight, handedness).
*   **Probabilistic Forecasts:** It doesn't just give one number. It uses the distribution of outcomes from those comparable players to generate a range of possibilities (e.g., 10th percentile, 50th percentile, 90th percentile outcomes). This allows it to identify "breakout" candidates or "bust" risks better than linear models.

### Key Inputs
*   **Minor League Data:** heavily weighted, including translated stats.
*   **Physical Attributes:** Height, weight, and handedness are crucial for finding comps.
*   **Batted Ball Data:** Uses peripheral statistics (strikeout rates, walk rates, groundball rates) which are more predictive than ERA or Batting Average.

### Best Use Case
*   **Prospect Analysis & Dynasty Leagues:** Its ability to find comps for minor leaguers makes it excellent for predicting how prospects will translate to the majors.
*   **Risk Assessment:** The percentile projections help fantasy managers understand the floor and ceiling of a player, useful for draft strategy (e.g., "high risk, high reward").

---

## 2. ZiPS (Szymborski Projection System)
**Creator:** Dan Szymborski (FanGraphs/ESPN)

### Core Algorithm: Weighted Historical Data & Proprietary Aging Curves
ZiPS is a sophisticated regression-based system that excels at handling historical data.
*   **Weighted History:** It uses a multi-year baseline (usually 3-4 years) but weights recent performance much more heavily. The weighting isn't static; it adjusts based on the specific stat and the player's consistency.
*   **Player-Specific Aging:** Unlike systems that apply a generic "aging curve" to everyone, ZiPS constructs aging curves based on the player's type. A speedy centerfielder ages differently than a slugging first baseman, and ZiPS accounts for this.

### Key Inputs
*   **Multi-Year Statistics:** Deep historical context is key.
*   **Player Type/Archetype:** Used to determine the aging curve.
*   **Velocity/Pitch Mix:** (In recent iterations) incorporates changes in velocity to adjust aging expectations for pitchers.

### Best Use Case
*   **Long-Term Contracts & Career Projections:** Because of its nuanced aging curves, it's often used to project how a player will perform over the life of a 5-7 year contract.
*   **General Baseline:** It is widely considered one of the most accurate "standard" projection systems for established major leaguers.

---

## 3. Steamer
**Creators:** Jared Cross, Dash Davidson, Peter Rosenbloom

### Core Algorithm: Regression to the Mean & Component-Based Projections
Steamer is often cited as one of the most accurate systems for in-season and next-season projections.
*   **Regression to the Mean:** It aggressively regresses performance to the mean, especially for smaller sample sizes. It assumes that extreme performances (good or bad) are likely to normalize.
*   **Component-Based:** It projects individual components (K%, BB%, HR/FB, BABIP) separately rather than projecting the topline stats (ERA, AVG) directly. It then combines these components to build the final stat line. This is more stable because components stabilize faster than results.

### Key Inputs
*   **Minor League Data:** Steamer was one of the first to rigorously translate and integrate minor league stats for all players, not just rookies.
*   **Pitch Tracking Data:** Incorporates pitch type and velocity data to refine pitcher projections.

### Best Use Case
*   **Daily Fantasy Sports (DFS) & Streaming:** Its responsiveness and accuracy for the immediate future make it a favorite for daily decision-making.
*   **In-Season Updates:** Steamer updates daily during the season, making it excellent for tracking changing values.

---

## 4. THE BAT / THE BAT X
**Creator:** Derek Carty

### Core Algorithm: Context-Dependent & Statcast Integration
THE BAT (and its advanced sibling THE BAT X) focuses on the "context" surrounding a player and, for THE BAT X, the underlying quality of contact.
*   **Contextual Adjustments:** Most systems adjust for park factors, but THE BAT goes deeper, adjusting for umpire tendencies, weather (temperature, air density), catcher framing, and defense.
*   **Statcast Integration (THE BAT X):** THE BAT X uses Statcast data (Exit Velocity, Launch Angle, Sprint Speed) rather than just results. Since Statcast metrics stabilize very quickly, THE BAT X can identify breakouts or declines much faster than systems relying on traditional stats.

### Key Inputs
*   **Statcast Data:** Barrel rates, exit velocity, launch angle (for THE BAT X).
*   **Environmental Factors:** Park dimensions, weather, altitude.
*   **Umpire & Catcher Data:** Strike zone tendencies and framing skills.

### Best Use Case
*   **Daily Fantasy Sports (DFS) & Betting:** The heavy focus on daily context (weather, umps, matchups) makes it the gold standard for daily games.
*   **Identifying Breakouts:** THE BAT X is often the first to catch a player whose underlying skills have changed (e.g., a pitcher adding 2mph or a hitter changing their launch angle) before the surface stats reflect it.

---

## 5. ATC (Average Total Cost)
**Creator:** Ariel Cohen

### Core Algorithm: "Wisdom of the Crowds" Aggregation
ATC is not a projection system in the sense of generating its own raw numbers from player data. Instead, it is an **aggregator**.
*   **Smart Aggregation:** It combines other projection systems (Steamer, ZiPS, THE BAT, etc.) but not with a simple average. It assigns weights to each system based on that system's historical accuracy for *specific statistics*.
*   **Volatility Adjustment:** It recognizes that some systems are better at projecting strikeouts, while others are better at home runs, and weights them accordingly.

### Key Inputs
*   **Other Projection Systems:** Steamer, ZiPS, THE BAT, PECOTA, etc.
*   **Historical Accuracy Weights:** A proprietary set of weights derived from back-testing how well each system predicts specific categories.

### Best Use Case
*   **Fantasy Drafts:** For a "set it and forget it" draft list, ATC is often the most accurate single source because it smooths out the idiosyncrasies and errors of individual models.
*   **Safe Floor:** It minimizes the risk of relying on a model that might be "blind" to a specific player's nuance.
