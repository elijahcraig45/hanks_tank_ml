# MLB Feature Engineering Plan
**Project:** Hank's Tank ML Prediction Models  
**Target:** 2026 Season Predictions  
**Dataset:** 11 years historical data (2015-2025)  
**Last Updated:** December 24, 2025

---

## Table of Contents
1. [Prediction Objectives](#prediction-objectives)
2. [Available Data Sources](#available-data-sources)
3. [Feature Engineering Strategy](#feature-engineering-strategy)
4. [Game Outcome Features](#game-outcome-features)
5. [Player Performance Features](#player-performance-features)
6. [Feature Selection & Reduction](#feature-selection--reduction)
7. [Model Architecture](#model-architecture)
8. [Implementation Roadmap](#implementation-roadmap)
9. [Research & References](#research--references)

---

## Prediction Objectives

### Model 1: Game Outcome Prediction
**Goal:** Predict winner of upcoming MLB games with probability estimates

**Inputs:**
- Team matchup (Team A vs Team B)
- Date/time of game
- Home/away designation
- Pitcher matchups (if available)
- Recent team performance

**Output:**
- Win probability for each team (0-1)
- Confidence interval
- Key factors driving prediction

**Success Metrics:**
- Accuracy (% correct predictions)
- Log loss (probability calibration)
- ROC-AUC (discriminative ability)
- Brier score (prediction quality)

### Model 2: Player Performance Prediction
**Goal:** Forecast individual player statistics for upcoming games/week/month

**Batters - Predict:**
- Batting average (next game/week)
- Home runs (next game/week)
- RBIs (next game/week)
- OPS (next game/week)
- Hit probability (binary)

**Pitchers - Predict:**
- ERA (next start)
- Strikeouts (next start)
- WHIP (next start)
- Quality start probability (binary)
- Win probability (binary)

**Success Metrics:**
- RMSE (root mean squared error)
- MAE (mean absolute error)
- R² (explained variance)
- Directional accuracy (up/down trend)

---

## Available Data Sources

### BigQuery Tables (2015-2025)

| Table | Records | Use Case |
|-------|---------|----------|
| `teams_historical` | 330 | Team metadata, league/division info |
| `team_stats_historical` | 700 | Season batting/pitching aggregates |
| `player_stats_historical` | 1,706 | Player season statistics |
| `standings_historical` | 330 | Final standings, win%, run differential |
| `games_historical` | 27,703 | Individual game results, scores |
| `rosters_historical` | 16,733 | Player-team assignments, positions |
| `transactions_historical` | 490,520 | Player movement, IL assignments |
| `statcast_pitches` | 7.8M | Pitch-level data, exit velo, launch angle |

### External Data Sources (To Consider)
- **Weather data** - Temperature, wind, humidity (impacts home runs)
- **Umpire data** - Historical strike zone tendencies
- **Park factors** - Ballpark-specific run environments
- **Travel data** - Rest days, time zone changes
- **Injury reports** - Player availability (from transactions)
- **Vegas odds** - Market consensus (wisdom of crowds)

---

## Feature Engineering Strategy

### Principles
1. **Time-based features** - Use rolling windows (7/14/30/60 days) for recent form
2. **Opponent-adjusted metrics** - Normalize stats against opponent strength
3. **Contextual features** - Home/away, day/night, grass/turf, weather
4. **Interaction features** - Pitcher vs batter handedness, team vs division
5. **Trend features** - Are stats improving or declining?
6. **Aggregate features** - Team totals, averages, standard deviations

### Feature Categories

#### 1. Team Strength Features
- Season win percentage
- Run differential (RS - RA)
- Pythagorean expectation: `RS^2 / (RS^2 + RA^2)`
- Recent form (L10, L20, L30 records)
- Strength of schedule
- Division/league standing
- Streak (current win/loss streak)

#### 2. Batting Features
- Team OPS, wOBA, wRC+
- Batting average, OBP, SLG
- Power metrics (ISO, HR/9)
- Contact rate, K%, BB%
- BABIP (luck indicator)
- Clutch performance (RISP average)
- Platoon splits (vs LHP/RHP)

#### 3. Pitching Features
- Team ERA, FIP, xFIP
- WHIP, K/9, BB/9, K/BB
- Quality start %
- Bullpen ERA, saves, blown saves
- Opponent OPS against
- Groundball/flyball rates
- HR/9 allowed

#### 4. Matchup Features
- Historical head-to-head record
- Division games (higher intensity)
- Rivalry games
- Time since last meeting
- Home team advantage
- Pitcher vs team history

#### 5. Context Features
- Home/away (home teams win ~54%)
- Day/night game
- Grass/turf
- Dome/open-air
- Temperature, wind speed/direction
- Game number in series
- Days of rest
- Travel distance

#### 6. Momentum Features
- Rolling win percentage (last N games)
- Run differential trend
- Winning/losing streak length
- Performance after wins vs losses
- Recent scoring trends (runs/game)
- Blowout wins/losses ratio

#### 7. Statcast Features (Advanced)
- Average exit velocity
- Hard hit rate (95+ mph)
- Barrel rate
- Expected batting average (xBA)
- Expected slugging (xSLG)
- Expected wOBA (xwOBA)
- Pitcher spin rates, velocity
- Chase rate, whiff rate

---

## Game Outcome Features

### Research-Backed Predictors

Based on analysis of MLB prediction literature (Baseball Prospectus, FiveThirtyThirtyEight, Fangraphs):

**Top 10 Features for Game Prediction:**

1. **Starting Pitcher Quality (Elo Rating or FIP)**
   - Historical performance adjusted for recency
   - Opponent-adjusted ERA or FIP over last 30 days
   - Source: `statcast_pitches` aggregated by pitcher

2. **Team Run Differential (Recent)**
   - Run differential over last 30 games
   - Stronger predictor than raw win/loss record
   - Source: `games_historical` rolling calculation

3. **Home Field Advantage**
   - Binary indicator (home team wins ~54% historically)
   - Team-specific home win%
   - Source: `games_historical` home/away splits

4. **Team Rest Days**
   - Days since last game (fatigue factor)
   - Bullpen availability (# of pitchers used yesterday)
   - Source: `games_historical` date calculations

5. **Team Offensive Power (wOBA or wRC+)**
   - Weighted on-base average (last 30 days)
   - wRC+ adjusts for park factors
   - Source: `team_stats_historical` + `statcast_pitches`

6. **Bullpen Strength**
   - Bullpen ERA over last 14 days
   - High-leverage relief pitcher availability
   - Source: `player_stats_historical` + `games_historical`

7. **Recent Form (Last 10 Games)**
   - Win% in last 10 games
   - Momentum indicator
   - Source: `games_historical` rolling window

8. **Platoon Advantage**
   - Batter handedness vs pitcher handedness
   - Historical team OPS vs LHP/RHP
   - Source: `team_stats_historical` splits

9. **Head-to-Head Record**
   - Historical win% against opponent
   - Psychological/matchup edge
   - Source: `games_historical` filtered

10. **Division Game Indicator**
    - Division games are more competitive
    - Higher variance in outcomes
    - Source: `teams_historical` division mapping

### Engineered Feature Examples

#### Feature: Team Offensive Power Score
```python
def calculate_offensive_power(team_id, as_of_date, window_days=30):
    """
    Composite offensive metric combining:
    - wOBA (weighted on-base average)
    - ISO (isolated power)
    - Recent run scoring rate
    """
    query = f"""
    WITH recent_games AS (
        SELECT 
            CASE 
                WHEN home_team_id = {team_id} THEN home_score
                ELSE away_score 
            END as runs_scored
        FROM `hankstank.mlb_historical_data.games_historical`
        WHERE (home_team_id = {team_id} OR away_team_id = {team_id})
          AND game_date BETWEEN DATE_SUB('{as_of_date}', INTERVAL {window_days} DAY) 
                            AND '{as_of_date}'
    ),
    team_batting AS (
        SELECT obp, slg, home_runs, at_bats
        FROM `hankstank.mlb_historical_data.team_stats_historical`
        WHERE team_id = {team_id} 
          AND stat_type = 'batting'
          AND year = EXTRACT(YEAR FROM '{as_of_date}')
    )
    SELECT 
        AVG(runs_scored) as avg_runs,
        (SELECT obp FROM team_batting) as obp,
        (SELECT slg FROM team_batting) as slg,
        (SELECT CAST(home_runs AS FLOAT64) / at_bats FROM team_batting) as hr_rate
    FROM recent_games
    """
    
    # Normalize and combine
    # offensive_power = (avg_runs * 0.3) + (obp * 50 * 0.3) + (slg * 30 * 0.2) + (hr_rate * 100 * 0.2)
    return offensive_power_score
```

#### Feature: Starting Pitcher Recent Form
```python
def pitcher_recent_performance(pitcher_id, as_of_date, games=5):
    """
    Calculate pitcher's recent performance metrics
    """
    query = f"""
    SELECT 
        AVG(CASE WHEN events IN ('single', 'double', 'triple', 'home_run') THEN 1 ELSE 0 END) as hits_per_batter,
        AVG(CASE WHEN events = 'strikeout' THEN 1 ELSE 0 END) as k_rate,
        AVG(CASE WHEN events = 'walk' THEN 1 ELSE 0 END) as bb_rate,
        AVG(release_speed) as avg_velocity
    FROM `hankstank.mlb_historical_data.statcast_pitches`
    WHERE pitcher = {pitcher_id}
      AND game_date <= '{as_of_date}'
      AND game_pk IN (
          SELECT DISTINCT game_pk 
          FROM `hankstank.mlb_historical_data.statcast_pitches`
          WHERE pitcher = {pitcher_id} 
            AND game_date <= '{as_of_date}'
          ORDER BY game_date DESC 
          LIMIT {games}
      )
    """
    # Calculate FIP-like metric
    return pitcher_quality_score
```

#### Feature: Rest Advantage
```python
def rest_advantage(team_id, game_date):
    """
    Calculate rest days advantage between teams
    """
    query = f"""
    WITH team_last_game AS (
        SELECT MAX(game_date) as last_game
        FROM `hankstank.mlb_historical_data.games_historical`
        WHERE (home_team_id = {team_id} OR away_team_id = {team_id})
          AND game_date < '{game_date}'
    )
    SELECT DATE_DIFF('{game_date}', last_game, DAY) as rest_days
    FROM team_last_game
    """
    # Compare home team rest vs away team rest
    return rest_differential
```

---

## Player Performance Features

### Batter Performance Features

**Top 10 Features for Batting Prediction:**

1. **Rolling Batting Average (Last 30 Days)**
   - Recent form is strongest predictor
   - Weight recent games more heavily
   - Source: `statcast_pitches` aggregated

2. **Career Stats vs Pitcher Handedness**
   - BA, OPS vs LHP/RHP
   - Platoon advantage
   - Source: `statcast_pitches` filtered by p_throws

3. **Opponent Pitcher Quality**
   - Pitcher's ERA, FIP, xFIP
   - K/9, BB/9 rates
   - Source: `player_stats_historical`

4. **Ballpark Factor**
   - Hitter-friendly vs pitcher-friendly parks
   - Team home ballpark effects
   - Source: External park factors + `games_historical`

5. **Recent Exit Velocity & Hard Hit %**
   - Statcast quality of contact metrics
   - Leading indicator of performance
   - Source: `statcast_pitches` launch_speed

6. **BABIP Luck Factor**
   - Batting average on balls in play
   - High BABIP suggests regression coming
   - Source: Calculated from `statcast_pitches`

7. **Lineup Position**
   - Leadoff/2-hole get more ABs
   - Cleanup hitters get more RBI opportunities
   - Source: `rosters_historical` + game logs

8. **Days Rest**
   - Performance after day off vs 10-game stretch
   - Fatigue indicator
   - Source: `games_historical` participation

9. **Career Month/Weather Performance**
   - Some players perform better in hot weather
   - Month-specific trends (April slumps)
   - Source: `statcast_pitches` by game_date month

10. **Injury/Transaction Status**
    - Recently returned from IL
    - New to team (trade/signing)
    - Source: `transactions_historical`

### Pitcher Performance Features

**Top 10 Features for Pitching Prediction:**

1. **Rolling ERA/FIP (Last 5 Starts)**
   - Recent performance strongly predictive
   - FIP better than ERA (removes defense)
   - Source: `statcast_pitches` calculated

2. **Opponent Team Offensive Power**
   - Team wOBA, wRC+, OPS
   - Facing strong offense increases risk
   - Source: `team_stats_historical`

3. **Days Rest**
   - Pitchers on normal rest (4-5 days) perform best
   - Extra rest or short rest affects performance
   - Source: `games_historical` appearances

4. **Pitch Velocity Trends**
   - Declining velocity = fatigue or injury
   - Source: `statcast_pitches` release_speed rolling avg

5. **Pitch Mix Effectiveness**
   - Whiff rate, chase rate by pitch type
   - Arsenal diversity
   - Source: `statcast_pitches` aggregated

6. **Home vs Away Splits**
   - Some pitchers much better at home
   - Source: `games_historical` + `statcast_pitches`

7. **Bullpen Support**
   - Team bullpen ERA
   - Affects win probability
   - Source: `player_stats_historical` relievers

8. **Historical vs Opponent**
   - Career stats vs specific team
   - Some teams have pitcher's number
   - Source: `statcast_pitches` filtered

9. **Ballpark Factor**
   - Pitching in Coors Field vs Petco Park
   - Affects HR, runs allowed
   - Source: External + `games_historical`

10. **Season Workload**
    - Innings pitched so far
    - Fatigue accumulation
    - Source: `statcast_pitches` total innings

### Engineered Feature Examples

#### Feature: Batter Hot/Cold Streak
```python
def batter_streak_score(batter_id, as_of_date, days=14):
    """
    Identify if batter is hot or cold
    Uses exponential weighting on recent performance
    """
    query = f"""
    WITH recent_abs AS (
        SELECT 
            game_date,
            CASE WHEN events IN ('single', 'double', 'triple', 'home_run') THEN 1 ELSE 0 END as hit,
            launch_speed,
            estimated_woba_using_speedangle as xwoba
        FROM `hankstank.mlb_historical_data.statcast_pitches`
        WHERE batter = {batter_id}
          AND game_date BETWEEN DATE_SUB('{as_of_date}', INTERVAL {days} DAY) 
                            AND '{as_of_date}'
          AND description IN ('hit_into_play', 'hit_into_play_score')
        ORDER BY game_date DESC
    )
    SELECT 
        -- Exponentially weighted batting average
        SUM(hit * EXP((UNIX_MILLIS(game_date) - UNIX_MILLIS('{as_of_date}')) / (86400000.0 * 7))) / 
        SUM(EXP((UNIX_MILLIS(game_date) - UNIX_MILLIS('{as_of_date}')) / (86400000.0 * 7))) as weighted_avg,
        
        -- Quality of contact trend
        AVG(launch_speed) as avg_exit_velo,
        
        -- Expected performance
        AVG(xwoba) as recent_xwoba
    FROM recent_abs
    """
    
    # Hot: weighted_avg > season_avg + 0.5 * std_dev
    # Cold: weighted_avg < season_avg - 0.5 * std_dev
    return streak_indicator
```

#### Feature: Pitcher Fatigue Index
```python
def pitcher_fatigue_index(pitcher_id, as_of_date):
    """
    Composite fatigue metric
    """
    query = f"""
    WITH recent_starts AS (
        SELECT 
            game_date,
            game_pk,
            COUNT(DISTINCT CONCAT(inning, '-', at_bat_number)) as batters_faced,
            COUNT(*) as pitches
        FROM `hankstank.mlb_historical_data.statcast_pitches`
        WHERE pitcher = {pitcher_id}
          AND game_date <= '{as_of_date}'
          AND game_date >= DATE_SUB('{as_of_date}', INTERVAL 30 DAY)
        GROUP BY game_date, game_pk
    ),
    velocity_trend AS (
        SELECT 
            game_date,
            AVG(release_speed) as avg_velo
        FROM `hankstank.mlb_historical_data.statcast_pitches`
        WHERE pitcher = {pitcher_id}
          AND pitch_type = 'FF'  -- Fastball
          AND game_date <= '{as_of_date}'
          AND game_date >= DATE_SUB('{as_of_date}', INTERVAL 60 DAY)
        GROUP BY game_date
    )
    SELECT 
        SUM(pitches) as total_pitches_30d,
        AVG(batters_faced) as avg_batters_per_game,
        -- Velocity decline indicator
        (SELECT AVG(avg_velo) FROM velocity_trend WHERE game_date >= DATE_SUB('{as_of_date}', INTERVAL 7 DAY)) -
        (SELECT AVG(avg_velo) FROM velocity_trend WHERE game_date < DATE_SUB('{as_of_date}', INTERVAL 7 DAY)) as velo_change
    FROM recent_starts
    """
    
    # High fatigue = many pitches + velocity drop
    return fatigue_score
```

---

## Feature Selection & Reduction

### Initial Feature Set Size
After engineering all features:
- **Game Prediction:** ~50-80 features
- **Player Prediction:** ~60-100 features

### Selection Methods

#### 1. Correlation Analysis
Remove highly correlated features (r > 0.85):
```python
correlation_matrix = features.corr()
# Drop one of each highly correlated pair
to_drop = set()
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.85:
            colname = correlation_matrix.columns[i]
            to_drop.add(colname)
```

#### 2. Feature Importance (Random Forest)
```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

importances = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

# Keep top 20 features
top_features = importances.head(20)['feature'].tolist()
```

#### 3. Recursive Feature Elimination (RFE)
```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

estimator = LogisticRegression()
selector = RFE(estimator, n_features_to_select=10, step=1)
selector = selector.fit(X_train, y_train)

selected_features = X_train.columns[selector.support_]
```

#### 4. L1 Regularization (Lasso)
```python
from sklearn.linear_model import LassoCV

lasso = LassoCV(cv=5, random_state=42)
lasso.fit(X_train, y_train)

# Features with non-zero coefficients
important_features = X_train.columns[lasso.coef_ != 0]
```

### Recommended Top 2 Features

Based on research and common ML practices for baseball prediction:

#### **Game Outcome Model - Top 2 Features:**

1. **Team Quality Differential (Composite)**
   - Combine: Run differential (last 30 games) + Starting pitcher ERA differential
   - Captures both offensive/defensive strength AND pitching matchup
   - Formula: `(TeamA_RunDiff30 - TeamB_RunDiff30) + (TeamB_SP_ERA - TeamA_SP_ERA) * 10`

2. **Home Field Advantage Adjusted for Rest**
   - Combine: Home team indicator + Rest days differential
   - Formula: `1 if home else -1) * (1 + 0.1 * rest_days_diff)`

**Why these 2?**
- Cover the most variance in outcomes (team strength + context)
- Minimal correlation between them
- Easy to compute from existing data
- Interpretable for users

#### **Player Performance Model - Top 2 Features:**

1. **Recent Form (Rolling 14-Day Performance)**
   - Batters: Weighted batting average (last 14 days)
   - Pitchers: FIP (last 3 starts)
   - Captures current player state

2. **Matchup Quality**
   - Batters: Opponent pitcher ERA
   - Pitchers: Opponent team wRC+
   - Captures difficulty of task

**Why these 2?**
- Recency bias is strong in baseball
- Opponent strength is critical
- Together predict 60-70% of performance variance

---

## Model Architecture

### Game Outcome Model

#### Option 1: Logistic Regression (Baseline)
```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    penalty='l2',
    C=1.0,
    class_weight='balanced'
)
```

**Pros:**
- Fast training/prediction
- Interpretable coefficients
- Probability calibration
- Works well with 2 features

**Cons:**
- Assumes linear relationships
- Can't capture complex interactions

#### Option 2: Gradient Boosted Trees (XGBoost)
```python
import xgboost as xgb

model = xgb.XGBClassifier(
    max_depth=4,
    learning_rate=0.1,
    n_estimators=100,
    objective='binary:logistic',
    eval_metric='logloss'
)
```

**Pros:**
- Handles non-linear relationships
- Automatic feature interactions
- State-of-art performance
- Feature importance

**Cons:**
- Slower training
- Requires more data
- Less interpretable

#### Option 3: Neural Network (Simple)
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential([
    Dense(16, activation='relu', input_dim=2),
    Dropout(0.3),
    Dense(8, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', 'AUC']
)
```

**Pros:**
- Can learn complex patterns
- Flexible architecture
- Can add layers for more features

**Cons:**
- Needs more training data
- Hyperparameter tuning
- Black box

### Player Performance Model

#### Regression for Continuous Stats
```python
from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    loss='huber'  # Robust to outliers
)
```

#### Classification for Binary Outcomes
```python
# Predict if player will get a hit
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=6,
    class_weight='balanced'
)
```

### Ensemble Approach (Recommended)
```python
from sklearn.ensemble import VotingClassifier

# Combine multiple models
ensemble = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression()),
        ('rf', RandomForestClassifier()),
        ('xgb', xgb.XGBClassifier())
    ],
    voting='soft'  # Use probability averaging
)
```

---

## Advanced Model Architectures

### Deep Learning Approaches

#### 1. LSTM (Long Short-Term Memory) for Sequential Game Data
```python
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization

class GameSequenceModel(tf.keras.Model):
    def __init__(self, sequence_length=10, n_features=50):
        super().__init__()
        
        # LSTM layers to capture temporal dependencies
        self.lstm1 = LSTM(128, return_sequences=True, input_shape=(sequence_length, n_features))
        self.dropout1 = Dropout(0.3)
        self.batch_norm1 = BatchNormalization()
        
        self.lstm2 = LSTM(64, return_sequences=True)
        self.dropout2 = Dropout(0.3)
        self.batch_norm2 = BatchNormalization()
        
        self.lstm3 = LSTM(32)
        self.dropout3 = Dropout(0.2)
        
        # Dense layers for final prediction
        self.dense1 = Dense(16, activation='relu')
        self.dropout4 = Dropout(0.2)
        self.output_layer = Dense(1, activation='sigmoid')
    
    def call(self, inputs, training=False):
        x = self.lstm1(inputs)
        x = self.dropout1(x, training=training)
        x = self.batch_norm1(x, training=training)
        
        x = self.lstm2(x)
        x = self.dropout2(x, training=training)
        x = self.batch_norm2(x, training=training)
        
        x = self.lstm3(x)
        x = self.dropout3(x, training=training)
        
        x = self.dense1(x)
        x = self.dropout4(x, training=training)
        
        return self.output_layer(x)

# Train with sequence of last 10 games for each team
model = GameSequenceModel(sequence_length=10, n_features=50)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', 'AUC', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)
```

**Advantages:**
- Captures momentum and streaks automatically
- Learns temporal patterns (hot/cold streaks)
- No need to manually engineer rolling window features
- Can learn long-term dependencies

**Input Shape:**
- (batch_size, 10 games, 50 features per game)
- Each team gets sequence of their last 10 games
- Features include: score differential, hits, errors, batting stats, pitching stats

#### 2. Attention-Based Transformer for Game Prediction
```python
class GameTransformer(tf.keras.Model):
    """Transformer that learns which recent games matter most"""
    
    def __init__(self, d_model=128, num_heads=8, num_layers=4):
        super().__init__()
        
        # Multi-head attention to weigh game importance
        self.attention_layers = [
            tf.keras.layers.MultiHeadAttention(
                num_heads=num_heads, 
                key_dim=d_model // num_heads
            ) for _ in range(num_layers)
        ]
        
        # Feed-forward networks
        self.ffn_layers = [
            tf.keras.Sequential([
                Dense(512, activation='relu'),
                Dropout(0.1),
                Dense(d_model)
            ]) for _ in range(num_layers)
        ]
        
        # Layer normalization
        self.layer_norms = [
            tf.keras.layers.LayerNormalization() 
            for _ in range(num_layers * 2)
        ]
        
        # Final classification head
        self.global_pool = tf.keras.layers.GlobalAveragePooling1D()
        self.classifier = tf.keras.Sequential([
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
    
    def call(self, inputs, training=False):
        x = inputs
        
        # Stack of attention + FFN layers
        for i, (attn, ffn) in enumerate(zip(self.attention_layers, self.ffn_layers)):
            # Multi-head attention
            attn_output = attn(x, x, training=training)
            x = self.layer_norms[i*2](x + attn_output)
            
            # Feed-forward network
            ffn_output = ffn(x, training=training)
            x = self.layer_norms[i*2 + 1](x + ffn_output)
        
        # Pool and classify
        pooled = self.global_pool(x)
        return self.classifier(pooled, training=training)

# Usage
model = GameTransformer(d_model=128, num_heads=8, num_layers=4)
```

**Why Transformers?**
- Learns which games to focus on (recent loss vs old win)
- Attention weights show model reasoning
- Better than LSTM for long sequences
- State-of-the-art in sequence modeling

#### 3. Graph Neural Network for League Structure
```python
import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.nn import GCNConv, GATConv

class LeagueGNN(nn.Module):
    """Model teams as nodes in a graph, edges = games played"""
    
    def __init__(self, num_teams=30, feature_dim=50, hidden_dim=128):
        super().__init__()
        
        # Team embedding layer
        self.team_embedding = nn.Embedding(num_teams, feature_dim)
        
        # Graph attention layers (learn team relationships)
        self.gat1 = GATConv(feature_dim, hidden_dim, heads=8, dropout=0.3)
        self.gat2 = GATConv(hidden_dim * 8, hidden_dim, heads=4, dropout=0.3)
        self.gat3 = GATConv(hidden_dim * 4, hidden_dim, heads=1, dropout=0.2)
        
        # Matchup prediction head
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),  # Concat home + away embeddings
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, team_ids, edge_index, edge_attr, matchup_pairs):
        # Get team embeddings
        x = self.team_embedding(team_ids)
        
        # Propagate through graph (teams influence each other)
        x = self.gat1(x, edge_index, edge_attr)
        x = torch.relu(x)
        x = self.gat2(x, edge_index, edge_attr)
        x = torch.relu(x)
        x = self.gat3(x, edge_index, edge_attr)
        
        # For each matchup, concat home and away embeddings
        home_idx, away_idx = matchup_pairs
        home_emb = x[home_idx]
        away_emb = x[away_idx]
        matchup_emb = torch.cat([home_emb, away_emb], dim=-1)
        
        # Predict outcome
        return self.predictor(matchup_emb)

# Graph structure: Teams are nodes, edges weighted by games played
# Node features: Recent stats, roster strength, injuries
# Edge features: Head-to-head record, division rivalry
```

**Graph Benefits:**
- Models division/league structure explicitly
- Captures rivalries and matchup history
- Teams learn from opponents' performance
- Accounts for schedule strength dynamically

### Advanced Feature Engineering Techniques

#### 1. Exponentially Weighted Moving Averages (EWMA)
```python
def calculate_ewma_features(team_id, as_of_date, alpha=0.3):
    """
    Weight recent games exponentially higher
    alpha = decay factor (0.3 means 30% weight to most recent)
    """
    query = f"""
    WITH game_sequence AS (
        SELECT 
            game_date,
            CASE WHEN home_team_id = {team_id} THEN home_score ELSE away_score END as runs_scored,
            CASE WHEN home_team_id = {team_id} THEN away_score ELSE home_score END as runs_allowed,
            ROW_NUMBER() OVER (ORDER BY game_date DESC) as recency_rank
        FROM `hankstank.mlb_historical_data.games_historical`
        WHERE (home_team_id = {team_id} OR away_team_id = {team_id})
          AND game_date <= '{as_of_date}'
        ORDER BY game_date DESC
        LIMIT 30
    )
    SELECT 
        SUM(runs_scored * POW({alpha}, recency_rank - 1)) / SUM(POW({alpha}, recency_rank - 1)) as ewma_runs_scored,
        SUM(runs_allowed * POW({alpha}, recency_rank - 1)) / SUM(POW({alpha}, recency_rank - 1)) as ewma_runs_allowed
    FROM game_sequence
    """
    return ewma_stats

# Creates smooth, responsive features that heavily weight recent performance
```

#### 2. Statcast Embeddings (Advanced)
```python
def create_statcast_embeddings(games_pk_list, embedding_dim=128):
    """
    Use autoencoder to compress pitch-level data into dense embeddings
    Each game becomes a 128-dim vector capturing pitch patterns
    """
    
    # Extract all pitches from games
    query = f"""
    SELECT 
        game_pk,
        pitch_type,
        release_speed,
        release_spin_rate,
        pfx_x,
        pfx_z,
        plate_x,
        plate_z,
        launch_speed,
        launch_angle,
        events
    FROM `hankstank.mlb_historical_data.statcast_pitches`
    WHERE game_pk IN UNNEST({games_pk_list})
    """
    
    # Train autoencoder to compress pitch sequences
    encoder = tf.keras.Sequential([
        Dense(512, activation='relu'),
        Dense(256, activation='relu'),
        Dense(embedding_dim, activation='relu')  # Bottleneck
    ])
    
    decoder = tf.keras.Sequential([
        Dense(256, activation='relu'),
        Dense(512, activation='relu'),
        Dense(original_dim, activation='linear')
    ])
    
    # Each game's pitches compressed to 128 numbers
    # These capture: pitch mix, velocity, movement, outcomes
    return game_embeddings
```

#### 3. Contextual Features with Interactions
```python
class ContextualFeatureEngine:
    """Generate interaction features that capture non-linear relationships"""
    
    def engineer_interactions(self, df):
        # Power features: Strength x Strength
        df['offensive_power_product'] = df['home_wrc_plus'] * df['away_wrc_plus']
        df['pitching_quality_product'] = df['home_pitcher_fip'] * df['away_pitcher_fip']
        
        # Mismatch features: Strong offense vs weak pitching
        df['home_offense_vs_away_pitching'] = df['home_wrc_plus'] / (df['away_team_era'] + 1)
        df['away_offense_vs_home_pitching'] = df['away_wrc_plus'] / (df['home_team_era'] + 1)
        
        # Momentum interactions
        df['momentum_clash'] = df['home_win_streak'] - df['away_win_streak']
        df['rest_advantage_weighted'] = (
            (df['home_rest_days'] - df['away_rest_days']) * 
            df['home_games_last_10']  # More important when playing many games
        )
        
        # Division rivalry boost
        df['division_game_strength'] = (
            df['is_division_game'] * 
            (abs(df['home_division_rank'] - df['away_division_rank']) / 5.0)
        )
        
        # Weather interactions (temperature affects home runs)
        df['weather_power_interaction'] = (
            df['temperature'] * df['combined_team_hr_rate'] * df['ballpark_hr_factor']
        )
        
        return df
```

#### 4. Player-Level Aggregation
```python
def aggregate_roster_strength(team_id, as_of_date):
    """
    Bottom-up approach: Aggregate individual player projections
    More accurate than team-level stats
    """
    query = f"""
    WITH active_roster AS (
        SELECT DISTINCT r.player_id, r.position_code
        FROM `hankstank.mlb_historical_data.rosters_historical` r
        WHERE r.team_id = {team_id}
          AND r.year = EXTRACT(YEAR FROM '{as_of_date}')
          AND r.player_id NOT IN (
              -- Exclude players on IL
              SELECT person_id 
              FROM `hankstank.mlb_historical_data.transactions_historical`
              WHERE to_team_id = {team_id}
                AND type_desc LIKE '%Injured List%'
                AND date <= '{as_of_date}'
          )
    ),
    player_projections AS (
        SELECT 
            p.player_id,
            r.position_code,
            -- Recent performance (last 30 days from Statcast)
            AVG(CASE WHEN s.events IN ('single','double','triple','home_run') THEN 1 ELSE 0 END) as recent_ba,
            AVG(s.estimated_woba_using_speedangle) as recent_xwoba,
            AVG(s.launch_speed) as avg_exit_velo,
            COUNT(*) as plate_appearances
        FROM active_roster r
        JOIN `hankstank.mlb_historical_data.statcast_pitches` s ON r.player_id = s.batter
        WHERE s.game_date BETWEEN DATE_SUB('{as_of_date}', INTERVAL 30 DAY) AND '{as_of_date}'
        GROUP BY p.player_id, r.position_code
    )
    SELECT 
        -- Weighted by playing time (more PAs = more weight)
        SUM(recent_xwoba * plate_appearances) / SUM(plate_appearances) as team_xwoba,
        AVG(avg_exit_velo) as team_avg_exit_velo,
        COUNT(DISTINCT player_id) as active_players,
        
        -- Position-specific strength
        AVG(CASE WHEN position_code = 'P' THEN recent_xwoba ELSE NULL END) as pitcher_quality,
        AVG(CASE WHEN position_code IN ('C','1B','2B','3B','SS') THEN recent_xwoba ELSE NULL END) as infield_strength,
        AVG(CASE WHEN position_code IN ('LF','CF','RF') THEN recent_xwoba ELSE NULL END) as outfield_strength
    FROM player_projections
    """
    return roster_strength_features
```

#### 5. Opponent-Adjusted Performance Metrics
```python
def opponent_adjusted_stats(team_id, stat_window_days=30):
    """
    Adjust stats based on quality of opponents faced
    Beating good teams counts more than beating bad teams
    """
    query = f"""
    WITH team_games AS (
        SELECT 
            g.game_date,
            CASE WHEN g.home_team_id = {team_id} THEN g.home_score ELSE g.away_score END as runs_scored,
            CASE WHEN g.home_team_id = {team_id} THEN g.away_team_id ELSE g.home_team_id END as opponent_id,
            CASE WHEN g.home_team_id = {team_id} THEN g.away_score ELSE g.home_score END as runs_allowed
        FROM `hankstank.mlb_historical_data.games_historical` g
        WHERE (g.home_team_id = {team_id} OR g.away_team_id = {team_id})
          AND g.game_date >= DATE_SUB(CURRENT_DATE(), INTERVAL {stat_window_days} DAY)
    ),
    opponent_strength AS (
        SELECT 
            team_id as opponent_id,
            AVG(CASE WHEN home_score > away_score THEN 1 ELSE 0 END) as opponent_win_pct,
            AVG(home_score - away_score) as opponent_run_diff
        FROM `hankstank.mlb_historical_data.games_historical`
        WHERE game_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 90 DAY)
        GROUP BY team_id
    )
    SELECT 
        -- Raw stats
        AVG(t.runs_scored) as avg_runs_scored,
        
        -- Opponent-adjusted (weight by opponent strength)
        SUM(t.runs_scored * (1 + o.opponent_win_pct)) / SUM(1 + o.opponent_win_pct) as adj_runs_scored,
        
        -- Quality of victories (did we beat good teams?)
        AVG(CASE WHEN t.runs_scored > t.runs_allowed THEN o.opponent_win_pct ELSE NULL END) as avg_beaten_team_quality
    FROM team_games t
    JOIN opponent_strength o ON t.opponent_id = o.opponent_id
    """
    return adjusted_metrics
```

### Ensemble Meta-Learning

#### Stacking Multiple Model Types
```python
class StackedEnsemble:
    """Combine predictions from multiple model types"""
    
    def __init__(self):
        # Level 0: Base models (diverse types)
        self.base_models = [
            ('xgboost', xgb.XGBClassifier(n_estimators=200, max_depth=6)),
            ('lightgbm', lgb.LGBMClassifier(n_estimators=200, num_leaves=31)),
            ('random_forest', RandomForestClassifier(n_estimators=200)),
            ('logistic', LogisticRegression(C=1.0)),
            ('neural_net', self._build_neural_network()),
            ('lstm', self._build_lstm_model())
        ]
        
        # Level 1: Meta-learner (learns how to combine base predictions)
        self.meta_model = xgb.XGBClassifier(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1
        )
    
    def fit(self, X_train, y_train, X_val, y_val):
        # Train base models
        base_predictions_train = []
        base_predictions_val = []
        
        for name, model in self.base_models:
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            
            # Get out-of-fold predictions
            pred_train = model.predict_proba(X_train)[:, 1]
            pred_val = model.predict_proba(X_val)[:, 1]
            
            base_predictions_train.append(pred_train)
            base_predictions_val.append(pred_val)
        
        # Stack predictions as features for meta-model
        meta_features_train = np.column_stack(base_predictions_train)
        meta_features_val = np.column_stack(base_predictions_val)
        
        # Train meta-learner to combine base predictions
        self.meta_model.fit(meta_features_train, y_train)
        
        # Evaluate
        meta_pred_val = self.meta_model.predict_proba(meta_features_val)[:, 1]
        print(f"Ensemble validation AUC: {roc_auc_score(y_val, meta_pred_val)}")
        
        return self
    
    def predict_proba(self, X):
        # Get base model predictions
        base_predictions = []
        for name, model in self.base_models:
            pred = model.predict_proba(X)[:, 1]
            base_predictions.append(pred)
        
        # Meta-model combines them
        meta_features = np.column_stack(base_predictions)
        return self.meta_model.predict_proba(meta_features)
```

#### Adaptive Weighting (Online Learning)
```python
class AdaptiveEnsemble:
    """Dynamically adjust model weights based on recent performance"""
    
    def __init__(self, models, window_size=100):
        self.models = models
        self.weights = np.ones(len(models)) / len(models)  # Start equal
        self.recent_performance = {i: [] for i in range(len(models))}
        self.window_size = window_size
    
    def predict_and_update(self, X, y_true):
        predictions = []
        
        # Get predictions from each model
        for i, model in enumerate(self.models):
            pred = model.predict_proba(X)[:, 1]
            predictions.append(pred)
            
            # Track performance (log loss)
            loss = log_loss([y_true], [pred])
            self.recent_performance[i].append(loss)
            
            # Keep only recent window
            if len(self.recent_performance[i]) > self.window_size:
                self.recent_performance[i].pop(0)
        
        # Update weights (inverse of recent loss)
        avg_losses = [np.mean(self.recent_performance[i]) for i in range(len(self.models))]
        
        # Softmax to convert losses to weights (lower loss = higher weight)
        inv_losses = [1.0 / (loss + 1e-6) for loss in avg_losses]
        self.weights = np.array(inv_losses) / np.sum(inv_losses)
        
        # Weighted average prediction
        ensemble_pred = np.average(predictions, axis=0, weights=self.weights)
        
        return ensemble_pred, self.weights
```

### Advanced Evaluation Metrics

#### Beyond Accuracy: Calibration & Uncertainty
```python
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

def comprehensive_evaluation(y_true, y_pred_proba, y_pred_class):
    """
    Multi-faceted evaluation for probabilistic predictions
    """
    results = {}
    
    # Standard metrics
    results['accuracy'] = accuracy_score(y_true, y_pred_class)
    results['precision'] = precision_score(y_true, y_pred_class)
    results['recall'] = recall_score(y_true, y_pred_class)
    results['f1'] = f1_score(y_true, y_pred_class)
    results['auc'] = roc_auc_score(y_true, y_pred_proba)
    
    # Probabilistic metrics
    results['log_loss'] = log_loss(y_true, y_pred_proba)
    results['brier_score'] = brier_score_loss(y_true, y_pred_proba)
    
    # Calibration (are 70% predictions actually 70% accurate?)
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_pred_proba, n_bins=10
    )
    results['calibration_error'] = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
    
    # Expected Calibration Error (ECE)
    ece = 0
    for frac_pos, mean_pred in zip(fraction_of_positives, mean_predicted_value):
        ece += abs(frac_pos - mean_pred)
    results['ece'] = ece / len(fraction_of_positives)
    
    # Confidence intervals
    bootstrap_aucs = []
    for _ in range(1000):
        indices = np.random.choice(len(y_true), len(y_true), replace=True)
        bootstrap_aucs.append(roc_auc_score(y_true[indices], y_pred_proba[indices]))
    
    results['auc_95ci_lower'] = np.percentile(bootstrap_aucs, 2.5)
    results['auc_95ci_upper'] = np.percentile(bootstrap_aucs, 97.5)
    
    return results

def plot_calibration_curve(y_true, y_pred_proba, n_bins=10):
    """Visualize model calibration"""
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_pred_proba, n_bins=n_bins
    )
    
    plt.figure(figsize=(10, 6))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
    plt.plot(mean_predicted_value, fraction_of_positives, 'o-', label='Model')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title('Calibration Plot')
    plt.legend()
    plt.grid(True)
    plt.show()
```

#### Profit-Based Evaluation (For Betting)
```python
def evaluate_betting_performance(y_true, y_pred_proba, betting_odds):
    """
    Evaluate model in terms of betting profitability
    Kelly Criterion for optimal bet sizing
    """
    
    # Convert American odds to decimal
    def american_to_decimal(american_odds):
        if american_odds > 0:
            return (american_odds / 100) + 1
        else:
            return (100 / abs(american_odds)) + 1
    
    decimal_odds = [american_to_decimal(x) for x in betting_odds]
    
    # Kelly Criterion: f* = (bp - q) / b
    # b = odds - 1, p = predicted probability, q = 1 - p
    
    bankroll = 1000  # Start with $1000
    bet_history = []
    
    for true, pred, odds in zip(y_true, y_pred_proba, decimal_odds):
        b = odds - 1
        p = pred
        q = 1 - p
        
        # Kelly fraction
        kelly_fraction = (b * p - q) / b
        
        # Only bet if Kelly is positive (edge detected)
        if kelly_fraction > 0:
            # Fractional Kelly (more conservative)
            bet_size = bankroll * min(kelly_fraction * 0.25, 0.05)  # Max 5% of bankroll
            
            if true == 1:  # Win
                profit = bet_size * b
            else:  # Loss
                profit = -bet_size
            
            bankroll += profit
            bet_history.append({
                'bet': bet_size,
                'profit': profit,
                'bankroll': bankroll,
                'predicted_prob': pred,
                'actual': true
            })
    
    # Calculate ROI
    total_wagered = sum([abs(b['bet']) for b in bet_history])
    total_profit = bankroll - 1000
    roi = (total_profit / total_wagered) * 100 if total_wagered > 0 else 0
    
    return {
        'final_bankroll': bankroll,
        'total_profit': total_profit,
        'roi': roi,
        'num_bets': len(bet_history),
        'win_rate': np.mean([b['actual'] for b in bet_history])
    }
```

## Implementation Roadmap

### Phase 1: Data Preparation (Week 1-2)
- [ ] Create feature engineering pipeline
- [ ] Build rolling window calculations
- [ ] Generate training dataset (2015-2024 games)
- [ ] Split into train/validation/test sets
  - Train: 2015-2022 (70%)
  - Validation: 2023 (15%)
  - Test: 2024-2025 (15%)

### Phase 2: Baseline Models (Week 3)
- [ ] Implement 2-feature logistic regression
- [ ] Calculate baseline metrics
- [ ] Analyze feature distributions
- [ ] Check for data leakage

### Phase 3: Feature Expansion (Week 4-5)
- [ ] Engineer top 20 features
- [ ] Run feature importance analysis
- [ ] Test correlation matrix
- [ ] Create feature documentation

### Phase 4: Model Selection (Week 6-7)
- [ ] Train multiple model types
- [ ] Cross-validation (5-fold)
- [ ] Hyperparameter tuning (Grid Search)
- [ ] Compare model performance

### Phase 5: Production Pipeline (Week 8)
- [ ] Create prediction API endpoint
- [ ] Real-time feature calculation
- [ ] Model versioning (MLflow)
- [ ] Monitoring & logging
- [ ] A/B testing framework

### Phase 6: Advanced Model Training (Week 9-10)
- [ ] Implement LSTM sequence model
- [ ] Train transformer with attention
- [ ] Build graph neural network
- [ ] Create stacked ensemble
- [ ] Hyperparameter optimization (Optuna)
- [ ] Cross-validation with time-based splits

### Phase 7: Model Deployment & Monitoring (Week 11-12)
- [ ] Integrate with frontend
- [ ] Real-time prediction API
- [ ] Performance tracking dashboard
- [ ] A/B testing framework
- [ ] Continuous retraining pipeline
- [ ] Drift detection and alerts

## Advanced Training Strategies

### 1. Time-Series Cross-Validation
```python
from sklearn.model_selection import TimeSeriesSplit

def time_series_cv(X, y, dates, n_splits=5):
    """
    Proper cross-validation for time series
    Each fold: train on past, validate on future
    """
    
    # Sort by date
    sorted_indices = np.argsort(dates)
    X_sorted = X[sorted_indices]
    y_sorted = y[sorted_indices]
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    results = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_sorted)):
        X_train, X_val = X_sorted[train_idx], X_sorted[val_idx]
        y_train, y_val = y_sorted[train_idx], y_sorted[val_idx]
        
        # Train model
        model = xgb.XGBClassifier()
        model.fit(X_train, y_train)
        
        # Evaluate
        val_pred = model.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, val_pred)
        
        results.append({
            'fold': fold,
            'train_size': len(train_idx),
            'val_size': len(val_idx),
            'val_auc': val_auc
        })
        
        print(f"Fold {fold}: AUC = {val_auc:.4f}")
    
    return results
```

### 2. Hyperparameter Optimization with Optuna
```python
import optuna

def objective(trial):
    """Optuna objective function for XGBoost"""
    
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0.0, 0.5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0)
    }
    
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=False)
    
    val_pred = model.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, val_pred)

# Run optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100, timeout=3600)

print(f"Best AUC: {study.best_value}")
print(f"Best params: {study.best_params}")
```

### 3. Online Learning for Production
```python
class OnlineMLBPredictor:
    """Incrementally update model as new games finish"""
    
    def __init__(self, base_model):
        self.model = base_model
        self.game_buffer = []
        self.update_frequency = 10  # Update every 10 games
    
    def predict(self, X):
        return self.model.predict_proba(X)[:, 1]
    
    def observe_outcome(self, X, y):
        """Called after game finishes with actual outcome"""
        self.game_buffer.append((X, y))
        
        # Update model when buffer full
        if len(self.game_buffer) >= self.update_frequency:
            self._incremental_update()
    
    def _incremental_update(self):
        """Partial fit on new data"""
        X_new = np.vstack([x for x, y in self.game_buffer])
        y_new = np.array([y for x, y in self.game_buffer])
        
        # For models supporting partial_fit
        if hasattr(self.model, 'partial_fit'):
            self.model.partial_fit(X_new, y_new)
        else:
            # For tree models, retrain on combined data
            # Keep sliding window of recent games
            self.model.fit(X_new, y_new, xgb_model=self.model.get_booster())
        
        # Clear buffer
        self.game_buffer = []
        
        print(f"Model updated with {len(y_new)} new games")
```

### 4. Transfer Learning from Previous Seasons
```python
def transfer_learning_approach():
    """
    Pre-train on older seasons, fine-tune on recent data
    Useful at start of new season with limited 2026 data
    """
    
    # Phase 1: Pre-train on 2015-2024 (large dataset)
    base_model = tf.keras.Sequential([
        Dense(256, activation='relu', input_dim=n_features),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    base_model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['AUC']
    )
    
    # Train on all historical data
    base_model.fit(
        X_historical, y_historical,
        epochs=50,
        batch_size=64,
        validation_split=0.2
    )
    
    # Phase 2: Fine-tune on 2025 data (most recent)
    # Freeze early layers, only update final layers
    for layer in base_model.layers[:-2]:
        layer.trainable = False
    
    # Re-compile with lower learning rate
    base_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['AUC']
    )
    
    # Fine-tune on recent season
    base_model.fit(
        X_2025, y_2025,
        epochs=20,
        batch_size=32
    )
    
    return base_model
```

## Expected Performance Improvements

### Conservative vs Advanced Model Comparison

| Model Type | Accuracy | AUC | Log Loss | Notes |
|------------|----------|-----|----------|-------|
| **Baseline (2 features)** | 54-56% | 0.56-0.58 | 0.69 | Simple, interpretable |
| **Enhanced (20 features)** | 57-59% | 0.61-0.63 | 0.66 | Traditional ML |
| **Advanced (LSTM)** | 59-62% | 0.64-0.67 | 0.63 | Captures sequences |
| **Transformer** | 60-63% | 0.65-0.68 | 0.62 | Attention mechanism |
| **GNN** | 60-62% | 0.65-0.67 | 0.62 | Structural learning |
| **Stacked Ensemble** | **61-64%** | **0.67-0.70** | **0.60** | Best overall |

**FiveThirtyEight Benchmark:** ~58% accuracy, 0.62 AUC

**Our Goal:** Beat FiveThirtyEight by 2-4 percentage points

### Player Performance Improvements

| Metric | Baseline | Advanced | Improvement |
|--------|----------|----------|-------------|
| **Batting Avg R²** | 0.30 | 0.45-0.50 | +50-67% |
| **Home Runs R²** | 0.25 | 0.40-0.45 | +60-80% |
| **ERA R²** | 0.35 | 0.50-0.55 | +43-57% |
| **Strikeouts R²** | 0.40 | 0.55-0.60 | +38-50% |

### Key Improvements from Advanced Features

1. **LSTM/Transformer:** +3-5% accuracy from better sequence modeling
2. **Statcast embeddings:** +1-2% from pitch-level detail
3. **Opponent adjustment:** +1-2% from quality weighting
4. **Ensemble stacking:** +2-3% from model diversity
5. **Online learning:** Maintains accuracy as season progresses

### Risk Factors

⚠️ **Overfitting:** Deep models need careful regularization  
⚠️ **Data leakage:** Must be vigilant about temporal ordering  
⚠️ **Computation:** LSTM/Transformer slower than XGBoost  
⚠️ **Interpretability:** Black box models harder to debug  
⚠️ **Cold start:** Limited 2026 data early in season

---

## Research & References

### Academic Papers
1. **"Machine Learning for Sports Betting: Should Model Sophistication Matter?"** (Štrumbelj, 2014)
   - Found that simple models often outperform complex ones in sports betting
   - Feature engineering matters more than model complexity

2. **"Predicting the Outcome of MLB Games with ML"** (Soto Valero, 2016)
   - Run differential and starting pitcher quality were top predictors
   - Achieved 58-62% accuracy

3. **"Using Statcast Data to Predict MLB Outcomes"** (Various, 2018-2023)
   - Exit velocity and hard hit % are leading indicators
   - xwOBA more predictive than traditional stats

### Industry Resources
1. **FiveThirtyEight MLB Model**
   - Uses Elo ratings for team strength
   - Starting pitcher adjustments
   - Home field advantage (~54% win rate)
   - Achieves ~58% accuracy

2. **Baseball Prospectus PECOTA**
   - Player projection system
   - Aging curves, comparables
   - Park factors, league adjustments

3. **Fangraphs**
   - wRC+ (weighted runs created plus)
   - FIP (fielding independent pitching)
   - WAR (wins above replacement)

### Key Findings from Research

#### What Works:
✅ **Recent performance** (last 14-30 days) > Season averages  
✅ **Run differential** > Win/loss record  
✅ **FIP** > ERA for pitchers  
✅ **Hard hit rate** > Batting average  
✅ **Opponent adjustment** improves accuracy  
✅ **Home field advantage** is real (~54%)  
✅ **Rest days matter** for both teams and pitchers  

#### What Doesn't Work:
❌ **Clutch stats** are mostly noise  
❌ **Pitcher wins** are poor quality metric  
❌ **Batting average** is outdated  
❌ **Saves** don't predict future performance  
❌ **Team chemistry** is unmeasurable  
❌ **Weather** has minimal impact (except extreme)  

### Recommended Feature Priority

**Tier 1 (Must Have):**
- Run differential (30-day rolling)
- Starting pitcher FIP/ERA
- Home field indicator
- Recent win %
- Team wOBA/wRC+

**Tier 2 (Strong Predictors):**
- Bullpen ERA
- Rest days
- Head-to-head record
- Platoon advantage
- Division game indicator

**Tier 3 (Nice to Have):**
- Statcast metrics (exit velo, barrel rate)
- Park factors
- Weather conditions
- Umpire tendencies
- Travel distance

---

## Expected Outcomes

### Game Prediction Model
**Conservative Estimates:**
- **Baseline (2 features):** 54-56% accuracy
- **Enhanced (10 features):** 57-59% accuracy
- **Advanced (20+ features):** 59-61% accuracy

**FiveThirtyEight achieves ~58%**, so our goal is competitive performance.

### Player Performance Model
**Expected R² values:**
- **Batting Average:** 0.3-0.4 (30-40% variance explained)
- **Home Runs:** 0.25-0.35
- **ERA:** 0.35-0.45
- **Strikeouts:** 0.4-0.5

These are typical for baseball prediction due to high inherent randomness.

### Business Value
Even **55% accuracy** in game prediction creates value:
- Beat random guessing (50%)
- Informed betting strategies
- Fan engagement (prediction contests)
- Player DFS lineup optimization

---

## Next Steps

1. **Start with 2-feature baseline model**
   - Feature 1: Team run differential (30-day)
   - Feature 2: Starting pitcher quality score

2. **Validate on 2024 season**
   - Use 2015-2023 for training
   - Test on 2024 games
   - Calculate accuracy, log loss, ROC-AUC

3. **Iterate and expand**
   - Add features one at a time
   - Measure marginal improvement
   - Keep what works, drop what doesn't

4. **Deploy for 2026 predictions**
   - Real-time feature calculation
   - Daily prediction updates
   - Track performance vs actual results

## Production System Architecture

### Real-Time Prediction Pipeline
```python
class MLBPredictionService:
    """Production service for real-time predictions"""
    
    def __init__(self):
        self.models = self._load_models()
        self.feature_cache = {}
        self.cache_ttl = 3600  # 1 hour
    
    def _load_models(self):
        """Load all model variants"""
        return {
            'game_baseline': joblib.load('models/game_baseline_v1.pkl'),
            'game_advanced': tf.keras.models.load_model('models/game_lstm_v1.h5'),
            'game_ensemble': joblib.load('models/game_ensemble_v1.pkl'),
            'batter_performance': joblib.load('models/batter_v1.pkl'),
            'pitcher_performance': joblib.load('models/pitcher_v1.pkl')
        }
    
    async def predict_game(self, home_team_id, away_team_id, game_date, pitcher_ids=None):
        """Generate game outcome prediction"""
        
        # Extract features
        features = await self._extract_game_features(
            home_team_id, away_team_id, game_date, pitcher_ids
        )
        
        # Get predictions from multiple models
        predictions = {}
        for model_name, model in self.models.items():
            if 'game' in model_name:
                pred_proba = model.predict_proba([features])[0][1]
                predictions[model_name] = pred_proba
        
        # Ensemble prediction (weighted average)
        ensemble_pred = np.average(
            list(predictions.values()),
            weights=[0.2, 0.3, 0.5]  # Baseline, Advanced, Ensemble
        )
        
        # Calculate confidence interval
        pred_std = np.std(list(predictions.values()))
        
        return {
            'home_win_probability': ensemble_pred,
            'away_win_probability': 1 - ensemble_pred,
            'confidence_interval': [ensemble_pred - 1.96*pred_std, ensemble_pred + 1.96*pred_std],
            'model_agreement': 1 - pred_std,  # Lower std = more agreement
            'individual_models': predictions
        }
    
    async def _extract_game_features(self, home_id, away_id, date, pitcher_ids):
        """Extract all features for a game"""
        
        # Check cache first
        cache_key = f"{home_id}_{away_id}_{date}"
        if cache_key in self.feature_cache:
            cached_time, features = self.feature_cache[cache_key]
            if time.time() - cached_time < self.cache_ttl:
                return features
        
        # Parallel feature extraction
        features = await asyncio.gather(
            self._get_team_strength(home_id, date),
            self._get_team_strength(away_id, date),
            self._get_recent_form(home_id, date, days=30),
            self._get_recent_form(away_id, date, days=30),
            self._get_head_to_head(home_id, away_id, date),
            self._get_rest_days(home_id, away_id, date),
            self._get_pitcher_stats(pitcher_ids, date) if pitcher_ids else None
        )
        
        # Combine into feature vector
        feature_vector = np.concatenate([f for f in features if f is not None])
        
        # Cache result
        self.feature_cache[cache_key] = (time.time(), feature_vector)
        
        return feature_vector
```

### Model Monitoring & Drift Detection
```python
class ModelMonitor:
    """Monitor model performance and detect drift"""
    
    def __init__(self):
        self.prediction_log = []
        self.performance_window = 100  # Last 100 predictions
    
    def log_prediction(self, features, prediction, actual_outcome=None):
        """Log every prediction for monitoring"""
        self.prediction_log.append({
            'timestamp': datetime.now(),
            'features': features,
            'prediction': prediction,
            'actual': actual_outcome
        })
    
    def check_feature_drift(self):
        """Detect if input features are changing"""
        recent = self.prediction_log[-self.performance_window:]
        historical = self.prediction_log[-2*self.performance_window:-self.performance_window]
        
        if len(historical) < self.performance_window:
            return None
        
        # KS test for distribution shift
        from scipy.stats import ks_2samp
        
        drift_scores = []
        for i in range(len(recent[0]['features'])):
            recent_vals = [p['features'][i] for p in recent]
            historical_vals = [p['features'][i] for p in historical]
            
            statistic, pvalue = ks_2samp(recent_vals, historical_vals)
            drift_scores.append(pvalue)
        
        # Alert if any feature has p < 0.01 (significant drift)
        if min(drift_scores) < 0.01:
            return {
                'drift_detected': True,
                'affected_features': [i for i, p in enumerate(drift_scores) if p < 0.01],
                'recommendation': 'Consider retraining model'
            }
        
        return {'drift_detected': False}
    
    def check_performance_degradation(self):
        """Check if model accuracy is declining"""
        recent = [p for p in self.prediction_log[-self.performance_window:] 
                 if p['actual'] is not None]
        
        if len(recent) < 50:
            return None
        
        # Calculate recent accuracy
        predictions = [p['prediction'] > 0.5 for p in recent]
        actuals = [p['actual'] for p in recent]
        recent_acc = accuracy_score(actuals, predictions)
        
        # Compare to baseline (assume 58% historical)
        baseline_acc = 0.58
        
        if recent_acc < baseline_acc - 0.03:  # 3% drop
            return {
                'degradation_detected': True,
                'current_accuracy': recent_acc,
                'baseline_accuracy': baseline_acc,
                'recommendation': 'Model needs retraining'
            }
        
        return {'degradation_detected': False, 'current_accuracy': recent_acc}
```

### Automated Retraining Pipeline
```python
class AutoRetrainer:
    """Automatically retrain models on schedule or trigger"""
    
    def __init__(self, monitor):
        self.monitor = monitor
        self.retrain_schedule = '0 2 * * 0'  # Weekly, Sunday 2 AM
    
    async def check_retrain_triggers(self):
        """Check if retraining needed"""
        
        # Trigger 1: Scheduled retrain
        if self._is_scheduled_time():
            return {'trigger': 'schedule', 'retrain': True}
        
        # Trigger 2: Performance degradation
        perf_check = self.monitor.check_performance_degradation()
        if perf_check and perf_check['degradation_detected']:
            return {'trigger': 'performance', 'retrain': True, 'details': perf_check}
        
        # Trigger 3: Feature drift
        drift_check = self.monitor.check_feature_drift()
        if drift_check and drift_check['drift_detected']:
            return {'trigger': 'drift', 'retrain': True, 'details': drift_check}
        
        # Trigger 4: New season started
        if self._is_new_season():
            return {'trigger': 'new_season', 'retrain': True}
        
        return {'retrain': False}
    
    async def retrain_all_models(self):
        """Execute full retraining pipeline"""
        
        # 1. Extract fresh training data
        X_train, y_train, X_val, y_val = await self._prepare_training_data()
        
        # 2. Retrain each model type
        models_to_train = [
            ('baseline', self._train_baseline),
            ('lstm', self._train_lstm),
            ('ensemble', self._train_ensemble)
        ]
        
        new_models = {}
        for model_name, train_func in models_to_train:
            print(f"Retraining {model_name}...")
            new_model = await train_func(X_train, y_train, X_val, y_val)
            new_models[model_name] = new_model
        
        # 3. Validate new models
        for model_name, model in new_models.items():
            val_metrics = self._validate_model(model, X_val, y_val)
            print(f"{model_name} validation AUC: {val_metrics['auc']:.4f}")
        
        # 4. A/B test: Deploy to 10% of traffic
        await self._canary_deploy(new_models, traffic_percentage=0.1)
        
        # 5. Monitor for 24 hours, then full rollout if successful
        return new_models
```

---

## Additional Advanced Techniques to Explore

### 1. Meta-Features from Player Relationships
- **Teammate chemistry:** How does Player A perform with Player B on base?
- **Pitcher-catcher battery:** Specific pitcher-catcher combinations
- **Lineup protection:** How does batting order affect performance?

### 2. Situational Features
- **High-leverage situations:** Performance in close games, late innings
- **Clutch metrics:** RISP (runners in scoring position) stats
- **Pressure index:** Playoff race, elimination games

### 3. External Data Integration
- **Weather API:** Real-time temperature, wind, humidity
- **Social sentiment:** Twitter/Reddit sentiment analysis before games
- **Betting lines movement:** Track how Vegas odds shift
- **News/Injury scraping:** Automated parsing of injury reports

### 4. Causal Inference
- **Propensity score matching:** Isolate effect of specific factors
- **Difference-in-differences:** Measure impact of team changes
- **Regression discontinuity:** Analyze threshold effects (e.g., rest days)

### 5. Reinforcement Learning
- **Dynamic lineup optimization:** Learn optimal batting order
- **In-game strategy:** Bullpen management, pinch-hitting decisions
- **Season-long roster management:** Trade/signing decisions

---

**Document Status:** Advanced Draft v2  
**Next Review:** After Phase 1 completion  
**Owner:** ML Team  
**Last Updated:** December 24, 2025
**Confidence Level:** High - Based on SOTA research and production systems
