"""
Train ensemble models for CF defensive analysis.

Three complementary models:
1. Bayesian Anticipation: Posterior skill updates based on Reaction Efficiency
2. Geometric Execution: Route deviation scores  
3. Stochastic Deterrence: Monte Carlo simulation of arm value
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BayesianAnticipationModel:
    """Bayesian model for CF Reaction Efficiency."""
    
    def __init__(self, features_df, train_year=2025):
        """Initialize with features DataFrame."""
        self.df = features_df.copy()
        self.train_year = train_year
        self.train = features_df[features_df['game_year'] == train_year].copy()
        self.test = features_df[features_df['game_year'] != train_year].copy()
        
        # Prior parameters for Beta-Binomial
        self.prior_alpha = 2
        self.prior_beta = 2
        self.player_posteriors = {}
    
    def fit(self):
        """Fit by updating player posteriors."""
        logger.info("Fitting Bayesian Anticipation Model...")
        
        for player_name in self.train['fielder_name'].unique():
            player_data = self.train[self.train['fielder_name'] == player_name]
            
            catches = player_data['caught'].sum()
            opportunities = len(player_data)
            
            # Beta-Binomial posterior
            posterior_alpha = self.prior_alpha + catches
            posterior_beta = self.prior_beta + (opportunities - catches)
            
            self.player_posteriors[player_name] = {
                'alpha': posterior_alpha,
                'beta': posterior_beta,
                'mean': posterior_alpha / (posterior_alpha + posterior_beta),
                'std': np.sqrt(
                    (posterior_alpha * posterior_beta) / 
                    ((posterior_alpha + posterior_beta)**2 * (posterior_alpha + posterior_beta + 1))
                ),
                'catches': catches,
                'opportunities': opportunities
            }
        
        logger.info(f"Fitted {len(self.player_posteriors)} CF players")
        return self
    
    def reaction_efficiency_score(self, player_name):
        """Calculate Reaction Efficiency for a player."""
        if player_name not in self.player_posteriors:
            return 0.0
        
        posterior = self.player_posteriors[player_name]
        prior_mean = self.prior_alpha / (self.prior_alpha + self.prior_beta)
        efficiency = (posterior['mean'] - prior_mean) / (0.01 + prior_mean)
        return efficiency
    
    def evaluate(self):
        """Evaluate on test set."""
        logger.info("Evaluating Bayesian model on test set...")
        
        test_results = []
        for player_name in self.test['fielder_name'].unique():
            if player_name not in self.player_posteriors:
                continue
            
            player_test = self.test[self.test['fielder_name'] == player_name]
            player_catches = player_test['caught'].sum()
            player_opps = len(player_test)
            
            posterior = self.player_posteriors[player_name]
            efficiency = self.reaction_efficiency_score(player_name)
            
            test_results.append({
                'fielder_name': player_name,
                'test_catches': player_catches,
                'test_opportunities': player_opps,
                'test_catch_rate': player_catches / player_opps if player_opps > 0 else 0,
                'posterior_mean': posterior['mean'],
                'posterior_std': posterior['std'],
                'reaction_efficiency': efficiency
            })
        
        return pd.DataFrame(test_results)


class GeometricExecutionModel:
    """Gradient Boosting model for route efficiency."""
    
    def __init__(self, features_df, train_year=2025):
        """Initialize with features DataFrame."""
        self.df = features_df.copy()
        self.train_year = train_year
        self.train = features_df[features_df['game_year'] == train_year].copy()
        self.test = features_df[features_df['game_year'] != train_year].copy()
        
        self.model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
    
    def fit(self):
        """Fit model to predict route deviation."""
        logger.info("Fitting Geometric Execution Model...")
        
        # Features: hang time and distance
        X = self.train[['hang_time_sec', 'distance_ft']].values
        y = self.train['route_deviation'].values
        
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        
        logger.info(f"Fitted on {len(X)} training samples")
        return self
    
    def route_efficiency_score(self, player_name, test_data):
        """Calculate execution efficiency for a player."""
        player_data = test_data[test_data['fielder_name'] == player_name]
        
        if len(player_data) == 0:
            return 0.0
        
        X = player_data[['hang_time_sec', 'distance_ft']].values
        y_actual = player_data['route_deviation'].values
        
        X_scaled = self.scaler.transform(X)
        y_pred = self.model.predict(X_scaled)
        
        # Efficiency = (Predicted - Actual) avg
        # Negative residual = more efficient than expected
        residuals = y_actual - y_pred
        efficiency = -residuals.mean()  # Negate so positive = efficient
        
        return efficiency
    
    def evaluate(self):
        """Evaluate on test set."""
        logger.info("Evaluating Geometric model on test set...")
        
        test_results = []
        for player_name in self.test['fielder_name'].unique():
            efficiency = self.route_efficiency_score(player_name, self.test)
            player_test = self.test[self.test['fielder_name'] == player_name]
            
            test_results.append({
                'fielder_name': player_name,
                'test_plays': len(player_test),
                'route_efficiency': efficiency,
                'avg_route_deviation': player_test['route_deviation'].mean()
            })
        
        return pd.DataFrame(test_results)


class StochasticDeterrenceModel:
    """Monte Carlo simulation for arm deterrence value."""
    
    def __init__(self, features_df, train_year=2025):
        """Initialize with features DataFrame."""
        self.df = features_df.copy()
        self.train_year = train_year
        self.train = features_df[features_df['game_year'] == train_year].copy()
        self.test = features_df[features_df['game_year'] != train_year].copy()
        
        self.player_arm_profiles = {}
    
    def fit(self):
        """Build arm profiles for each player."""
        logger.info("Fitting Stochastic Deterrence Model...")
        
        for player_name in self.train['fielder_name'].unique():
            player_data = self.train[self.train['fielder_name'] == player_name]
            
            # Arm strength proxy: catch rate on deep balls
            deep_balls = player_data[player_data['arm_opportunity_ft'] > 200]
            arm_strength = deep_balls['caught'].mean() if len(deep_balls) > 0 else 0.5
            
            # Release quickness proxy: route efficiency
            route_quality = 1.0 / (1.0 + player_data['route_deviation'].mean())
            
            self.player_arm_profiles[player_name] = {
                'arm_strength': arm_strength,
                'release_quickness': route_quality,
                'deep_opportunities': len(deep_balls)
            }
        
        logger.info(f"Built profiles for {len(self.player_arm_profiles)} players")
        return self
    
    def simulate_deterrence_value(self, player_name, n_simulations=1000):
        """Monte Carlo simulation for deterrence value."""
        if player_name not in self.player_arm_profiles:
            return 0.0
        
        profile = self.player_arm_profiles[player_name]
        arm_strength = profile['arm_strength']
        release_quickness = profile['release_quickness']
        
        # Probability runner is thrown out
        p_out = arm_strength * release_quickness
        
        # Base runner MDP: advance if E[value] > 0
        # Simplified: runner advances 30% of time on tag-up, expects 0.3 runs
        # Gets out with p_out, expects -0.3 runs
        runs_deterred = 0
        for _ in range(n_simulations):
            if np.random.random() < p_out:
                # Runner out
                runs_deterred += 0.3  # Prevented 0.3 runs
        
        return runs_deterred / n_simulations
    
    def evaluate(self):
        """Evaluate deterrence value on test set."""
        logger.info("Evaluating Stochastic Deterrence model on test set...")
        
        test_results = []
        for player_name in self.test['fielder_name'].unique():
            deterrence_value = self.simulate_deterrence_value(player_name, n_simulations=2000)
            player_test = self.test[self.test['fielder_name'] == player_name]
            deep_opps = len(player_test[player_test['arm_opportunity_ft'] > 200])
            
            test_results.append({
                'fielder_name': player_name,
                'deterrence_value': deterrence_value,
                'deep_opportunities': deep_opps,
                'avg_arm_distance': player_test['arm_opportunity_ft'].mean()
            })
        
        return pd.DataFrame(test_results)


def main():
    """Train all three models."""
    logger.info("Loading features...")
    df_features = pd.read_parquet("cf_features.parquet")
    
    # Train Bayesian
    bayes_model = BayesianAnticipationModel(df_features)
    bayes_model.fit()
    bayes_results = bayes_model.evaluate()
    bayes_results.to_csv("bayes_anticipation_results.csv", index=False)
    logger.info(f"Bayesian results saved: {len(bayes_results)} players")
    
    # Train Geometric
    geom_model = GeometricExecutionModel(df_features)
    geom_model.fit()
    geom_results = geom_model.evaluate()
    geom_results.to_csv("geom_execution_results.csv", index=False)
    logger.info(f"Geometric results saved: {len(geom_results)} players")
    
    # Train Stochastic
    stoch_model = StochasticDeterrenceModel(df_features)
    stoch_model.fit()
    stoch_results = stoch_model.evaluate()
    stoch_results.to_csv("stoch_deterrence_results.csv", index=False)
    logger.info(f"Stochastic results saved: {len(stoch_results)} players")
    
    return bayes_model, geom_model, stoch_model


if __name__ == "__main__":
    main()
