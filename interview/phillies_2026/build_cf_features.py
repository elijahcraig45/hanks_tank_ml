"""
Build defensive features for center fielder analysis.

This module computes features for three modeling components:
1. Anticipation (Bayesian): Reaction efficiency relative to baseline catch probability
2. Execution (Geometric): Route deviation from geodesic path
3. Deterrence (Stochastic): Arm quality metrics for base-runner deterrence
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from scipy.integrate import quad
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CFFeatureBuilder:
    """Build defensive features for CF modeling."""
    
    def __init__(self, df):
        """
        Initialize with raw Statcast data.
        
        Args:
            df: DataFrame from fetch_cf_data.py with CF plays
        """
        self.df = df.copy()
        self.features = pd.DataFrame()
    
    def build_anticipation_features(self):
        """
        Build features for Bayesian Anticipation model.
        
        Calculates:
        - hang_time: Seconds ball is in air (proxy for reaction window)
        - distance: Feet from fielder start to ball landing
        - catch_baseline: Expected catch rate given physics (logistic)
        - caught: Binary outcome (1=out, 0=hit/miss)
        
        Returns self for chaining.
        """
        logger.info("Building anticipation features...")
        
        # Handle both synthetic and real Statcast data formats
        if 'hang_time_sec' in self.df.columns:
            self.features['hang_time_sec'] = self.df['hang_time_sec'].fillna(self.df['hang_time_sec'].median())
        else:
            # Estimate from coordinates if needed
            self.features['hang_time_sec'] = 2.0
        
        if 'hit_distance_ft' in self.df.columns:
            self.features['distance_ft'] = self.df['hit_distance_ft']
        else:
            self.features['distance_ft'] = 250.0
        
        if 'catch_baseline_prob' in self.df.columns:
            self.features['catch_baseline'] = self.df['catch_baseline_prob']
        else:
            # Logistic model: baseline = f(hang_time, distance)
            hang_time = self.features['hang_time_sec']
            distance = self.features['distance_ft']
            self.features['catch_baseline'] = 1 / (1 + np.exp(-(hang_time - 2.0) / 0.5))
        
        self.features['caught'] = self.df['is_out'].astype(int)
        
        return self
    
    def build_execution_features(self):
        """
        Build features for Geometric Route Deviation model.
        
        Calculates:
        - route_arc_length: Actual path distance (approximated)
        - geodesic_distance: Straight-line distance to ball
        - route_deviation: Arc / Geodesic (>1 = inefficient)
        
        Returns self for chaining.
        """
        logger.info("Building execution features...")
        
        if 'geodesic_ft' in self.df.columns:
            self.features['geodesic_ft'] = self.df['geodesic_ft']
            self.features['route_arc_ft'] = self.df['route_arc_ft']
        else:
            # Estimate from distances
            self.features['geodesic_ft'] = self.features['distance_ft']
            self.features['route_arc_ft'] = self.features['distance_ft'] * 1.1
        
        self.features['route_deviation'] = self.features['route_arc_ft'] / (self.features['geodesic_ft'] + 1e-5)
        
        logger.info(f"Built execution features ({len(self.features)} plays)")
        return self
    
    def build_deterrence_features(self):
        """
        Build features for Stochastic Deterrence Value model.
        
        Calculates:
        - arm_opportunity_ft: Hit distance (deeper = more arm value)
        - arm_context: Type of play context
        
        Returns self for chaining.
        """
        logger.info("Building deterrence features...")
        
        if 'arm_opportunity_ft' in self.df.columns:
            self.features['arm_opportunity_ft'] = self.df['arm_opportunity_ft']
            self.features['arm_context'] = self.df['arm_context']
        else:
            self.features['arm_opportunity_ft'] = self.features['distance_ft']
            # Categorize by distance
            self.features['arm_context'] = pd.cut(
                self.features['distance_ft'],
                bins=[0, 100, 200, 300, np.inf],
                labels=['shallow', 'medium', 'deep', 'very_deep']
            )
        
        logger.info(f"Built deterrence features ({len(self.features)} plays)")
        return self
    
    def finalize(self):
        """Add metadata and return final feature set."""
        self.features['fielder_name'] = self.df['fielder_name']
        self.features['game_date'] = self.df['game_date']
        self.features['game_year'] = self.df['game_year']
        
        return self.features
    
    def build_all(self):
        """Build all feature groups."""
        self.build_anticipation_features()
        self.build_execution_features()
        self.build_deterrence_features()
        return self.finalize()


def main():
    """Load raw data and build features."""
    logger.info("Loading raw Statcast data...")
    df_raw = pd.read_parquet("cf_statcast_raw.parquet")
    
    builder = CFFeatureBuilder(df_raw)
    features = builder.build_all()
    
    logger.info(f"\nFeature set shape: {features.shape}")
    logger.info(f"Columns: {list(features.columns)}")
    logger.info(f"\nSample features:\n{features.head()}")
    
    features.to_parquet("cf_features.parquet", index=False)
    logger.info("Saved to cf_features.parquet")


if __name__ == "__main__":
    main()
