"""
Main orchestration pipeline for CF elite defensive analysis.

Runs the complete workflow:
1. Fetch Statcast data from BigQuery
2. Engineer defensive features
3. Train ensemble models (Bayesian, Geometric, Stochastic)
4. Generate insights and rankings
"""

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run complete pipeline."""
    
    logger.info("="*70)
    logger.info("ELITE CENTER FIELDER ANALYSIS PIPELINE")
    logger.info("="*70)
    
    try:
        # Step 1: Fetch data
        logger.info("\n[STEP 1] Fetching CF Statcast data from BigQuery...")
        from fetch_cf_data import fetch_cf_statcast_data
        
        df_raw = fetch_cf_statcast_data(year_range=(2025, 2026))
        logger.info(f"Fetched {len(df_raw)} CF plays")
        logger.info(f"  Unique players: {df_raw['fielder_name'].nunique()}")
        logger.info(f"  Date range: {df_raw['game_date'].min()} to {df_raw['game_date'].max()}")
        logger.info(f"  Overall out rate: {df_raw['is_out'].mean():.2%}")
        df_raw.to_parquet("cf_statcast_raw.parquet", index=False)
        
    except Exception as e:
        logger.error(f"Error in data fetch: {e}")
        logger.info("(If BigQuery connection fails, ensure credentials are set)")
        return
    
    try:
        # Step 2: Feature engineering
        logger.info("\n[STEP 2] Engineering defensive features...")
        from build_cf_features import CFFeatureBuilder
        import pandas as pd
        
        df_raw = pd.read_parquet("cf_statcast_raw.parquet")
        builder = CFFeatureBuilder(df_raw)
        features = builder.build_all()
        
        logger.info(f"Feature matrix shape: {features.shape}")
        logger.info(f"Features: {list(features.columns)}")
        features.to_parquet("cf_features.parquet", index=False)
        
    except Exception as e:
        logger.error(f"Error in feature engineering: {e}")
        return
    
    try:
        # Step 3: Train models
        logger.info("\n[STEP 3] Training ensemble models...")
        logger.info("  - Bayesian Anticipation Model (Reaction Efficiency)")
        logger.info("  - Geometric Execution Model (Route Deviation)")
        logger.info("  - Stochastic Deterrence Model (Arm Value)")
        
        from train_cf_ensemble import (
            BayesianAnticipationModel,
            GeometricExecutionModel,
            StochasticDeterrenceModel
        )
        
        df_features = pd.read_parquet("cf_features.parquet")
        
        # Bayesian
        bayes = BayesianAnticipationModel(df_features, train_year=2025)
        bayes.fit()
        bayes_results = bayes.evaluate()
        bayes_results.to_csv("bayes_anticipation_results.csv", index=False)
        logger.info(f"✓ Bayesian model fitted ({len(bayes.player_posteriors)} players)")
        
        # Geometric
        geom = GeometricExecutionModel(df_features, train_year=2025)
        geom.fit()
        geom_results = geom.evaluate()
        geom_results.to_csv("geom_execution_results.csv")
        logger.info(f"✓ Geometric model fitted and evaluated")
        
        # Stochastic
        stoch = StochasticDeterrenceModel(df_features, train_year=2025)
        stoch.fit()
        stoch_results = stoch.evaluate()
        stoch_results.to_csv("stoch_deterrence_results.csv", index=False)
        logger.info(f"✓ Stochastic model fitted ({len(stoch.player_arm_profiles)} profiles)")
        
    except Exception as e:
        logger.error(f"Error in model training: {e}")
        return
    
    try:
        # Step 4: Generate insights
        logger.info("\n[STEP 4] Generating insights and rankings...")
        from analyze_cf_insights import CFInsightEngine
        
        engine = CFInsightEngine()
        ranking = engine.generate_report()
        
        logger.info(f"\n✓ Analysis complete! Elite ranking saved to cf_elite_ranking.csv")
        
    except Exception as e:
        logger.error(f"Error in insights: {e}")
        return
    
    logger.info("\n" + "="*70)
    logger.info("PIPELINE COMPLETE")
    logger.info("="*70)
    logger.info("\nOutputs:")
    logger.info("  • cf_elite_ranking.csv - Top CF with composite scores")
    logger.info("  • bayes_anticipation_results.csv - Reaction efficiency rankings")
    logger.info("  • geom_execution_results.csv - Route efficiency analysis")
    logger.info("  • stoch_deterrence_results.csv - Arm deterrence values")
    logger.info("\nUse cf_elite_ranking.csv for final Phillies recommendation.")


if __name__ == "__main__":
    main()
