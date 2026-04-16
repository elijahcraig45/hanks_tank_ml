"""
Generate insights and rankings from CF ensemble models.

This module combines results from the three models (Anticipation, Execution, Deterrence)
to produce an integrated ranking and actionable insights for decision-making.
"""

import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CFInsightEngine:
    """Aggregate model results and generate rankings."""
    
    def __init__(self):
        """Load all model results."""
        self.bayes = pd.read_csv("bayes_anticipation_results.csv")
        self.geom = pd.read_csv("geom_execution_results.csv")
        self.stoch = pd.read_csv("stoch_deterrence_results.csv")
    
    def normalize_scores(self):
        """Normalize scores to 0-100 scale for comparison."""
        logger.info("Normalizing model scores...")
        
        # Anticipation: higher reaction_efficiency is better
        self.bayes['anticipation_score'] = (
            (self.bayes['reaction_efficiency'] - self.bayes['reaction_efficiency'].min()) /
            (self.bayes['reaction_efficiency'].max() - self.bayes['reaction_efficiency'].min() + 1e-6) * 100
        ).fillna(50)
        
        # Execution: higher route_efficiency is better
        self.geom['execution_score'] = (
            (self.geom['route_efficiency'] - self.geom['route_efficiency'].min()) /
            (self.geom['route_efficiency'].max() - self.geom['route_efficiency'].min() + 1e-6) * 100
        ).fillna(50)
        
        # Deterrence: higher deterrence_value is better
        self.stoch['deterrence_score'] = (
            (self.stoch['deterrence_value'] - self.stoch['deterrence_value'].min()) /
            (self.stoch['deterrence_value'].max() - self.stoch['deterrence_value'].min() + 1e-6) * 100
        ).fillna(50)
    
    def create_integrated_ranking(self):
        """Combine three models into single ranking."""
        logger.info("Creating integrated CF ranking...")
        
        # Merge all results on fielder name
        ranking = self.bayes[['fielder_name', 'anticipation_score', 'test_opportunities']].copy()
        ranking.columns = ['fielder_name', 'anticipation_score', 'total_opportunities']
        
        # Merge execution scores
        geom_scores = self.geom[['fielder_name', 'execution_score']].copy()
        ranking = ranking.merge(geom_scores, on='fielder_name', how='left')
        
        # Merge deterrence scores
        stoch_scores = self.stoch[['fielder_name', 'deterrence_score']].copy()
        ranking = ranking.merge(stoch_scores, on='fielder_name', how='left')
        
        # Fill missing scores with league average (50)
        ranking['anticipation_score'] = ranking['anticipation_score'].fillna(50)
        ranking['execution_score'] = ranking['execution_score'].fillna(50)
        ranking['deterrence_score'] = ranking['deterrence_score'].fillna(50)
        
        # Composite score: weighted average
        # Anticipation 40%, Execution 35%, Deterrence 25%
        ranking['composite_score'] = (
            ranking['anticipation_score'] * 0.40 +
            ranking['execution_score'] * 0.35 +
            ranking['deterrence_score'] * 0.25
        )
        
        ranking = ranking.sort_values('composite_score', ascending=False)
        return ranking
    
    def generate_player_profile(self, player_name, ranking):
        """Generate detailed profile for a single player."""
        player = ranking[ranking['fielder_name'] == player_name].iloc[0]
        
        profile = f"""
{'='*70}
ELITE CENTER FIELDER PROFILE: {player_name}
{'='*70}

OVERALL RANKING SCORE: {player['composite_score']:.1f}/100

Component Scores:
  Anticipation (Reaction Efficiency):    {player['anticipation_score']:.1f}/100
  Execution (Route Efficiency):          {player['execution_score']:.1f}/100
  Deterrence (Arm Value):                {player['deterrence_score']:.1f}/100

Plays Analyzed: {player['total_opportunities']:.0f}

INSIGHTS:
"""
        
        # Anticipation insight
        if player['anticipation_score'] > 70:
            profile += "  • ELITE ANTICIPATION: Exceptional first-step reactions and ball reading.\n"
            profile += "    Can play shallower than position average, generating more outs.\n"
        elif player['anticipation_score'] > 55:
            profile += "  • ABOVE-AVERAGE ANTICIPATION: Solid first-step and positioning.\n"
        else:
            profile += "  • NEEDS IMPROVEMENT: Below-average anticipation metrics.\n"
        
        # Execution insight
        if player['execution_score'] > 70:
            profile += "  • ELITE EXECUTION: Extremely efficient routes. Makes difficult plays routine.\n"
            profile += "    Low injury risk due to optimal path selection.\n"
        elif player['execution_score'] > 55:
            profile += "  • EFFICIENT ROUTES: Good path management relative to ball physics.\n"
        else:
            profile += "  • LOOSE ROUTES: Tends to take inefficient paths, higher diving/injury risk.\n"
        
        # Deterrence insight
        if player['deterrence_score'] > 70:
            profile += "  • ELITE ARM DETERRENCE: Strong throw velocity + release quickness.\n"
            profile += "    Prevents extra-base advancement, suppresses runs scored.\n"
        elif player['deterrence_score'] > 55:
            profile += "  • ADEQUATE ARM: Average deterrence value for position.\n"
        else:
            profile += "  • WEAK ARM: Limited deterrence impact on base-runners.\n"
        
        profile += f"\n{'='*70}\n"
        return profile
    
    def generate_report(self):
        """Generate full report with rankings and profiles."""
        self.normalize_scores()
        ranking = self.create_integrated_ranking()
        
        logger.info("\n" + "="*70)
        logger.info("ELITE CENTER FIELDER RANKING (2025-2026 Data)")
        logger.info("="*70)
        logger.info(ranking[['fielder_name', 'composite_score', 'anticipation_score', 
                            'execution_score', 'deterrence_score']].head(15).to_string())
        
        # Generate detailed profiles for top 5
        logger.info("\n" + "="*70)
        logger.info("DETAILED PLAYER PROFILES - TOP 5")
        logger.info("="*70)
        
        for i, player_name in enumerate(ranking['fielder_name'].head(5)):
            logger.info(self.generate_player_profile(player_name, ranking))
        
        # Save results
        ranking.to_csv("cf_elite_ranking.csv", index=False)
        logger.info("\nFull ranking saved to cf_elite_ranking.csv")
        
        return ranking


def main():
    """Run insight generation."""
    engine = CFInsightEngine()
    ranking = engine.generate_report()


if __name__ == "__main__":
    main()
