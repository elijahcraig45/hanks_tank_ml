"""
Main entry point for ML Agent System
CLI interface for asking agent questions and managing models
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Setup paths
sys.path.insert(0, str(Path(__file__).parent))

from core.agent_manager import ModelManagementAgent, AgentDecision
from config.agent_config import config
from knowledge.knowledge_base import BaseballKnowledgeBase


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.AGENT_DECISION_LOG),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AgentCLI:
    """Command-line interface for agent interactions"""
    
    def __init__(self):
        """Initialize agent and knowledge base"""
        self.agent = ModelManagementAgent(api_key=config.ANTHROPIC_API_KEY)
        self.knowledge_base = BaseballKnowledgeBase()
        logger.info("Agent CLI initialized")
    
    def print_header(self):
        """Print welcome header"""
        print("\n" + "="*70)
        print("  ML MODEL MANAGEMENT AGENT - Interactive CLI")
        print("  Baseball Prediction System (2026 Season)")
        print("  Type 'help' for commands, 'exit' to quit")
        print("="*70 + "\n")
    
    def print_help(self):
        """Print available commands"""
        help_text = """
Available Commands:
  help                    Show this help message
  status                  Check system status (models, data, config)
  query <question>        Ask agent a question (use quotes for multi-word)
  retrain                 Check if models need retraining
  analyze                 Analyze model confidence and performance
  features                Suggest new features to engineer
  health                  Check data quality and freshness
  knowledge <query>       Search baseball knowledge base
  config                  Show current configuration
  logs                    Show recent agent decisions
  examples                Show example queries
  exit/quit               Exit the CLI

Examples:
  query "Should I retrain the V3 model?"
  query "What new features should we engineer?"
  analyze
  health
  knowledge pitcher fatigue
  retrain
        """
        print(help_text)
    
    def show_status(self):
        """Show system status"""
        print("\n" + "-"*70)
        print("SYSTEM STATUS")
        print("-"*70)
        
        print(f"\n📊 MODEL CONFIGURATION:")
        print(f"  Primary Model:        {config.PRIMARY_MODEL_VERSION}")
        print(f"  Accuracy Threshold:   {config.RETRAIN_THRESHOLD_ACCURACY:.1%}")
        print(f"  Model Versions:       {', '.join(config.MODEL_VERSIONS)}")
        print(f"  Feature Counts:       V1={config.V1_FEATURE_COUNT}, V2={config.V2_FEATURE_COUNT}, V3={config.V3_FEATURE_COUNT}")
        
        print(f"\n📈 CONFIDENCE SETTINGS:")
        print(f"  High Percentile:      {config.CONFIDENCE_HIGH_PERCENTILE:.0%}")
        print(f"  High Accuracy:        {config.CONFIDENCE_HIGH_ACCURACY:.1%}")
        print(f"  Medium Percentile:    {config.CONFIDENCE_MEDIUM_PERCENTILE:.0%}")
        print(f"  Medium Accuracy:      {config.CONFIDENCE_MEDIUM_ACCURACY:.1%}")
        
        print(f"\n🏦 DATA CONFIGURATION:")
        print(f"  BigQuery Project:     {config.GCP_PROJECT_ID}")
        print(f"  Dataset:              {config.BIGQUERY_DATASET}")
        print(f"  Training Years:       {config.TRAINING_YEARS[0]}-{config.TRAINING_YEARS[1]}")
        print(f"  Validation Year:      {config.VALIDATION_YEAR}")
        print(f"  Data Freshness:       {config.DATA_FRESHNESS_THRESHOLD_DAYS} days")
        
        print(f"\n⚙️  AGENT SETTINGS:")
        print(f"  API Model:            {config.ANTHROPIC_MODEL}")
        print(f"  Local LLM Enabled:    {config.USE_LOCAL_LLM}")
        print(f"  Max Iterations:       {config.AGENT_MAX_ITERATIONS}")
        print(f"  Confirmation Timeout: {config.AGENT_CONFIRMATION_TIMEOUT_SEC}s")
        
        print(f"\n📁 LOG DIRECTORY:       {config.AGENT_LOG_DIRECTORY}")
        print()
    
    def show_config(self):
        """Show full configuration"""
        print("\n" + "-"*70)
        print("FULL CONFIGURATION")
        print("-"*70 + "\n")
        
        cfg_dict = config.to_dict()
        for key, value in sorted(cfg_dict.items()):
            print(f"  {key:35} = {value}")
        print()
    
    def show_examples(self):
        """Show example queries"""
        examples = [
            ("retrain_check", "Should I retrain the V3 model? Check for performance degradation."),
            ("feature_analysis", "What new features should we engineer based on recent performance?"),
            ("prediction_confidence", "Analyze the confidence distribution for recent predictions."),
            ("baseball_insights", "What baseball domain insights should we incorporate?"),
            ("data_health", "Check data quality: Is BigQuery data fresh and valid?"),
        ]
        
        print("\n" + "-"*70)
        print("EXAMPLE QUERIES")
        print("-"*70 + "\n")
        
        for name, query in examples:
            print(f"  $ query \"{query}\"")
            print()
    
    def show_logs(self, n: int = 5):
        """Show recent decisions from log"""
        log_path = Path(config.AGENT_DECISION_LOG)
        
        if not log_path.exists():
            print(f"No decisions logged yet. Log path: {log_path}")
            return
        
        print("\n" + "-"*70)
        print(f"RECENT DECISIONS (last {n})")
        print("-"*70 + "\n")
        
        with open(log_path, "r") as f:
            lines = f.readlines()
            for line in lines[-n:]:
                import json
                try:
                    decision = json.loads(line)
                    print(f"[{decision['timestamp']}]")
                    print(f"  Agent: {decision['agent_role']}")
                    print(f"  Decision: {decision['decision'][:100]}...")
                    print(f"  Confidence: {decision['confidence']:.1%}")
                    print()
                except:
                    pass
    
    def search_knowledge(self, query: str):
        """Search knowledge base"""
        print(f"\n🔍 Searching knowledge base for: '{query}'\n")
        
        results = self.knowledge_base.search(query, top_k=5)
        
        if not results:
            print("No results found.")
            return
        
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['title']}")
            print(f"   Category: {result['category']}")
            print(f"   Score: {result['score']:.1f}")
            print(f"   Tags: {', '.join(result['tags'])}")
            print(f"   Source: {result['source']}")
            print()
    
    def run_query(self, question: str):
        """Execute agent query"""
        print(f"\n🤖 Agent Processing...\n")
        logger.info(f"User query: {question}")
        
        try:
            response = self.agent.query(question)
            print("\n" + "-"*70)
            print("AGENT RESPONSE")
            print("-"*70 + "\n")
            print(response)
            print()
        except Exception as e:
            logger.error(f"Query failed: {e}", exc_info=True)
            print(f"\n❌ Error: {e}")
    
    def run_interactive(self):
        """Run interactive CLI loop"""
        self.print_header()
        
        while True:
            try:
                user_input = input("agent> ").strip()
                
                if not user_input:
                    continue
                
                # Parse command
                parts = user_input.split(None, 1)
                command = parts[0].lower()
                args = parts[1] if len(parts) > 1 else ""
                
                if command in ["exit", "quit", "q"]:
                    print("\nGoodbye! 👋\n")
                    break
                
                elif command == "help":
                    self.print_help()
                
                elif command == "status":
                    self.show_status()
                
                elif command == "config":
                    self.show_config()
                
                elif command == "examples":
                    self.show_examples()
                
                elif command == "logs":
                    n = int(args) if args.isdigit() else 5
                    self.show_logs(n)
                
                elif command == "knowledge":
                    if args:
                        self.search_knowledge(args)
                    else:
                        print("Usage: knowledge <query>")
                
                elif command == "query":
                    if args:
                        self.run_query(args)
                    else:
                        print("Usage: query <question>")
                
                elif command == "retrain":
                    self.run_query(
                        "Check if the V3 model needs retraining. "
                        "Analyze recent performance and data quality."
                    )
                
                elif command == "analyze":
                    self.run_query(
                        "Analyze model confidence and performance across all percentiles. "
                        "What are the key findings?"
                    )
                
                elif command == "features":
                    self.run_query(
                        "Suggest new features to engineer. Look at current feature importance "
                        "and baseball domain knowledge."
                    )
                
                elif command == "health":
                    self.run_query(
                        "Check data health: Is BigQuery data fresh? "
                        "Are there quality issues in recent data?"
                    )
                
                else:
                    print("Unknown command. Type 'help' for available commands.")
            
            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋\n")
                break
            except Exception as e:
                logger.error(f"Error in CLI loop: {e}", exc_info=True)
                print(f"Error: {e}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="ML Model Management Agent - Interactive AI for Baseball Predictions"
    )
    parser.add_argument(
        "-q", "--query",
        help="Run a single query and exit"
    )
    parser.add_argument(
        "-c", "--command",
        help="Run a built-in command (retrain, analyze, features, health)"
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Run non-interactively"
    )
    parser.add_argument(
        "--knowledge-search",
        help="Search knowledge base"
    )
    
    args = parser.parse_args()
    
    # Validate configuration
    is_valid, message = config.validate()
    if not is_valid:
        print(f"⚠️  Configuration issues: {message}")
        print("Edit .env.agent to fix issues, or run: python agent/setup.py")
    
    cli = AgentCLI()
    
    if args.query:
        cli.run_query(args.query)
    
    elif args.command:
        if args.command == "retrain":
            cli.run_query(
                "Check if the V3 model needs retraining. "
                "Analyze recent performance and data quality."
            )
        elif args.command == "analyze":
            cli.run_query(
                "Analyze model confidence and performance across all percentiles."
            )
        elif args.command == "features":
            cli.run_query(
                "Suggest new features to engineer."
            )
        elif args.command == "health":
            cli.run_query(
                "Check data health in BigQuery."
            )
    
    elif args.knowledge_search:
        cli.search_knowledge(args.knowledge_search)
    
    elif not args.no_interactive:
        cli.run_interactive()


if __name__ == "__main__":
    main()
