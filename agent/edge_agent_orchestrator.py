"""
Edge-Optimized Agent Orchestrator
Runs locally on Pi3/Pi5/Laptop with minimal overhead
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime
from enum import Enum

from agent.config.edge_device_config import EdgeAgentConfig, DeviceProfile
from agent.knowledge.local_knowledge_system import LocalBaseballKnowledgeBase, LocalFineTuningPipeline
from agent.tools.minimal_web_access import MinimalWebAccess


class AgentMode(Enum):
    """Agent operation modes"""
    OFFLINE = "offline"        # Pure local, no internet
    SCHEDULED = "scheduled"    # Scheduled web syncs only
    INTERACTIVE = "interactive" # User can trigger web calls
    TRAINING = "training"      # Fine-tuning mode


class EdgeAgentOrchestrator:
    """
    Main agent orchestrator optimized for edge deployment
    Manages: local LLM, knowledge base, model training, web access
    """
    
    def __init__(self, device_profile: Optional[str] = None, mode: AgentMode = AgentMode.OFFLINE):
        """Initialize edge agent"""
        
        # Configuration
        self.config = EdgeAgentConfig(device_profile)
        self.mode = mode
        
        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger("EdgeAgent")
        
        # Initialize components
        self.knowledge_base = LocalBaseballKnowledgeBase()
        self.fine_tuning_pipeline = LocalFineTuningPipeline(self.knowledge_base)
        self.web_access = MinimalWebAccess()
        
        # State tracking
        self.initialized_at = datetime.now()
        self.query_history = []
        self.decisions_log = []
        
        self.logger.info(f"Edge Agent initialized on {self.config.device_config.name}")
        self.logger.info(f"Operating in {mode.value} mode")
    
    def _setup_logging(self):
        """Setup logging to console and file"""
        
        log_dir = Path("./logs")
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"edge_agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def query(self, question: str) -> Dict:
        """
        Process user query locally
        Returns: decision, reasoning, confidence
        """
        
        self.logger.info(f"Query: {question}")
        
        result = {
            "query": question,
            "timestamp": datetime.now().isoformat(),
            "mode": self.mode.value,
            "decision": None,
            "reasoning": "",
            "confidence": 0,
            "knowledge_used": [],
        }
        
        # Step 1: Search knowledge base
        relevant_knowledge = self.knowledge_base.search_local(question, top_k=3)
        result["knowledge_used"] = [
            self.knowledge_base.get_entry(entry_id)
            for entry_id, _ in relevant_knowledge
        ]
        
        self.logger.info(f"Found {len(result['knowledge_used'])} relevant knowledge entries")
        
        # Step 2: Prepare context for local LLM
        context = self._prepare_llm_context(question, result["knowledge_used"])
        
        # Step 3: Call local LLM (placeholder - real call would use ollama/llamacpp)
        llm_response = self._call_local_llm(context)
        
        # Step 4: Log decision
        result["decision"] = llm_response.get("decision", "unknown")
        result["reasoning"] = llm_response.get("reasoning", "")
        result["confidence"] = llm_response.get("confidence", 0.5)
        
        # Step 5: Web access if needed and mode allows
        if self.mode in [AgentMode.SCHEDULED, AgentMode.INTERACTIVE]:
            if llm_response.get("needs_web_data"):
                result["web_data"] = self._fetch_web_data(
                    llm_response.get("web_data_type")
                )
        
        # Log to history
        self.query_history.append(result)
        self.decisions_log.append(result)
        
        return result
    
    def _prepare_llm_context(self, question: str, knowledge: List[Dict]) -> str:
        """Prepare context for LLM call"""
        
        system_prompt = f"""
You are a baseball analysis agent optimized for game outcome prediction.
You have access to the following knowledge base and historical data.

Device: {self.config.device_config.name}
Mode: {self.mode.value}

Your task: Analyze the query and provide a confident prediction or decision.
"""
        
        knowledge_context = "\n\n".join([
            f"[{k['category'].upper()}] {k['topic']}\n{k['content']}"
            for k in knowledge if k
        ])
        
        user_message = f"""
Query: {question}

Relevant knowledge:
{knowledge_context}

Provide your analysis in JSON format with:
- decision: your prediction/recommendation
- reasoning: explanation
- confidence: 0.0-1.0 confidence score
- needs_web_data: boolean if you need current web data
- web_data_type: type of data needed (standings, injuries, etc.)
"""
        
        return system_prompt + "\n\n" + user_message
    
    def _call_local_llm(self, context: str) -> Dict:
        """Call local LLM model"""
        
        # In production, this would use:
        # - ollama library with local Mistral/Phi/TinyLLaMA
        # - or llama-cpp-python with GGUF quantized models
        # - or other edge LLM frameworks
        
        # For now, placeholder that shows structure
        self.logger.debug("Calling local LLM...")
        
        # This would be replaced with actual LLM call
        # For testing: return structured response
        
        response = {
            "decision": "prediction_pending",
            "reasoning": "Local LLM would analyze here",
            "confidence": 0.55,
            "needs_web_data": False,
        }
        
        return response
    
    def _fetch_web_data(self, data_type: Optional[str]) -> Dict:
        """Fetch web data based on request"""
        
        if data_type == "standings":
            return self.web_access.get_standings()
        elif data_type == "injuries":
            return self.web_access.get_injury_updates()
        elif data_type == "game_context":
            return self.web_access.get_game_predictions_context("ANY")
        else:
            return {}
    
    def prepare_fine_tuning(self) -> Path:
        """Prepare data for fine-tuning on baseball domain"""
        
        self.logger.info("Preparing fine-tuning dataset...")
        
        return self.fine_tuning_pipeline.prepare_fine_tuning_data()
    
    def get_fine_tuning_config(self) -> Dict:
        """Get LoRA fine-tuning config for current device"""
        
        return self.fine_tuning_pipeline.get_fine_tuning_config(
            self.config.device_config.profile.value
        )
    
    def get_status(self) -> Dict:
        """Get agent status and resource usage"""
        
        import psutil
        
        process = psutil.Process()
        
        return {
            "agent": {
                "initialized_at": self.initialized_at.isoformat(),
                "mode": self.mode.value,
                "device": self.config.device_config.name,
                "model": self.config.MODEL_SELECTION,
                "queries_processed": len(self.query_history),
                "decisions_logged": len(self.decisions_log),
            },
            "resources": {
                "memory_mb": round(process.memory_info().rss / (1024 * 1024), 2),
                "memory_percent": process.memory_percent(),
                "cpu_percent": process.cpu_percent(interval=1),
            },
            "knowledge": {
                "total_entries": self.knowledge_base._count_entries(),
                "fine_tuning_examples": len(self.knowledge_base.get_fine_tuning_dataset()),
            },
            "web": {
                "bandwidth_stats": self.web_access.get_bandwidth_stats(),
            },
            "config": self.config.to_dict(),
        }
    
    def export_decisions_log(self, output_path: Optional[Path] = None) -> Path:
        """Export decisions log to JSON"""
        
        if output_path is None:
            log_dir = Path("./logs")
            log_dir.mkdir(exist_ok=True)
            output_path = log_dir / f"decisions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        
        with open(output_path, 'w') as f:
            for decision in self.decisions_log:
                f.write(json.dumps(decision) + "\n")
        
        self.logger.info(f"Exported {len(self.decisions_log)} decisions to {output_path}")
        return output_path
    
    def print_summary(self):
        """Print agent summary"""
        
        status = self.get_status()
        
        print(f"""
╔════════════════════════════════════════════════════════════╗
║          EDGE AGENT STATUS                                 ║
╚════════════════════════════════════════════════════════════╝

Device:          {status['agent']['device']}
Model:           {status['agent']['model']}
Mode:            {status['agent']['mode']}
Uptime:          {(datetime.now() - self.initialized_at).seconds}s

PROCESSING:
  Queries:       {status['agent']['queries_processed']}
  Decisions:     {status['agent']['decisions_logged']}

RESOURCES:
  Memory:        {status['resources']['memory_mb']} MB
  CPU:           {status['resources']['cpu_percent']}%

KNOWLEDGE:
  Entries:       {status['knowledge']['total_entries']}
  Fine-tune Ex:  {status['knowledge']['fine_tuning_examples']}

WEB ACCESS:
  Total Used:    {status['web']['bandwidth_stats'].get('total_mb', 0)} MB
  Target:        {status['web']['bandwidth_stats'].get('target_mb', 0)} MB/week
        """)


class EdgeAgentCLI:
    """Interactive CLI for edge agent"""
    
    def __init__(self):
        self.agent = None
    
    def start(self):
        """Start interactive session"""
        
        print("""
╔════════════════════════════════════════════════════════════╗
║        EDGE AGENT - LOCAL BASEBALL AI                      ║
╚════════════════════════════════════════════════════════════╝
        """)
        
        # Initialize agent
        device = input("Device profile (pi3/pi5/laptop) [auto]: ").strip() or None
        mode = input("Mode (offline/scheduled/interactive) [offline]: ").strip() or "offline"
        
        self.agent = EdgeAgentOrchestrator(
            device_profile=device,
            mode=AgentMode[mode.upper()]
        )
        
        self.agent.print_summary()
        
        # Interactive loop
        while True:
            try:
                query = input("\nQuery > ").strip()
                
                if query.lower() == "exit":
                    break
                elif query.lower() == "status":
                    self.agent.print_summary()
                elif query.lower() == "export":
                    path = self.agent.export_decisions_log()
                    print(f"Exported to {path}")
                else:
                    result = self.agent.query(query)
                    print(f"\nDecision: {result['decision']}")
                    print(f"Confidence: {result['confidence']:.2%}")
                    print(f"Reasoning: {result['reasoning']}")
            
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
        
        # Export on exit
        self.agent.export_decisions_log()
        print("\nEdge Agent shutting down. Decisions logged.")


if __name__ == "__main__":
    # Test agent
    print("Starting Edge Agent...\n")
    
    # Test mode 1: Direct instantiation
    agent = EdgeAgentOrchestrator(device_profile="laptop")
    agent.print_summary()
    
    # Test mode 2: Query
    result = agent.query("How does pitcher rest affect winning?")
    print(f"\nTest Query Result:")
    print(f"  Decision: {result['decision']}")
    print(f"  Confidence: {result['confidence']}")
    print(f"  Knowledge sources: {len(result['knowledge_used'])}")
    
    # Test mode 3: Fine-tuning prep
    ft_config = agent.get_fine_tuning_config()
    print(f"\nFine-tuning config: {json.dumps(ft_config, indent=2)}")
