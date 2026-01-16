# Edge-Optimized Agent Implementation Guide

## Overview

This guide covers the **fully local, cost-optimized AI agent** designed to run on:
- **Raspberry Pi 3** (512MB RAM) - $35 hardware cost
- **Raspberry Pi 5** (8GB RAM) - $80 hardware cost  
- **Older Laptop** (4-8GB RAM) - Free to $200
- **Modern Laptop** (16GB+ RAM) - Full feature set

**Operating Cost: $2-10/month** (electricity only) vs $300+/month with cloud LLM APIs

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│         EDGE AGENT ORCHESTRATOR                     │
│  (Local control, decision-making, logging)          │
└──────────────┬──────────────────────────────────────┘
               │
       ┌───────┴────────┬──────────────┬──────────────┐
       │                │              │              │
       ▼                ▼              ▼              ▼
┌─────────────┐  ┌──────────────┐  ┌──────────┐  ┌─────────────┐
│  LOCAL LLM  │  │ KNOWLEDGE    │  │  WEB     │  │  INFERENCE  │
│             │  │  BASE        │  │  ACCESS  │  │  ENGINE     │
│ Mistral 7B  │  │  (SQLite)    │  │  (Cache) │  │  (ONNX)     │
│ or Phi 2.7B │  │              │  │          │  │             │
│ q4/q8       │  │ + LoRA       │  │ Minimal  │  │  Streaming  │
└─────────────┘  │ Fine-tuning  │  │ Scheduled│  │  Inference  │
                 └──────────────┘  └──────────┘  └─────────────┘
```

### Key Components

#### 1. **EdgeAgentConfig** (`edge_device_config.py`)
- Auto-detects hardware capabilities
- Selects optimal model size per device
- Applies device-specific optimizations
- Manages quantization settings

**Device Profiles:**
```
Pi3 (512MB)      → TinyLLaMA 1.1B int8 (600MB)
Pi5 (8GB)        → Phi-2 2.7B q4 (1.6GB) or Mistral 7B q4 (4GB)
Laptop (8GB+)    → Mistral 7B q4 (4GB)
Laptop (16GB+)   → Full models or larger
```

#### 2. **LocalBaseballKnowledgeBase** (`local_knowledge_system.py`)
- SQLite-backed persistent knowledge store
- 8 foundational baseball topics
- Fine-tuning example collection
- Semantic search (text-based, ONNX-ready)
- Exports to JSONL for LoRA training

**Embedded Knowledge:**
- MLB Rules & Core Rulebook
- Batting Sabermetrics (BA, OBP, SLG, OPS, wOBA)
- Pitching Statistics (ERA, FIP, K/9, WHIP)
- Park Factors & Weather Effects
- Pitcher Fatigue & Rest Cycles
- Injury Impact Analysis
- Momentum & Streaks
- Advanced Analytics

#### 3. **LocalFineTuningPipeline** (`local_knowledge_system.py`)
- Prepares dataset for LoRA fine-tuning
- Device-optimized training configs
- LoRA rank/alpha tuning per hardware
- Supports training on laptop, inference on Pi

#### 4. **MinimalWebAccess** (`minimal_web_access.py`)
- **Scheduled-only** API calls (weekly standings, daily injuries)
- SQLite cache with TTL (7-day default)
- Bandwidth tracking (~100MB/week target)
- On-demand manual triggers only
- Uses free APIs: MLB StatsAPI, ESPN

#### 5. **EdgeAgentOrchestrator** (`edge_agent_orchestrator.py`)
- Main control loop
- Query processing with local LLM
- Knowledge integration
- Decision logging
- Web data fetching (mode-based)
- Status monitoring

---

## Installation & Setup

### Prerequisites

**System Requirements:**
```bash
# Pi3/Pi5
- 512MB+ RAM
- 32GB+ SD card
- Network connection (setup only)

# Laptop
- 4GB+ RAM
- 5GB+ SSD space
- Network connection (setup only)

# Python
- Python 3.8+ installed
- pip or conda
```

### Quick Start (30 minutes)

#### Step 1: Clone & Install

```bash
cd /path/to/hanks_tank_ml
pip install -r agent/requirements.txt
```

**Key dependencies:**
```
requests          # Web API calls (minimal)
sqlite3           # Built-in, no install
onnx              # Embeddings (optional initially)
onnxruntime       # Embeddings (optional initially)
psutil            # System monitoring
```

#### Step 2: Initialize Agent

```bash
python agent/edge_agent_orchestrator.py
```

This creates:
- `data/baseball_knowledge.db` (SQLite, ~5MB)
- `data/embeddings/` (ONNX cache)
- `logs/` (decision logs)

#### Step 3: First Query

```python
from agent.edge_agent_orchestrator import EdgeAgentOrchestrator, AgentMode

# Initialize
agent = EdgeAgentOrchestrator(device_profile="pi5", mode=AgentMode.OFFLINE)

# Query
result = agent.query("How does pitcher rest affect ERA?")
print(result['decision'])  # Local LLM response
print(result['confidence'])  # 0.0-1.0
```

#### Step 4: Prepare Fine-tuning (Optional)

```bash
python -c "
from agent.edge_agent_orchestrator import EdgeAgentOrchestrator

agent = EdgeAgentOrchestrator(device_profile='laptop')
ft_path = agent.prepare_fine_tuning()
ft_config = agent.get_fine_tuning_config()
print(f'Fine-tuning data ready at {ft_path}')
print(f'Config: {ft_config}')
"
```

---

## Device Profiles & Configuration

### Profile Selection

Auto-detection based on system RAM:
```python
# Automatic (recommended)
agent = EdgeAgentOrchestrator()  # Detects hardware

# Or explicit
agent = EdgeAgentOrchestrator(device_profile="pi5")
```

### Configuration Reference

```python
from agent.config.edge_device_config import EdgeAgentConfig

config = EdgeAgentConfig()

# Get current config
print(config.MODEL_SELECTION)           # e.g., "Mistral 7B"
print(config.MODEL_QUANT)               # e.g., "q4"
print(config.MAX_TOKENS)                # Context length
print(config.BATCH_SIZE)                # Inference batch
print(config.ENABLE_FINE_TUNING)        # Can tune this device?
print(config.ALLOW_API_CALLS_PRODUCTION) # Web calls allowed?

# Export full config
config_dict = config.to_dict()
```

### Memory Budget per Device

**Raspberry Pi 3 (512MB total):**
```
Model (TinyLLaMA int8):    400 MB
LoRA weights:               50 MB
Inference buffer:           30 MB
Knowledge cache:            20 MB
OS + system:                12 MB
─────────────────────────────────
Total:                     ~512 MB (fits exactly!)
```

**Raspberry Pi 5 (8GB available):**
```
Model (Mistral 7B q4):    4000 MB
LoRA weights:              200 MB
Inference buffer:          500 MB
Knowledge base:            100 MB
Cache (sqlite):             50 MB
OS + headroom:            3150 MB
─────────────────────────────────
Total used:              ~5000 MB (well within budget)
```

**Laptop (8GB+ available):**
```
All optimizations optional
Can use full-precision models
Multiple models in parallel possible
```

---

## Operating Modes

### 1. **OFFLINE Mode** (Recommended for Production)
- ✅ 100% local operation
- ✅ Zero internet dependency after setup
- ✅ Fastest inference (no network latency)
- ✅ Maximum privacy
- ❌ Cannot fetch live standings/injuries

```python
agent = EdgeAgentOrchestrator(mode=AgentMode.OFFLINE)
```

### 2. **SCHEDULED Mode** (Balanced)
- ✅ Scheduled web syncs (weekly standings, daily injuries)
- ✅ Predictable bandwidth (~100MB/week)
- ✅ Cache-first, network fallback
- ✅ Good for continuous operation

```python
agent = EdgeAgentOrchestrator(mode=AgentMode.SCHEDULED)
```

### 3. **INTERACTIVE Mode** (Development)
- ✅ User can trigger web lookups on-demand
- ✅ Full data freshness when needed
- ✅ Good for experimentation
- ❌ Higher bandwidth use

```python
agent = EdgeAgentOrchestrator(mode=AgentMode.INTERACTIVE)
```

### 4. **TRAINING Mode** (Fine-tuning)
- ✅ Optimized for LoRA training
- ✅ Can use larger context
- ✅ Suitable for laptops (not Pi)
- ✅ Prepares domain-specific models

```python
agent = EdgeAgentOrchestrator(mode=AgentMode.TRAINING)
```

---

## Knowledge Base Usage

### Accessing Knowledge

```python
from agent.knowledge.local_knowledge_system import LocalBaseballKnowledgeBase

kb = LocalBaseballKnowledgeBase()

# Search
results = kb.search_local("pitcher fatigue", top_k=3)
for entry_id, score in results:
    entry = kb.get_entry(entry_id)
    print(f"{entry['topic']}: {entry['content'][:100]}...")

# Get fine-tuning examples
examples = kb.get_fine_tuning_dataset()
print(f"Have {len(examples)} training examples")

# Export for training
kb.export_for_training("./training_data.jsonl")
```

### Adding Custom Knowledge

```python
from agent.knowledge.local_knowledge_system import LocalBaseballKnowledgeBase, KnowledgeEntry
from datetime import datetime

kb = LocalBaseballKnowledgeBase()

# Add custom knowledge
entry = KnowledgeEntry(
    id="my_custom_insight",
    category="strategy",
    topic="Bullpen Usage Patterns",
    content="High-leverage situations use closer. Low-leverage use setup men. Mid-game use middle relievers.",
    source="Personal Analysis",
    importance=8,
    created_at=datetime.now().isoformat(),
)

kb.add_entry(entry)

# Add training example
kb.add_fine_tuning_example(
    category="team_strategy",
    question="When does a team use their closer?",
    answer="Closers are used in high-leverage situations, typically 1 run lead or tied late in games.",
    related_topics=["bullpen_usage", "close_games"]
)
```

### Knowledge Export

```python
from pathlib import Path

kb = LocalBaseballKnowledgeBase()

# Export as JSONL for fine-tuning
kb.export_for_training(Path("baseball_knowledge.jsonl"))

# Check file
with open("baseball_knowledge.jsonl") as f:
    for i, line in enumerate(f):
        if i < 3:
            print(line.strip()[:80] + "...")
```

---

## Web Access Management

### Bandwidth Tracking

```python
from agent.tools.minimal_web_access import MinimalWebAccess

web = MinimalWebAccess()

# Check bandwidth
stats = web.get_bandwidth_stats()
print(f"Used: {stats['total_mb']} MB / {stats['target_mb']} MB")

# Get standings (cached, weekly)
standings = web.get_standings()

# Get injuries (cached, daily)
injuries = web.get_injury_updates()

# Manual refresh (force)
standings = web.get_standings(force_refresh=True)
```

### API Call Limits

| Endpoint | Interval | Target | Reason |
|----------|----------|--------|--------|
| Standings | 7 days | 10 MB | Division/playoff changes |
| Injuries | 1 day | 5 MB | IL updates |
| Game Data | 1 day | 20 MB | Upcoming games |
| Player Stats | 7 days | 10 MB | Performance trends |

**Total Target:** ~100 MB/week (~430 MB/month vs GB/day for cloud LLM)

---

## Fine-Tuning Pipeline

### Preparing Data

```python
from agent.edge_agent_orchestrator import EdgeAgentOrchestrator
from pathlib import Path

# Initialize agent
agent = EdgeAgentOrchestrator(device_profile="laptop", mode=AgentMode.TRAINING)

# Prepare fine-tuning dataset
ft_path = agent.prepare_fine_tuning()
print(f"Dataset ready at {ft_path}")

# Get device-optimized config
ft_config = agent.get_fine_tuning_config()
print(f"LoRA Rank: {ft_config['r']}")  # Device-specific
print(f"Batch Size: {ft_config['batch_size']}")
print(f"Epochs: {ft_config['num_epochs']}")
```

### LoRA Configuration per Device

**Pi3 (Extreme Constraints):**
```python
{
    "r": 4,              # Minimal rank (4 low-rank matrices)
    "lora_alpha": 8,
    "batch_size": 1,     # One example at a time
    "num_epochs": 1,     # Single pass only
    "learning_rate": 1e-4,
}
```

**Pi5 (Balanced):**
```python
{
    "r": 8,
    "lora_alpha": 16,
    "batch_size": 2,
    "num_epochs": 2,
    "learning_rate": 2e-4,
}
```

**Laptop (Development):**
```python
{
    "r": 16,
    "lora_alpha": 32,
    "batch_size": 4,
    "num_epochs": 3,
    "learning_rate": 5e-4,
}
```

### Running Fine-tuning

```python
# Using llama-cpp-python with LoRA
import subprocess

# On laptop, train model
subprocess.run([
    "python", "-m", "llama_cpp_python.lora",
    "--model", "path/to/model.gguf",
    "--train-data", "training_data.jsonl",
    "--output", "baseball_lora.bin",
    "--lora-r", "16",
    "--lora-alpha", "32",
    "--epochs", "3",
], check=True)

# Copy trained LoRA weights to Pi5/Pi3
# Model + LoRA combined uses ~50MB more
```

---

## Status Monitoring

### Get Agent Status

```python
agent = EdgeAgentOrchestrator()

status = agent.get_status()

print(f"Device: {status['agent']['device']}")
print(f"Model: {status['agent']['model']}")
print(f"Queries: {status['agent']['queries_processed']}")
print(f"Memory: {status['resources']['memory_mb']} MB")
print(f"CPU: {status['resources']['cpu_percent']}%")
print(f"Knowledge Entries: {status['knowledge']['total_entries']}")
print(f"Bandwidth: {status['web']['bandwidth_stats']['total_mb']} MB used")

# Print formatted summary
agent.print_summary()
```

### Exporting Decisions

```python
# Automatic export on exit
agent.export_decisions_log()

# Or manual export
log_path = agent.export_decisions_log(Path("./my_decisions.jsonl"))
print(f"Decisions exported to {log_path}")

# Read decisions
import json
with open(log_path) as f:
    for line in f:
        decision = json.loads(line)
        print(f"{decision['query']}: {decision['decision']}")
```

---

## Performance Expectations

### Inference Speed

| Device | Model | Tokens/sec | Context | Note |
|--------|-------|-----------|---------|------|
| Pi3 | TinyLLaMA 1.1B q8 | 2-3 | 512 | Very slow but works |
| Pi5 | Phi-2 2.7B q4 | 5-8 | 2048 | Reasonable for queries |
| Laptop | Mistral 7B q4 | 12-20 | 4096 | Very good |
| Modern | Full precision | 25-50+ | 8192 | Excellent |

### Memory Usage

| Component | Pi3 | Pi5 | Laptop |
|-----------|-----|-----|--------|
| Model | 600MB | 1600MB | 4000MB |
| LoRA | 50MB | 200MB | 200MB |
| Cache | 20MB | 50MB | 200MB |
| Knowledge | 10MB | 50MB | 100MB |
| **Total** | **~680MB** | **~1900MB** | **~4500MB** |

### Latency per Query

| Device | Time | Components |
|--------|------|------------|
| Pi3 | 15-30s | Knowledge search (instant) + LLM (~15s) |
| Pi5 | 3-5s | Knowledge search (instant) + LLM (~2s) |
| Laptop | 1-2s | Knowledge search (instant) + LLM (~0.5s) |

---

## Troubleshooting

### Out of Memory on Pi3

**Problem:** Agent crashes with MemoryError

**Solutions:**
1. Reduce context length to 256 tokens
2. Use aggressive batch size 1 only
3. Disable caching temporarily
4. Pre-quantize model to int4

```python
# Manual config adjustment
from agent.config.edge_device_config import PI3_CONFIG
PI3_CONFIG.max_context_length = 256  # Instead of 512
PI3_CONFIG.batch_size = 1
```

### Slow Inference on Pi5

**Problem:** Inference taking 30+ seconds

**Solutions:**
1. Check if background tasks running
2. Reduce context window
3. Use float16 instead of int8
4. Add heat sink if thermal throttling

```bash
# Monitor CPU/temp
watch -n 1 'vcgencmd measure_temp; ps aux'
```

### API Calls Failing

**Problem:** "Connection timeout" or "Cannot reach server"

**Solutions:**
1. Check internet connection
2. Verify API endpoint status (MLB StatsAPI)
3. Fall back to cache (enabled by default)
4. Switch to OFFLINE mode if not needed

```python
# Check cache state
status = agent.get_status()
bw = status['web']['bandwidth_stats']
print(f"Cache available: {bw.get('total_mb', 0)} MB")
```

---

## Next Steps

### Immediate (Week 1)

1. ✅ Install requirements
2. ✅ Run first query
3. ✅ Verify knowledge base loads
4. ✅ Check web access (if not OFFLINE)

### Short-term (Week 2-3)

1. ⏳ Prepare fine-tuning dataset
2. ⏳ Run LoRA fine-tuning on laptop
3. ⏳ Test inference with trained model
4. ⏳ Add custom knowledge for your specific analysis

### Medium-term (Week 4-8)

1. ⏳ Deploy to target device (Pi3/Pi5/Laptop)
2. ⏳ Run 24/7 in production
3. ⏳ Integrate with existing game prediction models
4. ⏳ Add continuous learning loop

### Long-term

1. ⏳ Multi-device cluster (multiple Pi5s)
2. ⏳ Advanced RAG (retrieval-augmented generation)
3. ⏳ Custom fine-tuning for specific teams
4. ⏳ Autonomous trading/betting system

---

## Cost Analysis

### Hardware One-Time Cost
- **Pi3**: $35 (8GB SD + case)
- **Pi5**: $80 (8GB RAM + 64GB SD + case)
- **Laptop**: $0-200 (repurposed existing)
- **Desktop**: $200-500

### Operating Costs (Monthly)
- **Pi3/Pi5/Laptop**: $2-10 (electricity only)
  - Pi3: ~5W × 730h/month ÷ 1000 × $0.12/kWh = $0.44
  - Pi5: ~8W × 730h/month ÷ 1000 × $0.12/kWh = $0.70
  - Laptop: ~15W × 730h/month ÷ 1000 × $0.12/kWh = $1.31

### Cloud Comparison
- **Claude API**: ~$300-500/month (continuous inference)
- **OpenAI API**: ~$200-400/month
- **Local LLM**: $2-10/month
- **Savings**: 80-95% cost reduction ✅

---

## License & Attribution

- MLB data: MLB StatsAPI (free public API)
- Sabermetrics knowledge: FanGraphs, Baseball Reference
- Model quantization: GGML framework
- LoRA fine-tuning: Meta/Hugging Face

---

## Support & Community

For issues or questions:

1. Check troubleshooting section above
2. Review edge_device_config.py for device settings
3. Check logs/ directory for detailed error logs
4. Review EDGE_ARCHITECTURE.md for design details

---

*Last updated: December 2024*
*Target deployments: Raspberry Pi 3/5, older laptops*
*Operating cost: $2-10/month*
