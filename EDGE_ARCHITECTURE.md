"""
EDGE-OPTIMIZED ML AGENT ARCHITECTURE
Designed for local-first operation on limited hardware (Pi3-Pi5, older laptops)
Minimal internet, maximum efficiency, fine-tuned for baseball domain
"""

# LEAN AGENT SYSTEM - EDGE ARCHITECTURE

## DESIGN PRINCIPLES

### 1. HARDWARE-AWARE OPERATION
```
Device Profile Selection:
  ├─ Pi3 (512MB RAM)     → TinyLLaMA 1.1B quantized
  ├─ Pi5 (8GB RAM)       → Phi-2 2.7B or Mistral 7B q4
  ├─ Laptop (8GB+ RAM)   → Mistral 7B, local fine-tuning
  └─ Hybrid Cluster      → Distributed inference across devices

Performance Targets:
  • Pi3: 2-3 tokens/sec (inference only)
  • Pi5: 5-8 tokens/sec (with fine-tuning capability)
  • Laptop: 10-20 tokens/sec (full local training)
```

### 2. LOCAL-FIRST OPERATION
```
After Initial Setup:
  ✓ 100% offline operation possible
  ✓ All models local (~4-5GB storage total with quantization)
  ✓ Knowledge base embedded locally
  ✓ No API calls required for core functionality
  
Optional Web Access (Scheduled, Minimal):
  • Weekly: Fetch latest MLB standings/stats
  • Daily: Check injury updates
  • On-demand: Search baseball research (manual triggers only)
```

### 3. KNOWLEDGE-CENTRIC DESIGN
```
Baseball Domain Training:
  1. Curated MLB Rules Corpus (official rulebook)
  2. Sabermetrics Papers (100+ key papers)
  3. Historical Analysis (1990-2025)
  4. Player/Team Data (embeddings)
  5. Park Factors (dynamic calculation)
  
Training Methods:
  • LoRA (Low-Rank Adaptation): 1-2% of model params
  • Knowledge Distillation: Compress to smaller models
  • Embeddings Fine-tuning: Semantic search on domain
  • Rule Engine: Deterministic baseball logic
```

## SYSTEM ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────┐
│              USER INTERFACE LAYER                            │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐   │
│  │ CLI         │  │ Simple Web   │  │ Scheduled Tasks │   │
│  │ (always)    │  │ (optional)   │  │ (cron)          │   │
│  └─────────────┘  └──────────────┘  └─────────────────┘   │
└──────────────┬────────────────────────────────────────────┘
               │
┌──────────────▼────────────────────────────────────────────┐
│        DEVICE-AWARE AGENT ORCHESTRATOR                    │
│  ┌─────────────────────────────────────────────────────┐ │
│  │ • Detect hardware capability on startup             │ │
│  │ • Load appropriate model for device                 │ │
│  │ • Stream inference for memory efficiency            │ │
│  │ • Queue tasks if memory constrained                 │ │
│  │ • Cache recent decisions in SQLite                  │ │
│  └─────────────────────────────────────────────────────┘ │
└──────────────┬────────────────────────────────────────────┘
               │
    ┌──────────┴───────────────────────────┐
    │                                      │
┌───▼─────────────────────┐    ┌──────────▼──────────────────┐
│ LOCAL LLM INFERENCE     │    │ TOOLS & SERVICES            │
│ (Hardware-Optimized)    │    │ ┌────────────────────────┐  │
│ ┌────────────────────┐  │    │ │ BigQuery Connector     │  │
│ │ TinyLLaMA 1.1B q8  │  │    │ │ (cached, episodic)     │  │
│ │ (Pi3)              │  │    │ │                        │  │
│ │ ┌──────────────┐   │  │    │ │ Baseball Stats API     │  │
│ │ │ LoRA weights │   │  │    │ │ (weekly sync)          │  │
│ │ │ (baseball)   │   │  │    │ │                        │  │
│ │ └──────────────┘   │  │    │ │ Local Knowledge DB     │  │
│ │                    │  │    │ │ (embedded, queried)    │  │
│ │ ─────────────────  │  │    │ │                        │  │
│ │ Phi-2 2.7B q4      │  │    │ │ Rule Engine            │  │
│ │ (Pi5 / Laptop)     │  │    │ │ (deterministic logic)  │  │
│ │ ┌──────────────┐   │  │    │ │                        │  │
│ │ │ LoRA weights │   │  │    │ │ Model Management       │  │
│ │ │ (baseball)   │   │  │    │ │ (training, validation) │  │
│ │ └──────────────┘   │  │    │ │                        │  │
│ │                    │  │    │ │ Cache Layer (SQLite)   │  │
│ │ ─────────────────  │  │    │ │                        │  │
│ │ Full models        │  │    │ │ Web Access Controller  │  │
│ │ (Laptop only)      │  │    │ │ (minimal, scheduled)   │  │
│ └────────────────────┘  │    │ └────────────────────────┘  │
└────────────────────────┘    └─────────────────────────────┘
               │                          │
               └──────────────┬───────────┘
                              │
                    ┌─────────▼──────────┐
                    │  LOCAL DATA LAYER   │
                    │ ┌────────────────┐ │
                    │ │ SQLite Cache   │ │
                    │ ├────────────────┤ │
                    │ │ ONNX Embeddings│ │
                    │ ├────────────────┤ │
                    │ │ Knowledge Base │ │
                    │ │ (vectors, text)│ │
                    │ ├────────────────┤ │
                    │ │ Rule Engine    │ │
                    │ ├────────────────┤ │
                    │ │ Model Weights  │ │
                    │ └────────────────┘ │
                    └────────────────────┘
```

## DEVICE PROFILES

### Profile 1: Raspberry Pi 3 (512MB RAM, ~1GHz)
```
Model: TinyLLaMA 1.1B (quantized int8)
Size: 600MB model + 200MB LoRA
Inference: 2-3 tokens/sec
Suitable For:
  ✓ Autonomous scheduled checks
  ✓ Simple queries (cached when possible)
  ✓ Decision logging
  ✗ Real-time complex reasoning
  ✗ Fine-tuning (too slow)

Setup: "pi3_minimal"
  python agent/setup.py --device pi3
```

### Profile 2: Raspberry Pi 5 (8GB RAM, ~3GHz)
```
Model: Phi-2 2.7B or Mistral 7B q4
Size: 2GB model + 500MB LoRA
Inference: 5-8 tokens/sec
Fine-tuning: Possible (8-12 hours for LoRA)
Suitable For:
  ✓ Autonomous operation with reasoning
  ✓ LoRA fine-tuning (~5% params)
  ✓ Complex queries
  ✓ Model experimentation

Setup: "pi5_full"
  python agent/setup.py --device pi5
```

### Profile 3: Older Laptop (4-8GB RAM, i5/i7)
```
Model: Mistral 7B q4 or Llama 2 7B
Size: 4-5GB model + 1GB LoRA
Inference: 10-20 tokens/sec
Fine-tuning: Efficient (4-6 hours for full LoRA)
Suitable For:
  ✓ Development and testing
  ✓ Full LoRA fine-tuning
  ✓ Large-scale analysis
  ✓ Running cluster coordinator

Setup: "laptop_dev"
  python agent/setup.py --device laptop
```

### Profile 4: Modern Laptop (16GB+ RAM, M1/i7+)
```
Model: Any open-source model (7B-13B)
Size: 7-13GB models
Inference: 20-50 tokens/sec
Fine-tuning: Fast (1-2 hours for full LoRA)
Suitable For:
  ✓ Development, testing, training
  ✓ Full model fine-tuning
  ✓ Cluster coordinator
  ✓ Production on single machine

Setup: "laptop_production"
  python agent/setup.py --device laptop_prod
```

## KEY OPTIMIZATIONS

### 1. QUANTIZATION & COMPRESSION
```
Model Sizes (4-Bit Quantization):
  • TinyLLaMA 1.1B → 600MB
  • Phi-2 2.7B → 1.6GB
  • Mistral 7B → 4GB
  • Llama 2 7B → 4GB
  
LoRA Weights (1-2%):
  • Baseball domain fine-tuning: ~100-200MB
  • Can update without reloading base model
  • Enables efficient transfer learning
```

### 2. STREAMING INFERENCE
```
Process Tokens as They Arrive:
  • Don't buffer entire response
  • Stream to cache/output immediately
  • Reduce peak memory usage by 50%+
  • More responsive on Pi3

Example:
  for token in model.generate_streaming(prompt):
      output.write(token)  # Write immediately
      cache.append(token)  # Log incrementally
```

### 3. LOCAL CACHING
```
SQLite Cache (lightweight):
  ├─ Query results (model accuracy, stats)
  ├─ Decision history (with embeddings)
  ├─ API responses (expires daily)
  ├─ Inference cache (exact prompts)
  └─ Performance metrics

Benefits:
  ✓ Avoid re-computation
  ✓ Fast offline access
  ✓ Minimal storage (~50MB)
  ✓ No external DB needed
```

### 4. KNOWLEDGE DISTILLATION
```
Train Smaller Model on Larger Model's Outputs:
  
  Step 1: Use Claude API to generate reasoning
  Step 2: Collect expert examples
  Step 3: Fine-tune local TinyLLaMA on examples
  Step 4: Use local model for routine queries
  
Efficiency:
  • 90% smaller model
  • 80% accuracy of original
  • Can run on Pi3
```

## LOCAL FINE-TUNING PIPELINE

### Data Preparation
```python
# 1. Curate Baseball Knowledge
knowledge_sources = [
    "MLB_Official_Rulebook.pdf",           # Official rules
    "Sabermetrics_101_Papers.txt",         # Key research
    "Historical_Analysis_1990_2025.json",  # Data
    "Park_Factors_Database.csv",           # Park-specific
    "Player_Profiles_Database.json",       # Player data
]

# 2. Convert to Training Format
training_data = [
    {
        "prompt": "What is the infield fly rule?",
        "response": "[detailed official explanation]",
        "source": "MLB_Official_Rulebook.pdf",
        "category": "rules"
    },
    # ... 1000s of examples
]

# 3. Fine-tune with LoRA
model.train_lora(
    training_data,
    epochs=3,
    lr=5e-5,
    rank=16,
    save_path="models/baseball_lora.pt"
)
```

### Training on Limited Hardware
```
Laptop (8GB):
  • Full LoRA: 4-6 hours
  • QLoRA (4-bit): 2-3 hours
  
Pi5 (8GB):
  • QLoRA: 8-12 hours
  • Batch size: 1 (memory constrained)
  
Pi3 (512MB):
  • Not recommended for training
  • Can use knowledge distillation results
```

## WEB ACCESS STRATEGY

### Minimal, Scheduled, Predictable

```python
# 1. WEEKLY: Fetch standings/schedule
schedule.every().sunday.at("02:00").do(
    fetch_standings_and_schedule
)

# 2. DAILY: Injury updates
schedule.every().day.at("08:00").do(
    update_injury_status
)

# 3. ON-DEMAND: Research queries (manual only)
# User explicitly requests: "Search for pitcher fatigue studies"
# Agent fetches specific papers only

# 4. CACHE EVERYTHING
cache = LocalCache(ttl_hours=24*7)  # 1 week for most data
results = cache.get_or_fetch(
    key="mlb_standings_2026",
    fetch_fn=lambda: fetch_standings(),
    ttl_hours=24  # Daily refresh
)
```

### APIs Used (Minimal Set)
```
1. MLB StatsAPI (free, reliable)
   - Games, schedules, standings
   - Player stats, team rosters
   - Called: Weekly for refresh

2. ESPN API (free)
   - Injury reports
   - Latest news headlines
   - Called: Daily for updates

3. Optional: FanGraphs (if free tier)
   - Advanced stats
   - Called: Monthly for refresh

No Anthropic API calls in production after setup!
```

## INITIALIZATION WORKFLOW

```
First Time Setup (30 minutes on laptop, 2 hours on Pi5):

1. User runs: python agent/setup.py --device [profile]
   
2. System detects hardware:
   ├─ RAM available
   ├─ CPU cores
   ├─ Storage
   └─ Network
   
3. Downloads & quantizes model (~30 min on laptop)
   └─ Saves: models/[device]/base_model.gguf (4GB)
   
4. Creates LoRA weights for baseball domain (1-2 hours)
   └─ Saves: models/baseball_lora.gguf (200MB)
   
5. Embeds knowledge base (10 min)
   └─ Saves: knowledge/embeddings.onnx (50MB)
   
6. Syncs initial data from APIs (5 min)
   ├─ MLB standings & schedule
   ├─ Team rosters
   ├─ Recent games
   └─ Saved to: cache/initial_data.db (10MB)
   
7. Ready for offline operation
   └─ Total storage: ~5GB (4GB model + 1GB data)
   └─ Total time: 30 min (laptop) to 4 hours (Pi3)
```

After initialization: **100% works offline** with weekly syncs

## MEMORY MANAGEMENT

### Pi3 (512MB available)
```
TinyLLaMA 1.1B loaded:
├─ Model weights: 400MB (int8 quantized)
├─ LoRA weights: 50MB
├─ Inference buffer: 30MB
├─ Cache/state: 20MB
└─ OS overhead: 12MB
───────────────────
Total: ~512MB (just fits!)

Optimization: Process in chunks
  • Batch size: 1
  • Max context: 512 tokens
  • Streaming output
  • Regular cache clearing
```

### Pi5 (8GB available)
```
Phi-2 2.7B or Mistral 7B q4:
├─ Model weights: 2-4GB (int4 quantized)
├─ LoRA weights: 200MB (when active)
├─ Inference buffer: 200MB
├─ Cache: 500MB
├─ Knowledge DB: 100MB
└─ OS/system: 1GB
───────────────────
Total: 4-6GB (comfortable)

Optimization: Batch size 2-4
  • Max context: 2048 tokens
  • Can run background tasks
  • Can fine-tune (slow but works)
```

## EXAMPLE: RUNNING ON PI3

```bash
# 1. Initial setup (2-3 hours)
python agent/setup.py --device pi3 --offline-only

# 2. Start agent (takes 30 seconds)
python agent/main.py

# 3. Try a query
agent> query "What's the recent performance of Yankees pitchers?"

# Agent response (~30 seconds):
# - Loads TinyLLaMA 1.1B (400MB) ✓
# - Checks local cache (milliseconds)
# - Streams response (token by token)
# - Logs decision to SQLite
# Total time: 15-30 seconds
```

## COST BREAKDOWN

### Hardware Investment
```
Option 1: Raspberry Pi 5 + case + power
  • Raspberry Pi 5 (8GB): $80
  • Case + cooling: $20
  • Power supply: $15
  • SD card (128GB): $20
  • Total: $135 one-time

Option 2: Raspberry Pi 3 + case
  • Raspberry Pi 3 (512MB): $35
  • Case: $10
  • Power supply: $10
  • SD card (32GB): $10
  • Total: $65 one-time

Option 3: Used laptop (eBay)
  • Intel i5 + 8GB RAM: $200-400
  • Total: $200-400 one-time
```

### Monthly Operating Cost
```
Electricity:
  • Pi5: ~3W continuous = $2-3/month
  • Pi3: ~1.5W continuous = $1-2/month
  • Laptop: ~15W continuous = $5-8/month

Internet:
  • Minimal usage (~100MB/week) = included in regular ISP
  • No extra cost

API Costs:
  • Zero after setup (local models only!)
  
Total: $2-10/month (vs $300+ for cloud LLM)
```

## NEXT STEPS FOR IMPLEMENTATION

### Phase 1: Hardware Profiling (This week)
- [ ] Test TinyLLaMA 1.1B on Pi3 simulator
- [ ] Profile memory/CPU for each device
- [ ] Create device detection script
- [ ] Benchmark inference speeds

### Phase 2: LoRA Fine-tuning (Week 2)
- [ ] Curate baseball knowledge corpus
- [ ] Create fine-tuning dataset
- [ ] Implement LoRA training
- [ ] Test on laptop first, then Pi5

### Phase 3: Local-First Design (Week 3)
- [ ] Implement SQLite caching
- [ ] Create offline mode
- [ ] Add knowledge base embeddings
- [ ] Test 48 hours offline operation

### Phase 4: Web Access Layer (Week 4)
- [ ] Implement scheduled API calls
- [ ] Create caching mechanism
- [ ] Build update daemon
- [ ] Test minimal bandwidth operation

---

**This approach gives you a fully autonomous system that costs $2-10/month to run, works on hardware you already have, and requires no API calls after setup. All models, knowledge, and reasoning happen locally.**
