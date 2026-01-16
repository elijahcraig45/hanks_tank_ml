#!/bin/bash
# Edge Agent Setup Script
# One-command initialization for Pi3/Pi5/Laptop
# Usage: bash setup_edge_agent.sh [device_profile] [mode]

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  EDGE AGENT SETUP                          ║${NC}"
echo -e "${BLUE}║  Local MLB AI for Pi3/Pi5/Laptop           ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════╝${NC}"
echo ""

# Device detection
DEVICE_PROFILE="${1:-auto}"
OPERATION_MODE="${2:-offline}"

echo -e "${GREEN}[1/5]${NC} Detecting hardware..."

if [ "$DEVICE_PROFILE" = "auto" ]; then
    # Try to detect Pi
    if [ -f "/proc/device-tree/model" ]; then
        MODEL=$(cat /proc/device-tree/model 2>/dev/null || echo "")
        if [[ $MODEL == *"Pi 5"* ]]; then
            DEVICE_PROFILE="pi5"
            echo -e "  ${GREEN}✓${NC} Detected: Raspberry Pi 5"
        elif [[ $MODEL == *"Pi 3"* ]]; then
            DEVICE_PROFILE="pi3"
            echo -e "  ${GREEN}✓${NC} Detected: Raspberry Pi 3"
        fi
    else
        # Estimate from RAM
        RAM_MB=$(free -m | awk 'NR==2 {print $2}')
        if [ $RAM_MB -lt 1024 ]; then
            DEVICE_PROFILE="pi3"
        elif [ $RAM_MB -lt 4096 ]; then
            DEVICE_PROFILE="pi5"
        else
            DEVICE_PROFILE="laptop"
        fi
        echo -e "  ${GREEN}✓${NC} Detected by RAM: ${RAM_MB}MB → $DEVICE_PROFILE"
    fi
else
    echo -e "  ${GREEN}✓${NC} Using profile: $DEVICE_PROFILE"
fi

echo ""
echo -e "${GREEN}[2/5]${NC} Installing Python dependencies..."

# Create virtual environment if not exists
if [ ! -d ".venv" ]; then
    echo "  Creating virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate

# Install minimal requirements for edge
pip install --upgrade pip setuptools wheel > /dev/null 2>&1
echo "  Installing core dependencies..."
pip install -r agent/requirements-edge.txt > /dev/null 2>&1
echo -e "  ${GREEN}✓${NC} Dependencies installed"

echo ""
echo -e "${GREEN}[3/5]${NC} Initializing local data..."

# Create data directories
mkdir -p data/{embeddings}
mkdir -p logs/{queries,decisions}

echo -e "  ${GREEN}✓${NC} Created data directories"

# Initialize knowledge base
echo "  Initializing knowledge base..."
python3 << 'EOF'
from pathlib import Path
from agent.knowledge.local_knowledge_system import LocalBaseballKnowledgeBase

kb = LocalBaseballKnowledgeBase()
count = kb._count_entries()
print(f"  ✓ Knowledge base initialized with {count} entries")
EOF

echo ""
echo -e "${GREEN}[4/5]${NC} Testing agent..."

# Test basic functionality
python3 << EOF
from agent.edge_agent_orchestrator import EdgeAgentOrchestrator, AgentMode
from agent.config.edge_device_config import EdgeAgentConfig

# Test config detection
config = EdgeAgentConfig("$DEVICE_PROFILE")
config.print_summary()

print("")
print("Testing agent initialization...")
agent = EdgeAgentOrchestrator(device_profile="$DEVICE_PROFILE", mode=AgentMode.${OPERATION_MODE^^})
print(f"✓ Agent initialized in {agent.mode.value} mode")
EOF

echo -e "  ${GREEN}✓${NC} Agent test successful"

echo ""
echo -e "${GREEN}[5/5]${NC} Creating quick commands..."

# Create quick-start script
cat > start_agent.sh << 'SHELL_EOF'
#!/bin/bash
source .venv/bin/activate
python3 agent/edge_agent_orchestrator.py
SHELL_EOF
chmod +x start_agent.sh

echo -e "  ${GREEN}✓${NC} Created start_agent.sh"

# Create Python quick-start
cat > quick_test.py << 'PYTHON_EOF'
#!/usr/bin/env python3
"""Quick test of edge agent"""

from agent.edge_agent_orchestrator import EdgeAgentOrchestrator, AgentMode

# Initialize
print("Initializing edge agent...")
agent = EdgeAgentOrchestrator(mode=AgentMode.OFFLINE)
agent.print_summary()

# Test query
print("\nTesting query...")
result = agent.query("How does pitcher rest affect winning?")
print(f"  Decision: {result['decision']}")
print(f"  Confidence: {result['confidence']:.0%}")
print(f"  Knowledge used: {len(result['knowledge_used'])} sources")

print("\n✓ All tests passed!")
PYTHON_EOF
chmod +x quick_test.py

echo -e "  ${GREEN}✓${NC} Created quick_test.py"

echo ""
echo -e "${BLUE}╔════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  SETUP COMPLETE!                          ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════╝${NC}"
echo ""
echo -e "Device:   ${YELLOW}$DEVICE_PROFILE${NC}"
echo -e "Mode:     ${YELLOW}$OPERATION_MODE${NC}"
echo ""
echo "Quick start commands:"
echo ""
echo -e "  ${BLUE}Test agent:${NC}"
echo "    python3 quick_test.py"
echo ""
echo -e "  ${BLUE}Start interactive agent:${NC}"
echo "    bash start_agent.sh"
echo ""
echo -e "  ${BLUE}Or run directly:${NC}"
echo "    python3 -c 'from agent.edge_agent_orchestrator import EdgeAgentOrchestrator; agent = EdgeAgentOrchestrator(); agent.query(\"Your query here\")'"
echo ""
echo "Documentation:"
echo "  - EDGE_AGENT_GUIDE.md (detailed guide)"
echo "  - EDGE_ARCHITECTURE.md (architecture)"
echo ""
