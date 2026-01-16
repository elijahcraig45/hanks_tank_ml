@echo off
REM Edge Agent Setup Script for Windows
REM One-command initialization for laptops
REM Usage: setup_edge_agent.bat [device_profile] [mode]

setlocal enabledelayedexpansion

echo.
echo ============================================================
echo  EDGE AGENT SETUP
echo  Local MLB AI for Windows Laptop
echo ============================================================
echo.

REM Get parameters or use defaults
set DEVICE_PROFILE=%1
if "!DEVICE_PROFILE!"=="" set DEVICE_PROFILE=auto

set OPERATION_MODE=%2
if "!OPERATION_MODE!"=="" set OPERATION_MODE=offline

echo [1/5] Detecting hardware...

REM Auto-detect device capability
if "!DEVICE_PROFILE!"=="auto" (
    REM Get available RAM
    for /f "tokens=2" %%A in ('tasklist /v /fo table /nh ^| find /c /v ""') do (
        set DEVICE_PROFILE=laptop
    )
    echo  [✓] Detected: Windows Laptop
) else (
    echo  [✓] Using profile: !DEVICE_PROFILE!
)

echo.
echo [2/5] Installing Python dependencies...

REM Check if venv exists
if not exist ".venv" (
    echo  Creating virtual environment...
    python -m venv .venv
)

REM Activate venv
call .venv\Scripts\activate.bat

REM Install dependencies
echo  Installing core dependencies...
pip install --upgrade pip setuptools wheel >nul 2>&1
pip install -r agent\requirements-edge.txt >nul 2>&1
echo  [✓] Dependencies installed

echo.
echo [3/5] Initializing local data...

REM Create directories
if not exist "data\embeddings" mkdir data\embeddings
if not exist "logs\queries" mkdir logs\queries
if not exist "logs\decisions" mkdir logs\decisions

echo  [✓] Created data directories

REM Initialize knowledge base
echo  Initializing knowledge base...
python -c "from agent.knowledge.local_knowledge_system import LocalBaseballKnowledgeBase; kb = LocalBaseballKnowledgeBase(); print(f'  [OK] Knowledge base initialized with {kb._count_entries()} entries')" >nul 2>&1
echo  [✓] Knowledge base initialized

echo.
echo [4/5] Testing agent...

REM Run test
python << PYTHON_EOF
from agent.edge_agent_orchestrator import EdgeAgentOrchestrator, AgentMode
from agent.config.edge_device_config import EdgeAgentConfig

config = EdgeAgentConfig("!DEVICE_PROFILE!")
config.print_summary()

print("\nTesting agent initialization...")
agent = EdgeAgentOrchestrator(device_profile="!DEVICE_PROFILE!", mode=AgentMode.!OPERATION_MODE!.upper())
print(f"[OK] Agent initialized in {agent.mode.value} mode")
PYTHON_EOF

echo  [✓] Agent test successful

echo.
echo [5/5] Creating quick commands...

REM Create batch script
(
  echo @echo off
  echo call .venv\Scripts\activate.bat
  echo python agent\edge_agent_orchestrator.py
  echo pause
) > start_agent.bat

REM Create Python quick test
(
  echo #!/usr/bin/env python3
  echo """Quick test of edge agent"""
  echo.
  echo from agent.edge_agent_orchestrator import EdgeAgentOrchestrator, AgentMode
  echo.
  echo print("Initializing edge agent...")
  echo agent = EdgeAgentOrchestrator(mode=AgentMode.OFFLINE^)
  echo agent.print_summary()
  echo.
  echo print("\nTesting query...")
  echo result = agent.query("How does pitcher rest affect winning?"^)
  echo print(f"  Decision: {result['decision']}"^)
  echo print(f"  Confidence: {result['confidence']:.0%%}"^)
  echo print(f"  Knowledge used: {len(result['knowledge_used'])} sources"^)
  echo.
  echo print("\n[OK] All tests passed!"^)
) > quick_test.py

echo  [✓] Created start_agent.bat
echo  [✓] Created quick_test.py

echo.
echo ============================================================
echo  SETUP COMPLETE!
echo ============================================================
echo.
echo Device:   !DEVICE_PROFILE!
echo Mode:     !OPERATION_MODE!
echo.
echo Quick start commands:
echo.
echo  [Test agent]:
echo    python quick_test.py
echo.
echo  [Start interactive agent]:
echo    start_agent.bat
echo.
echo  [Or run directly]:
echo    python -c "from agent.edge_agent_orchestrator import EdgeAgentOrchestrator; agent = EdgeAgentOrchestrator(); agent.query('Your query here')"
echo.
echo Documentation:
echo  - EDGE_AGENT_GUIDE.md (detailed guide^)
echo  - EDGE_ARCHITECTURE.md (architecture^)
echo.
echo Virtual environment activated. Type 'deactivate' to exit.
echo.

call .venv\Scripts\activate.bat
