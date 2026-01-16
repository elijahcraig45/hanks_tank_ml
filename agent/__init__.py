"""
ML Agent System Package
Autonomous model management for baseball predictions
"""

__version__ = "0.1.0"
__author__ = "ML Engineering Team"

from .core.agent_manager import ModelManagementAgent, AgentDecision, AgentRole
from .config.agent_config import config
from .knowledge.knowledge_base import BaseballKnowledgeBase, KnowledgeEntry

__all__ = [
    "ModelManagementAgent",
    "AgentDecision",
    "AgentRole",
    "config",
    "BaseballKnowledgeBase",
    "KnowledgeEntry",
]
