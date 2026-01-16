"""
Baseball Knowledge Base with RAG (Retrieval Augmented Generation)
Embeds domain knowledge for agent decision-making
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeEntry:
    """Single entry in knowledge base"""
    id: str
    title: str
    category: str
    content: str
    source: str
    date: str
    relevance_tags: List[str]


class BaseballKnowledgeBase:
    """Embeddings-based knowledge base for baseball domain knowledge"""
    
    def __init__(self, vector_db_path: str = "agent/knowledge/vectors.db"):
        """Initialize knowledge base"""
        self.vector_db_path = vector_db_path
        self.knowledge_entries: List[KnowledgeEntry] = []
        self.embeddings: Dict[str, np.ndarray] = {}
        
        # Create directory if needed
        os.makedirs(os.path.dirname(vector_db_path), exist_ok=True)
        
        self._load_foundational_knowledge()
        logger.info("Initialized Baseball Knowledge Base")
    
    def _load_foundational_knowledge(self):
        """Load key baseball domain knowledge"""
        foundational = [
            {
                "title": "Pitcher Fatigue and Performance",
                "category": "Player Factors",
                "content": """
                Pitcher fatigue significantly impacts performance. Key findings:
                - ERA increases by ~0.5 for pitchers throwing >100 pitches vs <80 pitches
                - Rest days matter: 4+ days rest improves performance vs 3 or fewer
                - Workload over time: pitchers with high innings in previous month show decline
                - Bullpen usage affects starter availability
                Related metrics: pitch count, days rest, innings last 7/14/30 days
                """,
                "relevance_tags": ["pitcher", "fatigue", "performance", "injury_risk"]
            },
            {
                "title": "Park Factors in Run Scoring",
                "category": "Environmental",
                "content": """
                Different parks significantly affect run production:
                - Fenway Park (BOS): +15% runs, small field, high wall (Green Monster)
                - Coors Field (COL): +20% runs, high altitude affects ball flight
                - Comerica Park (DET): -8% runs, large field
                - Rogers Centre (TOR): Dome affects wind, climate control
                Factor formula: (Team Runs at Home / Home Games) / (Team Runs Away / Away Games)
                """,
                "relevance_tags": ["park", "environment", "run_scoring", "home_field"]
            },
            {
                "title": "Batter Hot/Cold Streaks",
                "category": "Player Form",
                "content": """
                Offensive trends show meaningful patterns:
                - 14-day rolling average is predictive of near-term performance
                - Hot streak advantage: +12% win probability in next game
                - Slump (cold streak): -15% win probability
                - Regression to mean: extreme performance tends to normalize in 30 days
                Key stats: recent batting average, ISO (isolated power), strikeout rate
                """,
                "relevance_tags": ["batter", "form", "hot_streak", "momentum"]
            },
            {
                "title": "Strength of Schedule",
                "category": "Opponent Quality",
                "content": """
                Upcoming opponent quality affects win probability:
                - Playing against bottom-10 teams: +4% win probability
                - Playing against top-10 teams: -6% win probability
                - Travel fatigue: opponents arriving same day show -3% performance
                - Home/away splits: teams have avg 54% win% at home vs 46% away
                Measure: opponent win% in current season, recent opponent form
                """,
                "relevance_tags": ["schedule", "opponent", "difficulty"]
            },
            {
                "title": "Weather Effects on Game Outcomes",
                "category": "Environmental",
                "content": """
                Weather measurably impacts baseball:
                - Temperature: warmer temps (+15F) increase fly ball distance ~5-7 feet
                - Wind: 10+ mph tailwind adds 1-2 runs per team
                - Humidity: higher humidity (75%+) slightly reduces offense
                - Rain: light rain affects grip, impacts scoring variance
                Impact: 10-degree temp increase ~2% to home run rate
                """,
                "relevance_tags": ["weather", "environment", "scoring"]
            },
            {
                "title": "Injury Impact on Team Performance",
                "category": "Roster Changes",
                "content": """
                Key player injuries affect win probability:
                - Star pitcher loss (2.0+ WAR): -6% team win probability
                - Star position player loss (3.0+ WAR): -4% team win probability
                - Backup/reserve loss: <1% impact
                - Cumulative injuries: multiple injuries compound effect
                Recovery timeline: avg 30 days for DL stint, varies by injury type
                """,
                "relevance_tags": ["injury", "roster", "availability"]
            },
            {
                "title": "Moneyball Principles in Modern Analytics",
                "category": "Analytics Framework",
                "content": """
                Evidence-based approach to baseball:
                - On-base percentage (OBP) undervalued, stronger than batting average
                - Defensive efficiency varies by park and pitcher
                - Bullpen inconsistency greater than starter variability
                - Draft prospects for potential (age-adjusted) vs immediate production
                Application: focus on high-value underappreciated metrics
                """,
                "relevance_tags": ["analytics", "sabermetrics", "valuation"]
            },
            {
                "title": "Streak Dynamics and Probability",
                "category": "Statistical Patterns",
                "content": """
                Team winning streaks show patterns:
                - 4-game win streak increases momentum, +8% expected next win probability
                - 3-game loss streak triggers lineup/strategy changes
                - Median streak length: 1-2 games, extreme streaks (5+) are statistical noise
                - Streak probability follows binomial distribution with team true talent
                """,
                "relevance_tags": ["streak", "momentum", "probability"]
            }
        ]
        
        for idx, entry_dict in enumerate(foundational):
            entry = KnowledgeEntry(
                id=f"foundation_{idx}",
                title=entry_dict["title"],
                category=entry_dict["category"],
                content=entry_dict["content"],
                source="Foundational Baseball Analytics",
                date="2025-01-01",
                relevance_tags=entry_dict["relevance_tags"]
            )
            self.knowledge_entries.append(entry)
        
        logger.info(f"Loaded {len(foundational)} foundational knowledge entries")
    
    def add_entry(self, title: str, category: str, content: str, 
                  source: str, tags: List[str]):
        """Add new knowledge entry"""
        entry_id = f"entry_{len(self.knowledge_entries)}"
        entry = KnowledgeEntry(
            id=entry_id,
            title=title,
            category=category,
            content=content,
            source=source,
            date=str(np.datetime64('today')),
            relevance_tags=tags
        )
        self.knowledge_entries.append(entry)
        logger.info(f"Added knowledge entry: {title}")
        return entry_id
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search knowledge base by query"""
        # Simple text-based search (can be upgraded to semantic/embedding search)
        results = []
        query_lower = query.lower()
        
        for entry in self.knowledge_entries:
            # Score based on title, tags, and content
            score = 0.0
            
            # Title match (highest weight)
            if query_lower in entry.title.lower():
                score += 3.0
            
            # Tag match
            for tag in entry.relevance_tags:
                if query_lower in tag.lower():
                    score += 2.0
            
            # Content match
            if query_lower in entry.content.lower():
                content_matches = entry.content.lower().count(query_lower)
                score += 0.5 * min(content_matches, 5)
            
            if score > 0:
                results.append({
                    "id": entry.id,
                    "title": entry.title,
                    "category": entry.category,
                    "content": entry.content,
                    "source": entry.source,
                    "score": score,
                    "tags": entry.relevance_tags
                })
        
        # Sort by score and return top_k
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
    
    def get_context_for_decision(self, decision_type: str, context: Dict) -> str:
        """Get relevant knowledge for specific decision type"""
        contexts = {
            "retrain_model": [
                "Streak Dynamics and Probability",
                "Pitcher Fatigue and Performance",
                "Batter Hot/Cold Streaks"
            ],
            "feature_engineering": [
                "Park Factors in Run Scoring",
                "Weather Effects on Game Outcomes",
                "Moneyball Principles in Modern Analytics"
            ],
            "injury_impact": [
                "Injury Impact on Team Performance",
                "Strength of Schedule"
            ],
            "prediction_uncertainty": [
                "Statistical Patterns",
                "Weather Effects on Game Outcomes"
            ]
        }
        
        relevant_topics = contexts.get(decision_type, [])
        knowledge_text = ""
        
        for topic in relevant_topics:
            matching = [e for e in self.knowledge_entries if e.title == topic]
            if matching:
                entry = matching[0]
                knowledge_text += f"\n### {entry.title}\n{entry.content}\n"
        
        return knowledge_text if knowledge_text else "No relevant knowledge found."
    
    def export_knowledge_base(self, filepath: str = "agent/knowledge/kb_export.json"):
        """Export knowledge base for backup/sharing"""
        export_data = {
            "entries": [
                {
                    "id": e.id,
                    "title": e.title,
                    "category": e.category,
                    "content": e.content,
                    "source": e.source,
                    "date": e.date,
                    "tags": e.relevance_tags
                }
                for e in self.knowledge_entries
            ],
            "export_date": str(np.datetime64('today')),
            "total_entries": len(self.knowledge_entries)
        }
        
        with open(filepath, "w") as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported knowledge base to {filepath}")
        return filepath


# Example usage
if __name__ == "__main__":
    kb = BaseballKnowledgeBase()
    
    # Search examples
    search_queries = [
        "pitcher fatigue impact",
        "park factors home runs",
        "weather effect on scoring",
        "injury risk"
    ]
    
    for query in search_queries:
        results = kb.search(query, top_k=3)
        print(f"\nSearch: '{query}'")
        print(f"Found {len(results)} results:")
        for result in results:
            print(f"  - {result['title']} (score: {result['score']:.1f})")
    
    # Add new knowledge
    kb.add_entry(
        title="Recent 2026 Season Observations",
        category="Live Updates",
        content="Update with actual 2026 season data as it becomes available",
        source="2026 Season Tracking",
        tags=["2026", "live_data", "seasonal"]
    )
    
    # Export
    kb.export_knowledge_base()
