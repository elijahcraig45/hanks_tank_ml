"""
Baseball Domain Knowledge System with Local Fine-tuning
Designed for edge devices - zero external dependencies after setup
"""

import json
import sqlite3
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import hashlib


@dataclass
class KnowledgeEntry:
    """Single piece of baseball knowledge"""
    id: str
    category: str  # "rules", "sabermetrics", "strategy", "history", etc.
    topic: str
    content: str
    source: str  # Where it came from
    importance: int  # 1-10 scale
    created_at: str
    embedding_vector: Optional[List[float]] = None


class LocalBaseballKnowledgeBase:
    """
    Lightweight baseball knowledge system optimized for edge devices
    Uses SQLite for persistent storage, supports semantic search with ONNX embeddings
    """
    
    def __init__(self, data_dir: Path = Path("./data")):
        """Initialize knowledge base with SQLite backend"""
        
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        self.db_path = self.data_dir / "baseball_knowledge.db"
        self.embeddings_path = self.data_dir / "embeddings"
        self.embeddings_path.mkdir(exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # Load foundational knowledge if empty
        if self._count_entries() == 0:
            self._load_foundational_knowledge()
    
    def _init_database(self):
        """Create SQLite schema for knowledge base"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS knowledge_entries (
            id TEXT PRIMARY KEY,
            category TEXT NOT NULL,
            topic TEXT NOT NULL,
            content TEXT NOT NULL,
            source TEXT NOT NULL,
            importance INTEGER,
            created_at TEXT,
            embedding_id TEXT
        )
        """)
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS embedding_vectors (
            id TEXT PRIMARY KEY,
            entry_id TEXT NOT NULL,
            vector BLOB,
            created_at TEXT,
            FOREIGN KEY(entry_id) REFERENCES knowledge_entries(id)
        )
        """)
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS fine_tuning_examples (
            id TEXT PRIMARY KEY,
            category TEXT,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            related_entries TEXT,
            created_at TEXT
        )
        """)
        
        conn.commit()
        conn.close()
    
    def _count_entries(self) -> int:
        """Count total knowledge entries"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM knowledge_entries")
        count = cursor.fetchone()[0]
        conn.close()
        return count
    
    def add_entry(self, entry: KnowledgeEntry) -> bool:
        """Add single knowledge entry"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
            INSERT OR REPLACE INTO knowledge_entries 
            (id, category, topic, content, source, importance, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                entry.id,
                entry.category,
                entry.topic,
                entry.content,
                entry.source,
                entry.importance,
                entry.created_at
            ))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error adding entry: {e}")
            return False
    
    def add_fine_tuning_example(self, category: str, question: str, 
                                answer: str, related_topics: List[str]) -> bool:
        """Add Q&A pair for fine-tuning dataset"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            example_id = hashlib.md5(
                f"{question}{answer}".encode()
            ).hexdigest()
            
            cursor.execute("""
            INSERT OR REPLACE INTO fine_tuning_examples
            (id, category, question, answer, related_entries, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """, (
                example_id,
                category,
                question,
                answer,
                json.dumps(related_topics),
                datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error adding fine-tuning example: {e}")
            return False
    
    def search_local(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Search knowledge base using simple text matching
        For edge devices without ONNX runtime initially
        """
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT id, topic, content FROM knowledge_entries")
        all_entries = cursor.fetchall()
        conn.close()
        
        # Simple relevance scoring
        scored_entries = []
        for entry_id, topic, content in all_entries:
            content_lower = (topic + " " + content).lower()
            score = sum(content_lower.count(word) for word in query_words)
            
            if score > 0:
                scored_entries.append((entry_id, float(score)))
        
        # Sort by score and return top_k
        scored_entries.sort(key=lambda x: x[1], reverse=True)
        return scored_entries[:top_k]
    
    def get_entry(self, entry_id: str) -> Optional[Dict]:
        """Get full knowledge entry"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
        SELECT id, category, topic, content, source, importance, created_at
        FROM knowledge_entries WHERE id = ?
        """, (entry_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                "id": row[0],
                "category": row[1],
                "topic": row[2],
                "content": row[3],
                "source": row[4],
                "importance": row[5],
                "created_at": row[6],
            }
        return None
    
    def get_fine_tuning_dataset(self) -> List[Dict]:
        """Get all fine-tuning examples for training"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
        SELECT category, question, answer, related_entries
        FROM fine_tuning_examples
        ORDER BY created_at DESC
        """)
        
        examples = []
        for row in cursor.fetchall():
            examples.append({
                "category": row[0],
                "question": row[1],
                "answer": row[2],
                "related_topics": json.loads(row[3]),
            })
        
        conn.close()
        return examples
    
    def export_for_training(self, output_path: Path) -> bool:
        """Export knowledge base and fine-tuning examples as JSONL for training"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get all entries
            cursor.execute("SELECT * FROM knowledge_entries")
            entries = cursor.fetchall()
            
            # Get all fine-tuning examples
            cursor.execute("SELECT * FROM fine_tuning_examples")
            examples = cursor.fetchall()
            
            conn.close()
            
            # Write as JSONL (one JSON object per line)
            with open(output_path, 'w') as f:
                # Write knowledge entries as system prompts
                for entry in entries:
                    obj = {
                        "type": "knowledge",
                        "category": entry[1],
                        "topic": entry[2],
                        "content": entry[3],
                        "importance": entry[5],
                    }
                    f.write(json.dumps(obj) + "\n")
                
                # Write fine-tuning examples as conversation turns
                for example in examples:
                    obj = {
                        "type": "fine_tune_example",
                        "category": example[1],
                        "question": example[2],
                        "answer": example[3],
                    }
                    f.write(json.dumps(obj) + "\n")
            
            return True
        except Exception as e:
            print(f"Error exporting training data: {e}")
            return False
    
    def _load_foundational_knowledge(self):
        """Load initial baseball knowledge corpus"""
        
        foundational_entries = [
            KnowledgeEntry(
                id="mlb_rules_core",
                category="rules",
                topic="MLB Core Rulebook",
                content="""
                Key MLB Rules:
                - 9 innings standard game (extra innings if tied after 9)
                - 27 outs per game (3 per inning x 9)
                - Batter out on: 3 strikes, 4 balls = walk, caught fly, tag out
                - Pitcher must throw over half of plate (strike zone) or ball is called
                - Home run on ball clearing fence in fair territory
                - Must touch all bases to score run (1st, 2nd, 3rd, home)
                """,
                source="MLB Official Rulebook",
                importance=10,
                created_at=datetime.now().isoformat(),
            ),
            
            KnowledgeEntry(
                id="batting_sabermetrics",
                category="sabermetrics",
                topic="Batting Statistics",
                content="""
                Key Batting Metrics:
                - BA (Batting Average) = H / AB (hits per at-bat) - simple but flawed
                - OBP (On-Base Percentage) = (H + BB + HBP) / PA - better than BA
                - SLG (Slugging Percentage) = TB / AB (total bases per AB)
                - OPS = OBP + SLG - combines getting on base and power
                - wOBA (weighted On-Base Average) - advanced version of OBP
                - Isolated Power (ISO) = SLG - BA (pure power measure)
                - WHIPs, exit velo, barrel rate - modern metrics
                """,
                source="Fangraphs Advanced Stats",
                importance=9,
                created_at=datetime.now().isoformat(),
            ),
            
            KnowledgeEntry(
                id="pitching_fundamentals",
                category="sabermetrics",
                topic="Pitching Statistics",
                content="""
                Key Pitching Metrics:
                - ERA (Earned Run Average) = (ER x 9) / IP - runs per 9 innings
                - FIP (Fielding Independent Pitching) - ERA based only on pitcher control
                - K/9 (Strikeout Rate) - strikeouts per 9 innings
                - BB/9 (Walk Rate) - walks per 9 innings
                - LOB% (Left On Base %) - % of baserunners pitcher strands
                - WHIP = (W + H) / IP - hits + walks per inning
                - Pitch velocity, spin rate, release angle - modern technology data
                """,
                source="Fangraphs Advanced Stats",
                importance=9,
                created_at=datetime.now().isoformat(),
            ),
            
            KnowledgeEntry(
                id="park_factors",
                category="strategy",
                topic="Park Effects on Outcomes",
                content="""
                Park Factor Impact:
                - Fenway Park (Boston) - Green Monster favors LHH power
                - Coors Field (Denver) - High altitude boosts hitting, suppresses pitching
                - Petco Park (San Diego) - Large dimensions suppress home runs
                - Yankee Stadium (NY) - Short porch in right field (315 ft)
                - Wind patterns affect outcomes (especially in Chicago, San Francisco)
                - Humidity affects flight distance (dry = carries farther)
                - Day vs night games: performance varies by player preference
                """,
                source="Baseball Reference Park Factors",
                importance=8,
                created_at=datetime.now().isoformat(),
            ),
            
            KnowledgeEntry(
                id="pitcher_fatigue",
                category="strategy",
                topic="Pitcher Workload and Rest",
                content="""
                Pitcher Fatigue Patterns:
                - Optimal rest between starts: 4-5 days
                - Less than 3 days rest = significantly higher ERA
                - 2+ months without injury = fresher closer to season end
                - High pitch count (>110 pitches) = increased fatigue
                - Bullpen usage the day before = pitcher may not be available
                - Back-to-back appearances by reliever = reduced effectiveness
                - Early season: lower velocity/control vs mid-season peak
                """,
                source="Pitcher Rest Analysis",
                importance=9,
                created_at=datetime.now().isoformat(),
            ),
            
            KnowledgeEntry(
                id="weather_effects",
                category="strategy",
                topic="Weather Impact on Game Outcomes",
                content="""
                Weather Effects on Baseball:
                - Temperature: +5°F = ~5 feet more distance for fly balls
                - Wind: Headwind suppresses HR distance, tailwind increases
                - Humidity: Dry air carries ball ~30 ft farther than humid
                - Rain: Game likely cancelled, muddy field changes play
                - Dew on grass: reduces ball movement off bat (harder hit)
                - Barometric pressure: Low pressure = ball carries less
                - Fog/visibility: Can affect outfielder tracking
                """,
                source="Advanced Baseball Analytics",
                importance=8,
                created_at=datetime.now().isoformat(),
            ),
            
            KnowledgeEntry(
                id="streaks_momentum",
                category="strategy",
                topic="Win Streaks and Momentum",
                content="""
                Streak Analysis:
                - Win streaks (3+) indicate playing well AND variance
                - 5+ game win streaks often regress next 5 games (mean reversion)
                - Hot bats often cool within 2-3 weeks (statistical noise)
                - Momentum is real but overestimated by betters
                - Recent games (7-10) more predictive than season average
                - Division games have different dynamics than regular season
                - Home field advantage: +3-4% win rate on average
                """,
                source="Moneyball Analytics",
                importance=7,
                created_at=datetime.now().isoformat(),
            ),
            
            KnowledgeEntry(
                id="injury_impact",
                category="strategy",
                topic="Injury Impact on Team Performance",
                content="""
                Injury Effects:
                - Star player (2+ WAR) out: ~2% team win rate reduction
                - Multiple injuries to same position: bigger impact
                - Bullpen injuries: higher variance, more blowouts
                - Pitcher injury: more severe than position player
                - Return from injury: 1-2 games reduced performance (rustiness)
                - IL stint < 15 days: likely minor injury, quick recovery
                - IL stint 30+ days: significant injury or surgery
                - Backup callups: usually -15% performance vs starter
                """,
                source="Injury Impact Analysis",
                importance=9,
                created_at=datetime.now().isoformat(),
            ),
        ]
        
        for entry in foundational_entries:
            self.add_entry(entry)
        
        # Add initial fine-tuning examples
        self.add_fine_tuning_example(
            category="pitcher_analysis",
            question="How does rest affect a pitcher's ERA?",
            answer="Pitchers with 4-5 days rest have optimal ERA. Less than 3 days rest significantly increases ERA. This is one of the strongest predictive factors.",
            related_topics=["pitcher_fatigue", "pitching_fundamentals"]
        )
        
        self.add_fine_tuning_example(
            category="park_effects",
            question="Which ballpark favors hitters most?",
            answer="Coors Field in Denver has the highest park factor for hits due to high altitude. Petco Park in San Diego suppresses home runs due to large dimensions.",
            related_topics=["park_factors", "batting_sabermetrics"]
        )
        
        print(f"Initialized knowledge base with {self._count_entries()} entries")


class LocalFineTuningPipeline:
    """
    Fine-tuning pipeline optimized for edge devices
    Works with LoRA (Low-Rank Adaptation) for minimal memory
    """
    
    def __init__(self, knowledge_base: LocalBaseballKnowledgeBase):
        self.kb = knowledge_base
        self.training_data_path = knowledge_base.data_dir / "training_data.jsonl"
    
    def prepare_fine_tuning_data(self, output_path: Optional[Path] = None) -> Path:
        """Prepare dataset for LoRA fine-tuning"""
        
        if output_path is None:
            output_path = self.training_data_path
        
        # Export knowledge base to JSONL
        self.kb.export_for_training(output_path)
        
        print(f"Prepared fine-tuning data at {output_path}")
        return output_path
    
    def get_fine_tuning_config(self, device_profile: str) -> Dict:
        """Get LoRA config optimized for device"""
        
        configs = {
            "pi3": {
                "r": 4,  # LoRA rank - smaller for Pi3
                "lora_alpha": 8,
                "lora_dropout": 0.1,
                "learning_rate": 1e-4,
                "num_epochs": 1,
                "batch_size": 1,
                "warmup_steps": 50,
            },
            "pi5": {
                "r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "learning_rate": 2e-4,
                "num_epochs": 2,
                "batch_size": 2,
                "warmup_steps": 100,
            },
            "laptop": {
                "r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "learning_rate": 5e-4,
                "num_epochs": 3,
                "batch_size": 4,
                "warmup_steps": 200,
            },
        }
        
        return configs.get(device_profile, configs["laptop"])


if __name__ == "__main__":
    # Test knowledge base
    kb = LocalBaseballKnowledgeBase()
    
    print("\n=== KNOWLEDGE BASE TEST ===\n")
    print(f"Total entries: {kb._count_entries()}")
    
    # Test search
    results = kb.search_local("pitcher rest")
    print(f"\nSearch results for 'pitcher rest':")
    for entry_id, score in results:
        entry = kb.get_entry(entry_id)
        if entry:
            print(f"  - {entry['topic']} (score: {score})")
    
    # Test fine-tuning dataset
    ft_examples = kb.get_fine_tuning_dataset()
    print(f"\nFine-tuning examples: {len(ft_examples)}")
    
    # Test export
    kb.export_for_training(Path("./test_training_data.jsonl"))
    print("\nExported training data to test_training_data.jsonl")
