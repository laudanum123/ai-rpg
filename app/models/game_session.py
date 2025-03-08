from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import uuid
import json
from datetime import datetime

@dataclass
class GameSession:
    character_id: str
    game_world: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    history: List[Dict] = field(default_factory=list)
    current_location: Dict = field(default_factory=dict)
    npcs: Dict[str, Dict] = field(default_factory=dict)
    locations: Dict[str, Dict] = field(default_factory=dict)
    plot_hooks: List[Dict] = field(default_factory=list)
    active_quests: List[Dict] = field(default_factory=list)
    completed_quests: List[Dict] = field(default_factory=list)
    in_combat: bool = False
    combat_state: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        """Convert session to dictionary."""
        return {
            "id": self.id,
            "character_id": self.character_id,
            "game_world": self.game_world,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "current_location": self.current_location,
            "in_combat": self.in_combat,
            "active_quests": self.active_quests,
            "completed_quests": len(self.completed_quests)
        }
    
    def to_json(self) -> str:
        """Convert session to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'GameSession':
        """Create session from dictionary."""
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'GameSession':
        """Create session from JSON string."""
        return cls.from_dict(json.loads(json_str))
    
    def add_message_to_history(self, role: str, content: str) -> None:
        """Add message to conversation history."""
        self.history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        self.updated_at = datetime.now().isoformat()
    
    def add_npc(self, npc_id: str, npc_data: Dict) -> None:
        """Add or update an NPC."""
        self.npcs[npc_id] = npc_data
        self.updated_at = datetime.now().isoformat()
    
    def get_npc(self, npc_id: str) -> Optional[Dict]:
        """Get NPC data by ID."""
        return self.npcs.get(npc_id)
    
    def add_location(self, location_id: str, location_data: Dict) -> None:
        """Add or update a location."""
        self.locations[location_id] = location_data
        self.updated_at = datetime.now().isoformat()
    
    def get_location(self, location_id: str) -> Optional[Dict]:
        """Get location data by ID."""
        return self.locations.get(location_id)
    
    def set_current_location(self, location_data: Dict) -> None:
        """Set the current location."""
        self.current_location = location_data
        self.updated_at = datetime.now().isoformat()
    
    def add_plot_hook(self, plot_hook: Dict) -> None:
        """Add a plot hook."""
        self.plot_hooks.append(plot_hook)
        self.updated_at = datetime.now().isoformat()
    
    def start_combat(self, enemies: List[Dict]) -> None:
        """Start a combat encounter."""
        self.in_combat = True
        self.combat_state = {
            "round": 1,
            "enemies": enemies,
            "turn_order": [],  # Will be populated with character and enemies
            "current_turn": 0,
            "log": []
        }
        self.updated_at = datetime.now().isoformat()
    
    def end_combat(self) -> None:
        """End the current combat encounter."""
        self.in_combat = False
        combat_log = self.combat_state.get("log", []) if self.combat_state else []
        
        # Add combat summary to history
        if combat_log:
            summary = f"Combat ended after {self.combat_state.get('round', 0)} rounds."
            self.add_message_to_history("system", summary)
        
        self.combat_state = None
        self.updated_at = datetime.now().isoformat()
    
    def next_combat_round(self) -> None:
        """Advance to the next combat round."""
        if self.combat_state:
            self.combat_state["round"] += 1
            self.combat_state["current_turn"] = 0
            self.updated_at = datetime.now().isoformat()
    
    def add_combat_log(self, message: str) -> None:
        """Add entry to combat log."""
        if self.combat_state:
            self.combat_state["log"].append({
                "round": self.combat_state["round"],
                "message": message,
                "timestamp": datetime.now().isoformat()
            })
            self.updated_at = datetime.now().isoformat()
    
    def add_quest(self, quest: Dict) -> None:
        """Add a new quest to active quests."""
        self.active_quests.append(quest)
        self.updated_at = datetime.now().isoformat()
    
    def complete_quest(self, quest_id: str) -> Optional[Dict]:
        """Mark a quest as completed."""
        for i, quest in enumerate(self.active_quests):
            if quest.get("id") == quest_id:
                completed_quest = self.active_quests.pop(i)
                self.completed_quests.append(completed_quest)
                self.updated_at = datetime.now().isoformat()
                return completed_quest
        return None 