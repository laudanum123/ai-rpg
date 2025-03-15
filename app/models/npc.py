import json
import random
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class NPC:
    name: str
    type: str  # e.g., "shopkeeper", "quest_giver", "enemy"
    description: str
    health: int = 50
    max_health: int = 50
    strength: int = 8
    dexterity: int = 8
    constitution: int = 8
    intelligence: int = 8
    wisdom: int = 8
    charisma: int = 8
    inventory: List[Dict] = field(default_factory=list)
    abilities: List[Dict] = field(default_factory=list)
    dialogue: Dict[str, List[str]] = field(default_factory=dict)
    hostile: bool = False
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> Dict:
        """Convert NPC to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "health": self.health,
            "max_health": self.max_health,
            "strength": self.strength,
            "dexterity": self.dexterity,
            "constitution": self.constitution,
            "intelligence": self.intelligence,
            "wisdom": self.wisdom,
            "charisma": self.charisma,
            "inventory": self.inventory,
            "abilities": self.abilities,
            "hostile": self.hostile,
        }

    def to_json(self) -> str:
        """Convert NPC to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict) -> "NPC":
        """Create NPC from dictionary."""
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> "NPC":
        """Create NPC from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def get_ability_modifier(self, ability: str) -> int:
        """Calculate ability modifier."""
        ability_score = getattr(self, ability.lower(), 10)
        return (ability_score - 10) // 2

    def roll_attack(self, weapon_bonus: int = 0) -> int:
        """Roll an attack (d20 + strength mod + weapon bonus)."""
        str_mod = self.get_ability_modifier("strength")
        return random.randint(1, 20) + str_mod + weapon_bonus

    def take_damage(self, amount: int) -> None:
        """Take damage and update health."""
        self.health = max(0, self.health - amount)

    def heal(self, amount: int) -> None:
        """Heal NPC by amount."""
        self.health = min(self.max_health, self.health + amount)

    def is_alive(self) -> bool:
        """Check if NPC is alive."""
        return self.health > 0

    def get_random_dialogue(self, dialogue_type: str = "greeting") -> Optional[str]:
        """Get a random dialogue line of specified type."""
        if dialogue_type in self.dialogue and self.dialogue[dialogue_type]:
            return random.choice(self.dialogue[dialogue_type])
        return None

    def add_dialogue(self, dialogue_type: str, line: str) -> None:
        """Add a dialogue line to the NPC."""
        if dialogue_type not in self.dialogue:
            self.dialogue[dialogue_type] = []
        self.dialogue[dialogue_type].append(line)

    def make_hostile(self) -> None:
        """Make the NPC hostile."""
        self.hostile = True

    def make_friendly(self) -> None:
        """Make the NPC friendly."""
        self.hostile = False
