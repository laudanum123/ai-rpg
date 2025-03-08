from dataclasses import dataclass, field
from typing import Dict, List, Optional
import uuid
import json

@dataclass
class Character:
    name: str
    character_class: str
    level: int = 1
    health: int = 100
    max_health: int = 100
    strength: int = 10
    dexterity: int = 10
    constitution: int = 10
    intelligence: int = 10
    wisdom: int = 10
    charisma: int = 10
    inventory: List[Dict] = field(default_factory=list)
    abilities: List[Dict] = field(default_factory=list)
    experience: int = 0
    gold: int = 10
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def to_dict(self) -> Dict:
        """Convert character to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "character_class": self.character_class,
            "level": self.level,
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
            "experience": self.experience,
            "gold": self.gold
        }
    
    def to_json(self) -> str:
        """Convert character to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Character':
        """Create character from dictionary."""
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Character':
        """Create character from JSON string."""
        return cls.from_dict(json.loads(json_str))
    
    def add_item(self, item: Dict) -> None:
        """Add item to inventory."""
        self.inventory.append(item)
    
    def remove_item(self, item_id: str) -> Optional[Dict]:
        """Remove item from inventory by ID."""
        for i, item in enumerate(self.inventory):
            if item.get('id') == item_id:
                return self.inventory.pop(i)
        return None
    
    def get_ability_modifier(self, ability: str) -> int:
        """Calculate ability modifier."""
        ability_score = getattr(self, ability.lower(), 10)
        return (ability_score - 10) // 2
    
    def roll_attack(self, weapon_bonus: int = 0) -> int:
        """Roll an attack (d20 + strength mod + weapon bonus)."""
        import random
        str_mod = self.get_ability_modifier('strength')
        return random.randint(1, 20) + str_mod + weapon_bonus
    
    def take_damage(self, amount: int) -> None:
        """Take damage and update health."""
        self.health = max(0, self.health - amount)
    
    def heal(self, amount: int) -> None:
        """Heal character by amount."""
        self.health = min(self.max_health, self.health + amount)
    
    def is_alive(self) -> bool:
        """Check if character is alive."""
        return self.health > 0
    
    def add_experience(self, amount: int) -> bool:
        """Add experience and check for level up."""
        self.experience += amount
        level_threshold = self.level * 100
        if self.experience >= level_threshold:
            self.level_up()
            return True
        return False
    
    def level_up(self) -> None:
        """Increase character level and update stats."""
        self.level += 1
        self.max_health += 10
        self.health = self.max_health
        
        # Increase one random stat
        import random
        stats = ['strength', 'dexterity', 'constitution', 
                'intelligence', 'wisdom', 'charisma']
        stat_to_increase = random.choice(stats)
        current_value = getattr(self, stat_to_increase)
        setattr(self, stat_to_increase, current_value + 1) 