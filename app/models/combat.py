from dataclasses import dataclass, field
from typing import Dict, List, Optional
import uuid
import random


@dataclass
class Enemy:
    name: str
    description: str
    health: int
    max_health: int
    armor_class: int
    strength: int
    dexterity: int
    constitution: int
    intelligence: int
    wisdom: int
    charisma: int
    attacks: List[Dict] = field(default_factory=list)
    abilities: List[Dict] = field(default_factory=list)
    loot: List[Dict] = field(default_factory=list)
    experience_reward: int = 10
    gold_reward: int = 5
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> Dict:
        """Convert enemy to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "health": self.health,
            "max_health": self.max_health,
            "armor_class": self.armor_class,
            "strength": self.strength,
            "dexterity": self.dexterity,
            "constitution": self.constitution,
            "intelligence": self.intelligence,
            "wisdom": self.wisdom,
            "charisma": self.charisma,
            "attacks": self.attacks,
            "abilities": self.abilities,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Enemy":
        """Create enemy from dictionary."""
        return cls(**data)

    def get_ability_modifier(self, ability: str) -> int:
        """Calculate ability modifier."""
        ability_score = getattr(self, ability.lower(), 10)
        return (ability_score - 10) // 2

    def roll_attack(self, attack_name: Optional[str] = None) -> Dict:
        """Roll an attack from the enemy's attack list or use a default."""
        if not self.attacks or attack_name is None:
            # Default attack if none specified or none available
            attack = {
                "name": "Strike",
                "to_hit_bonus": self.get_ability_modifier("strength"),
                "damage_dice": "1d6",
                "damage_bonus": self.get_ability_modifier("strength"),
            }
        else:
            # Find the named attack or use the first one
            attack = next(
                (a for a in self.attacks if a.get("name") == attack_name),
                self.attacks[0],
            )

        # Roll to hit
        to_hit_roll = random.randint(1, 20)
        to_hit_bonus = attack.get("to_hit_bonus", self.get_ability_modifier("strength"))
        to_hit_total = to_hit_roll + to_hit_bonus

        # Parse and roll damage dice (e.g., "2d6" means roll 2 six-sided dice)
        damage_dice = attack.get("damage_dice", "1d6")
        dice_count, dice_sides = map(int, damage_dice.split("d"))
        damage_rolls = [random.randint(1, dice_sides) for _ in range(dice_count)]
        damage_bonus = attack.get("damage_bonus", self.get_ability_modifier("strength"))
        damage_total = sum(damage_rolls) + damage_bonus

        return {
            "attack_name": attack.get("name", "Strike"),
            "to_hit_roll": to_hit_roll,
            "to_hit_bonus": to_hit_bonus,
            "to_hit_total": to_hit_total,
            "damage_rolls": damage_rolls,
            "damage_bonus": damage_bonus,
            "damage_total": max(0, damage_total),  # Ensure non-negative damage
        }

    def take_damage(self, amount: int) -> None:
        """Take damage and update health."""
        self.health = max(0, self.health - amount)

    def is_alive(self) -> bool:
        """Check if enemy is alive."""
        return self.health > 0


@dataclass
class CombatEncounter:
    enemies: List[Enemy]
    difficulty: str = "normal"  # "easy", "normal", "hard", "boss"
    environment: str = "dungeon"
    description: str = ""
    special_rules: List[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> Dict:
        """Convert encounter to dictionary."""
        return {
            "id": self.id,
            "enemies": [enemy.to_dict() for enemy in self.enemies],
            "difficulty": self.difficulty,
            "environment": self.environment,
            "description": self.description,
            "special_rules": self.special_rules,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "CombatEncounter":
        """Create encounter from dictionary."""
        enemy_data = data.pop("enemies", [])
        enemies = [Enemy.from_dict(enemy) for enemy in enemy_data]
        return cls(enemies=enemies, **data)

    def get_total_xp(self) -> int:
        """Calculate total XP reward for the encounter."""
        return sum(enemy.experience_reward for enemy in self.enemies)

    def get_total_gold(self) -> int:
        """Calculate total gold reward for the encounter."""
        return sum(enemy.gold_reward for enemy in self.enemies)

    def all_enemies_defeated(self) -> bool:
        """Check if all enemies are defeated."""
        return all(not enemy.is_alive() for enemy in self.enemies)

    def get_active_enemies(self) -> List[Enemy]:
        """Get list of enemies that are still alive."""
        return [enemy for enemy in self.enemies if enemy.is_alive()]


def roll_dice(dice_notation: str) -> int:
    """Roll dice based on standard dice notation (e.g., '2d6+3')."""
    # Split the dice notation into its components
    if "+" in dice_notation:
        dice_part, bonus_part = dice_notation.split("+")
        bonus = int(bonus_part)
    elif "-" in dice_notation:
        dice_part, penalty_part = dice_notation.split("-")
        bonus = -int(penalty_part)
    else:
        dice_part = dice_notation
        bonus = 0

    # Parse the dice part
    dice_count, dice_sides = map(int, dice_part.split("d"))

    # Roll the dice and calculate the total
    rolls = [random.randint(1, dice_sides) for _ in range(dice_count)]
    total = sum(rolls) + bonus

    return max(total, 0)  # Ensure non-negative result
