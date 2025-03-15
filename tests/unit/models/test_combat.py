from unittest.mock import patch

from app.models.combat import CombatEncounter, Enemy, roll_dice


def test_enemy_creation():
    """Test creating an enemy with valid attributes."""
    enemy = Enemy(
        id="test-enemy-1",
        name="Goblin",
        description="A small, green creature",
        health=20,
        max_health=20,
        armor_class=12,
        strength=8,
        dexterity=14,
        constitution=10,
        intelligence=10,
        wisdom=8,
        charisma=8,
        attacks=[{"name": "Dagger", "to_hit_bonus": 4, "damage_dice": "1d4", "damage_bonus": 2}],
        abilities=[{"name": "Nimble Escape", "description": "Can disengage as a bonus action"}],
        loot=[{"id": "dagger-1", "name": "Rusty Dagger", "value": 2}],
        experience_reward=25,
        gold_reward=5
    )

    assert enemy.id == "test-enemy-1"
    assert enemy.name == "Goblin"
    assert enemy.description == "A small, green creature"
    assert enemy.health == 20
    assert enemy.max_health == 20
    assert enemy.armor_class == 12
    assert enemy.strength == 8
    assert enemy.dexterity == 14
    assert enemy.constitution == 10
    assert enemy.intelligence == 10
    assert enemy.wisdom == 8
    assert enemy.charisma == 8
    assert len(enemy.attacks) == 1
    assert enemy.attacks[0]["name"] == "Dagger"
    assert len(enemy.abilities) == 1
    assert enemy.abilities[0]["name"] == "Nimble Escape"
    assert len(enemy.loot) == 1
    assert enemy.loot[0]["name"] == "Rusty Dagger"
    assert enemy.experience_reward == 25
    assert enemy.gold_reward == 5


def test_enemy_to_dict():
    """Test converting enemy to dictionary."""
    enemy = Enemy(
        name="Orc",
        description="A brutish humanoid",
        health=30,
        max_health=30,
        armor_class=13,
        strength=16,
        dexterity=12,
        constitution=16,
        intelligence=7,
        wisdom=11,
        charisma=10
    )

    enemy_dict = enemy.to_dict()

    assert isinstance(enemy_dict, dict)
    assert enemy_dict["name"] == "Orc"
    assert enemy_dict["description"] == "A brutish humanoid"
    assert enemy_dict["health"] == 30
    assert enemy_dict["max_health"] == 30
    assert enemy_dict["strength"] == 16
    assert "attacks" in enemy_dict
    assert "abilities" in enemy_dict


def test_enemy_from_dict():
    """Test creating an enemy from a dictionary."""
    enemy_data = {
        "id": "test-enemy-2",
        "name": "Skeleton",
        "description": "An animated pile of bones",
        "health": 15,
        "max_health": 15,
        "armor_class": 13,
        "strength": 10,
        "dexterity": 14,
        "constitution": 15,
        "intelligence": 6,
        "wisdom": 8,
        "charisma": 5,
        "attacks": [{"name": "Shortsword", "damage_dice": "1d6"}],
        "experience_reward": 50
    }

    enemy = Enemy.from_dict(enemy_data)

    assert enemy.id == "test-enemy-2"
    assert enemy.name == "Skeleton"
    assert enemy.health == 15
    assert enemy.armor_class == 13
    assert enemy.strength == 10
    assert len(enemy.attacks) == 1
    assert enemy.attacks[0]["name"] == "Shortsword"
    assert enemy.experience_reward == 50


def test_enemy_get_ability_modifier():
    """Test ability modifier calculation for enemies."""
    enemy = Enemy(
        name="Test Enemy",
        description="For testing",
        health=10,
        max_health=10,
        armor_class=10,
        strength=16,  # Modifier should be +3
        dexterity=14,  # Modifier should be +2
        constitution=12,  # Modifier should be +1
        intelligence=8,  # Modifier should be -1
        wisdom=10,  # Modifier should be +0
        charisma=6,  # Modifier should be -2
    )

    assert enemy.get_ability_modifier("strength") == 3
    assert enemy.get_ability_modifier("dexterity") == 2
    assert enemy.get_ability_modifier("constitution") == 1
    assert enemy.get_ability_modifier("intelligence") == -1
    assert enemy.get_ability_modifier("wisdom") == 0
    assert enemy.get_ability_modifier("charisma") == -2

    # Test with non-existent ability (should default to modifier for 10)
    assert enemy.get_ability_modifier("nonexistent") == 0


@patch('random.randint')
def test_enemy_roll_attack_default(mock_randint):
    """Test enemy's default attack roll when no attacks are specified."""
    # Set fixed values for dice rolls
    mock_randint.return_value = 15

    enemy = Enemy(
        name="Test Enemy",
        description="For testing",
        health=10,
        max_health=10,
        armor_class=10,
        strength=14,  # Modifier +2
        dexterity=10,
        constitution=10,
        intelligence=10,
        wisdom=10,
        charisma=10,
        attacks=[]  # No attacks defined, will use default
    )

    attack_result = enemy.roll_attack()

    assert attack_result["attack_name"] == "Strike"
    assert attack_result["to_hit_roll"] == 15
    assert attack_result["to_hit_bonus"] == 2  # Strength modifier
    assert attack_result["to_hit_total"] == 17  # 15 + 2
    assert attack_result["damage_bonus"] == 2  # Strength modifier
    assert attack_result["damage_total"] > 0


@patch('random.randint')
def test_enemy_roll_attack_specific(mock_randint):
    """Test enemy's attack roll with a specific named attack."""
    # This test will need to mock two different random.randint calls:
    # 1. For the to_hit roll
    # 2. For each damage dice roll
    mock_randint.side_effect = [18, 4, 5]  # to_hit roll, then damage dice rolls

    enemy = Enemy(
        name="Test Enemy",
        description="For testing",
        health=10,
        max_health=10,
        armor_class=10,
        strength=10,
        dexterity=10,
        constitution=10,
        intelligence=10,
        wisdom=10,
        charisma=10,
        attacks=[
            {
                "name": "Greataxe",
                "to_hit_bonus": 5,
                "damage_dice": "2d6",
                "damage_bonus": 3
            }
        ]
    )

    attack_result = enemy.roll_attack("Greataxe")

    assert attack_result["attack_name"] == "Greataxe"
    assert attack_result["to_hit_roll"] == 18
    assert attack_result["to_hit_bonus"] == 5
    assert attack_result["to_hit_total"] == 23  # 18 + 5
    assert attack_result["damage_rolls"] == [4, 5]
    assert attack_result["damage_bonus"] == 3
    assert attack_result["damage_total"] == 12  # 4 + 5 + 3


def test_enemy_take_damage():
    """Test enemy taking damage."""
    enemy = Enemy(
        name="Test Enemy",
        description="For testing",
        health=30,
        max_health=30,
        armor_class=10,
        strength=10,
        dexterity=10,
        constitution=10,
        intelligence=10,
        wisdom=10,
        charisma=10
    )

    initial_health = enemy.health

    # Test normal damage
    enemy.take_damage(10)
    assert enemy.health == initial_health - 10

    # Test excessive damage
    enemy.take_damage(enemy.health + 10)
    assert enemy.health == 0
    assert not enemy.is_alive()


def test_enemy_is_alive():
    """Test enemy alive/dead status."""
    enemy = Enemy(
        name="Test Enemy",
        description="For testing",
        health=1,
        max_health=10,
        armor_class=10,
        strength=10,
        dexterity=10,
        constitution=10,
        intelligence=10,
        wisdom=10,
        charisma=10
    )

    # Initially alive
    assert enemy.is_alive()

    # After taking damage that reduces health to 0
    enemy.take_damage(1)
    assert not enemy.is_alive()


def test_combat_encounter_creation():
    """Test creating a combat encounter with enemies."""
    goblin = Enemy(
        name="Goblin",
        description="A small, green creature",
        health=15,
        max_health=15,
        armor_class=12,
        strength=8,
        dexterity=14,
        constitution=10,
        intelligence=10,
        wisdom=8,
        charisma=8
    )

    orc = Enemy(
        name="Orc",
        description="A brutish humanoid",
        health=30,
        max_health=30,
        armor_class=13,
        strength=16,
        dexterity=12,
        constitution=16,
        intelligence=7,
        wisdom=11,
        charisma=10
    )

    encounter = CombatEncounter(
        enemies=[goblin, orc],
        difficulty="hard",
        environment="forest",
        description="A goblin and orc patrol",
        special_rules=["Surprise attack: Enemies get advantage on first round"]
    )

    assert len(encounter.enemies) == 2
    assert encounter.enemies[0].name == "Goblin"
    assert encounter.enemies[1].name == "Orc"
    assert encounter.difficulty == "hard"
    assert encounter.environment == "forest"
    assert encounter.description == "A goblin and orc patrol"
    assert len(encounter.special_rules) == 1
    assert encounter.special_rules[0] == "Surprise attack: Enemies get advantage on first round"


def test_combat_encounter_to_dict():
    """Test converting combat encounter to dictionary."""
    goblin = Enemy(
        name="Goblin",
        description="A small, green creature",
        health=15,
        max_health=15,
        armor_class=12,
        strength=8,
        dexterity=14,
        constitution=10,
        intelligence=10,
        wisdom=8,
        charisma=8
    )

    encounter = CombatEncounter(
        enemies=[goblin],
        difficulty="normal",
        environment="cave"
    )

    encounter_dict = encounter.to_dict()

    assert isinstance(encounter_dict, dict)
    assert len(encounter_dict["enemies"]) == 1
    assert encounter_dict["enemies"][0]["name"] == "Goblin"
    assert encounter_dict["difficulty"] == "normal"
    assert encounter_dict["environment"] == "cave"


def test_combat_encounter_from_dict():
    """Test creating a combat encounter from a dictionary."""
    encounter_data = {
        "id": "test-encounter-1",
        "enemies": [
            {
                "name": "Zombie",
                "description": "An undead creature",
                "health": 22,
                "max_health": 22,
                "armor_class": 8,
                "strength": 13,
                "dexterity": 6,
                "constitution": 16,
                "intelligence": 3,
                "wisdom": 6,
                "charisma": 5
            },
            {
                "name": "Skeleton",
                "description": "An animated pile of bones",
                "health": 13,
                "max_health": 13,
                "armor_class": 13,
                "strength": 10,
                "dexterity": 14,
                "constitution": 15,
                "intelligence": 6,
                "wisdom": 8,
                "charisma": 5
            }
        ],
        "difficulty": "normal",
        "environment": "graveyard",
        "description": "The restless dead have risen",
        "special_rules": ["Undead fortitude: Zombies have a chance to survive lethal damage"]
    }

    encounter = CombatEncounter.from_dict(encounter_data)

    assert encounter.id == "test-encounter-1"
    assert len(encounter.enemies) == 2
    assert encounter.enemies[0].name == "Zombie"
    assert encounter.enemies[1].name == "Skeleton"
    assert encounter.difficulty == "normal"
    assert encounter.environment == "graveyard"
    assert encounter.description == "The restless dead have risen"
    assert len(encounter.special_rules) == 1


def test_combat_encounter_get_total_xp():
    """Test calculating total XP reward for an encounter."""
    goblin = Enemy(
        name="Goblin",
        description="A small, green creature",
        health=15,
        max_health=15,
        armor_class=12,
        strength=8,
        dexterity=14,
        constitution=10,
        intelligence=10,
        wisdom=8,
        charisma=8,
        experience_reward=25
    )

    orc = Enemy(
        name="Orc",
        description="A brutish humanoid",
        health=30,
        max_health=30,
        armor_class=13,
        strength=16,
        dexterity=12,
        constitution=16,
        intelligence=7,
        wisdom=11,
        charisma=10,
        experience_reward=100
    )

    encounter = CombatEncounter(enemies=[goblin, orc])

    assert encounter.get_total_xp() == 125  # 25 + 100


def test_combat_encounter_get_total_gold():
    """Test calculating total gold reward for an encounter."""
    goblin = Enemy(
        name="Goblin",
        description="A small, green creature",
        health=15,
        max_health=15,
        armor_class=12,
        strength=8,
        dexterity=14,
        constitution=10,
        intelligence=10,
        wisdom=8,
        charisma=8,
        gold_reward=5
    )

    orc = Enemy(
        name="Orc",
        description="A brutish humanoid",
        health=30,
        max_health=30,
        armor_class=13,
        strength=16,
        dexterity=12,
        constitution=16,
        intelligence=7,
        wisdom=11,
        charisma=10,
        gold_reward=15
    )

    encounter = CombatEncounter(enemies=[goblin, orc])

    assert encounter.get_total_gold() == 20  # 5 + 15


def test_combat_encounter_all_enemies_defeated():
    """Test checking if all enemies in an encounter are defeated."""
    goblin = Enemy(
        name="Goblin",
        description="A small, green creature",
        health=15,
        max_health=15,
        armor_class=12,
        strength=8,
        dexterity=14,
        constitution=10,
        intelligence=10,
        wisdom=8,
        charisma=8
    )

    orc = Enemy(
        name="Orc",
        description="A brutish humanoid",
        health=30,
        max_health=30,
        armor_class=13,
        strength=16,
        dexterity=12,
        constitution=16,
        intelligence=7,
        wisdom=11,
        charisma=10
    )

    encounter = CombatEncounter(enemies=[goblin, orc])

    # Initially, no enemies are defeated
    assert not encounter.all_enemies_defeated()

    # Defeat one enemy
    goblin.take_damage(goblin.health)
    assert not encounter.all_enemies_defeated()

    # Defeat all enemies
    orc.take_damage(orc.health)
    assert encounter.all_enemies_defeated()


def test_combat_encounter_get_active_enemies():
    """Test getting list of active (alive) enemies in an encounter."""
    goblin = Enemy(
        name="Goblin",
        description="A small, green creature",
        health=15,
        max_health=15,
        armor_class=12,
        strength=8,
        dexterity=14,
        constitution=10,
        intelligence=10,
        wisdom=8,
        charisma=8
    )

    orc = Enemy(
        name="Orc",
        description="A brutish humanoid",
        health=30,
        max_health=30,
        armor_class=13,
        strength=16,
        dexterity=12,
        constitution=16,
        intelligence=7,
        wisdom=11,
        charisma=10
    )

    skeleton = Enemy(
        name="Skeleton",
        description="An animated pile of bones",
        health=13,
        max_health=13,
        armor_class=13,
        strength=10,
        dexterity=14,
        constitution=15,
        intelligence=6,
        wisdom=8,
        charisma=5
    )

    encounter = CombatEncounter(enemies=[goblin, orc, skeleton])

    # Initially, all enemies are active
    active_enemies = encounter.get_active_enemies()
    assert len(active_enemies) == 3

    # Defeat one enemy
    goblin.take_damage(goblin.health)
    active_enemies = encounter.get_active_enemies()
    assert len(active_enemies) == 2
    assert goblin not in active_enemies
    assert orc in active_enemies
    assert skeleton in active_enemies

    # Defeat another enemy
    skeleton.take_damage(skeleton.health)
    active_enemies = encounter.get_active_enemies()
    assert len(active_enemies) == 1
    assert orc in active_enemies


@patch('random.randint')
def test_roll_dice_simple(mock_randint):
    """Test rolling dice with a simple notation (e.g., '2d6')."""
    mock_randint.side_effect = [3, 5]  # Two d6 rolls

    result = roll_dice("2d6")

    assert result == 8  # 3 + 5


@patch('random.randint')
def test_roll_dice_with_bonus(mock_randint):
    """Test rolling dice with a bonus (e.g., '1d20+5')."""
    mock_randint.return_value = 15  # One d20 roll

    result = roll_dice("1d20+5")

    assert result == 20  # 15 + 5


@patch('random.randint')
def test_roll_dice_with_penalty(mock_randint):
    """Test rolling dice with a penalty (e.g., '1d20-2')."""
    mock_randint.return_value = 10  # One d20 roll

    result = roll_dice("1d20-2")

    assert result == 8  # 10 - 2


@patch('random.randint')
def test_roll_dice_minimum_zero(mock_randint):
    """Test that dice rolls have a minimum result of 0 even with penalties."""
    mock_randint.return_value = 1  # One d4 roll

    result = roll_dice("1d4-5")

    assert result == 0  # 1 - 5 would be -4, but minimum is 0
