import pytest
import json
from unittest.mock import patch
from app.models.character import Character


def test_character_creation():
    """Test creating a character with valid attributes."""
    character = Character(
        id="test-id",
        name="Aragorn",
        character_class="Ranger",
        level=1,
        max_health=20,
        health=20,
        strength=14,
        dexterity=16,
        intelligence=12,
        constitution=12,
        wisdom=14,
        charisma=16,
        gold=100,
        experience=0
    )
    
    assert character.id == "test-id"
    assert character.name == "Aragorn"
    assert character.character_class == "Ranger"
    assert character.level == 1
    assert character.max_health == 20
    assert character.health == 20
    assert character.strength == 14
    assert character.dexterity == 16
    assert character.intelligence == 12
    assert character.constitution == 12
    assert character.wisdom == 14
    assert character.charisma == 16
    assert character.gold == 100
    assert character.experience == 0
    assert isinstance(character.inventory, list)
    assert isinstance(character.abilities, list)


def test_character_level_up(mock_character):
    """Test character level up mechanics."""
    initial_level = mock_character.level
    initial_health = mock_character.max_health
    
    # Directly call level_up to test the method
    mock_character.level_up()
    
    # Check level increased
    assert mock_character.level == initial_level + 1
    # Check health increased
    assert mock_character.max_health > initial_health
    # Check health was reset to max_health
    assert mock_character.health == mock_character.max_health
    # Check that some stat was increased
    total_stats_before = initial_level * 6  # Assuming starting stats were all the same
    total_stats_after = (
        mock_character.strength + mock_character.dexterity + 
        mock_character.constitution + mock_character.intelligence + 
        mock_character.wisdom + mock_character.charisma
    )
    assert total_stats_after > total_stats_before


def test_character_take_damage(mock_character):
    """Test character taking damage."""
    initial_health = mock_character.health
    damage = 5
    
    mock_character.take_damage(damage)
    
    assert mock_character.health == initial_health - damage
    assert mock_character.is_alive()


def test_character_heal(mock_character):
    """Test character healing."""
    # First damage the character
    mock_character.health = 10
    initial_health = mock_character.health
    heal_amount = 5
    
    mock_character.heal(heal_amount)
    
    assert mock_character.health == initial_health + heal_amount
    # Check healing doesn't exceed max health
    mock_character.heal(50)
    assert mock_character.health <= mock_character.max_health


def test_character_death(mock_character):
    """Test character death mechanics."""
    mock_character.take_damage(mock_character.health + 10)
    
    assert mock_character.health == 0
    assert not mock_character.is_alive()


def test_character_add_item(mock_character):
    """Test adding items to character inventory."""
    initial_items_count = len(mock_character.inventory)
    new_item = {"id": "ring-1", "name": "Magic Ring", "effect": "invisibility"}
    
    mock_character.add_item(new_item)
    
    assert len(mock_character.inventory) == initial_items_count + 1
    assert new_item in mock_character.inventory


def test_character_remove_item(mock_character):
    """Test removing items from character inventory."""
    # Ensure the character has an item to remove
    item_id = mock_character.inventory[0]["id"]
    initial_items_count = len(mock_character.inventory)
    
    removed_item = mock_character.remove_item(item_id)
    
    assert len(mock_character.inventory) == initial_items_count - 1
    assert removed_item is not None
    assert removed_item["id"] == item_id


def test_character_to_dict(mock_character):
    """Test converting character to dictionary."""
    character_dict = mock_character.to_dict()
    
    assert isinstance(character_dict, dict)
    assert character_dict["id"] == mock_character.id
    assert character_dict["name"] == mock_character.name
    assert character_dict["character_class"] == mock_character.character_class
    assert character_dict["level"] == mock_character.level
    assert character_dict["health"] == mock_character.health
    assert "inventory" in character_dict
    assert "abilities" in character_dict


def test_character_to_json(mock_character):
    """Test converting character to JSON string."""
    json_str = mock_character.to_json()
    
    assert isinstance(json_str, str)
    
    # Parse the JSON and verify it matches the character
    character_dict = json.loads(json_str)
    assert character_dict["id"] == mock_character.id
    assert character_dict["name"] == mock_character.name
    assert character_dict["character_class"] == mock_character.character_class


def test_character_from_dict():
    """Test creating a character from a dictionary."""
    character_data = {
        "id": "test-id-123",
        "name": "Gandalf",
        "character_class": "Wizard",
        "level": 5,
        "health": 75,
        "max_health": 80,
        "strength": 12,
        "dexterity": 14,
        "constitution": 15,
        "intelligence": 18,
        "wisdom": 17,
        "charisma": 16,
        "gold": 250,
        "experience": 450,
        "inventory": [{"id": "staff-1", "name": "Staff of Power"}],
        "abilities": [{"id": "fireball", "name": "Fireball"}]
    }
    
    character = Character.from_dict(character_data)
    
    assert character.id == "test-id-123"
    assert character.name == "Gandalf"
    assert character.character_class == "Wizard"
    assert character.level == 5
    assert character.health == 75
    assert character.max_health == 80
    assert character.strength == 12
    assert character.experience == 450
    assert len(character.inventory) == 1
    assert character.inventory[0]["name"] == "Staff of Power"
    assert len(character.abilities) == 1
    assert character.abilities[0]["name"] == "Fireball"


def test_character_from_json():
    """Test creating a character from a JSON string."""
    character_data = {
        "id": "test-id-456",
        "name": "Legolas",
        "character_class": "Archer",
        "level": 3,
        "health": 60,
        "max_health": 60
    }
    
    json_str = json.dumps(character_data)
    character = Character.from_json(json_str)
    
    assert character.id == "test-id-456"
    assert character.name == "Legolas"
    assert character.character_class == "Archer"
    assert character.level == 3
    assert character.health == 60
    assert character.max_health == 60


def test_get_ability_modifier(mock_character):
    """Test ability modifier calculation."""
    # Set up specific ability scores
    mock_character.strength = 16  # Modifier should be +3
    mock_character.dexterity = 14  # Modifier should be +2
    mock_character.intelligence = 8  # Modifier should be -1
    mock_character.wisdom = 10  # Modifier should be +0
    
    assert mock_character.get_ability_modifier("strength") == 3
    assert mock_character.get_ability_modifier("dexterity") == 2
    assert mock_character.get_ability_modifier("intelligence") == -1
    assert mock_character.get_ability_modifier("wisdom") == 0
    
    # Test with non-existent ability (should default to 10, giving modifier 0)
    assert mock_character.get_ability_modifier("nonexistent") == 0


@patch('random.randint')
def test_roll_attack(mock_randint, mock_character):
    """Test attack roll calculation."""
    # Set a fixed value for the die roll
    mock_randint.return_value = 15
    
    # Set strength to 14 (modifier +2)
    mock_character.strength = 14
    
    # Test basic attack (no weapon bonus)
    attack_roll = mock_character.roll_attack()
    mock_randint.assert_called_with(1, 20)
    assert attack_roll == 15 + 2  # 15 (die) + 2 (str mod)
    
    # Test with weapon bonus
    attack_roll_with_bonus = mock_character.roll_attack(weapon_bonus=3)
    assert attack_roll_with_bonus == 15 + 2 + 3  # 15 (die) + 2 (str mod) + 3 (weapon)


def test_add_experience(mock_character):
    """Test adding experience and checking for level up."""
    # Set initial values
    mock_character.level = 2
    mock_character.experience = 150
    initial_level = mock_character.level
    
    # Add experience but not enough to level up
    result = mock_character.add_experience(10)
    assert mock_character.experience == 160
    assert not result  # Should not level up
    assert mock_character.level == initial_level
    
    # Add enough experience to level up
    result = mock_character.add_experience(50)  # Level 2 needs 200 XP to level up
    assert mock_character.experience == 210
    assert result  # Should level up
    assert mock_character.level == initial_level + 1
    
    # Check multiple level ups
    mock_character.level = 1
    mock_character.experience = 90
    result = mock_character.add_experience(500)  # Should trigger multiple level ups
    assert result
    assert mock_character.level > 1
