import pytest
import json
from unittest.mock import patch, MagicMock
from app.models.npc import NPC


@pytest.fixture
def sample_npc():
    """Create a sample NPC for testing."""
    return NPC(
        name="Gandalf",
        type="quest_giver",
        description="A wise wizard with a long beard",
        health=100,
        max_health=100,
        strength=14,
        dexterity=12,
        constitution=16,
        intelligence=18,
        wisdom=20,
        charisma=16,
        inventory=[{"id": "item-1", "name": "Staff", "type": "weapon"}],
        abilities=[{"id": "ability-1", "name": "Fireball", "damage": 20}],
        dialogue={"greeting": ["Hello, adventurer!"]},
        hostile=False,
        id="npc-123"
    )


def test_npc_creation():
    """Test basic NPC creation."""
    npc = NPC(
        name="Aragorn",
        type="ally",
        description="A ranger from the North"
    )
    
    assert npc.name == "Aragorn"
    assert npc.type == "ally"
    assert npc.description == "A ranger from the North"
    assert npc.health == 50  # Default value
    assert npc.max_health == 50  # Default value
    assert npc.inventory == []  # Empty list by default
    assert npc.abilities == []  # Empty list by default
    assert npc.dialogue == {}  # Empty dict by default
    assert npc.hostile is False  # Default value
    assert npc.id is not None  # Should be auto-generated


def test_npc_to_dict(sample_npc):
    """Test converting NPC to dictionary."""
    npc_dict = sample_npc.to_dict()
    
    assert npc_dict["id"] == "npc-123"
    assert npc_dict["name"] == "Gandalf"
    assert npc_dict["type"] == "quest_giver"
    assert npc_dict["description"] == "A wise wizard with a long beard"
    assert npc_dict["health"] == 100
    assert npc_dict["max_health"] == 100
    assert npc_dict["strength"] == 14
    assert npc_dict["inventory"] == [{"id": "item-1", "name": "Staff", "type": "weapon"}]
    assert npc_dict["abilities"] == [{"id": "ability-1", "name": "Fireball", "damage": 20}]
    assert npc_dict["hostile"] is False


def test_npc_to_json(sample_npc):
    """Test converting NPC to JSON string."""
    npc_json = sample_npc.to_json()
    
    # Parse the JSON string back to a dictionary for comparison
    npc_dict = json.loads(npc_json)
    
    assert npc_dict["id"] == "npc-123"
    assert npc_dict["name"] == "Gandalf"
    assert npc_dict["type"] == "quest_giver"
    assert isinstance(npc_json, str)


def test_npc_from_dict():
    """Test creating NPC from dictionary."""
    npc_data = {
        "id": "npc-456",
        "name": "Legolas",
        "type": "ally",
        "description": "An elf with great archery skills",
        "health": 80,
        "max_health": 80,
        "strength": 12,
        "dexterity": 18,
        "constitution": 14,
        "intelligence": 14,
        "wisdom": 16,
        "charisma": 15,
        "inventory": [{"id": "item-2", "name": "Bow", "type": "weapon"}],
        "abilities": [{"id": "ability-2", "name": "Rapid Shot", "damage": 15}],
        "hostile": False
    }
    
    npc = NPC.from_dict(npc_data)
    
    assert npc.id == "npc-456"
    assert npc.name == "Legolas"
    assert npc.type == "ally"
    assert npc.description == "An elf with great archery skills"
    assert npc.health == 80
    assert npc.max_health == 80
    assert npc.strength == 12
    assert npc.dexterity == 18
    assert len(npc.inventory) == 1
    assert npc.inventory[0]["name"] == "Bow"
    assert len(npc.abilities) == 1
    assert npc.abilities[0]["name"] == "Rapid Shot"
    assert npc.hostile is False


def test_npc_from_json():
    """Test creating NPC from JSON string."""
    npc_json = json.dumps({
        "id": "npc-789",
        "name": "Gimli",
        "type": "ally",
        "description": "A dwarf with a mighty axe",
        "health": 120,
        "max_health": 120,
        "strength": 16,
        "dexterity": 10,
        "constitution": 18,
        "intelligence": 12,
        "wisdom": 14,
        "charisma": 12,
        "inventory": [{"id": "item-3", "name": "Axe", "type": "weapon"}],
        "abilities": [{"id": "ability-3", "name": "Cleave", "damage": 25}],
        "hostile": False
    })
    
    npc = NPC.from_json(npc_json)
    
    assert npc.id == "npc-789"
    assert npc.name == "Gimli"
    assert npc.type == "ally"
    assert npc.description == "A dwarf with a mighty axe"
    assert npc.health == 120
    assert npc.max_health == 120
    assert npc.strength == 16
    assert len(npc.inventory) == 1
    assert npc.inventory[0]["name"] == "Axe"
    assert len(npc.abilities) == 1
    assert npc.abilities[0]["name"] == "Cleave"
    assert npc.hostile is False


def test_get_ability_modifier():
    """Test calculating ability modifiers."""
    npc = NPC(
        name="Test NPC",
        type="test",
        description="Test description",
        strength=10,      # Modifier should be 0
        dexterity=12,     # Modifier should be 1
        constitution=14,  # Modifier should be 2
        intelligence=16,  # Modifier should be 3
        wisdom=18,        # Modifier should be 4
        charisma=20       # Modifier should be 5
    )
    
    assert npc.get_ability_modifier("strength") == 0
    assert npc.get_ability_modifier("dexterity") == 1
    assert npc.get_ability_modifier("constitution") == 2
    assert npc.get_ability_modifier("intelligence") == 3
    assert npc.get_ability_modifier("wisdom") == 4
    assert npc.get_ability_modifier("charisma") == 5
    
    # Test with non-existing ability (should default to 0)
    assert npc.get_ability_modifier("nonexistent") == 0


@patch("random.randint")
def test_roll_attack(mock_randint):
    """Test rolling attack."""
    # Mock the random roll to return a fixed value
    mock_randint.return_value = 15
    
    npc = NPC(
        name="Test NPC",
        type="test",
        description="Test description",
        strength=14  # Modifier should be 2
    )
    
    # With no weapon bonus, the attack roll should be 15 (die) + 2 (str mod) = 17
    assert npc.roll_attack() == 17
    
    # With a weapon bonus of 3, the attack roll should be 15 (die) + 2 (str mod) + 3 (bonus) = 20
    assert npc.roll_attack(weapon_bonus=3) == 20
    
    # Make sure randint was called with correct arguments
    mock_randint.assert_called_with(1, 20)


def test_take_damage(sample_npc):
    """Test taking damage."""
    initial_health = sample_npc.health
    
    # Take 20 damage
    sample_npc.take_damage(20)
    assert sample_npc.health == initial_health - 20
    
    # Take more damage
    sample_npc.take_damage(30)
    assert sample_npc.health == initial_health - 50
    
    # Take damage that exceeds current health
    sample_npc.take_damage(100)
    assert sample_npc.health == 0  # Health should not go below 0


def test_heal(sample_npc):
    """Test healing."""
    # First damage the NPC
    sample_npc.take_damage(50)
    assert sample_npc.health == 50
    
    # Heal for 20 points
    sample_npc.heal(20)
    assert sample_npc.health == 70
    
    # Heal for more than missing health
    sample_npc.heal(50)
    assert sample_npc.health == sample_npc.max_health  # Health should not exceed max_health


def test_is_alive(sample_npc):
    """Test is_alive method."""
    # Initially the NPC is alive
    assert sample_npc.is_alive() is True
    
    # Set health to 1
    sample_npc.health = 1
    assert sample_npc.is_alive() is True
    
    # Set health to 0
    sample_npc.health = 0
    assert sample_npc.is_alive() is False
    
    # Set health to negative (should not happen normally)
    sample_npc.health = -10
    assert sample_npc.is_alive() is False


def test_get_random_dialogue(sample_npc):
    """Test getting random dialogue."""
    # Sample NPC has one greeting
    greeting = sample_npc.get_random_dialogue("greeting")
    assert greeting == "Hello, adventurer!"
    
    # Add another greeting
    sample_npc.dialogue["greeting"].append("Well met!")
    
    # With mock to ensure we get a specific greeting
    with patch("random.choice", return_value="Well met!"):
        greeting = sample_npc.get_random_dialogue("greeting")
        assert greeting == "Well met!"
    
    # Test with non-existent dialogue type
    assert sample_npc.get_random_dialogue("farewell") is None
    
    # Test with empty dialogue list
    sample_npc.dialogue["empty"] = []
    assert sample_npc.get_random_dialogue("empty") is None


def test_add_dialogue(sample_npc):
    """Test adding dialogue."""
    # Add to existing dialogue type
    sample_npc.add_dialogue("greeting", "Greetings, traveler!")
    assert "Greetings, traveler!" in sample_npc.dialogue["greeting"]
    assert len(sample_npc.dialogue["greeting"]) == 2
    
    # Add to new dialogue type
    sample_npc.add_dialogue("farewell", "Farewell, adventurer!")
    assert "farewell" in sample_npc.dialogue
    assert "Farewell, adventurer!" in sample_npc.dialogue["farewell"]
    assert len(sample_npc.dialogue["farewell"]) == 1


def test_make_hostile(sample_npc):
    """Test making NPC hostile."""
    # Initially the NPC is not hostile
    assert sample_npc.hostile is False
    
    # Make hostile
    sample_npc.make_hostile()
    assert sample_npc.hostile is True
    
    # Call make_hostile again (should remain hostile)
    sample_npc.make_hostile()
    assert sample_npc.hostile is True


def test_make_friendly(sample_npc):
    """Test making NPC friendly."""
    # First make the NPC hostile
    sample_npc.make_hostile()
    assert sample_npc.hostile is True
    
    # Make friendly
    sample_npc.make_friendly()
    assert sample_npc.hostile is False
    
    # Call make_friendly again (should remain friendly)
    sample_npc.make_friendly()
    assert sample_npc.hostile is False
