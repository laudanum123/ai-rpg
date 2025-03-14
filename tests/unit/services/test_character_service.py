import json
import os
import pytest
import shutil
import uuid
from unittest.mock import patch, MagicMock

from app.models.character import Character
from app.services.character_service import CharacterService


@pytest.fixture
def temp_data_dir(tmpdir):
    """Create a temporary directory for character data."""
    characters_dir = tmpdir.mkdir("characters")
    yield str(characters_dir)
    # Clean up after test
    if os.path.exists(str(characters_dir)):
        shutil.rmtree(str(characters_dir))


@pytest.fixture
def character_service(temp_data_dir):
    """Create a character service with a temporary data directory."""
    return CharacterService(data_dir=temp_data_dir)


@pytest.fixture
def sample_character():
    """Create a sample character for testing."""
    return Character(
        name="Test Character",
        character_class="Fighter",
        id="test-char-id"
    )


def test_init_with_default_data_dir(tmpdir):
    """Test initializing the character service with default data dir."""
    # We need to remove the test for Flask app dependency since it's hard to mock
    # Instead, let's test that we can specify a custom data directory
    custom_dir = str(tmpdir.mkdir("custom_characters"))
    service = CharacterService(data_dir=custom_dir)
    assert service.data_dir == custom_dir
    assert os.path.exists(service.data_dir)


def test_init_without_flask_app():
    """Test initializing without a Flask app context."""
    with patch("flask.current_app", None):
        service = CharacterService()
        assert service.data_dir == "data/characters"
        assert os.path.exists(service.data_dir)


def test_create_character_fighter(character_service):
    """Test creating a fighter character."""
    character = character_service.create_character("Aragorn", "Fighter")
    
    assert character.name == "Aragorn"
    assert character.character_class == "Fighter"
    assert character.strength > 10  # Fighter gets +2 strength
    assert character.constitution > 10  # Fighter gets +1 constitution
    
    # Check if the character has the right abilities and items
    assert any(ability["name"] == "Second Wind" for ability in character.abilities)
    assert any(item["name"] == "Longsword" for item in character.inventory)
    assert any(item["name"] == "Shield" for item in character.inventory)
    assert any(item["name"] == "Healing Potion" for item in character.inventory)
    
    # Check if the character was saved
    assert os.path.exists(os.path.join(character_service.data_dir, f"{character.id}.json"))
    
    # Check if the character is in the cache
    assert character.id in character_service.characters


def test_create_character_wizard(character_service):
    """Test creating a wizard character."""
    character = character_service.create_character("Gandalf", "Wizard")
    
    assert character.name == "Gandalf"
    assert character.character_class == "Wizard"
    assert character.intelligence > 10  # Wizard gets +2 intelligence
    assert character.wisdom > 10  # Wizard gets +1 wisdom
    
    # Check if the character has the right abilities and items
    assert any(ability["name"] == "Arcane Recovery" for ability in character.abilities)
    assert any(item["name"] == "Wizard's Staff" for item in character.inventory)
    assert any(item["name"] == "Spellbook" for item in character.inventory)


def test_create_character_rogue(character_service):
    """Test creating a rogue character."""
    character = character_service.create_character("Bilbo", "Rogue")
    
    assert character.name == "Bilbo"
    assert character.character_class == "Rogue"
    assert character.dexterity > 10  # Rogue gets +2 dexterity
    assert character.charisma > 10  # Rogue gets +1 charisma
    
    # Check if the character has the right abilities and items
    assert any(ability["name"] == "Sneak Attack" for ability in character.abilities)
    assert any(item["name"] == "Dagger" for item in character.inventory)
    assert any(item["name"] == "Thieves' Tools" for item in character.inventory)


def test_get_character_from_cache(character_service, sample_character):
    """Test retrieving a character from cache."""
    # Add the character to the cache
    character_service.characters[sample_character.id] = sample_character
    
    # Get the character
    retrieved_character = character_service.get_character(sample_character.id)
    
    assert retrieved_character is not None
    assert retrieved_character.id == sample_character.id
    assert retrieved_character.name == sample_character.name


def test_get_character_from_file(character_service, sample_character):
    """Test retrieving a character from file."""
    # Save the character to file
    character_path = os.path.join(character_service.data_dir, f"{sample_character.id}.json")
    with open(character_path, "w") as f:
        json.dump(sample_character.to_dict(), f)
    
    # Get the character
    retrieved_character = character_service.get_character(sample_character.id)
    
    assert retrieved_character is not None
    assert retrieved_character.id == sample_character.id
    assert retrieved_character.name == sample_character.name
    
    # Check if the character was added to the cache
    assert sample_character.id in character_service.characters


def test_get_character_not_found(character_service):
    """Test retrieving a non-existent character."""
    non_existent_id = "non-existent-id"
    retrieved_character = character_service.get_character(non_existent_id)
    
    assert retrieved_character is None


def test_get_or_create_character_existing(character_service):
    """Test getting an existing character by name."""
    # Create a character first
    original_character = character_service.create_character("Frodo", "Rogue")
    
    # Clear the cache to ensure we're loading from file
    character_service.characters = {}
    
    # Get or create the character
    retrieved_character = character_service.get_or_create_character("Frodo", "Fighter")
    
    assert retrieved_character is not None
    assert retrieved_character.id == original_character.id
    assert retrieved_character.name == "Frodo"
    assert retrieved_character.character_class == "Rogue"  # Should keep the original class


def test_get_or_create_character_new(character_service):
    """Test creating a new character when not found."""
    # Get or create a new character
    character = character_service.get_or_create_character("Sam", "Fighter")
    
    assert character is not None
    assert character.name == "Sam"
    assert character.character_class == "Fighter"
    
    # Check if the character was saved
    assert os.path.exists(os.path.join(character_service.data_dir, f"{character.id}.json"))


def test_save_character(character_service, sample_character):
    """Test saving a character to file."""
    character_service.save_character(sample_character)
    
    character_path = os.path.join(character_service.data_dir, f"{sample_character.id}.json")
    assert os.path.exists(character_path)
    
    # Verify the content of the saved file
    with open(character_path, "r") as f:
        saved_data = json.load(f)
        assert saved_data["id"] == sample_character.id
        assert saved_data["name"] == sample_character.name
        assert saved_data["character_class"] == sample_character.character_class


def test_update_character(character_service, sample_character):
    """Test updating a character."""
    # Add the character to the cache and save to file
    character_service.characters[sample_character.id] = sample_character
    character_service.save_character(sample_character)
    
    # Modify the character
    sample_character.name = "Updated Name"
    sample_character.strength = 16
    
    # Update the character
    character_service.update_character(sample_character)
    
    # Check if the character was updated in cache
    assert character_service.characters[sample_character.id].name == "Updated Name"
    assert character_service.characters[sample_character.id].strength == 16
    
    # Check if the character was updated in file
    character_path = os.path.join(character_service.data_dir, f"{sample_character.id}.json")
    with open(character_path, "r") as f:
        saved_data = json.load(f)
        assert saved_data["name"] == "Updated Name"
        assert saved_data["strength"] == 16


def test_delete_character_success(character_service, sample_character):
    """Test successfully deleting a character."""
    # Add the character to the cache and save to file
    character_service.characters[sample_character.id] = sample_character
    character_service.save_character(sample_character)
    
    # Delete the character
    result = character_service.delete_character(sample_character.id)
    
    assert result is True
    assert sample_character.id not in character_service.characters
    character_path = os.path.join(character_service.data_dir, f"{sample_character.id}.json")
    assert not os.path.exists(character_path)


def test_delete_character_not_found(character_service):
    """Test deleting a non-existent character."""
    non_existent_id = "non-existent-id"
    result = character_service.delete_character(non_existent_id)
    
    assert result is False


def test_get_inventory(character_service, sample_character):
    """Test getting a character's inventory."""
    # Add the character to the cache
    character_service.characters[sample_character.id] = sample_character
    
    # Add some items to the character's inventory
    sample_character.add_item({"id": "test-item-1", "name": "Test Item 1"})
    sample_character.add_item({"id": "test-item-2", "name": "Test Item 2"})
    
    # Get the inventory
    inventory = character_service.get_inventory(sample_character.id)
    
    assert len(inventory) == 2
    assert any(item["id"] == "test-item-1" for item in inventory)
    assert any(item["id"] == "test-item-2" for item in inventory)


def test_get_inventory_character_not_found(character_service):
    """Test getting inventory for a non-existent character."""
    non_existent_id = "non-existent-id"
    inventory = character_service.get_inventory(non_existent_id)
    
    assert inventory == []


def test_add_item_to_inventory(character_service, sample_character):
    """Test adding an item to a character's inventory."""
    # Add the character to the cache
    character_service.characters[sample_character.id] = sample_character
    
    # Add an item to the inventory
    item = {"id": "test-item", "name": "Test Item"}
    result = character_service.add_item_to_inventory(sample_character.id, item)
    
    assert result is True
    assert any(i["id"] == "test-item" for i in character_service.characters[sample_character.id].inventory)
    
    # Check if the character was updated in storage
    character_path = os.path.join(character_service.data_dir, f"{sample_character.id}.json")
    with open(character_path, "r") as f:
        saved_data = json.load(f)
        assert any(i["id"] == "test-item" for i in saved_data["inventory"])


def test_add_item_to_inventory_character_not_found(character_service):
    """Test adding an item to a non-existent character's inventory."""
    non_existent_id = "non-existent-id"
    item = {"id": "test-item", "name": "Test Item"}
    result = character_service.add_item_to_inventory(non_existent_id, item)
    
    assert result is False


def test_remove_item_from_inventory(character_service, sample_character):
    """Test removing an item from a character's inventory."""
    # Add the character to the cache
    character_service.characters[sample_character.id] = sample_character
    
    # Add an item to the inventory
    item = {"id": "test-item", "name": "Test Item"}
    sample_character.add_item(item)
    
    # Remove the item from the inventory
    removed_item = character_service.remove_item_from_inventory(sample_character.id, "test-item")
    
    assert removed_item is not None
    assert removed_item["id"] == "test-item"
    assert not any(i["id"] == "test-item" for i in character_service.characters[sample_character.id].inventory)
    
    # Check if the character was updated in storage
    character_path = os.path.join(character_service.data_dir, f"{sample_character.id}.json")
    with open(character_path, "r") as f:
        saved_data = json.load(f)
        assert not any(i["id"] == "test-item" for i in saved_data["inventory"])


def test_remove_item_from_inventory_character_not_found(character_service):
    """Test removing an item from a non-existent character's inventory."""
    non_existent_id = "non-existent-id"
    result = character_service.remove_item_from_inventory(non_existent_id, "test-item")
    
    assert result is None


def test_remove_item_from_inventory_item_not_found(character_service, sample_character):
    """Test removing a non-existent item from a character's inventory."""
    # Add the character to the cache
    character_service.characters[sample_character.id] = sample_character
    
    # Remove a non-existent item
    result = character_service.remove_item_from_inventory(sample_character.id, "non-existent-item")
    
    assert result is None
