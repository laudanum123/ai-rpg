import os
import pytest
from unittest.mock import MagicMock, patch
from flask import Flask
from app import create_app
from app.models.character import Character
from app.models.game_session import GameSession
from app.models.combat import Enemy, CombatEncounter
from app.services.game_master import GameMaster
from app.services.character_service import CharacterService
from app.services.game_state_service import GameStateService
from app.services.memory_graph import MemoryGraph


@pytest.fixture
def app():
    """Create a Flask app configured for testing."""
    app = create_app(test_config={
        'TESTING': True,
        'API_DEBUG': False,
        'SQLALCHEMY_DATABASE_URI': 'sqlite:///:memory:'
    })
    
    # Setup application context
    with app.app_context():
        yield app


@pytest.fixture
def client(app):
    """Create a test client for the app."""
    with app.test_client() as client:
        yield client


@pytest.fixture
def mock_openai():
    """Mock the OpenAI client."""
    with patch('openai.OpenAI') as mock:
        # Configure mock to return a completion with mock content
        mock_completion = MagicMock()
        mock_completion.choices[0].message.content = "This is a mock AI response."
        mock_instance = mock.return_value
        mock_instance.chat.completions.create.return_value = mock_completion
        yield mock_instance


@pytest.fixture
def mock_character():
    """Create a mock character for testing."""
    character = Character(
        id="test-char-123",
        name="Test Character",
        character_class="Warrior",
        level=1,
        max_health=20,
        health=20,
        strength=12,
        dexterity=10,
        intelligence=8,
        constitution=10,
        wisdom=8,
        charisma=8,
        gold=50,
        experience=0
    )
    
    # Add items to inventory
    character.add_item({"id": "sword-1", "name": "Short Sword", "damage": 5})
    character.add_item({"id": "shield-1", "name": "Shield", "defense": 2})
    character.add_item({"id": "potion-1", "name": "Health Potion", "healing": 10})
    
    # Add abilities
    character.abilities = [
        {"id": "strike-1", "name": "Strike", "damage": 3, "cost": 0},
        {"id": "block-1", "name": "Block", "defense": 2, "cost": 0}
    ]
    
    return character


@pytest.fixture
def mock_game_session():
    """Create a mock game session for testing."""
    session = GameSession(
        id="test-session-456",
        character_id="test-char-123",
        game_world="Fantasy",
        in_combat=False,
        created_at="2025-03-13T00:00:00",
        updated_at="2025-03-13T00:00:00"
    )
    
    # Add some history entries
    session.history = [
        {
            "role": "assistant",
            "type": "narration",
            "content": "You enter the fantasy world of Eldoria.",
            "timestamp": "2025-03-13T00:01:00"
        },
        {
            "role": "player",
            "type": "player_action",
            "content": "I look around.",
            "timestamp": "2025-03-13T00:02:00"
        },
        {
            "role": "assistant",
            "type": "narration",
            "content": "You see a small village with a tavern.",
            "timestamp": "2025-03-13T00:03:00"
        }
    ]
    
    # Set current location
    session.current_location = {
        "id": "village-1",
        "name": "Eldoria Village",
        "description": "A peaceful village with thatched-roof cottages."
    }
    
    return session


@pytest.fixture
def mock_enemy():
    """Create a mock enemy for testing."""
    return Enemy(
        name="Goblin",
        description="A small, green, hostile creature",
        health=10,
        max_health=10,
        armor_class=12,
        strength=8,
        dexterity=14,
        constitution=10,
        intelligence=8,
        wisdom=8,
        charisma=8,
        attacks=[{"name": "Dagger", "damage": "1d4+2", "hit_bonus": 2}],
        abilities=[{"name": "Nimble Escape", "description": "Can disengage or hide as a bonus action"}],
        loot=[{"name": "Dagger", "value": 2}, {"name": "Goblin Ear", "value": 1}],
        experience_reward=25,
        gold_reward=5
    )


@pytest.fixture
def mock_combat_encounter(mock_enemy):
    """Create a mock combat encounter for testing."""
    enemies = [mock_enemy]
    return CombatEncounter(
        id="test-combat-789",
        enemies=enemies,
        difficulty="normal",
        environment="dungeon",
        description="A test combat encounter",
        special_rules=[]
    )


@pytest.fixture
def mock_game_master(mock_openai):
    """Create a GameMaster instance with mocked OpenAI."""
    game_master = GameMaster()
    # Use openai_client instead of client to match refactored implementation
    game_master.openai_client = mock_openai
    return game_master


@pytest.fixture
def mock_memory_graph():
    """Create a mocked memory graph."""
    with patch('app.services.memory_graph.MemoryGraph') as mock:
        mock_instance = mock.return_value
        mock_instance.get_relevant_context.return_value = "Mock memory context"
        mock_instance.nodes = []
        mock_instance.get_all_nodes.return_value = []
        yield mock_instance


@pytest.fixture
def mock_character_service(mock_character):
    """Create a mocked character service."""
    with patch('app.services.character_service.CharacterService') as mock:
        mock_instance = mock.return_value
        mock_instance.get_character.return_value = mock_character
        mock_instance.create_character.return_value = mock_character
        mock_instance.update_character.return_value = True
        yield mock_instance


@pytest.fixture
def mock_game_state_service(mock_game_session):
    """Create a mocked game state service."""
    with patch('app.services.game_state_service.GameStateService') as mock:
        mock_instance = mock.return_value
        mock_instance.get_session.return_value = mock_game_session
        mock_instance.create_session.return_value = "test-session-456"
        mock_instance.get_session_history.return_value = [
            {"role": "gm", "content": "Welcome to the game!"},
            {"role": "player", "content": "I look around."},
            {"role": "gm", "content": "You see a forest."}
        ]
        yield mock_instance
