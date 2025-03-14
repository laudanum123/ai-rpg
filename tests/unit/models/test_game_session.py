import pytest
import json
from unittest.mock import patch
from datetime import datetime
from app.models.game_session import GameSession


@pytest.fixture
def sample_game_session():
    """Create a sample game session for testing."""
    return GameSession(
        id="test-session-1",
        character_id="char-123",
        game_world="fantasy"
    )


def test_game_session_creation():
    """Test creating a game session with valid attributes."""
    session = GameSession(
        id="test-session-1",
        character_id="char-123",
        game_world="fantasy",
        created_at="2023-01-01T12:00:00",
        updated_at="2023-01-01T12:00:00"
    )
    
    assert session.id == "test-session-1"
    assert session.character_id == "char-123"
    assert session.game_world == "fantasy"
    assert session.created_at == "2023-01-01T12:00:00"
    assert session.updated_at == "2023-01-01T12:00:00"
    assert session.history == []
    assert session.current_location == {}
    assert session.npcs == {}
    assert session.locations == {}
    assert session.plot_hooks == []
    assert session.active_quests == []
    assert session.completed_quests == []
    assert not session.in_combat
    assert session.combat_state is None


def test_game_session_to_dict(sample_game_session):
    """Test converting a game session to a dictionary."""
    session_dict = sample_game_session.to_dict()
    
    assert isinstance(session_dict, dict)
    assert session_dict["id"] == "test-session-1"
    assert session_dict["character_id"] == "char-123"
    assert session_dict["game_world"] == "fantasy"
    assert "created_at" in session_dict
    assert "updated_at" in session_dict
    assert "history" in session_dict
    assert "current_location" in session_dict
    assert "npcs" in session_dict
    assert "locations" in session_dict
    assert "plot_hooks" in session_dict
    assert "active_quests" in session_dict
    assert "completed_quests" in session_dict
    assert "in_combat" in session_dict
    assert "combat_state" in session_dict


def test_game_session_to_dict_with_custom_attributes():
    """Test to_dict with custom attributes."""
    session = GameSession(
        id="test-session-1",
        character_id="char-123",
        game_world="fantasy"
    )
    
    # Add a custom attribute
    session.custom_attribute = "test_value"
    
    session_dict = session.to_dict()
    
    assert "custom_attribute" in session_dict
    assert session_dict["custom_attribute"] == "test_value"


def test_game_session_to_json(sample_game_session):
    """Test converting a game session to JSON."""
    json_str = sample_game_session.to_json()
    
    assert isinstance(json_str, str)
    
    # Verify we can decode it back to a dictionary
    data = json.loads(json_str)
    assert data["id"] == "test-session-1"
    assert data["character_id"] == "char-123"
    assert data["game_world"] == "fantasy"


def test_game_session_from_dict():
    """Test creating a game session from a dictionary."""
    session_data = {
        "id": "test-session-2",
        "character_id": "char-456",
        "game_world": "sci-fi",
        "created_at": "2023-02-01T10:00:00",
        "updated_at": "2023-02-01T11:00:00",
        "history": [{"role": "system", "content": "Welcome to the game", "timestamp": "2023-02-01T10:00:00"}],
        "current_location": {"id": "loc-1", "name": "Space Station"},
        "npcs": {"npc-1": {"name": "Captain"}},
        "active_quests": [{"id": "quest-1", "title": "Rescue Mission"}]
    }
    
    session = GameSession.from_dict(session_data)
    
    assert session.id == "test-session-2"
    assert session.character_id == "char-456"
    assert session.game_world == "sci-fi"
    assert session.created_at == "2023-02-01T10:00:00"
    assert session.updated_at == "2023-02-01T11:00:00"
    assert len(session.history) == 1
    assert session.history[0]["role"] == "system"
    assert session.current_location["name"] == "Space Station"
    assert "npc-1" in session.npcs
    assert len(session.active_quests) == 1
    assert session.active_quests[0]["title"] == "Rescue Mission"


def test_game_session_from_json():
    """Test creating a game session from a JSON string."""
    json_str = json.dumps({
        "id": "test-session-3",
        "character_id": "char-789",
        "game_world": "post-apocalyptic",
        "created_at": "2023-03-01T10:00:00",
        "updated_at": "2023-03-01T11:00:00"
    })
    
    session = GameSession.from_json(json_str)
    
    assert session.id == "test-session-3"
    assert session.character_id == "char-789"
    assert session.game_world == "post-apocalyptic"
    assert session.created_at == "2023-03-01T10:00:00"
    assert session.updated_at == "2023-03-01T11:00:00"


@patch('app.models.game_session.datetime')
def test_add_message_to_history(mock_datetime, sample_game_session):
    """Test adding a message to the game session history."""
    mock_time = "2023-04-01T12:34:56"
    mock_dt = datetime.fromisoformat(mock_time)
    mock_datetime.now.return_value = mock_dt
    mock_datetime.fromisoformat.return_value = mock_dt
    
    sample_game_session.add_message_to_history("player", "Hello, world!")
    
    assert len(sample_game_session.history) == 1
    assert sample_game_session.history[0]["role"] == "player"
    assert sample_game_session.history[0]["content"] == "Hello, world!"
    assert sample_game_session.history[0]["timestamp"] == mock_time
    assert sample_game_session.updated_at == mock_time


@patch('app.models.game_session.datetime')
def test_add_npc(mock_datetime, sample_game_session):
    """Test adding an NPC to the game session."""
    mock_time = "2023-04-01T12:34:56"
    mock_dt = datetime.fromisoformat(mock_time)
    mock_datetime.now.return_value = mock_dt
    
    npc_data = {
        "id": "npc-1",
        "name": "Gandalf",
        "description": "A wise wizard",
        "disposition": "friendly"
    }
    
    sample_game_session.add_npc("npc-1", npc_data)
    
    assert "npc-1" in sample_game_session.npcs
    assert sample_game_session.npcs["npc-1"] == npc_data
    assert sample_game_session.updated_at == mock_time


def test_get_npc(sample_game_session):
    """Test getting an NPC from the game session."""
    npc_data = {
        "id": "npc-2",
        "name": "Aragorn",
        "description": "A ranger from the North",
        "disposition": "friendly"
    }
    
    sample_game_session.npcs["npc-2"] = npc_data
    
    retrieved_npc = sample_game_session.get_npc("npc-2")
    
    assert retrieved_npc == npc_data
    
    # Test getting a non-existent NPC
    assert sample_game_session.get_npc("nonexistent-npc") is None


@patch('app.models.game_session.datetime')
def test_add_location(mock_datetime, sample_game_session):
    """Test adding a location to the game session."""
    mock_time = "2023-04-01T12:34:56"
    mock_dt = datetime.fromisoformat(mock_time)
    mock_datetime.now.return_value = mock_dt
    
    location_data = {
        "id": "loc-1",
        "name": "Rivendell",
        "description": "An elven outpost"
    }
    
    sample_game_session.add_location("loc-1", location_data)
    
    assert "loc-1" in sample_game_session.locations
    assert sample_game_session.locations["loc-1"] == location_data
    assert sample_game_session.updated_at == mock_time


def test_get_location(sample_game_session):
    """Test getting a location from the game session."""
    location_data = {
        "id": "loc-2",
        "name": "Mordor",
        "description": "A dark and dangerous land"
    }
    
    sample_game_session.locations["loc-2"] = location_data
    
    retrieved_location = sample_game_session.get_location("loc-2")
    
    assert retrieved_location == location_data
    
    # Test getting a non-existent location
    assert sample_game_session.get_location("nonexistent-location") is None


@patch('app.models.game_session.datetime')
def test_set_current_location(mock_datetime, sample_game_session):
    """Test setting the current location."""
    mock_time = "2023-04-01T12:34:56"
    mock_dt = datetime.fromisoformat(mock_time)
    mock_datetime.now.return_value = mock_dt
    
    location_data = {
        "id": "loc-3",
        "name": "The Shire",
        "description": "A peaceful region inhabited by hobbits"
    }
    
    sample_game_session.set_current_location(location_data)
    
    assert sample_game_session.current_location == location_data
    assert sample_game_session.updated_at == mock_time


@patch('app.models.game_session.datetime')
def test_add_plot_hook(mock_datetime, sample_game_session):
    """Test adding a plot hook."""
    mock_time = "2023-04-01T12:34:56"
    mock_dt = datetime.fromisoformat(mock_time)
    mock_datetime.now.return_value = mock_dt
    
    plot_hook = {
        "id": "hook-1",
        "title": "The One Ring",
        "description": "A powerful artifact that must be destroyed"
    }
    
    sample_game_session.add_plot_hook(plot_hook)
    
    assert len(sample_game_session.plot_hooks) == 1
    assert sample_game_session.plot_hooks[0] == plot_hook
    assert sample_game_session.updated_at == mock_time


@patch('app.models.game_session.datetime')
def test_start_combat(mock_datetime, sample_game_session):
    """Test starting a combat encounter."""
    mock_time = "2023-04-01T12:34:56"
    mock_dt = datetime.fromisoformat(mock_time)
    mock_datetime.now.return_value = mock_dt
    
    enemies = [
        {"id": "enemy-1", "name": "Orc", "health": 20},
        {"id": "enemy-2", "name": "Goblin", "health": 10}
    ]
    
    sample_game_session.start_combat(enemies)
    
    assert sample_game_session.in_combat is True
    assert sample_game_session.combat_state is not None
    assert sample_game_session.combat_state["round"] == 1
    assert sample_game_session.combat_state["enemies"] == enemies
    assert sample_game_session.combat_state["turn_order"] == []
    assert sample_game_session.combat_state["current_turn"] == 0
    assert sample_game_session.combat_state["log"] == []
    assert sample_game_session.updated_at == mock_time


@patch('app.models.game_session.datetime')
def test_end_combat_without_log(mock_datetime, sample_game_session):
    """Test ending a combat encounter without combat log."""
    mock_time = "2023-04-01T12:34:56"
    mock_dt = datetime.fromisoformat(mock_time)
    mock_datetime.now.return_value = mock_dt
    
    # Start combat first
    sample_game_session.start_combat([{"id": "enemy-1", "name": "Orc", "health": 20}])
    
    # Clear history to test that no message is added when log is empty
    sample_game_session.history = []
    
    sample_game_session.end_combat()
    
    assert sample_game_session.in_combat is False
    assert sample_game_session.combat_state is None
    assert len(sample_game_session.history) == 0  # No message should be added
    assert sample_game_session.updated_at == mock_time


@patch('app.models.game_session.datetime')
def test_end_combat_with_log(mock_datetime, sample_game_session):
    """Test ending a combat encounter with combat log."""
    mock_time = "2023-04-01T12:34:56"
    mock_dt = datetime.fromisoformat(mock_time)
    mock_datetime.now.return_value = mock_dt
    
    # Start combat first
    sample_game_session.start_combat([{"id": "enemy-1", "name": "Orc", "health": 20}])
    
    # Add some combat log entries
    sample_game_session.combat_state["log"] = [
        {"round": 1, "message": "Combat started", "timestamp": "2023-04-01T12:30:00"},
        {"round": 1, "message": "Player attacks Orc", "timestamp": "2023-04-01T12:31:00"}
    ]
    sample_game_session.combat_state["round"] = 2
    
    sample_game_session.end_combat()
    
    assert sample_game_session.in_combat is False
    assert sample_game_session.combat_state is None
    assert len(sample_game_session.history) == 1
    assert "Combat ended after 2 rounds" in sample_game_session.history[0]["content"]
    assert sample_game_session.updated_at == mock_time


@patch('app.models.game_session.datetime')
def test_next_combat_round(mock_datetime, sample_game_session):
    """Test advancing to the next combat round."""
    mock_time = "2023-04-01T12:34:56"
    mock_dt = datetime.fromisoformat(mock_time)
    mock_datetime.now.return_value = mock_dt
    
    # Start combat first
    sample_game_session.start_combat([{"id": "enemy-1", "name": "Orc", "health": 20}])
    
    # Set the current turn to something other than 0
    sample_game_session.combat_state["current_turn"] = 2
    
    initial_round = sample_game_session.combat_state["round"]
    
    sample_game_session.next_combat_round()
    
    assert sample_game_session.combat_state["round"] == initial_round + 1
    assert sample_game_session.combat_state["current_turn"] == 0
    assert sample_game_session.updated_at == mock_time


def test_next_combat_round_not_in_combat(sample_game_session):
    """Test next_combat_round when not in combat."""
    # Make sure we're not in combat
    sample_game_session.in_combat = False
    sample_game_session.combat_state = None
    
    # This should not raise any exceptions
    sample_game_session.next_combat_round()
    
    # Combat state should still be None
    assert sample_game_session.combat_state is None


@patch('app.models.game_session.datetime')
def test_add_combat_log(mock_datetime, sample_game_session):
    """Test adding an entry to the combat log."""
    mock_time = "2023-04-01T12:34:56"
    mock_dt = datetime.fromisoformat(mock_time)
    mock_datetime.now.return_value = mock_dt
    
    # Start combat first
    sample_game_session.start_combat([{"id": "enemy-1", "name": "Orc", "health": 20}])
    
    sample_game_session.add_combat_log("Player attacks Orc for 5 damage")
    
    assert len(sample_game_session.combat_state["log"]) == 1
    assert sample_game_session.combat_state["log"][0]["round"] == 1
    assert sample_game_session.combat_state["log"][0]["message"] == "Player attacks Orc for 5 damage"
    assert sample_game_session.combat_state["log"][0]["timestamp"] == mock_time
    assert sample_game_session.updated_at == mock_time


def test_add_combat_log_not_in_combat(sample_game_session):
    """Test add_combat_log when not in combat."""
    # Make sure we're not in combat
    sample_game_session.in_combat = False
    sample_game_session.combat_state = None
    
    # This should not raise any exceptions
    sample_game_session.add_combat_log("This message should not be logged")
    
    # Combat state should still be None
    assert sample_game_session.combat_state is None


@patch('app.models.game_session.datetime')
def test_add_quest(mock_datetime, sample_game_session):
    """Test adding a quest."""
    mock_time = "2023-04-01T12:34:56"
    mock_dt = datetime.fromisoformat(mock_time)
    mock_datetime.now.return_value = mock_dt
    
    quest = {
        "id": "quest-1",
        "title": "The Missing Artifact",
        "description": "Find the lost artifact in the ancient ruins",
        "reward": {"xp": 100, "gold": 50}
    }
    
    sample_game_session.add_quest(quest)
    
    assert len(sample_game_session.active_quests) == 1
    assert sample_game_session.active_quests[0] == quest
    assert sample_game_session.updated_at == mock_time


@patch('app.models.game_session.datetime')
def test_complete_quest(mock_datetime, sample_game_session):
    """Test completing a quest."""
    mock_time = "2023-04-01T12:34:56"
    mock_dt = datetime.fromisoformat(mock_time)
    mock_datetime.now.return_value = mock_dt
    
    quest1 = {
        "id": "quest-1",
        "title": "The Missing Artifact",
        "description": "Find the lost artifact in the ancient ruins"
    }
    
    quest2 = {
        "id": "quest-2",
        "title": "Defeat the Dragon",
        "description": "Slay the dragon terrorizing the village"
    }
    
    # Add quests to active quests
    sample_game_session.active_quests = [quest1, quest2]
    
    # Complete quest1
    completed_quest = sample_game_session.complete_quest("quest-1")
    
    assert completed_quest == quest1
    assert len(sample_game_session.active_quests) == 1
    assert sample_game_session.active_quests[0] == quest2
    assert len(sample_game_session.completed_quests) == 1
    assert sample_game_session.completed_quests[0] == quest1
    assert sample_game_session.updated_at == mock_time


def test_complete_nonexistent_quest(sample_game_session):
    """Test completing a quest that doesn't exist."""
    quest = {
        "id": "quest-1",
        "title": "The Missing Artifact",
        "description": "Find the lost artifact in the ancient ruins"
    }
    
    # Add a quest to active quests
    sample_game_session.active_quests = [quest]
    
    # Try to complete a non-existent quest
    result = sample_game_session.complete_quest("nonexistent-quest")
    
    assert result is None
    assert len(sample_game_session.active_quests) == 1
    assert sample_game_session.active_quests[0] == quest
    assert len(sample_game_session.completed_quests) == 0
