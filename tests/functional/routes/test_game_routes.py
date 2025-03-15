import json
from unittest.mock import patch


def test_index_route(client):
    """Test the index route returns the home page."""
    response = client.get('/')

    assert response.status_code == 200
    assert b'<!DOCTYPE html>' in response.data


def test_new_character_get(client):
    """Test the new character page loads correctly."""
    response = client.get('/new-character')

    assert response.status_code == 200
    assert b'Create New Character' in response.data or b'character' in response.data.lower()


def test_new_character_post(client, mock_character_service, mock_game_state_service, mock_game_master):
    """Test creating a new character."""
    # Set up the services to return appropriate values
    mock_character_service.create_character.return_value.id = "test-char-123"
    mock_game_state_service.create_session.return_value = "test-session-456"

    with patch.object(mock_game_master, 'start_game', return_value="Welcome to your adventure!"):
        response = client.post('/new-character', data={
            'name': 'Merlin',
            'character_class': 'Wizard',
            'game_world': 'Fantasy'
        }, follow_redirects=True)

    assert response.status_code == 200
    assert b'game' in response.data.lower()  # Should redirect to game page

    # Verify the services were called correctly
    mock_character_service.create_character.assert_called_with('Merlin', 'Wizard')
    mock_game_state_service.create_session.assert_called()
    mock_game_master.start_game.assert_called()


def test_game_without_session(client):
    """Test the game route redirects to new character if no session exists."""
    response = client.get('/game', follow_redirects=True)

    assert response.status_code == 200
    assert b'Create New Character' in response.data or b'character' in response.data.lower()


def test_game_with_session(client, mock_character_service, mock_game_state_service):
    """Test the game route loads the game page with an active session."""
    # Configure the session
    with client.session_transaction() as sess:
        sess['character_id'] = "test-char-123"
        sess['session_id'] = "test-session-456"

    response = client.get('/game')

    assert response.status_code == 200
    assert b'game' in response.data.lower()

    # Verify the services were called correctly
    mock_character_service.get_character.assert_called_with("test-char-123")
    mock_game_state_service.get_session.assert_called_with("test-session-456")


def test_character_sheet(client, mock_character_service):
    """Test the character sheet route displays character information."""
    # Configure the session
    with client.session_transaction() as sess:
        sess['character_id'] = "test-char-123"

    response = client.get('/character-sheet')

    assert response.status_code == 200
    assert b'Character Sheet' in response.data or b'character' in response.data.lower()

    # Verify the character service was called
    mock_character_service.get_character.assert_called_with("test-char-123")


def test_process_action(client, mock_character_service, mock_game_state_service, mock_game_master):
    """Test processing a player action via AJAX."""
    # Configure the session
    with client.session_transaction() as sess:
        sess['character_id'] = "test-char-123"
        sess['session_id'] = "test-session-456"

    # Set up the game master mock to return a response
    with patch.object(mock_game_master, 'process_player_action', return_value="You find a hidden door."):
        # Make the request
        response = client.post('/action',
                             json={'action': 'I search the room carefully'},
                             content_type='application/json')

    assert response.status_code == 200
    data = json.loads(response.data)
    assert "result" in data
    assert "You find a hidden door" in data["result"]

    # Verify services were called correctly
    mock_character_service.get_character.assert_called_with("test-char-123")
    mock_game_state_service.get_session.assert_called_with("test-session-456")
    mock_game_state_service.add_message_to_history.assert_called()
    mock_game_master.process_player_action.assert_called()


def test_start_combat(client, mock_character_service, mock_game_state_service, mock_game_master):
    """Test starting a combat encounter."""
    # Configure the session
    with client.session_transaction() as sess:
        sess['character_id'] = "test-char-123"
        sess['session_id'] = "test-session-456"

    # Set up the game master mock to return a combat result
    with patch.object(mock_game_master, 'start_combat') as mock_start_combat:
        mock_start_combat.return_value = {
            "description": "Goblins attack!",
            "encounter": {
                "enemies": [{"name": "Goblin", "health": 10, "max_health": 10}],
                "is_active": True
            }
        }

        # Make the request
        response = client.post('/start-combat',
                             json={'enemy_types': ['Goblin'], 'difficulty': 'medium'},
                             content_type='application/json')

    assert response.status_code == 200
    data = json.loads(response.data)
    assert "result" in data
    assert "Goblins attack" in data["result"]
    assert "in_combat" in data
    assert data["in_combat"] is True

    # Verify services were called correctly
    mock_character_service.get_character.assert_called_with("test-char-123")
    mock_game_state_service.get_session.assert_called_with("test-session-456")
    mock_game_master.start_combat.assert_called()
    mock_game_state_service.update_session.assert_called()
