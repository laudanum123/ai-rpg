import pytest
from unittest.mock import patch, MagicMock

def test_character_creation_and_session_start(mock_character_service, mock_game_state_service, mock_game_master):
    """Test the full flow of creating a character and starting a game session."""
    # Set up return values from service calls
    character_name = "Elric"
    character_class = "Sorcerer"
    game_world = "Dark Fantasy"
    
    # Create a character
    character = mock_character_service.create_character(character_name, character_class)
    assert character.name == character_name
    assert character.character_class == character_class
    
    # Create a game session
    session_id = mock_game_state_service.create_session(character.id, game_world)
    assert session_id is not None
    
    # Start the game
    with patch.object(mock_game_master, 'get_ai_response', return_value="Welcome to the dark world of magic and monsters!"):
        intro_text = mock_game_master.start_game(character, game_world, session_id)
        assert "Welcome" in intro_text
    
    # Verify game state is correctly updated
    mock_game_state_service.add_message_to_history.assert_called_with(
        session_id, "gm", intro_text
    )


def test_combat_flow(mock_character_service, mock_game_state_service, mock_game_master):
    """Test the flow of starting, executing, and ending combat."""
    # Get character and session
    character = mock_character_service.get_character("test-char-123")
    session = mock_game_state_service.get_session("test-session-456")
    
    # Ensure session isn't in combat yet
    session.in_combat = False
    
    # Start combat
    with patch.object(mock_game_master, 'get_ai_response', return_value="A band of orcs appears!"):
        start_result = mock_game_master.start_combat(
            session, character, ["Orc"], "medium"
        )
    
    # Verify combat started correctly
    assert session.in_combat
    assert session.current_combat is not None
    assert len(session.current_combat.enemies) > 0
    
    # Execute a combat action
    with patch.object(mock_game_master, 'get_ai_response', return_value="You hit the orc for 7 damage."):
        with patch.object(mock_game_master, 'parse_combat_result') as mock_parse:
            mock_parse.return_value = {
                "description": "You hit the orc for 7 damage.",
                "damage_dealt": 7,
                "enemy_defeated": False,
                "player_damage_taken": 0
            }
            
            action_result = mock_game_master.process_combat_action(
                session, character, "I attack with my sword"
            )
    
    # Verify action processed correctly
    assert "damage_dealt" in action_result
    assert action_result["damage_dealt"] == 7
    
    # End combat
    with patch.object(mock_game_master, 'get_ai_response', return_value="You've defeated all enemies and found 50 gold!"):
        end_result = mock_game_master.end_combat(
            session, character, victory=True
        )
    
    # Verify combat ended correctly
    assert not session.in_combat
    assert session.current_combat is None
    assert "experience_gained" in end_result


def test_memory_integration(mock_character_service, mock_game_state_service, mock_game_master, mock_memory_graph):
    """Test the integration of the memory system with game actions."""
    # Set up game elements
    character = mock_character_service.get_character("test-char-123")
    session_id = "test-session-456"
    
    # Set up the memory graph in the game master
    mock_game_master.memory_graphs[session_id] = mock_memory_graph
    
    # Player performs an action that should be recorded in memory
    player_action = "I talk to the innkeeper and ask about the missing villagers."
    
    with patch.object(mock_game_master, 'get_ai_response', return_value="The innkeeper tells you about rumors of kidnappings in the nearby forest."):
        # Process the action
        result = mock_game_master.process_player_action(
            mock_game_state_service.get_session(session_id),
            character,
            player_action
        )
    
    # Verify the result
    assert "innkeeper" in result.lower()
    assert "kidnappings" in result.lower()
    
    # Verify memory interaction
    mock_memory_graph.add_node.assert_called()
    
    # Now verify retrieval by simulating a follow-up question
    mock_memory_graph.get_relevant_context.return_value = "The innkeeper mentioned kidnappings in the forest."
    
    follow_up_action = "I ask if anyone has investigated the forest."
    
    with patch.object(mock_game_master, 'get_ai_response', return_value="Nobody has dared to enter the forest since the disappearances began."):
        # Process the follow-up action
        result = mock_game_master.process_player_action(
            mock_game_state_service.get_session(session_id),
            character,
            follow_up_action
        )
    
    # Verify memory was used in generating the response
    mock_memory_graph.get_relevant_context.assert_called()
    assert "forest" in result.lower()
    assert "disappearances" in result.lower()
