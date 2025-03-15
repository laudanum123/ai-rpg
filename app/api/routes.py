from flask import jsonify, request

from app.api import api_bp
from app.services.character_service import CharacterService
from app.services.game_master import GameMaster
from app.services.game_state_service import GameStateService

game_master = GameMaster()
character_service = CharacterService()
game_state_service = GameStateService()


@api_bp.route("/health", methods=["GET"])
def health_check():
    """Endpoint to check if the API is running."""
    return jsonify({"status": "ok"})


@api_bp.route("/session/start", methods=["POST"])
def start_session():
    """Start a new game session."""
    data = request.json
    character_name = data.get("character_name", "Adventurer")
    character_class = data.get("character_class", "Fighter")
    game_world = data.get("game_world", "Fantasy")

    # Create a new character if it doesn't exist
    character = character_service.get_or_create_character(
        character_name, character_class
    )

    # Initialize game state
    session_id = game_state_service.create_session(
        character_id=character.id, game_world=game_world
    )

    # Get intro message from game master
    intro_message = game_master.start_game(character, game_world)

    return jsonify(
        {
            "session_id": session_id,
            "character": character.to_dict(),
            "message": intro_message,
        }
    )


@api_bp.route("/action", methods=["POST"])
def take_action():
    """Process a player's action in the game."""
    data = request.json
    session_id = data.get("session_id")
    action = data.get("action")

    # Validate session
    if not game_state_service.is_valid_session(session_id):
        return jsonify({"error": "Invalid session"}), 400

    # Get session details
    session = game_state_service.get_session(session_id)
    character = character_service.get_character(session.character_id)

    # Process action with game master
    response = game_master.process_action(session, character, action)

    return jsonify(
        {
            "message": response,
            "character": character.to_dict(),
            "session": session.to_dict(),
        }
    )


@api_bp.route("/combat/start", methods=["POST"])
def start_combat():
    """Initialize a combat encounter."""
    data = request.json
    session_id = data.get("session_id")
    enemy_type = data.get("enemy_type", "random")

    # Validate session
    if not game_state_service.is_valid_session(session_id):
        return jsonify({"error": "Invalid session"}), 400

    # Get session details
    session = game_state_service.get_session(session_id)
    character = character_service.get_character(session.character_id)

    # Start combat with game master
    combat_data = game_master.start_combat(session, character, enemy_type)

    return jsonify(combat_data)


@api_bp.route("/combat/action", methods=["POST"])
def combat_action():
    """Take an action during combat."""
    data = request.json
    session_id = data.get("session_id")
    action = data.get("action")

    # Validate session
    if not game_state_service.is_valid_session(session_id):
        return jsonify({"error": "Invalid session"}), 400

    # Get session details
    session = game_state_service.get_session(session_id)
    character = character_service.get_character(session.character_id)

    # Process combat action with game master
    combat_result = game_master.process_combat_action(session, character, action)

    return jsonify(combat_result)


@api_bp.route("/character/<character_id>", methods=["GET"])
def get_character(character_id):
    """Retrieve character details."""
    character = character_service.get_character(character_id)
    if not character:
        return jsonify({"error": "Character not found"}), 404

    return jsonify(character.to_dict())


@api_bp.route("/inventory/<character_id>", methods=["GET"])
def get_inventory(character_id):
    """Retrieve character inventory."""
    inventory = character_service.get_inventory(character_id)
    return jsonify(inventory)
