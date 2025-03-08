from flask import Blueprint, render_template, redirect, url_for, request, session, jsonify, flash, current_app
from datetime import datetime
from app.services.character_service import CharacterService
from app.services.game_state_service import GameStateService
from app.services.game_master import GameMaster

main = Blueprint('main', __name__)
character_service = CharacterService()
game_state_service = GameStateService()
game_master = GameMaster()

@main.context_processor
def inject_now():
    return {'now': datetime.now()}

@main.route('/')
def index():
    """Home page."""
    return render_template('index.html')

@main.route('/new-character', methods=['GET', 'POST'])
def new_character():
    """Create new character page."""
    if request.method == 'POST':
        name = request.form.get('name')
        character_class = request.form.get('character_class')
        
        if not name or not character_class:
            flash('Please provide both name and class.')
            return redirect(url_for('main.new_character'))
        
        character = character_service.create_character(name, character_class)
        session['character_id'] = character.id
        
        # Create a new game session
        game_world = request.form.get('game_world', 'Fantasy')
        session_id = game_state_service.create_session(character.id, game_world)
        session['session_id'] = session_id
        
        return redirect(url_for('main.game'))
    
    return render_template('new_character.html')

@main.route('/game')
def game():
    """Main game page."""
    character_id = session.get('character_id')
    session_id = session.get('session_id')
    
    if not character_id or not session_id:
        return redirect(url_for('main.new_character'))
    
    character = character_service.get_character(character_id)
    game_session = game_state_service.get_session(session_id)
    
    if not character or not game_session:
        # Invalid session or character, start fresh
        session.pop('character_id', None)
        session.pop('session_id', None)
        return redirect(url_for('main.new_character'))
    
    return render_template(
        'game.html', 
        character=character,
        game_session=game_session,
        in_combat=game_session.in_combat
    )

@main.route('/character-sheet')
def character_sheet():
    """Character sheet page."""
    character_id = session.get('character_id')
    
    if not character_id:
        return redirect(url_for('main.new_character'))
    
    character = character_service.get_character(character_id)
    
    if not character:
        session.pop('character_id', None)
        return redirect(url_for('main.new_character'))
    
    return render_template('character_sheet.html', character=character)

@main.route('/action', methods=['POST'])
def process_action():
    """Process a player's action via AJAX."""
    character_id = session.get('character_id')
    session_id = session.get('session_id')
    
    if not character_id or not session_id:
        return jsonify({"error": "No active game session."}), 400
    
    action = request.json.get('action')
    if not action:
        return jsonify({"error": "No action provided."}), 400
    
    character = character_service.get_character(character_id)
    game_session = game_state_service.get_session(session_id)
    
    if not character or not game_session:
        return jsonify({"error": "Invalid session or character."}), 400
    
    # Record player action in history
    game_state_service.add_message_to_history(session_id, "player", action)
    
    # Process the action with the game master
    if game_session.in_combat:
        # Combat action
        result = game_master.process_combat_action(game_session, character, action)
        game_state_service.add_message_to_history(session_id, "gm", result["description"])
        
        # Update character if needed
        if "damage_taken" in result.get("effects", {}):
            character.take_damage(result["effects"]["damage_taken"])
            character_service.update_character(character)
        
        # Check if combat has ended
        if not result.get("combat_continues", True):
            game_state_service.end_combat(session_id)
        
        return jsonify({
            "message": result["description"],
            "character": character.to_dict(),
            "in_combat": game_session.in_combat,
            "combat_state": game_session.combat_state
        })
    else:
        # Regular action
        response = game_master.process_action(game_session, character, action)
        game_state_service.add_message_to_history(session_id, "gm", response)
        
        return jsonify({
            "message": response,
            "character": character.to_dict(),
            "location": game_session.current_location,
            "in_combat": game_session.in_combat
        })

@main.route('/start-combat', methods=['POST'])
def start_combat():
    """Start a combat encounter."""
    character_id = session.get('character_id')
    session_id = session.get('session_id')
    
    if not character_id or not session_id:
        return jsonify({"error": "No active game session."}), 400
    
    enemy_type = request.json.get('enemy_type', 'random')
    
    character = character_service.get_character(character_id)
    game_session = game_state_service.get_session(session_id)
    
    if not character or not game_session:
        return jsonify({"error": "Invalid session or character."}), 400
    
    if game_session.in_combat:
        return jsonify({"error": "Already in combat."}), 400
    
    # Start combat with the game master
    combat_data = game_master.start_combat(game_session, character, enemy_type)
    
    # Record combat start in history
    game_state_service.add_message_to_history(session_id, "gm", combat_data["message"])
    
    # Update game session with combat state
    game_state_service.start_combat(session_id, combat_data["encounter"]["enemies"])
    
    return jsonify({
        "message": combat_data["message"],
        "combat_state": game_session.combat_state,
        "in_combat": True
    })

@main.route('/end-combat', methods=['POST'])
def end_combat():
    """End the current combat encounter."""
    session_id = session.get('session_id')
    
    if not session_id:
        return jsonify({"error": "No active game session."}), 400
    
    game_session = game_state_service.get_session(session_id)
    
    if not game_session or not game_session.in_combat:
        return jsonify({"error": "No active combat."}), 400
    
    # End combat
    game_state_service.end_combat(session_id)
    
    # Record combat end in history
    game_state_service.add_message_to_history(session_id, "gm", "Combat has ended.")
    
    return jsonify({
        "message": "Combat has ended.",
        "in_combat": False
    })

@main.route('/logout')
def logout():
    """Clear session and start fresh."""
    session.clear()
    return redirect(url_for('main.index'))


@main.route('/debug', methods=['GET', 'POST'])
def debug():
    """API debug page to view OpenAI API requests and responses."""
    if request.method == 'POST':
        if 'toggle_debug' in request.form:
            current_app.config['API_DEBUG'] = not current_app.config.get('API_DEBUG', False)
            status = 'enabled' if current_app.config['API_DEBUG'] else 'disabled'
            flash(f'API debug mode {status}')
        elif 'clear_logs' in request.form:
            game_master.api_debug_logs.clear()
            flash('Debug logs cleared')
        
        return redirect(url_for('main.debug'))
    
    # Convert deque to list for template display
    api_logs = list(game_master.api_debug_logs)
    
    return render_template(
        'debug.html', 
        api_logs=api_logs,
        debug_enabled=current_app.config.get('API_DEBUG', False)
    ) 