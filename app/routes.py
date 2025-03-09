from flask import Blueprint, render_template, redirect, url_for, request, session, jsonify, flash, current_app
from datetime import datetime
import statistics
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
        
        # Initialize game with memory tracking
        intro_text = game_master.start_game(character, game_world, session_id=session_id)
        game_state_service.add_message_to_history(session_id, "gm", intro_text)
        
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
    
    # If this is a new session that hasn't been initialized with memory graph yet
    if not hasattr(game_session, 'intro_generated') or not game_session.intro_generated:
        intro_text = game_master.start_game(character, game_session.game_world, session_id=session_id)
        game_state_service.add_message_to_history(session_id, "gm", intro_text)
        game_state_service.set_session_attribute(session_id, 'intro_generated', True)
    
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
        result_desc = result["description"]
        game_state_service.add_message_to_history(session_id, "gm", result_desc)
        
        # Store combat events in memory graph (with session ID for context retrieval)
        combat_memory = f"Combat action: {action}\nResult: {result_desc}"
        game_master.memory_graph.add_node(
            content=combat_memory,
            node_type="combat",
            importance=0.9  # Combat events are high importance
        )
        
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

@main.route('/memory-debug', methods=['GET', 'POST'])
def memory_debug():
    """Memory graph debug page to visualize and test the memory system."""
    # Initialize variables
    query = ""
    node_limit = 5
    relevant_memories = []
    
    # Handle memory query form submission
    if request.method == 'POST':
        query = request.form.get('query', '')
        node_limit = int(request.form.get('node_limit', 5))
        
        if query:
            # Get relevant memories based on the query
            memory_context = game_master.memory_graph.get_relevant_context(query, node_limit, 2000)
            
            # Get the actual memory nodes for display
            # This requires some custom logic since get_relevant_context returns formatted text
            query_embedding = game_master.memory_graph.get_embedding(query)
            
            # Find nodes with highest similarity
            similarity_scores = {}
            for node_id, node in game_master.memory_graph.nodes.items():
                if node.embedding is not None:
                    import numpy as np
                    from sklearn.metrics.pairwise import cosine_similarity
                    similarity = cosine_similarity([query_embedding], [node.embedding])[0][0]
                    similarity_scores[node_id] = similarity
            
            # Get top nodes
            ranked_nodes = sorted(
                similarity_scores.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:node_limit]
            
            # Prepare nodes for display
            for node_id, _ in ranked_nodes:
                node = game_master.memory_graph.nodes[node_id]
                # Add color based on node type
                node_type_colors = {
                    'event': 'primary',
                    'character': 'success',
                    'location': 'warning',
                    'combat': 'danger',
                    'item': 'info'
                }
                node_copy = node.to_dict()
                node_copy['type_color'] = node_type_colors.get(node.metadata['type'], 'secondary')
                relevant_memories.append(node_copy)
    
    # Collect memory statistics
    memory_count = len(game_master.memory_graph.nodes)
    relation_count = sum(len(rel_list) for rel_list in game_master.memory_graph.relations.values())
    
    # Count nodes by type
    type_counts = {}
    importance_values = []
    
    for node in game_master.memory_graph.nodes.values():
        node_type = node.metadata['type']
        type_counts[node_type] = type_counts.get(node_type, 0) + 1
        importance_values.append(node.metadata['importance'])
    
    # Calculate average importance
    avg_importance = statistics.mean(importance_values) if importance_values else 0
    
    # Get timestamp of last update
    last_updated = max([datetime.fromisoformat(node.metadata['timestamp']) 
                       for node in game_master.memory_graph.nodes.values()]) if memory_count > 0 else 'No data'
    
    return render_template(
        'memory_debug.html',
        memory_count=memory_count,
        type_counts=type_counts,
        relation_count=relation_count,
        avg_importance=avg_importance,
        last_updated=last_updated,
        query=query,
        node_limit=node_limit,
        relevant_memories=relevant_memories
    )