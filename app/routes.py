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

@main.route('/sessions')
def sessions():
    """Manage game sessions page."""
    # Get all available sessions
    all_sessions = game_state_service.get_all_sessions()
    
    # Get character data for each session
    sessions_data = []
    for game_session in all_sessions:
        character = character_service.get_character(game_session.character_id)
        if character:
            # Convert ISO format strings to datetime objects
            created_at = datetime.fromisoformat(game_session.created_at)
            updated_at = datetime.fromisoformat(game_session.updated_at)
            
            sessions_data.append({
                "session": game_session,
                "character": character,
                "created_at": created_at,
                "last_updated": updated_at
            })
    
    # Sort sessions by last updated (most recent first)
    sessions_data.sort(key=lambda x: x["last_updated"], reverse=True)
    
    return render_template('sessions.html', sessions=sessions_data)

@main.route('/load-session/<session_id>')
def load_session(session_id):
    """Load an existing game session."""
    if not game_state_service.is_valid_session(session_id):
        flash('Invalid session ID.', 'error')
        return redirect(url_for('main.sessions'))
    
    # Get the session and set it as active
    game_session = game_state_service.get_session(session_id)
    session['session_id'] = session_id
    session['character_id'] = game_session.character_id
    
    flash(f'Session loaded successfully!', 'success')
    return redirect(url_for('main.game'))

@main.route('/delete-session/<session_id>', methods=['POST'])
def delete_session(session_id):
    """Delete a game session."""
    # Check if this is the active session
    is_active = session.get('session_id') == session_id
    
    # Delete the session
    success = game_state_service.delete_session(session_id)
    
    if success:
        # If we deleted the active session, remove it from the user's session
        if is_active:
            session.pop('session_id', None)
            # Don't remove character_id as they might want to start a new session with the same character
        
        flash('Session deleted successfully.', 'success')
    else:
        flash('Failed to delete session.', 'error')
    
    return redirect(url_for('main.sessions'))

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
        game_state_service.set_session_attribute(session_id, 'intro_generated', True)
        
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
    
    # Check if intro has been generated for this session
    # First try to get the attribute directly
    intro_generated = False
    if hasattr(game_session, 'intro_generated'):
        intro_generated = bool(game_session.intro_generated)
    
    # If this is a new session that hasn't generated an intro yet
    if not intro_generated:
        intro_text = game_master.start_game(character, game_session.game_world, session_id=session_id)
        game_state_service.add_message_to_history(session_id, "gm", intro_text)
        game_state_service.set_session_attribute(session_id, 'intro_generated', True)
        # Make sure to save the session with the updated attribute
        game_session = game_state_service.get_session(session_id)
    
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
        memory_graph = game_master.get_session_memory_graph(session_id)
        memory_graph.add_node(
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
        
        # Make sure character data is fully up-to-date after processing
        character = character_service.get_character(character_id)
        
        return jsonify({
            "message": response,
            "character": character.to_dict(),
            "location": game_session.current_location,
            "in_combat": game_session.in_combat,
            "inventory": character.inventory  # Explicitly include inventory for frontend
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

@main.route('/delete-memory-node', methods=['POST'])
def delete_memory_node():
    """Delete a specific memory node from a session."""
    session_id = request.form.get('session_id')
    node_id = request.form.get('node_id')
    
    if not session_id or not node_id:
        flash('Missing session ID or node ID', 'danger')
        return redirect(url_for('main.memory_debug'))
    
    # Attempt to delete the node
    success = game_master.delete_memory_node(session_id, node_id)
    
    if success:
        flash(f'Memory node deleted successfully', 'success')
    else:
        flash(f'Failed to delete memory node', 'danger')
    
    # Redirect back to memory debug with the same session and view mode
    return redirect(url_for('main.memory_debug', 
                           session_id=session_id, 
                           view_mode='browse'))


@main.route('/memory-debug', methods=['GET', 'POST'])
def memory_debug():
    """Memory graph debug page to visualize and test the memory system."""
    # Initialize variables
    query = ""
    node_limit = 5
    relevant_memories = []
    selected_session_id = request.args.get('session_id', None)
    
    # Get all available sessions
    all_sessions = game_state_service.get_all_sessions()
    sessions_data = []
    
    for game_session in all_sessions:
        character = character_service.get_character(game_session.character_id)
        if character:
            # Convert ISO format strings to datetime objects
            updated_at = datetime.fromisoformat(game_session.updated_at)
            
            sessions_data.append({
                "session_id": game_session.id,
                "character_name": character.name,
                "last_updated": updated_at
            })
    
    # Sort sessions by last updated (most recent first)
    sessions_data.sort(key=lambda x: x["last_updated"], reverse=True)
    
    # If no session is selected and there are sessions available, use the first one
    if not selected_session_id and sessions_data:
        selected_session_id = sessions_data[0]["session_id"]
        
    # Get the memory graph for the selected session
    memory_graph = None
    if selected_session_id:
        memory_graph = game_master.get_session_memory_graph(selected_session_id)
    
    # Handle memory query form submission
    if request.method == 'POST':
        query = request.form.get('query', '')
        node_limit = int(request.form.get('node_limit', 5))
        selected_session_id = request.form.get('session_id', selected_session_id)
        
        # Re-fetch the memory graph if the session ID came from the form
        if selected_session_id:
            memory_graph = game_master.get_session_memory_graph(selected_session_id)
        
        if query and memory_graph:
            # Get relevant memories based on the query
            memory_context = memory_graph.get_relevant_context(query, node_limit, 2000)
            
            # Get the actual memory nodes for display
            # This requires some custom logic since get_relevant_context returns formatted text
            query_embedding = memory_graph.get_embedding(query)
            
            # Find nodes with highest similarity
            similarity_scores = {}
            for node_id, node in memory_graph.nodes.items():
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
                node = memory_graph.nodes[node_id]
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
    
    # Collect memory statistics if we have a memory graph
    memory_count = 0
    relation_count = 0
    type_counts = {}
    importance_values = []
    last_updated = 'No data'
    
    if memory_graph:
        memory_count = len(memory_graph.nodes)
        relation_count = sum(len(rel_list) for rel_list in memory_graph.relations.values())
        
        # Count nodes by type
        for node in memory_graph.nodes.values():
            node_type = node.metadata['type']
            type_counts[node_type] = type_counts.get(node_type, 0) + 1
            importance_values.append(node.metadata['importance'])
        
        # Calculate average importance
        avg_importance = statistics.mean(importance_values) if importance_values else 0
        
        # Get timestamp of last update
        if memory_count > 0:
            last_updated = max([datetime.fromisoformat(node.metadata['timestamp']) 
                            for node in memory_graph.nodes.values()])
    
    # Get view mode from request args or form data
    view_mode = request.args.get('view_mode', request.form.get('view_mode', 'stats'))
    
    # Variables for browse mode
    memory_nodes = []
    sort_by = request.args.get('sort_by', 'timestamp')
    sort_order = request.args.get('sort_order', 'desc')
    selected_node = None
    node_relations = None
    
    # If in browse mode, get all memory nodes
    if view_mode == 'browse' and memory_graph:
        memory_nodes = game_master.get_memory_nodes(
            selected_session_id,
            sort_by=sort_by,
            reverse=(sort_order == 'desc')
        )
        
        # If a node is selected, get its details
        selected_node_id = request.args.get('node_id', None)
        if selected_node_id:
            # Find the selected node in memory_nodes
            for node in memory_nodes:
                if node['id'] == selected_node_id:
                    selected_node = node
                    break
            
            # Get relations for the selected node
            if selected_node:
                node_relations = game_master.get_node_relations(selected_session_id, selected_node_id)
    
    return render_template(
        'memory_debug.html',
        memory_count=memory_count,
        type_counts=type_counts,
        relation_count=relation_count,
        avg_importance=avg_importance if 'avg_importance' in locals() else 0,
        last_updated=last_updated,
        query=query,
        node_limit=node_limit,
        relevant_memories=relevant_memories,
        sessions=sessions_data,
        selected_session_id=selected_session_id,
        view_mode=view_mode,
        memory_nodes=memory_nodes,
        sort_by=sort_by,
        sort_order=sort_order,
        selected_node=selected_node,
        node_relations=node_relations
    )