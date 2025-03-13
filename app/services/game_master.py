import os
import json
from typing import Dict, List, Any
import random
from datetime import datetime
from openai import OpenAI
from collections import deque
from flask import current_app
from app.models.character import Character
from app.models.npc import NPC
from app.models.combat import Enemy, CombatEncounter, roll_dice
from app.services.memory_graph import MemoryGraph


class GameMaster:
    """Service for AI-powered game mastering using GPT-4o-mini."""

    def __init__(self):
        """Initialize the game master service."""
        # Initialize OpenAI client
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Store for API debug messages (max 50 entries)
        self.api_debug_logs = deque(maxlen=50)
        self.debug_enabled = False

        # Dictionary to store session-specific memory graphs
        self.memory_graphs = {}

        # Base configuration for memory graphs
        self.memory_graph_config = {
            "openai_client": self.client,
            "embedding_model": "text-embedding-3-large",
            "llm_model": "gpt-4o-mini",
        }

        # System prompt for the AI game master
        self.system_prompt = """You are an experienced Game Master for a fantasy RPG game.
        Your role is to create an engaging and dynamic adventure, manage NPCs, describe
        locations vividly, create interesting plot hooks, and run combat encounters.
        Always stay in character as a GM and maintain consistency in the game world.
        Focus on creating an immersive experience while following the game's rules."""

    def get_session_memory_graph(self, session_id: str) -> MemoryGraph:
        """Get or create a memory graph for the specified session.

        Args:
            session_id: The game session ID

        Returns:
            The memory graph for this session
        """
        if session_id not in self.memory_graphs:
            # Create a new memory graph for this session with a session-specific storage directory
            storage_dir = f"data/memory_graph/{session_id}"
            memory_graph = MemoryGraph(
                openai_client=self.memory_graph_config["openai_client"],
                embedding_model=self.memory_graph_config["embedding_model"],
                llm_model=self.memory_graph_config["llm_model"],
                storage_dir=storage_dir,
            )

            # Attempt to load existing memories for this session if available
            try:
                memory_graph.load()
                print(
                    f"Loaded memory graph for session {session_id} with {len(memory_graph.nodes)} nodes"
                )
            except Exception as e:
                print(
                    f"No existing memory graph found for session {session_id} or error loading: {e}"
                )

            self.memory_graphs[session_id] = memory_graph

        return self.memory_graphs[session_id]

    def delete_memory_node(self, session_id: str, node_id: str) -> bool:
        """Delete a specific memory node from a session's memory graph.

        Args:
            session_id: The game session ID
            node_id: The ID of the memory node to delete

        Returns:
            True if the node was deleted successfully, False otherwise
        """
        # Get the memory graph for this session
        memory_graph = self.get_session_memory_graph(session_id)

        # Attempt to delete the node
        return memory_graph.delete_node(node_id)

    def get_memory_nodes(
        self, session_id: str, sort_by: str = "timestamp", reverse: bool = True
    ) -> List[Dict[str, Any]]:
        """Get all memory nodes for a session, optionally sorted.

        Args:
            session_id: The game session ID
            sort_by: Field to sort by ('timestamp', 'importance', 'type')
            reverse: If True, sort in descending order

        Returns:
            List of memory nodes as dictionaries
        """
        memory_graph = self.get_session_memory_graph(session_id)
        return memory_graph.get_all_nodes(sort_by, reverse)

    def get_node_relations(
        self, session_id: str, node_id: str
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Get all relations for a specific memory node.

        Args:
            session_id: The game session ID
            node_id: The ID of the memory node

        Returns:
            Dictionary with incoming and outgoing relations
        """
        memory_graph = self.get_session_memory_graph(session_id)
        return memory_graph.get_node_relations(node_id)

    def get_ai_response(
        self,
        messages: List[Dict],
        session_id: str = None,
        recent_history_turns: int = 5,
    ) -> str:
        """Get a response from the AI model with contextual memory.

        Args:
            messages: List of conversation messages
            session_id: Current game session ID for memory retrieval
            recent_history_turns: Number of recent conversation turns to include (default: 5)

        Returns:
            AI model's response text
        """
        # Check if API debug mode is enabled in the app config
        try:
            self.debug_enabled = current_app.config.get("API_DEBUG", False)
        except RuntimeError:
            # Not in an application context, assume debug is False
            self.debug_enabled = False

        # Extract the current situation from the latest user message
        current_situation = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                current_situation = msg.get("content", "")
                break

        # Retrieve relevant memories if we have a session ID
        memory_context = ""
        if session_id and current_situation:
            # Get the session-specific memory graph
            memory_graph = self.get_session_memory_graph(session_id)
            memory_context = memory_graph.get_relevant_context(
                current_situation, node_limit=10, max_tokens=10000
            )

        # Start with the system message (including memory context)
        final_messages = [
            {
                "role": "system",
                "content": self.system_prompt
                + (
                    f"\n\nRelevant game history:\n{memory_context}"
                    if memory_context
                    else ""
                ),
            }
        ]

        # Get the user's current query as the last message
        current_user_query = None
        for msg in reversed(messages):
            if msg.get("role") == "user":
                current_user_query = msg
                break

        # If we have a session ID, we can add the most recent relevant conversation
        if session_id and recent_history_turns > 0 and current_user_query:
            from app.services.game_state_service import GameStateService

            game_state_service = GameStateService()
            history = game_state_service.get_session_history(session_id)

            if history and len(history) > 0:
                # Get recent conversation turns, but filter to just player/GM exchanges
                filtered_history = []
                for msg in history:
                    role = msg.get("role", "")
                    content = msg.get("content", "")

                    # Only include player statements and GM responses
                    # Skip system messages, status updates, etc.
                    if role in ["player", "gm"] and content:
                        # Further filter out inventory updates or status messages
                        # This is a simple heuristic - you may want to improve it
                        if not (
                            "inventory" in content.lower()
                            or "stats:" in content.lower()
                            or "health:" in content.lower()
                            or "updated" in content.lower()
                        ):
                            filtered_history.append(msg)

                # Take only recent turns based on parameter
                recent_turns = filtered_history[
                    -min(recent_history_turns * 2, len(filtered_history)) :
                ]

                # Add filtered conversation history to messages
                for msg in recent_turns:
                    role = msg.get("role", "")
                    # Convert internal roles to OpenAI format
                    openai_role = (
                        "user"
                        if role == "player"
                        else "assistant"
                        if role == "gm"
                        else "system"
                    )
                    final_messages.append(
                        {"role": openai_role, "content": msg.get("content", "")}
                    )

        # Add the current query as the final user message
        if current_user_query and (
            not final_messages or final_messages[-1]["role"] != "user"
        ):
            final_messages.append(current_user_query)

        try:
            # Use OpenAI 1.0+ API format
            # Store request data if debug is enabled
            if self.debug_enabled:
                debug_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "request": {
                        "model": "gpt-4o-mini",
                        "messages": final_messages,
                        "temperature": 0.7,
                        "max_tokens": 2000,
                    },
                    "response": None,
                    "error": None,
                }

            # Make the actual API call with the new client-based API
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=final_messages,
                temperature=0.7,
                max_tokens=2000,
            )
            response_content = response.choices[0].message.content

            # Store response data if debug is enabled
            if self.debug_enabled:
                debug_entry["response"] = response
                self.api_debug_logs.append(debug_entry)

            return response_content

        except Exception as e:
            error_msg = f"Error getting AI response: {str(e)}"

            # Store error data if debug is enabled
            if self.debug_enabled:
                debug_entry["error"] = error_msg
                self.api_debug_logs.append(debug_entry)

            return error_msg

    def start_game(
        self, character: Character, game_world: str, session_id: str = None
    ) -> str:
        """Start a new game session.

        Args:
            character: Player character
            game_world: Type of game world
            session_id: Unique identifier for the game session

        Returns:
            Introduction message from the GM
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": f"""Start a new game session with:
            - Character: {character.name}, a level {character.level} {character.character_class}
            - World: {game_world}
            Provide an engaging introduction and initial scene.""",
            },
        ]

        # Generate AI response - no history for first message
        response = self.get_ai_response(
            messages, session_id=session_id, recent_history_turns=0
        )

        # Store the game start as a high-importance memory node
        if session_id:
            start_content = f"""Game session started with {character.name}, a level {character.level} {character.character_class} in {game_world}.
            
            Initial scene: {response[:500]}... (truncated)"""

            # Get the session-specific memory graph
            memory_graph = self.get_session_memory_graph(session_id)
            memory_graph.add_node(
                content=start_content,
                node_type="event",
                importance=1.0,  # Highest importance - game initialization
            )

            # Save memory graph to ensure it's persisted
            memory_graph.save()

        return response

    def process_action(
        self,
        session: "GameSession",
        character: Character,
        action: str,
        npcs_present: List[Dict] = None,
    ) -> str:
        """Process a player's action.

        Args:
            session: Current game session
            character: Player character
            action: Player's action description
            npcs_present: List of NPCs present in the current location

        Returns:
            GM's response to the action
        """
        if npcs_present is None:
            npcs_present = []

        # Importance factor for the memory graph (1.0 is highest)
        importance = 0.5

        # Get recent game history
        messages = []
        for history_item in session.history[-10:]:  # Get last 10 items max
            messages.append(
                {
                    "role": "user" if history_item["role"] == "player" else "assistant",
                    "content": history_item["content"],
                }
            )

        # Add the current action
        messages.append({"role": "user", "content": action})

        # Load recent memories to provide context
        game_context = ""
        if hasattr(session, "id"):
            memory_graph = self.get_session_memory_graph(session.id)
            relevant_context = memory_graph.get_relevant_context(action, node_limit=5)
            if relevant_context and relevant_context != "No relevant memories found.":
                game_context = f"""Important context from previous interactions:
                
                {relevant_context}
                
                Current location: {session.current_location.get("name", "Unknown")}
                Description: {session.current_location.get("description", "")}
                """

                # Add game context as a system message
                messages.insert(
                    0,
                    {
                        "role": "system",
                        "content": self.system_prompt + "\n\n" + game_context,
                    },
                )
            else:
                messages.insert(0, {"role": "system", "content": self.system_prompt})
        else:
            messages.insert(0, {"role": "system", "content": self.system_prompt})

        # Check if this might be a movement action
        movement_words = [
            "go",
            "walk",
            "move",
            "travel",
            "head",
            "enter",
            "exit",
            "leave",
            "north",
            "south",
            "east",
            "west",
            "up",
            "down",
        ]
        potential_movement = any(
            word in action.lower().split() for word in movement_words
        )

        # Response variable
        response = ""

        # Define the output schema for OpenAI's structured output
        location_response_schema = {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "The game master's narrative response to the player's action",
                },
                "location": {
                    "type": "object",
                    "description": "Information about the current location after the action is processed",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "The name of the current location",
                        },
                        "description": {
                            "type": "string",
                            "description": "A detailed description of the current location",
                        },
                        "location_changed": {
                            "type": "boolean",
                            "description": "Whether the player has moved to a new location as a result of this action",
                        },
                    },
                    "required": ["name", "description", "location_changed"],
                },
                "inventory_changes": {
                    "type": "object",
                    "description": "Changes to the player's inventory as a result of this action",
                    "properties": {
                        "added_items": {
                            "type": "array",
                            "description": "Items added to the player's inventory",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {
                                        "type": "string",
                                        "description": "The name of the item",
                                    },
                                    "type": {
                                        "type": "string",
                                        "description": "The type of item (weapon, armor, consumable, quest, etc.)",
                                    },
                                    "description": {
                                        "type": "string",
                                        "description": "A description of the item",
                                    },
                                },
                                "required": ["name", "type"],
                            },
                        },
                        "removed_items": {
                            "type": "array",
                            "description": "Items removed from the player's inventory",
                            "items": {
                                "type": "string",
                                "description": "The name of the item to remove",
                            },
                        },
                        "items_used": {
                            "type": "array",
                            "description": "Items used by the player in this action (but not removed from inventory)",
                            "items": {
                                "type": "string",
                                "description": "The name of the item used",
                            },
                        },
                    },
                },
                "character_updates": {
                    "type": "object",
                    "description": "Updates to the player character as a result of this action",
                    "properties": {
                        "experience_gained": {
                            "type": "integer",
                            "description": "Amount of experience points gained from this action",
                        },
                        "gold_gained": {
                            "type": "integer",
                            "description": "Amount of gold gained from this action",
                        },
                        "gold_spent": {
                            "type": "integer",
                            "description": "Amount of gold spent during this action",
                        },
                        "health_change": {
                            "type": "integer",
                            "description": "Change in health points (positive for healing, negative for damage)",
                        },
                    },
                },
            },
            "required": ["message", "location"],
        }

        # Check if API debug mode is enabled in the app config
        try:
            self.debug_enabled = current_app.config.get("API_DEBUG", False)
        except RuntimeError:
            # Not in an application context, assume debug is False
            self.debug_enabled = False

        # Create debug entry if debug is enabled
        if self.debug_enabled:
            debug_entry = {
                "timestamp": datetime.now().isoformat(),
                "session_id": session.id if hasattr(session, "id") else None,
                "request": {
                    "model": "gpt-4o-mini",
                    "messages": messages,
                    "temperature": 0.7,
                    "function_call": True,
                },
                "response": None,
                "error": None,
            }

        # Use the OpenAI client to get a structured response using function calling
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.7,
                functions=[
                    {
                        "name": "process_response",
                        "description": "Process the structured response from the AI",
                        "parameters": location_response_schema,
                    }
                ],
                function_call={"name": "process_response"},
            )

            # Store response data if debug is enabled
            if self.debug_enabled:
                debug_entry["response"] = response
                self.api_debug_logs.append(debug_entry)

        except Exception as e:
            error_msg = f"Error getting AI response: {str(e)}"

            # Store error data if debug is enabled
            if self.debug_enabled:
                debug_entry["error"] = error_msg
                self.api_debug_logs.append(debug_entry)

            raise e

        # Extract the response json
        try:
            # When using function_call, the content will be in the function_call arguments
            if response.choices[0].message.function_call:
                function_args = json.loads(
                    response.choices[0].message.function_call.arguments
                )

                # Update location if changed and has correct structure
                if "location" in function_args and isinstance(
                    function_args["location"], dict
                ):
                    location_data = function_args["location"]
                    if location_data.get("location_changed", False):
                        session.current_location = {
                            "name": location_data.get("name", "Unknown"),
                            "description": location_data.get("description", ""),
                        }
                        # Persist the location change to storage
                        from app.services.game_state_service import GameStateService

                        game_state_service = GameStateService()
                        game_state_service.update_session(session)

                # Process inventory changes if any
                if "inventory_changes" in function_args and isinstance(
                    function_args["inventory_changes"], dict
                ):
                    inventory_changes = function_args["inventory_changes"]

                    # Process added items
                    if "added_items" in inventory_changes and isinstance(
                        inventory_changes["added_items"], list
                    ):
                        for item in inventory_changes["added_items"]:
                            if (
                                isinstance(item, dict)
                                and "name" in item
                                and "type" in item
                            ):
                                # Generate a unique ID for the item
                                import uuid

                                item_id = str(uuid.uuid4())

                                # Create a complete item object
                                new_item = {
                                    "id": item_id,
                                    "name": item["name"],
                                    "type": item["type"],
                                    "description": item.get(
                                        "description", f"A {item['type']}."
                                    ),
                                }

                                # Add to character inventory
                                character.add_item(new_item)

                                # Log item addition to memory graph
                                memory_graph = self.get_session_memory_graph(session.id)
                                memory_graph.add_node(
                                    content=f"Character acquired item: {item['name']} ({item['type']})",
                                    node_type="inventory",
                                    importance=0.7,
                                )

                    # Process removed items
                    if "removed_items" in inventory_changes and isinstance(
                        inventory_changes["removed_items"], list
                    ):
                        for item_name in inventory_changes["removed_items"]:
                            # Find the item in inventory
                            item_to_remove = None
                            for inv_item in character.inventory:
                                if inv_item.get("name") == item_name:
                                    item_to_remove = inv_item
                                    break

                            # Remove the item if found
                            if item_to_remove:
                                character.remove_item(item_to_remove.get("id"))

                                # Log item removal to memory graph
                                memory_graph = self.get_session_memory_graph(session.id)
                                memory_graph.add_node(
                                    content=f"Character lost item: {item_name}",
                                    node_type="inventory",
                                    importance=0.6,
                                )

                    # Track used items (no need to remove them)
                    if "items_used" in inventory_changes and isinstance(
                        inventory_changes["items_used"], list
                    ):
                        for item_name in inventory_changes["items_used"]:
                            # Verify item exists in inventory
                            item_exists = any(
                                inv_item.get("name") == item_name
                                for inv_item in character.inventory
                            )

                            if item_exists:
                                # Log item usage to memory graph
                                memory_graph = self.get_session_memory_graph(session.id)
                                memory_graph.add_node(
                                    content=f"Character used item: {item_name}",
                                    node_type="inventory",
                                    importance=0.5,
                                )

                # Process character updates (XP, gold, health)
                if "character_updates" in function_args and isinstance(
                    function_args["character_updates"], dict
                ):
                    char_updates = function_args["character_updates"]
                    updates_made = False
                    update_summary = []

                    # Handle XP gains
                    if "experience_gained" in char_updates and isinstance(
                        char_updates["experience_gained"], int
                    ):
                        xp_gained = char_updates["experience_gained"]
                        if xp_gained > 0:
                            # Add XP and check for level up
                            level_up = character.add_experience(xp_gained)
                            updates_made = True
                            update_summary.append(f"Gained {xp_gained} XP")

                            # Log to memory graph
                            memory_graph = self.get_session_memory_graph(session.id)
                            if level_up:
                                memory_graph.add_node(
                                    content=f"Character gained {xp_gained} XP and leveled up to level {character.level}",
                                    node_type="progression",
                                    importance=0.9,  # Level ups are important milestones
                                )
                            else:
                                memory_graph.add_node(
                                    content=f"Character gained {xp_gained} XP",
                                    node_type="progression",
                                    importance=0.6,
                                )

                    # Handle gold changes
                    if "gold_gained" in char_updates and isinstance(
                        char_updates["gold_gained"], int
                    ):
                        gold_gained = char_updates["gold_gained"]
                        if gold_gained > 0:
                            character.gold += gold_gained
                            updates_made = True
                            update_summary.append(f"Gained {gold_gained} gold")

                            # Log to memory graph
                            memory_graph = self.get_session_memory_graph(session.id)
                            memory_graph.add_node(
                                content=f"Character gained {gold_gained} gold",
                                node_type="transaction",
                                importance=0.5,
                            )

                    if "gold_spent" in char_updates and isinstance(
                        char_updates["gold_spent"], int
                    ):
                        gold_spent = char_updates["gold_spent"]
                        if gold_spent > 0:
                            # Only deduct if character has enough gold
                            if character.gold >= gold_spent:
                                character.gold -= gold_spent
                                updates_made = True
                                update_summary.append(f"Spent {gold_spent} gold")

                                # Log to memory graph
                                memory_graph = self.get_session_memory_graph(session.id)
                                memory_graph.add_node(
                                    content=f"Character spent {gold_spent} gold",
                                    node_type="transaction",
                                    importance=0.5,
                                )

                    # Handle health changes
                    if "health_change" in char_updates and isinstance(
                        char_updates["health_change"], int
                    ):
                        health_change = char_updates["health_change"]
                        if health_change > 0:
                            # Healing
                            character.heal(health_change)
                            updates_made = True
                            update_summary.append(f"Healed {health_change} HP")
                        elif health_change < 0:
                            # Damage
                            character.take_damage(
                                -health_change
                            )  # Convert to positive for take_damage
                            updates_made = True
                            update_summary.append(f"Took {-health_change} damage")

                    # If any updates were made, save the character
                    if updates_made:
                        from app.services.character_service import CharacterService

                        character_service = CharacterService()
                        character_service.update_character(character)

                        # Add a combined update node if multiple changes occurred
                        if len(update_summary) > 1:
                            memory_graph = self.get_session_memory_graph(session.id)
                            memory_graph.add_node(
                                content=f"Character updates: {', '.join(update_summary)}",
                                node_type="character_update",
                                importance=0.7,
                            )

                # Set the final response text to the message portion
                if "message" in function_args:
                    response = function_args["message"]
                else:
                    # Fallback if message not found
                    response = "I don't understand what you mean. Please try again."
            else:
                # Fallback if function wasn't called - try to parse direct content
                response_text = response.choices[0].message.content
                try:
                    parsed_response = json.loads(response_text)
                    if "message" in parsed_response:
                        response = parsed_response["message"]
                    # Handle location changes
                    if "location" in parsed_response and parsed_response[
                        "location"
                    ].get("location_changed", False):
                        session.current_location = {
                            "name": parsed_response["location"]["name"],
                            "description": parsed_response["location"]["description"],
                        }
                        # Persist the location change to storage
                        from app.services.game_state_service import GameStateService

                        game_state_service = GameStateService()
                        game_state_service.update_session(session)

                    # Handle inventory changes for fallback path
                    if "inventory_changes" in parsed_response and isinstance(
                        parsed_response["inventory_changes"], dict
                    ):
                        inventory_changes = parsed_response["inventory_changes"]

                        # Process added items
                        if "added_items" in inventory_changes and isinstance(
                            inventory_changes["added_items"], list
                        ):
                            for item in inventory_changes["added_items"]:
                                if (
                                    isinstance(item, dict)
                                    and "name" in item
                                    and "type" in item
                                ):
                                    # Generate a unique ID for the item
                                    import uuid

                                    item_id = str(uuid.uuid4())

                                    # Create a complete item object
                                    new_item = {
                                        "id": item_id,
                                        "name": item["name"],
                                        "type": item["type"],
                                        "description": item.get(
                                            "description", f"A {item['type']}."
                                        ),
                                    }

                                    # Add to character inventory
                                    character.add_item(new_item)

                                    # Log item addition to memory graph
                                    memory_graph = self.get_session_memory_graph(
                                        session.id
                                    )
                                    memory_graph.add_node(
                                        content=f"Character acquired item: {item['name']} ({item['type']})",
                                        node_type="inventory",
                                        importance=0.7,
                                    )

                        # Process removed items
                        if "removed_items" in inventory_changes and isinstance(
                            inventory_changes["removed_items"], list
                        ):
                            for item_name in inventory_changes["removed_items"]:
                                # Find the item in inventory
                                item_to_remove = None
                                for inv_item in character.inventory:
                                    if inv_item.get("name") == item_name:
                                        item_to_remove = inv_item
                                        break

                                # Remove the item if found
                                if item_to_remove:
                                    character.remove_item(item_to_remove.get("id"))

                                    # Log item removal to memory graph
                                    memory_graph = self.get_session_memory_graph(
                                        session.id
                                    )
                                    memory_graph.add_node(
                                        content=f"Character lost item: {item_name}",
                                        node_type="inventory",
                                        importance=0.6,
                                    )

                        # Track used items (no need to remove them)
                        if "items_used" in inventory_changes and isinstance(
                            inventory_changes["items_used"], list
                        ):
                            for item_name in inventory_changes["items_used"]:
                                # Verify item exists in inventory
                                item_exists = any(
                                    inv_item.get("name") == item_name
                                    for inv_item in character.inventory
                                )

                                if item_exists:
                                    # Log item usage to memory graph
                                    memory_graph = self.get_session_memory_graph(
                                        session.id
                                    )
                                    memory_graph.add_node(
                                        content=f"Character used item: {item_name}",
                                        node_type="inventory",
                                        importance=0.5,
                                    )

                        # Save character changes
                        from app.services.character_service import CharacterService

                        character_service = CharacterService()
                        character_service.update_character(character)

                    # Process character updates for fallback path
                    if "character_updates" in parsed_response and isinstance(
                        parsed_response["character_updates"], dict
                    ):
                        char_updates = parsed_response["character_updates"]
                        updates_made = False
                        update_summary = []

                        # Handle XP gains
                        if "experience_gained" in char_updates and isinstance(
                            char_updates["experience_gained"], int
                        ):
                            xp_gained = char_updates["experience_gained"]
                            if xp_gained > 0:
                                # Add XP and check for level up
                                level_up = character.add_experience(xp_gained)
                                updates_made = True
                                update_summary.append(f"Gained {xp_gained} XP")

                                # Log to memory graph
                                memory_graph = self.get_session_memory_graph(session.id)
                                if level_up:
                                    memory_graph.add_node(
                                        content=f"Character gained {xp_gained} XP and leveled up to level {character.level}",
                                        node_type="progression",
                                        importance=0.9,
                                    )
                                else:
                                    memory_graph.add_node(
                                        content=f"Character gained {xp_gained} XP",
                                        node_type="progression",
                                        importance=0.6,
                                    )

                        # Handle gold changes
                        if "gold_gained" in char_updates and isinstance(
                            char_updates["gold_gained"], int
                        ):
                            gold_gained = char_updates["gold_gained"]
                            if gold_gained > 0:
                                character.gold += gold_gained
                                updates_made = True
                                update_summary.append(f"Gained {gold_gained} gold")

                        if "gold_spent" in char_updates and isinstance(
                            char_updates["gold_spent"], int
                        ):
                            gold_spent = char_updates["gold_spent"]
                            if gold_spent > 0 and character.gold >= gold_spent:
                                character.gold -= gold_spent
                                updates_made = True
                                update_summary.append(f"Spent {gold_spent} gold")

                        # Handle health changes
                        if "health_change" in char_updates and isinstance(
                            char_updates["health_change"], int
                        ):
                            health_change = char_updates["health_change"]
                            if health_change > 0:
                                character.heal(health_change)
                                updates_made = True
                                update_summary.append(f"Healed {health_change} HP")
                            elif health_change < 0:
                                character.take_damage(-health_change)
                                updates_made = True
                                update_summary.append(f"Took {-health_change} damage")

                        # Save character if updates were made
                        if updates_made:
                            character_service = CharacterService()
                            character_service.update_character(character)
                except json.JSONDecodeError:
                    # If not JSON, just use the raw response
                    response = response_text

        except Exception as e:
            print(f"Error parsing response: {e}")
            # Fallback to standard processing
            response = self.get_ai_response(messages, session_id=session.id)

        # Store player action and GM response in memory graph
        if hasattr(session, "id"):
            memory_content = f"""Location: {session.current_location.get("name")}
            
            Player action: {action}
            
            Game Master response: {response}
            """

            # Add memory node for this interaction
            memory_graph = self.get_session_memory_graph(session.id)
            memory_graph.add_node(
                content=memory_content, node_type="event", importance=importance
            )

            # Check for potential character or location memories to create
            if any(
                loc_word in action.lower()
                for loc_word in ["place", "area", "room", "location"]
            ):
                # This might be a significant location description
                location_content = f"""Location: {session.current_location.get("name")}
                Description: {session.current_location.get("description")}
                Discovered during action: {action}
                """
                memory_graph.add_node(
                    content=location_content, node_type="location", importance=0.7
                )

            # Check for character interactions
            for npc in npcs_present:
                if npc.get("name", "").lower() in action.lower():
                    npc_content = f"""Character: {npc.get("name")}
                    Type: {npc.get("type")}
                    Description: {npc.get("description")}
                    Interaction: {action}
                    """
                    memory_graph.add_node(
                        content=npc_content, node_type="character", importance=0.7
                    )

            # Save memory graph periodically (every ~5 actions)
            if random.random() < 0.2:  # 20% chance to save
                memory_graph.save()

        return response

    def generate_npc(self, npc_type: str, location: str) -> NPC:
        """Generate a new NPC.

        Args:
            npc_type: Type of NPC (e.g., "shopkeeper", "quest_giver")
            location: Location where the NPC is found

        Returns:
            Generated NPC object
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": f"""Create a detailed NPC:
            Type: {npc_type}
            Location: {location}
            Include:
            - Name
            - Description
            - Personality
            - Key information or quest hooks
            Format as JSON.""",
            },
        ]

        response = self.get_ai_response(messages)
        try:
            npc_data = json.loads(response)
            return NPC(
                name=npc_data.get("name", "Unknown"),
                type=npc_type,
                description=npc_data.get("description", ""),
                dialogue=npc_data.get("dialogue", {}),
            )
        except json.JSONDecodeError:
            # Fallback NPC if AI response isn't valid JSON
            return NPC(
                name=f"{npc_type.title()} #{random.randint(1, 1000)}",
                type=npc_type,
                description=f"A typical {npc_type} in {location}.",
            )

    def generate_location(self, location_name: str, location_type: str) -> Dict:
        """Generate a new location.

        Args:
            location_name: Name of the location
            location_type: Type of location (e.g., "tavern", "dungeon")

        Returns:
            Generated location data
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": f"""Create a detailed location:
            Name: {location_name}
            Type: {location_type}
            Include:
            - Description
            - Notable features
            - Possible encounters
            - Connected locations
            Format as JSON.""",
            },
        ]

        response = self.get_ai_response(messages)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Fallback location if AI response isn't valid JSON
            return {
                "name": location_name,
                "type": location_type,
                "description": f"A typical {location_type}.",
                "features": [],
                "possible_encounters": [],
                "exits": [],
            }

    def generate_quest(self, character: Character, difficulty: str = "normal") -> Dict:
        """Generate a new quest.

        Args:
            character: Player character
            difficulty: Quest difficulty

        Returns:
            Generated quest data
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": f"""Create a quest for:
            Character: Level {character.level} {character.character_class}
            Difficulty: {difficulty}
            Include:
            - Title
            - Description
            - Objectives
            - Rewards
            Format as JSON.""",
            },
        ]

        response = self.get_ai_response(messages)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Fallback quest if AI response isn't valid JSON
            return {
                "id": str(random.randint(1000, 9999)),
                "title": f"Generic {difficulty.title()} Quest",
                "description": f"A typical quest for a level {character.level} character.",
                "objectives": ["Complete the task"],
                "rewards": {
                    "gold": character.level * 10,
                    "experience": character.level * 100,
                },
            }

    def start_combat(
        self, session: "GameSession", character: Character, enemy_type: str = "random"
    ) -> Dict:
        """Start a combat encounter.

        Args:
            session: Current game session
            character: Player character
            enemy_type: Type of enemies to generate

        Returns:
            Combat initialization data
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": f"""Create a combat encounter for:
            Character: Level {character.level} {character.character_class}
            Enemy type: {enemy_type}
            Location: {session.current_location.get("name")}
            Include:
            - Enemy details
            - Combat environment
            - Special conditions
            Format as JSON.""",
            },
        ]

        response = self.get_ai_response(messages)
        try:
            encounter_data = json.loads(response)
            enemies = [
                Enemy(**enemy_data) for enemy_data in encounter_data.get("enemies", [])
            ]
        except (json.JSONDecodeError, TypeError):
            # Fallback enemies if AI response isn't valid
            enemies = [
                Enemy(
                    name="Generic Enemy",
                    description="A typical opponent",
                    health=20,
                    max_health=20,
                    armor_class=12,
                    strength=10,
                    dexterity=10,
                    constitution=10,
                    intelligence=10,
                    wisdom=10,
                    charisma=10,
                )
            ]

        # Create combat encounter
        encounter = CombatEncounter(
            enemies=enemies,
            environment=session.current_location.get("name", "battlefield"),
            description=f"Combat with {len(enemies)} enemies",
        )

        # Determine initiative order
        initiatives = []

        # Roll for character
        char_init = roll_dice("1d20") + character.get_ability_modifier("dexterity")
        initiatives.append(("character", char_init))

        # Roll for enemies
        for i, enemy in enumerate(enemies):
            enemy_init = roll_dice("1d20") + enemy.get_ability_modifier("dexterity")
            initiatives.append((f"enemy_{i}", enemy_init))

        # Sort by initiative (highest first)
        initiatives.sort(key=lambda x: x[1], reverse=True)

        return {
            "encounter": encounter.to_dict(),
            "initiative_order": initiatives,
            "current_turn": initiatives[0][0],
            "round": 1,
            "message": f"Combat begins! Initiative order: {', '.join(f'{name} ({init})' for name, init in initiatives)}",
        }

    def process_combat_action(
        self, session: "GameSession", character: Character, action: str
    ) -> Dict:
        """Process a combat action.

        Args:
            session: Current game session
            character: Player character
            action: Combat action description

        Returns:
            Result of the combat action
        """
        if not session.combat_state:
            return {"error": "No active combat"}

        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": f"""Process combat action:
            Character: Level {character.level} {character.character_class}
            Health: {character.health}/{character.max_health}
            
            Inventory:
            {", ".join(item.get("name", "Unknown Item") for item in character.inventory) if character.inventory else "No items"}
            
            Combat stats:
            - Strength: {character.strength} (Modifier: {character.get_ability_modifier("strength")})
            - Dexterity: {character.dexterity} (Modifier: {character.get_ability_modifier("dexterity")})
            
            Action: {action}
            Combat state: Round {session.combat_state.get("round")}
            
            Describe the outcome and update combat state. If the player is trying to use an item from their inventory, incorporate it into the result.
            Format as JSON with 'description' and 'effects'.""",
            },
        ]

        response = self.get_ai_response(messages)
        try:
            result = json.loads(response)
            return {
                "description": result.get("description", "The action is resolved."),
                "effects": result.get("effects", {}),
                "combat_continues": True,  # Update based on combat state
            }
        except json.JSONDecodeError:
            return {
                "description": "The action is processed.",
                "effects": {},
                "combat_continues": True,
            }

    def generate_loot(self, enemy: Enemy, character_level: int) -> List[Dict]:
        """Generate loot from a defeated enemy.

        Args:
            enemy: Defeated enemy
            character_level: Player character's level

        Returns:
            List of loot items
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": f"""Generate loot for:
            Enemy: {enemy.name}
            Character Level: {character_level}
            Include:
            - Items
            - Gold
            - Special rewards
            Format as JSON.""",
            },
        ]

        response = self.get_ai_response(messages)
        try:
            loot_data = json.loads(response)
            return loot_data.get("items", [])
        except json.JSONDecodeError:
            # Fallback loot if AI response isn't valid JSON
            return [
                {
                    "id": f"gold_{random.randint(1000, 9999)}",
                    "name": "Gold",
                    "type": "currency",
                    "amount": enemy.gold_reward,
                }
            ]
