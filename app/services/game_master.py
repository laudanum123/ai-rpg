import json
import logging
import random
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from app.models.character import Character
from app.models.combat import CombatEncounter, Enemy, roll_dice
from app.models.game_session import GameSession
from app.models.npc import NPC
from app.services.ai_service import AIService
from app.services.character_service import CharacterService
from app.services.game_state_service import GameStateService
from app.services.memory_graph import MemoryGraph


class GameMaster:
    """Service for AI-powered game mastering using GPT-4o-mini."""

    def __init__(self, debug_enabled=False):
        """Initialize the game master service."""
        # Initialize OpenAI client
        self.openai_client = OpenAI()

        # Store for API debug messages (max 50 entries)
        self.api_debug_logs = deque(maxlen=50)
        self.debug_enabled = debug_enabled
        self.memory_graphs = {}

        # System prompt for the AI game master
        self.system_prompt = """You are an experienced Game Master for a fantasy RPG game.
        Your role is to create an engaging and dynamic adventure, manage NPCs, describe
        locations vividly, create interesting plot hooks, and run combat encounters.
        Always stay in character as a GM and maintain consistency in the game world.
        Focus on creating an immersive experience while following the game's rules."""  # noqa: E501

        # Set up AIService
        self.ai_service = AIService(self.openai_client, self.system_prompt)

        # Base configuration for memory graphs
        self.memory_graph_config = {
            "openai_client": self.openai_client,
            "embedding_model": "text-embedding-3-large",
            "llm_model": "gpt-4o-mini",
        }

    def get_session_memory_graph(self, session_id: str) -> MemoryGraph:
        """Get or create a memory graph for the specified session.

        Args:
            session_id: The game session ID

        Returns:
            The memory graph for this session
        """
        if session_id not in self.memory_graphs:
            # Create a new memory graph for this session with
            # a session-specific storage directory
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
                    f"Loaded memory graph for session {session_id} with {len(memory_graph.nodes)} nodes"  # noqa: E501
                )
            except Exception as e:
                print(
                    f"No existing memory graph found for session {session_id} or error loading: {e}"  # noqa: E501
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

    def parse_combat_result(self, ai_response: str) -> Dict[str, Any]:
        """Parse combat results from an AI response.

        Args:
            ai_response: The AI's response text containing combat results

        Returns:
            Dictionary with parsed combat information
        """
        result = {
            "description": ai_response,
            "damage_dealt": 0,
            "player_damage_taken": 0,
            "enemy_defeated": False
        }

        # Extract damage dealt
        import re
        damage_dealt_match = re.search(r"Player damage dealt:\s*(\d+)", ai_response)
        if damage_dealt_match:
            result["damage_dealt"] = int(damage_dealt_match.group(1))

        # Extract player damage taken
        damage_taken_match = re.search(r"Player damage taken:\s*(\d+)", ai_response)
        if damage_taken_match:
            result["player_damage_taken"] = int(damage_taken_match.group(1))

        # Check if enemy was defeated
        if any(phrase in ai_response.lower() for phrase in [
            "enemy defeated",
            "enemy is defeated",
            "enemy has been defeated",
            "enemy is killed",
            "enemy has been killed",
            "enemy is dead",
            "enemy lies dead",
            "defeated the enemy",
            "killed the enemy"
        ]):
            result["enemy_defeated"] = True

        return result

    # Removed _extract_current_situation method (moved to AIService)

    # Removed _get_current_user_query method (moved to AIService)

    # Removed _create_debug_entry method (moved to AIService)

    # Removed _call_openai_api method (moved to AIService)

    # Removed _handle_ai_response_error method (moved to AIService)

    # Removed _handle_original_ai_response_error method (moved to AIService)

    def get_ai_response(
        self,
        messages: List[Dict] or str,
        session_id: str = None,
        recent_history_turns: int = 5,
        model_name: str = "gpt-4o-mini",
        max_tokens: int = 2000,
    ) -> str:
        """Get a response from the AI model with contextual memory.

        Delegates to AIService for handling the AI interaction.

        Args:
            messages: List of conversation messages
            session_id: Current game session ID for memory retrieval
            recent_history_turns: Number of recent conversation turns to include (default: 5)
            model_name: The OpenAI model to use
            max_tokens: Maximum tokens for the response

        Returns:
            AI model's response text
        """
        # Get the memory graph if we have a session ID
        memory_graph = None
        if session_id:
            memory_graph = self.get_session_memory_graph(session_id)

        # Create game state service instance
        game_state_service = GameStateService()

        # Delegate to the AI service
        return self.ai_service.get_ai_response(
            messages,
            session_id,
            recent_history_turns,
            model_name,
            max_tokens,
            memory_graph,
            game_state_service
        )
        
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
        self, session: "GameSession", character: Character, action: str, npcs_present: List[Dict] = None
    ) -> str:
        """
        Process a player action within the game session.
        
        Args:
            session: Current game session
            character: Player character
            action: Action text
            npcs_present: List of NPCs present in the scene
        
        Returns:
            Response text
        """
        try:
            # Step 1: Prepare messages for AI request
            messages = self._prepare_action_messages(session, character, action, npcs_present)

            # Step 2: Add memory context
            memory_graph = self.get_session_memory_graph(session.id)
            messages = self._add_memory_context(messages, memory_graph, character, session)

            # Step 3: Create response schema for structured output
            schema = self._create_location_response_schema()

            # Step 4: Initialize debug entry if debugging is enabled
            model_name = "gpt-4o-mini"
            debug_entry = self._initialize_debug_for_action(model_name, messages)

            # Step 5: Get structured AI response
            function_args, text_response = self._get_structured_ai_response(
                messages, schema, debug_entry
            )

            # Step 6: Process the structured response
            self._process_structured_ai_response(function_args, session, character)

            return text_response

        except Exception as e:
            # Handle errors
            return self._handle_ai_response_error(e, debug_entry if 'debug_entry' in locals() else None)

    def _prepare_action_messages(self, session: "GameSession", character: Character,
                               action: str, npcs_present: List[Dict] = None) -> List[Dict[str, Any]]:
        """Prepare messages for the AI request based on session history and current action."""
        # Construct the system message and base messages
        messages = self._create_base_system_messages(character)

        # Add location and NPC context
        messages = self._add_environment_context(messages, session, npcs_present)

        # Add history context
        messages = self._add_history_context(messages, session, action)

        # Add the current action
        messages.append({"role": "user", "content": action})

        return messages

    def _create_base_system_messages(self, character: Character) -> List[Dict[str, Any]]:
        """Create the base system messages including character information."""
        system_message = {
            "role": "system",
            "content": (
                "You are a fantasy RPG game master. Respond in character as a narrator to user's "
                "actions. Keep responses lively, engaging and between 100-200 words unless more "
                "description is needed. Include environmental details and character reactions. "
                "If player attempts a questionable action, show reasonable consequences rather "
                "than refusing outright. Allow creative problem-solving."
            ),
        }

        messages = [system_message]

        # Add character context
        char_desc = (
            f"Your character is {character.name}, a level {character.level} "
            f"{character.character_class}."
        )
        messages.append({"role": "system", "content": char_desc})

        # Add character stats context
        stats_desc = (
            f"Stats - Health: {character.health}/{character.max_health}, "
            f"Gold: {character.gold}, Experience: {character.experience}, "
            f"Level: {character.level}"
        )
        messages.append({"role": "system", "content": stats_desc})

        # Add inventory context
        if character.inventory:
            inv_items = ", ".join(
                item.get("name", "Unknown Item") for item in character.inventory
            )
            inv_desc = f"Inventory: {inv_items}"
            messages.append({"role": "system", "content": inv_desc})

        return messages

    def _add_environment_context(self, messages: List[Dict[str, Any]],
                               session: "GameSession",
                               npcs_present: List[Dict] = None) -> List[Dict[str, Any]]:
        """Add location and NPC context to messages."""
        # Add location context if available
        if hasattr(session, "current_location") and session.current_location:
            location = session.current_location
            location_desc = (
                f"You are currently in {location.get('name', 'Unknown')}. "
                f"{location.get('description', '')}"
            )
            messages.append({"role": "system", "content": location_desc})

        # Add NPC context if present
        if npcs_present:
            npc_names = ", ".join(npc.get("name", "Unknown NPC") for npc in npcs_present)
            npc_desc = f"NPCs present: {npc_names}"
            messages.append({"role": "system", "content": npc_desc})

            # Add detailed NPC information
            for npc in npcs_present:
                if "description" in npc:
                    npc_info = f"{npc['name']}: {npc['description']}"
                    messages.append({"role": "system", "content": npc_info})

        return messages

    def _add_history_context(self, messages: List[Dict[str, Any]],
                           session: "GameSession",
                           action: str) -> List[Dict[str, Any]]:
        """Add conversation history context to messages."""
        # Add session history context
        if hasattr(session, "history") and len(session.history) > 0:
            history = session.history[-10:]  # Get last 10 items for context
        else:
            history = []

        # Calculate context tokens
        context_tokens = (
            len(action.split()) +
            sum(len(entry.get("action", "").split()) + len(entry.get("response", "").split())
                for entry in history)
        )

        # Add history context if token count allows
        if context_tokens < 2000 and history:
            for entry in history:
                if "action" in entry and entry["action"]:
                    messages.append({"role": "user", "content": entry["action"]})
                if "response" in entry and entry["response"]:
                    messages.append({"role": "assistant", "content": entry["response"]})

        return messages

    def _add_memory_context(self, messages: List[Dict[str, Any]], memory_graph,
                          character: Character, session: "GameSession") -> List[Dict[str, Any]]:
        """Add relevant context from the memory graph to the messages."""
        # Skip if no memory graph
        if not memory_graph:
            return messages

        # Query relevant memories
        action_text = messages[-1]["content"]
        relevant_nodes = memory_graph.get_relevant_context(action_text)

        if relevant_nodes:
            memory_context = "Relevant memories:\n"
            for node in relevant_nodes:
                # Handle both MemoryNode objects and strings
                if hasattr(node, 'content'):
                    # It's a MemoryNode object
                    memory_context += f"- {node.content}\n"
                else:
                    # It's already a string
                    memory_context += f"- {node}\n"

            # Insert memory context after system message
            messages.insert(1, {"role": "system", "content": memory_context})

        # Add quest information if available
        if hasattr(session, "active_quests") and session.active_quests:
            quest_context = "Active quests:\n"
            for quest in session.active_quests:
                quest_context += (
                    f"- {quest['title']}: {quest['description']} "
                    f"(Status: {quest['status']})\n"
                )
            messages.insert(1, {"role": "system", "content": quest_context})

        return messages

    def _create_location_response_schema(self) -> Dict:
        """Create the response schema for structured output from OpenAI."""
        return {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "The narrative response to the player's action"
                },
                "location": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "description": {"type": "string"},
                        "location_changed": {"type": "boolean"}
                    }
                },
                "inventory_changes": {
                    "type": "object",
                    "properties": {
                        "added_items": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "type": {"type": "string"},
                                    "description": {"type": "string"}
                                }
                            }
                        },
                        "removed_items": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "items_used": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    }
                },
                "character_updates": {
                    "type": "object",
                    "properties": {
                        "experience_gained": {"type": "integer"},
                        "gold_gained": {"type": "integer"},
                        "gold_spent": {"type": "integer"},
                        "health_change": {"type": "integer"}
                    }
                },
                "relationship_updates": {
                    "type": "object",
                    "properties": {
                        "npc_relationships": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "change": {"type": "integer"},
                                    "reason": {"type": "string"}
                                }
                            }
                        }
                    }
                },
                "quest_updates": {
                    "type": "object",
                    "properties": {
                        "new_quests": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "string"},
                                    "title": {"type": "string"},
                                    "description": {"type": "string"},
                                    "objective": {"type": "string"},
                                    "reward": {"type": "string"}
                                },
                                "required": ["id", "title"]
                            }
                        },
                        "completed_quests": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "updated_quests": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "string"},
                                    "progress": {"type": "string"}
                                },
                                "required": ["id"]
                            }
                        }
                    }
                }
            },
            "required": ["message"]
        }

    def _initialize_debug_for_action(self, model_name: str, messages: List[Dict[str, Any]]) -> Optional[Dict]:
        """Initialize a debug entry if debugging is enabled."""
        if not self.debug_enabled:
            return None

        return {
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "prompt": messages,
            "max_tokens": 1000,
            "temperature": 0.7,
            "response": None,
            "error": None,
        }

    def _get_structured_ai_response(
        self,
        messages: List[Dict[str, Any]],
        schema: Dict[str, Any],
        debug_entry: Optional[Dict] = None
    ) -> Tuple[Dict, str]:
        """Get structured AI response using OpenAI function calling.

        Delegates to AIService to handle the AI interaction.

        Args:
            messages: The messages to send to the API
            schema: The JSON schema for the function call
            debug_entry: Optional debug entry for logging

        Returns:
            Tuple of (function_args, text_response)
        """
        try:
            # Default parameters for the API call
            model_name = "gpt-4o-mini"
            max_tokens = 1000
            temperature = 0.7
            
            # Delegate to AIService
            return self.ai_service.get_structured_ai_response(
                messages=messages,
                schema=schema,
                debug_entry=debug_entry,
                model_name=model_name,
                max_tokens=max_tokens,
                temperature=temperature
            )
        except Exception as e:
            logging.error(f"Error getting structured AI response: {str(e)}")
            if debug_entry is not None:
                debug_entry["error"] = str(e)
                self.api_debug_logs.append(debug_entry)
            return {}, f"Error: {str(e)}"

    def _process_structured_ai_response(
        self,
        function_args: Dict,
        session: "GameSession",
        character: Character
    ) -> None:
        """Process the structured response from the AI."""
        # Process location updates
        self._process_location_updates(function_args, session)

        # Process inventory changes
        self._process_inventory_changes(function_args, session, character)

        # Process character updates
        self._process_character_updates(function_args, session, character)

        # Process relationship updates
        self._process_relationship_updates(function_args, session)

        # Process combat updates
        self._process_combat_updates(function_args, session, character)

        # Process quest updates
        self._process_quest_updates(function_args, session)

        # Add the interaction to memory
        self._add_interaction_to_memory(session, character, function_args)

    def _process_location_updates(self, function_args: Dict, session: "GameSession"):
        """Process location updates from the AI response."""
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
                game_state_service = GameStateService()
                game_state_service.update_session(session)

                # Add location change to memory graph
                memory_graph = self.get_session_memory_graph(session.id)
                memory_graph.add_node(
                    content=f"Character moved to {location_data.get('name', 'Unknown')}",
                    node_type="location",
                    importance=0.8,
                )

    def _process_inventory_changes(self, function_args: Dict, session: "GameSession", character: Character):
        """Process inventory changes from the AI response."""
        if "inventory_changes" not in function_args or not isinstance(
            function_args["inventory_changes"], dict
        ):
            return

        inventory_changes = function_args["inventory_changes"]

        # Process added items
        self._process_added_items(inventory_changes, session, character)

        # Process removed items
        self._process_removed_items(inventory_changes, session, character)

        # Process used items
        self._process_used_items(inventory_changes, session, character)

    def _process_added_items(self, inventory_changes: Dict, session: "GameSession", character: Character):
        """Process items added to the inventory."""
        if "added_items" not in inventory_changes or not isinstance(
            inventory_changes["added_items"], list
        ):
            return

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

    def _process_removed_items(self, inventory_changes: Dict, session: "GameSession", character: Character):
        """Process items removed from the inventory."""
        if "removed_items" not in inventory_changes or not isinstance(
            inventory_changes["removed_items"], list
        ):
            return

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

    def _process_used_items(self, inventory_changes: Dict, session: "GameSession", character: Character):
        """Process items used by the character."""
        if "items_used" not in inventory_changes or not isinstance(
            inventory_changes["items_used"], list
        ):
            return

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

    def _process_character_updates(self, function_args: Dict, session: "GameSession", character: Character):
        """Process character updates from the AI response."""
        if "character_updates" not in function_args or not isinstance(
            function_args["character_updates"], dict
        ):
            return

        char_updates = function_args["character_updates"]
        updates_made = False
        update_summary = []

        # Process XP gain
        if "experience_gained" in char_updates and isinstance(char_updates["experience_gained"], int):
            xp_gained = char_updates["experience_gained"]
            if xp_gained > 0:
                # Add XP and check for level up
                level_up = character.add_experience(xp_gained)
                updates_made = True
                update_summary.append(f"Gained {xp_gained} XP")

                # Log XP gain to memory graph
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

        # Process gold changes
        if "gold_gained" in char_updates and isinstance(char_updates["gold_gained"], int):
            gold_gained = char_updates["gold_gained"]
            if gold_gained > 0:
                character.gold += gold_gained
                updates_made = True
                update_summary.append(f"Gained {gold_gained} gold")

                # Log gold gain to memory graph
                memory_graph = self.get_session_memory_graph(session.id)
                memory_graph.add_node(
                    content=f"Character gained {gold_gained} gold",
                    node_type="transaction",
                    importance=0.5,
                )

        if "gold_spent" in char_updates and isinstance(char_updates["gold_spent"], int):
            gold_spent = char_updates["gold_spent"]
            if gold_spent > 0 and character.gold >= gold_spent:
                character.gold -= gold_spent
                updates_made = True
                update_summary.append(f"Spent {gold_spent} gold")

                # Log gold spending to memory graph
                memory_graph = self.get_session_memory_graph(session.id)
                memory_graph.add_node(
                    content=f"Character spent {gold_spent} gold",
                    node_type="transaction",
                    importance=0.5,
                )

        # Process health changes
        if "health_change" in char_updates and isinstance(char_updates["health_change"], int):
            health_change = char_updates["health_change"]
            if health_change != 0:
                if health_change > 0:
                    # Healing
                    character.heal(health_change)
                    update_summary.append(f"Healed {health_change} HP")
                else:
                    # Damage
                    character.take_damage(-health_change)  # Convert to positive for take_damage
                    update_summary.append(f"Took {-health_change} damage")

                updates_made = True

                # Log health change to memory graph
                memory_graph = self.get_session_memory_graph(session.id)
                if health_change > 0:
                    memory_graph.add_node(
                        content=f"Character healed for {health_change} health points",
                        node_type="character",
                        importance=0.6,
                    )
                else:
                    memory_graph.add_node(
                        content=f"Character took {abs(health_change)} damage",
                        node_type="character",
                        importance=0.7,
                    )

        # Save character state if updates were made
        if updates_made:
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

    def _process_combat_updates(self, function_args: Dict, session: "GameSession", character: Character):
        """Process combat updates from the AI response."""
        if "combat_updates" not in function_args or not isinstance(
            function_args["combat_updates"], dict
        ):
            return

        # Initialize combat state if not present or has incorrect structure
        if not hasattr(session, "combat_state") or not isinstance(session.combat_state, dict):
            session.combat_state = {"enemies": [], "round": 1}
        if "enemies" not in session.combat_state:
            session.combat_state["enemies"] = []
        if "round" not in session.combat_state:
            session.combat_state["round"] = 1

        combat_updates = function_args["combat_updates"]

        # Update enemies info
        if "enemies" in combat_updates and isinstance(combat_updates["enemies"], list):
            # Replace the entire enemy list for the combat state
            session.combat_state["enemies"] = combat_updates["enemies"]

        # Update combat round
        if "round" in combat_updates and isinstance(combat_updates["round"], int):
            session.combat_state["round"] = combat_updates["round"]

        # Process player damage
        if "player_damage_taken" in combat_updates and isinstance(combat_updates["player_damage_taken"], int):
            damage = combat_updates["player_damage_taken"]
            if damage > 0:
                character.take_damage(damage)

                # Log damage to memory graph
                memory_graph = self.get_session_memory_graph(session.id)
                memory_graph.add_node(
                    content=f"Character took {damage} damage in combat",
                    node_type="combat",
                    importance=0.7,
                )

        # Always update the session with combat state changes
        game_state_service = GameStateService()
        game_state_service.update_session(session)

    def _process_quest_updates(self, function_args: Dict, session: "GameSession"):
        """Process quest updates from the AI response."""
        if "quest_updates" not in function_args or not isinstance(
            function_args["quest_updates"], dict
        ):
            return

        quest_updates = function_args["quest_updates"]

        # Initialize quest lists if not present
        if not hasattr(session, "active_quests"):
            session.active_quests = []
        if not hasattr(session, "completed_quests"):
            session.completed_quests = []

        # Add new quests
        if "new_quests" in quest_updates and isinstance(quest_updates["new_quests"], list):
            for quest in quest_updates["new_quests"]:
                if isinstance(quest, dict) and "id" in quest and "title" in quest:
                    # Make sure the quest isn't already in active quests
                    if not any(q.get("id") == quest["id"] for q in session.active_quests):
                        # Ensure quest has all required fields
                        if "description" not in quest:
                            quest["description"] = ""
                        if "objective" not in quest:
                            quest["objective"] = ""
                        if "reward" not in quest:
                            quest["reward"] = ""

                        session.active_quests.append(quest)

                        # Add to memory graph
                        memory_graph = self.get_session_memory_graph(session.id)
                        memory_graph.add_node(
                            content=f"Received quest: {quest['title']}",
                            node_type="quest",
                            importance=0.8,
                        )

        # Update existing quests
        if "updated_quests" in quest_updates and isinstance(quest_updates["updated_quests"], list):
            for update in quest_updates["updated_quests"]:
                if isinstance(update, dict) and "id" in update:
                    for i, quest in enumerate(session.active_quests):
                        if quest.get("id") == update["id"]:
                            # Make a safe copy of the quest to update
                            updated_quest = quest.copy()

                            # Update the quest with all fields from the update
                            for key, value in update.items():
                                updated_quest[key] = value

                            # Ensure quest maintains required fields
                            if "title" not in updated_quest and "title" in quest:
                                updated_quest["title"] = quest["title"]
                            if "description" not in updated_quest:
                                updated_quest["description"] = quest.get("description", "")

                            session.active_quests[i] = updated_quest

                            # Add to memory if there's progress info
                            if "progress" in update:
                                memory_graph = self.get_session_memory_graph(session.id)
                                memory_graph.add_node(
                                    content=f"Quest progress for {updated_quest.get('title', 'Unknown quest')}: {update['progress']}",
                                    node_type="quest",
                                    importance=0.7,
                                )

        # Complete quests
        if "completed_quests" in quest_updates and isinstance(quest_updates["completed_quests"], list):
            for quest_id in quest_updates["completed_quests"]:
                for i, quest in enumerate(session.active_quests):
                    if quest.get("id") == quest_id:
                        # Move quest from active to completed
                        completed_quest = session.active_quests.pop(i)
                        session.completed_quests.append(completed_quest)

                        # Add to memory graph
                        memory_graph = self.get_session_memory_graph(session.id)
                        memory_graph.add_node(
                            content=f"Completed quest: {completed_quest.get('title', 'Unknown quest')}",
                            node_type="quest",
                            importance=0.9,
                        )
                        break

        # Update the session with quest changes
        game_state_service = GameStateService()
        game_state_service.update_session(session)

    def _process_relationship_updates(self, function_args: Dict, session: "GameSession"):
        """Process NPC relationship updates from the AI response."""
        if "relationship_updates" not in function_args or not isinstance(
            function_args["relationship_updates"], dict
        ):
            return

        rel_updates = function_args["relationship_updates"]

        # Process NPC relationship changes
        if "npc_relationships" in rel_updates and isinstance(rel_updates["npc_relationships"], list):
            for npc_rel in rel_updates["npc_relationships"]:
                if isinstance(npc_rel, dict) and "name" in npc_rel and "change" in npc_rel:
                    npc_name = npc_rel["name"]
                    change_value = npc_rel["change"]
                    reason = npc_rel.get("reason", "Recent interaction")

                    # Update NPC relationships in session
                    if not hasattr(session, "npc_relationships"):
                        session.npc_relationships = {}

                    if npc_name not in session.npc_relationships:
                        session.npc_relationships[npc_name] = 0

                    session.npc_relationships[npc_name] += change_value

                    # Log relationship change to memory graph
                    memory_graph = self.get_session_memory_graph(session.id)
                    if change_value > 0:
                        memory_graph.add_node(
                            content=f"Relationship with {npc_name} improved by {change_value}: {reason}",
                            node_type="relationship",
                            importance=0.6,
                        )
                    else:
                        memory_graph.add_node(
                            content=f"Relationship with {npc_name} worsened by {abs(change_value)}: {reason}",
                            node_type="relationship",
                            importance=0.6,
                        )

            # Persist relationship changes
            game_state_service = GameStateService()
            game_state_service.update_session(session)

    def _add_interaction_to_memory(self, session: "GameSession", character: Character, function_args: Dict):
        """Add the current interaction to the memory graph."""
        if not hasattr(session, "id"):
            return

        memory_graph = self.get_session_memory_graph(session.id)

        # Create a summary of the interaction
        interaction_summary = function_args.get("message", "")
        if len(interaction_summary) > 300:
            interaction_summary = interaction_summary[:297] + "..."

        # Determine importance based on content
        importance = 0.5  # Default importance

        # Detect if this is an important event (quest, combat, etc.)
        important_keywords = ["quest", "mission", "battle", "fight", "found", "discovered", "treasure", "secret"]
        if any(keyword in interaction_summary.lower() for keyword in important_keywords):
            importance = 0.8

        # Add this interaction to memory
        memory_graph.add_node(
            content=interaction_summary,
            node_type="interaction",
            importance=importance,
        )

        # Save the memory graph
        memory_graph.save()

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
