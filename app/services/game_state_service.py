import json
import os
from typing import Dict, List, Optional
from app.models.game_session import GameSession


class GameStateService:
    """Service for managing game sessions."""

    def __init__(self, data_dir: str = None):
        """Initialize the game state service.

        Args:
            data_dir: Directory to store session data. Defaults to 'data/sessions'.
        """
        if data_dir is None:
            # Create a default data directory in the app's instance folder
            from flask import current_app

            if current_app:
                data_dir = os.path.join(current_app.instance_path, "sessions")
            else:
                data_dir = "data/sessions"

        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

        # In-memory cache of sessions
        self.sessions = {}

    def create_session(self, character_id: str, game_world: str = "Fantasy") -> str:
        """Create a new game session.

        Args:
            character_id: ID of the character in the session
            game_world: Type of game world

        Returns:
            ID of the newly created session
        """
        # Create a new session
        session = GameSession(character_id=character_id, game_world=game_world)

        # Initialize with default starting location
        session.set_current_location(
            {
                "name": "Town Square",
                "description": "The bustling center of a small town. Various shops and buildings surround the square.",
                "exits": ["tavern", "blacksmith", "general_store", "town_gate"],
            }
        )

        # Save the session to storage
        self.save_session(session)

        # Cache the session in memory
        self.sessions[session.id] = session

        return session.id

    def get_session(self, session_id: str) -> Optional[GameSession]:
        """Get a session by ID.

        Args:
            session_id: Session ID

        Returns:
            The session if found, None otherwise
        """
        # Check if the session is in memory cache
        if session_id in self.sessions:
            return self.sessions[session_id]

        # Try to load the session from storage
        session_path = os.path.join(self.data_dir, f"{session_id}.json")
        if os.path.exists(session_path):
            try:
                with open(session_path, "r") as f:
                    try:
                        session_data = json.load(f)
                        session = GameSession.from_dict(session_data)
                        # Cache the session
                        self.sessions[session_id] = session
                        return session
                    except json.JSONDecodeError as e:
                        print(f"Error reading session {session_id}: {e}")
            except Exception as e:
                print(f"Error processing session {session_id}: {e}")

        return None

    def is_valid_session(self, session_id: str) -> bool:
        """Check if a session ID is valid.

        Args:
            session_id: Session ID to check

        Returns:
            True if the session exists, False otherwise
        """
        return self.get_session(session_id) is not None

    def get_all_sessions(self) -> List[GameSession]:
        """Get all available game sessions.

        Returns:
            List of all game sessions
        """
        sessions = []
        # Check the data directory for session files
        for filename in os.listdir(self.data_dir):
            if filename.endswith(".json"):
                try:
                    session_id = filename.replace(".json", "")
                    # Attempt to load the session directly rather than using get_session
                    # to handle potential JSON errors more gracefully
                    session_path = os.path.join(self.data_dir, filename)
                    with open(session_path, "r") as f:
                        try:
                            session_data = json.load(f)
                            session = GameSession.from_dict(session_data)
                            sessions.append(session)
                        except json.JSONDecodeError as e:
                            print(f"Error reading session file {filename}: {e}")
                except Exception as e:
                    print(f"Error processing session file {filename}: {e}")
        return sessions

    def delete_session(self, session_id: str) -> bool:
        """Delete a game session.

        Args:
            session_id: ID of the session to delete

        Returns:
            True if successful, False otherwise
        """
        session_path = os.path.join(self.data_dir, f"{session_id}.json")

        if os.path.exists(session_path):
            try:
                # Remove from disk
                os.remove(session_path)

                # Remove from memory cache if present
                if session_id in self.sessions:
                    del self.sessions[session_id]

                # Also delete the session's memory graph directory if it exists
                try:
                    from app import game_master

                    if (
                        hasattr(game_master, "memory_graph_config")
                        and "storage_dir" in game_master.memory_graph_config
                    ):
                        memory_dir = os.path.join(
                            game_master.memory_graph_config["storage_dir"], session_id
                        )
                        if os.path.exists(memory_dir):
                            import shutil

                            shutil.rmtree(memory_dir)
                except Exception as e:
                    print(
                        f"Warning: Could not delete memory graph for session {session_id}: {e}"
                    )

                return True
            except Exception as e:
                print(f"Error deleting session {session_id}: {e}")
                return False
        return False

    def save_session(self, session: GameSession) -> None:
        """Save a session to storage.

        Args:
            session: Session to save
        """
        session_path = os.path.join(self.data_dir, f"{session.id}.json")

        # Convert the session to a dictionary for serialization
        session_data = {
            "id": session.id,
            "character_id": session.character_id,
            "game_world": session.game_world,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
            "history": session.history,
            "current_location": session.current_location,
            "npcs": session.npcs,
            "locations": session.locations,
            "plot_hooks": session.plot_hooks,
            "active_quests": session.active_quests,
            "completed_quests": session.completed_quests,
            "in_combat": session.in_combat,
            "combat_state": session.combat_state,
        }

        with open(session_path, "w") as f:
            json.dump(session_data, f, indent=4)

    def update_session(self, session: GameSession) -> None:
        """Update a session in storage.

        Args:
            session: Session to update
        """
        self.sessions[session.id] = session
        self.save_session(session)

    def delete_session(self, session_id: str) -> bool:
        """Delete a session.

        Args:
            session_id: ID of the session to delete

        Returns:
            True if the session was deleted, False otherwise
        """
        session_path = os.path.join(self.data_dir, f"{session_id}.json")
        if os.path.exists(session_path):
            os.remove(session_path)
            # Remove from cache if present
            if session_id in self.sessions:
                del self.sessions[session_id]
            return True
        return False

    def add_message_to_history(self, session_id: str, role: str, content: str) -> bool:
        """Add a message to a session's history.

        Args:
            session_id: Session ID
            role: Message role (e.g., 'user', 'assistant', 'system')
            content: Message content

        Returns:
            True if the message was added, False otherwise
        """
        session = self.get_session(session_id)
        if session:
            session.add_message_to_history(role, content)
            self.update_session(session)
            return True
        return False

    def get_session_history(self, session_id: str) -> List[Dict]:
        """Get a session's conversation history.

        Args:
            session_id: Session ID

        Returns:
            List of messages in the session's history
        """
        session = self.get_session(session_id)
        if session:
            return session.history
        return []

    def add_npc_to_session(self, session_id: str, npc_id: str, npc_data: Dict) -> bool:
        """Add or update an NPC in a session.

        Args:
            session_id: Session ID
            npc_id: NPC ID
            npc_data: NPC data

        Returns:
            True if the NPC was added/updated, False otherwise
        """
        session = self.get_session(session_id)
        if session:
            session.add_npc(npc_id, npc_data)
            self.update_session(session)
            return True
        return False

    def add_location_to_session(
        self, session_id: str, location_id: str, location_data: Dict
    ) -> bool:
        """Add or update a location in a session.

        Args:
            session_id: Session ID
            location_id: Location ID
            location_data: Location data

        Returns:
            True if the location was added/updated, False otherwise
        """
        session = self.get_session(session_id)
        if session:
            session.add_location(location_id, location_data)
            self.update_session(session)
            return True
        return False

    def set_current_location(self, session_id: str, location_data: Dict) -> bool:
        """Set the current location in a session.

        Args:
            session_id: Session ID
            location_data: Location data

        Returns:
            True if the location was set, False otherwise
        """
        session = self.get_session(session_id)
        if session:
            session.set_current_location(location_data)
            self.update_session(session)
            return True
        return False

    def add_plot_hook(self, session_id: str, plot_hook: Dict) -> bool:
        """Add a plot hook to a session.

        Args:
            session_id: Session ID
            plot_hook: Plot hook data

        Returns:
            True if the plot hook was added, False otherwise
        """
        session = self.get_session(session_id)
        if session:
            session.add_plot_hook(plot_hook)
            self.update_session(session)
            return True
        return False

    def start_combat(self, session_id: str, enemies: List[Dict]) -> bool:
        """Start a combat encounter in a session.

        Args:
            session_id: Session ID
            enemies: List of enemy data

        Returns:
            True if combat was started, False otherwise
        """
        session = self.get_session(session_id)
        if session:
            session.start_combat(enemies)
            self.update_session(session)
            return True
        return False

    def end_combat(self, session_id: str) -> bool:
        """End the current combat encounter in a session.

        Args:
            session_id: Session ID

        Returns:
            True if combat was ended, False otherwise
        """
        session = self.get_session(session_id)
        if session and session.in_combat:
            session.end_combat()
            self.update_session(session)
            return True
        return False

    def next_combat_round(self, session_id: str) -> bool:
        """Advance to the next combat round in a session.

        Args:
            session_id: Session ID

        Returns:
            True if advanced to next round, False otherwise
        """
        session = self.get_session(session_id)
        if session and session.in_combat:
            session.next_combat_round()
            self.update_session(session)
            return True
        return False

    def add_combat_log(self, session_id: str, message: str) -> bool:
        """Add an entry to a session's combat log.

        Args:
            session_id: Session ID
            message: Log message

        Returns:
            True if the message was added, False otherwise
        """
        session = self.get_session(session_id)
        if session and session.in_combat:
            session.add_combat_log(message)
            self.update_session(session)
            return True
        return False

    def add_quest(self, session_id: str, quest: Dict) -> bool:
        """Add a quest to a session.

        Args:
            session_id: Session ID
            quest: Quest data

        Returns:
            True if the quest was added, False otherwise
        """
        session = self.get_session(session_id)
        if session:
            session.add_quest(quest)
            self.update_session(session)
            return True
        return False

    def complete_quest(self, session_id: str, quest_id: str) -> Optional[Dict]:
        """Mark a quest as completed in a session.

        Args:
            session_id: Session ID
            quest_id: Quest ID

        Returns:
            The completed quest if successful, None otherwise
        """
        session = self.get_session(session_id)
        if session:
            completed_quest = session.complete_quest(quest_id)
            if completed_quest:
                self.update_session(session)
                return completed_quest
        return None

    def set_session_attribute(self, session_id: str, attribute: str, value) -> bool:
        """Set a custom attribute on a session.

        Args:
            session_id: Session ID
            attribute: Attribute name
            value: Attribute value

        Returns:
            True if the attribute was set, False otherwise
        """
        session = self.get_session(session_id)
        if session:
            # Set the attribute directly on the object
            setattr(session, attribute, value)
            # Update the session
            self.update_session(session)
            return True
        return False
