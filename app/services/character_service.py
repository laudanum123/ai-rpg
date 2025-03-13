import json
import os
from typing import Dict, List, Optional
from app.models.character import Character


class CharacterService:
    """Service for managing player characters."""

    def __init__(self, data_dir: str = None):
        """Initialize the character service.

        Args:
            data_dir: Directory to store character data. Defaults to 'data/characters'.
        """
        if data_dir is None:
            # Create a default data directory in the app's instance folder
            from flask import current_app

            if current_app:
                data_dir = os.path.join(current_app.instance_path, "characters")
            else:
                data_dir = "data/characters"

        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

        # In-memory cache of characters
        self.characters = {}

    def create_character(self, name: str, character_class: str) -> Character:
        """Create a new character.

        Args:
            name: Character name
            character_class: Character class (e.g., 'Fighter', 'Wizard')

        Returns:
            The newly created character
        """
        # Create a new character with default stats
        character = Character(name=name, character_class=character_class)

        # Adjust stats based on class
        if character_class.lower() == "fighter":
            character.strength += 2
            character.constitution += 1
            character.max_health += 10
            character.health = character.max_health
            character.abilities.append(
                {
                    "name": "Second Wind",
                    "description": "Recover 1d10 + level hit points. Once per short rest.",
                }
            )
        elif character_class.lower() == "wizard":
            character.intelligence += 2
            character.wisdom += 1
            character.abilities.append(
                {
                    "name": "Arcane Recovery",
                    "description": "Recover spell slots during a short rest.",
                }
            )
        elif character_class.lower() == "rogue":
            character.dexterity += 2
            character.charisma += 1
            character.abilities.append(
                {
                    "name": "Sneak Attack",
                    "description": "Deal extra damage when you have advantage on attack rolls.",
                }
            )

        # Add starting items based on class
        if character_class.lower() == "fighter":
            character.add_item(
                {
                    "id": "sword_1",
                    "name": "Longsword",
                    "type": "weapon",
                    "damage": "1d8",
                    "properties": ["versatile"],
                    "description": "A versatile sword that can be used with one or two hands.",
                }
            )
            character.add_item(
                {
                    "id": "shield_1",
                    "name": "Shield",
                    "type": "armor",
                    "armor_class": 2,
                    "description": "A sturdy shield that provides protection in combat.",
                }
            )
        elif character_class.lower() == "wizard":
            character.add_item(
                {
                    "id": "staff_1",
                    "name": "Wizard's Staff",
                    "type": "weapon",
                    "damage": "1d6",
                    "properties": ["casting_focus"],
                    "description": "A wooden staff that serves as a spellcasting focus.",
                }
            )
            character.add_item(
                {
                    "id": "spellbook_1",
                    "name": "Spellbook",
                    "type": "tool",
                    "description": "A book containing arcane knowledge and spells.",
                }
            )
        elif character_class.lower() == "rogue":
            character.add_item(
                {
                    "id": "dagger_1",
                    "name": "Dagger",
                    "type": "weapon",
                    "damage": "1d4",
                    "properties": ["finesse", "light", "thrown"],
                    "description": "A small, easily concealable blade.",
                }
            )
            character.add_item(
                {
                    "id": "thieves_tools_1",
                    "name": "Thieves' Tools",
                    "type": "tool",
                    "description": "Tools used for picking locks and disarming traps.",
                }
            )

        # Add common starting items
        character.add_item(
            {
                "id": "potion_1",
                "name": "Healing Potion",
                "type": "consumable",
                "effect": "Restore 2d4+2 hit points when consumed.",
                "description": "A red liquid that heals wounds when consumed.",
            }
        )

        # Save the character to storage
        self.save_character(character)

        # Cache the character in memory
        self.characters[character.id] = character

        return character

    def get_character(self, character_id: str) -> Optional[Character]:
        """Get a character by ID.

        Args:
            character_id: Character ID

        Returns:
            The character if found, None otherwise
        """
        # Check if the character is in memory cache
        if character_id in self.characters:
            return self.characters[character_id]

        # Try to load the character from storage
        character_path = os.path.join(self.data_dir, f"{character_id}.json")
        if os.path.exists(character_path):
            with open(character_path, "r") as f:
                character_data = json.load(f)
                character = Character.from_dict(character_data)
                # Cache the character
                self.characters[character_id] = character
                return character

        return None

    def get_or_create_character(self, name: str, character_class: str) -> Character:
        """Get a character by name or create a new one if not found.

        Args:
            name: Character name
            character_class: Character class

        Returns:
            The existing or newly created character
        """
        # Try to find an existing character with the same name
        character_files = os.listdir(self.data_dir)
        for file_name in character_files:
            if file_name.endswith(".json"):
                character_path = os.path.join(self.data_dir, file_name)
                with open(character_path, "r") as f:
                    character_data = json.load(f)
                    if character_data.get("name") == name:
                        character_id = character_data.get("id")
                        return self.get_character(character_id)

        # Character not found, create a new one
        return self.create_character(name, character_class)

    def save_character(self, character: Character) -> None:
        """Save a character to storage.

        Args:
            character: Character to save
        """
        character_path = os.path.join(self.data_dir, f"{character.id}.json")
        with open(character_path, "w") as f:
            json.dump(character.to_dict(), f, indent=4)

    def update_character(self, character: Character) -> None:
        """Update a character in storage.

        Args:
            character: Character to update
        """
        self.characters[character.id] = character
        self.save_character(character)

    def delete_character(self, character_id: str) -> bool:
        """Delete a character.

        Args:
            character_id: ID of the character to delete

        Returns:
            True if the character was deleted, False otherwise
        """
        character_path = os.path.join(self.data_dir, f"{character_id}.json")
        if os.path.exists(character_path):
            os.remove(character_path)
            # Remove from cache if present
            if character_id in self.characters:
                del self.characters[character_id]
            return True
        return False

    def get_inventory(self, character_id: str) -> List[Dict]:
        """Get a character's inventory.

        Args:
            character_id: Character ID

        Returns:
            List of items in the character's inventory
        """
        character = self.get_character(character_id)
        if character:
            return character.inventory
        return []

    def add_item_to_inventory(self, character_id: str, item: Dict) -> bool:
        """Add an item to a character's inventory.

        Args:
            character_id: Character ID
            item: Item to add

        Returns:
            True if the item was added, False otherwise
        """
        character = self.get_character(character_id)
        if character:
            character.add_item(item)
            self.update_character(character)
            return True
        return False

    def remove_item_from_inventory(
        self, character_id: str, item_id: str
    ) -> Optional[Dict]:
        """Remove an item from a character's inventory.

        Args:
            character_id: Character ID
            item_id: ID of the item to remove

        Returns:
            The removed item if successful, None otherwise
        """
        character = self.get_character(character_id)
        if character:
            item = character.remove_item(item_id)
            if item:
                self.update_character(character)
                return item
        return None
