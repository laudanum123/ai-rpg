import os
import json
from typing import Dict, List, Optional, Tuple, Deque
import random
from datetime import datetime
import openai
from collections import deque
from flask import current_app
from app.models.character import Character
from app.models.npc import NPC
from app.models.combat import Enemy, CombatEncounter, roll_dice

class GameMaster:
    """Service for AI-powered game mastering using GPT-4o-mini."""
    
    def __init__(self):
        """Initialize the game master service."""
        # Set OpenAI API key from environment
        openai.api_key = os.getenv('OPENAI_API_KEY')
        
        # Store for API debug messages (max 50 entries)
        self.api_debug_logs = deque(maxlen=50)
        self.debug_enabled = False
        
        # System prompt for the AI game master
        self.system_prompt = """You are an experienced Game Master for a fantasy RPG game.
        Your role is to create an engaging and dynamic adventure, manage NPCs, describe
        locations vividly, create interesting plot hooks, and run combat encounters.
        Always stay in character as a GM and maintain consistency in the game world.
        Focus on creating an immersive experience while following the game's rules."""
    
    def get_ai_response(self, messages: List[Dict]) -> str:
        """Get a response from the AI model.
        
        Args:
            messages: List of conversation messages
            
        Returns:
            AI model's response text
        """
        # Check if API debug mode is enabled in the app config
        try:
            self.debug_enabled = current_app.config.get('API_DEBUG', False)
        except RuntimeError:
            # Not in an application context, assume debug is False
            self.debug_enabled = False
            
        try:
            # Use OpenAI 0.28.0 API format
            request_data = {
                "model": "gpt-4o-mini",
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 2000
            }
            
            # Store request data if debug is enabled
            if self.debug_enabled:
                debug_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "request": request_data,
                    "response": None,
                    "error": None
                }
            
            # Make the actual API call
            response = openai.ChatCompletion.create(**request_data)
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
    
    def start_game(self, character: Character, game_world: str) -> str:
        """Start a new game session.
        
        Args:
            character: Player character
            game_world: Type of game world
            
        Returns:
            Introduction message from the GM
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"""Start a new game session with:
            - Character: {character.name}, a level {character.level} {character.character_class}
            - World: {game_world}
            Provide an engaging introduction and initial scene."""}
        ]
        
        return self.get_ai_response(messages)
    
    def process_action(self, session: 'GameSession', character: Character, action: str) -> str:
        """Process a player's action.
        
        Args:
            session: Current game session
            character: Player character
            action: Player's action description
            
        Returns:
            GM's response to the action
        """
        # Build context from session state
        context = f"""Current location: {session.current_location.get('name')}
        Location description: {session.current_location.get('description')}
        Available exits: {', '.join(session.current_location.get('exits', []))}
        """
        
        # Add relevant NPCs to context
        npcs_present = [npc for npc in session.npcs.values() 
                       if npc.get('location') == session.current_location.get('name')]
        if npcs_present:
            context += "\nNPCs present:\n" + "\n".join(
                f"- {npc.get('name')}: {npc.get('description')}"
                for npc in npcs_present
            )
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"""Game state:
            {context}
            
            Character: {character.name}, level {character.level} {character.character_class}
            Health: {character.health}/{character.max_health}
            Gold: {character.gold}
            
            Inventory:
            {', '.join(item.get('name', 'Unknown Item') for item in character.inventory) if character.inventory else 'No items'}
            
            Stats:
            - Strength: {character.strength} (Modifier: {character.get_ability_modifier('strength')})
            - Dexterity: {character.dexterity} (Modifier: {character.get_ability_modifier('dexterity')})
            - Constitution: {character.constitution} (Modifier: {character.get_ability_modifier('constitution')})
            - Intelligence: {character.intelligence} (Modifier: {character.get_ability_modifier('intelligence')})
            - Wisdom: {character.wisdom} (Modifier: {character.get_ability_modifier('wisdom')})
            - Charisma: {character.charisma} (Modifier: {character.get_ability_modifier('charisma')})
            
            Player action: {action}
            
            Respond as the GM, describing the outcome of this action. If the player is trying to use an item, make sure to reference it properly based on their inventory."""}
        ]
        
        return self.get_ai_response(messages)
    
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
            {"role": "user", "content": f"""Create a detailed NPC:
            Type: {npc_type}
            Location: {location}
            Include:
            - Name
            - Description
            - Personality
            - Key information or quest hooks
            Format as JSON."""}
        ]
        
        response = self.get_ai_response(messages)
        try:
            npc_data = json.loads(response)
            return NPC(
                name=npc_data.get("name", "Unknown"),
                type=npc_type,
                description=npc_data.get("description", ""),
                dialogue=npc_data.get("dialogue", {})
            )
        except json.JSONDecodeError:
            # Fallback NPC if AI response isn't valid JSON
            return NPC(
                name=f"{npc_type.title()} #{random.randint(1, 1000)}",
                type=npc_type,
                description=f"A typical {npc_type} in {location}."
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
            {"role": "user", "content": f"""Create a detailed location:
            Name: {location_name}
            Type: {location_type}
            Include:
            - Description
            - Notable features
            - Possible encounters
            - Connected locations
            Format as JSON."""}
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
                "exits": []
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
            {"role": "user", "content": f"""Create a quest for:
            Character: Level {character.level} {character.character_class}
            Difficulty: {difficulty}
            Include:
            - Title
            - Description
            - Objectives
            - Rewards
            Format as JSON."""}
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
                    "experience": character.level * 100
                }
            }
    
    def start_combat(self, session: 'GameSession', character: Character, 
                    enemy_type: str = "random") -> Dict:
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
            {"role": "user", "content": f"""Create a combat encounter for:
            Character: Level {character.level} {character.character_class}
            Enemy type: {enemy_type}
            Location: {session.current_location.get('name')}
            Include:
            - Enemy details
            - Combat environment
            - Special conditions
            Format as JSON."""}
        ]
        
        response = self.get_ai_response(messages)
        try:
            encounter_data = json.loads(response)
            enemies = [
                Enemy(**enemy_data)
                for enemy_data in encounter_data.get("enemies", [])
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
                    charisma=10
                )
            ]
        
        # Create combat encounter
        encounter = CombatEncounter(
            enemies=enemies,
            environment=session.current_location.get('name', 'battlefield'),
            description=f"Combat with {len(enemies)} enemies"
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
            "message": f"Combat begins! Initiative order: {', '.join(f'{name} ({init})' for name, init in initiatives)}"
        }
    
    def process_combat_action(self, session: 'GameSession', character: Character,
                            action: str) -> Dict:
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
            {"role": "user", "content": f"""Process combat action:
            Character: Level {character.level} {character.character_class}
            Health: {character.health}/{character.max_health}
            
            Inventory:
            {', '.join(item.get('name', 'Unknown Item') for item in character.inventory) if character.inventory else 'No items'}
            
            Combat stats:
            - Strength: {character.strength} (Modifier: {character.get_ability_modifier('strength')})
            - Dexterity: {character.dexterity} (Modifier: {character.get_ability_modifier('dexterity')})
            
            Action: {action}
            Combat state: Round {session.combat_state.get('round')}
            
            Describe the outcome and update combat state. If the player is trying to use an item from their inventory, incorporate it into the result.
            Format as JSON with 'description' and 'effects'."""}
        ]
        
        response = self.get_ai_response(messages)
        try:
            result = json.loads(response)
            return {
                "description": result.get("description", "The action is resolved."),
                "effects": result.get("effects", {}),
                "combat_continues": True  # Update based on combat state
            }
        except json.JSONDecodeError:
            return {
                "description": "The action is processed.",
                "effects": {},
                "combat_continues": True
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
            {"role": "user", "content": f"""Generate loot for:
            Enemy: {enemy.name}
            Character Level: {character_level}
            Include:
            - Items
            - Gold
            - Special rewards
            Format as JSON."""}
        ]
        
        response = self.get_ai_response(messages)
        try:
            loot_data = json.loads(response)
            return loot_data.get("items", [])
        except json.JSONDecodeError:
            # Fallback loot if AI response isn't valid JSON
            return [{
                "id": f"gold_{random.randint(1000, 9999)}",
                "name": "Gold",
                "type": "currency",
                "amount": enemy.gold_reward
            }]
