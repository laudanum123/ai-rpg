"""Mock responses for OpenAI API calls to ensure consistent test results."""

# Standard responses for different game master scenarios
GM_RESPONSES = {
    "game_start": {
        "fantasy": "Welcome, brave adventurer! You find yourself in the bustling town of Eldoria, a place where magic flows freely and danger lurks in the shadows. The local tavern, The Prancing Pegasus, seems like a good place to start your adventure. What would you like to do?",
        "sci_fi": "Your consciousness flickers to life in the cybernetic corridors of Neo-Terminus, a sprawling metropolis on the edge of known space. Your augmented reality display indicates several mission opportunities at the nearby trading hub. How do you proceed?",
        "horror": "The fog clings to your clothes as you step off the decrepit bus into the silent town of Ravenhollow. The buildings loom like ancient sentinels, and something feels... wrong. A chill runs down your spine as you notice the absence of any townsfolk. What's your first move?"
    },
    "combat_start": {
        "goblin": "From the shadows leap three goblins, their yellow eyes gleaming with malice! Their crude weapons are raised as they cackle with anticipation. Roll for initiative!",
        "undead": "The ground before you erupts as skeletal hands claw their way to the surface. Three rotting corpses pull themselves from shallow graves, their hollow eye sockets fixed on you. Combat begins!",
        "robot": "Warning klaxons blare as three security bots round the corner, their weapons systems powering up with an ominous hum. 'INTRUDER DETECTED. LETHAL FORCE AUTHORIZED.' Combat protocol initiated!"
    },
    "combat_action": {
        "hit": "Your weapon finds its mark! The {enemy} staggers back as your {weapon} connects with a satisfying thud, dealing {damage} points of damage. The creature looks wounded but remains standing.",
        "critical_hit": "A perfect strike! Your {weapon} arcs through the air and strikes a vital spot on the {enemy}. It howls in pain as you deal a critical hit for {damage} damage! The creature is severely wounded.",
        "miss": "Your attack goes wide as the {enemy} deftly sidesteps. Your {weapon} cuts through empty air, missing your target completely. The {enemy} prepares to counterattack!",
        "enemy_hit": "The {enemy} lashes out with surprising speed! Its {attack} catches you off guard, dealing {damage} points of damage. You wince in pain but maintain your fighting stance.",
        "enemy_miss": "The {enemy} lunges at you, but you react quickly and dodge out of the way. Its {attack} misses you completely, creating an opening for your next attack."
    },
    "combat_end": {
        "victory": "With a final, decisive blow, you defeat the last of your enemies! As the dust settles, you take a moment to catch your breath. The battle is won, and you've gained {xp} experience points. Searching the area, you find {gold} gold coins and {items}.",
        "defeat": "The world spins as you take one hit too many. Your vision blurs, and you collapse to the ground. Darkness claims you, but this is not the end. You awaken later, weakened but alive. Someone must have dragged you to safety. You've lost {gold} gold coins in the process."
    },
    "exploration": {
        "forest": "You push deeper into the ancient forest. Dappled sunlight filters through the dense canopy above, and the air is rich with the scent of moss and earth. Wildlife skitters unseen through the undergrowth. What do you wish to do in this verdant realm?",
        "dungeon": "The stone corridor stretches before you, dimly lit by guttering torches set in rusted sconces. Water drips somewhere in the distance, and the air is thick with the musty smell of age. The shadows seem to shift as if alive. How do you proceed?",
        "town": "The marketplace bustles with activity as merchants hawk their wares and townsfolk go about their daily business. The smell of fresh bread wafts from a nearby bakery, and you can hear a street performer playing a lively tune. Who would you like to interact with?"
    }
}

# Mock for memory context retrieval
MEMORY_CONTEXTS = {
    "basic": "You are in the town of Eldoria. You've spoken with the innkeeper about rumors of missing children in the nearby forest. The town guard seems unconcerned, but locals are frightened.",
    "quest_active": "You've accepted a quest to investigate the disappearances in the Whispering Woods. The local blacksmith gave you a silver dagger, saying it might help against 'unnatural things'. The missing children were last seen picking berries at the forest edge.",
    "combat_history": "You've already encountered and defeated a pack of wolves in the forest. One of them had an unusual collar with strange runes. You took minor damage in that fight but have since rested and recovered."
}

# Mock for OpenAI API behavior with different instructions
def mock_openai_completion(system_prompt, user_message, relevant_memory=None):
    """Simulate an OpenAI API call with deterministic responses based on input patterns."""

    # Include memory in system prompt if available
    full_system_prompt = system_prompt
    if relevant_memory:
        full_system_prompt += f"\n\nRelevant game history:\n{relevant_memory}"

    # Detect key phrases in the user message to determine response category
    message_lower = user_message.lower()

    # Game start detection
    if "start game" in message_lower or "new adventure" in message_lower:
        if "sci-fi" in message_lower or "future" in message_lower:
            return GM_RESPONSES["game_start"]["sci_fi"]
        elif "horror" in message_lower or "scary" in message_lower:
            return GM_RESPONSES["game_start"]["horror"]
        else:
            return GM_RESPONSES["game_start"]["fantasy"]

    # Combat detection
    elif "fight" in message_lower or "attack" in message_lower or "combat" in message_lower:
        if "undead" in message_lower or "skeleton" in message_lower or "zombie" in message_lower:
            return GM_RESPONSES["combat_start"]["undead"]
        elif "robot" in message_lower or "machine" in message_lower or "tech" in message_lower:
            return GM_RESPONSES["combat_start"]["robot"]
        else:
            return GM_RESPONSES["combat_start"]["goblin"]

    # Combat action detection
    elif "sword" in message_lower or "swing" in message_lower or "strike" in message_lower:
        if "critical" in full_system_prompt or "natural 20" in full_system_prompt:
            return GM_RESPONSES["combat_action"]["critical_hit"].format(
                enemy="goblin", weapon="sword", damage=12
            )
        elif "miss" in full_system_prompt or "failed roll" in full_system_prompt:
            return GM_RESPONSES["combat_action"]["miss"].format(
                enemy="goblin", weapon="sword"
            )
        else:
            return GM_RESPONSES["combat_action"]["hit"].format(
                enemy="goblin", weapon="sword", damage=6
            )

    # Exploration detection
    elif "explore" in message_lower or "look around" in message_lower or "investigate" in message_lower:
        if "forest" in message_lower or "woods" in message_lower:
            return GM_RESPONSES["exploration"]["forest"]
        elif "dungeon" in message_lower or "cave" in message_lower or "underground" in message_lower:
            return GM_RESPONSES["exploration"]["dungeon"]
        else:
            return GM_RESPONSES["exploration"]["town"]

    # Default response for other queries
    else:
        return "The Game Master ponders your action for a moment. \"An interesting choice. Let's see where this leads us...\""
