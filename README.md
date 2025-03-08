# AI Game Master RPG

A web-based role-playing game (RPG) with an AI-powered Game Master using GPT-4o-mini. This application provides an immersive RPG experience where players can create characters, explore locations, interact with NPCs, complete quests, and engage in combat, all managed by an intelligent AI Game Master.

## Features

- AI-powered Game Master using GPT-4o-mini
- Character creation and management
- Dynamic world exploration
- NPC interactions with realistic dialogue
- Quest generation and tracking
- Turn-based combat system
- Inventory management
- Experience and leveling system

## Requirements

- Python 3.8+
- uv package manager
- OpenAI API key (for GPT-4o-mini integration)
- Dependencies listed in pyproject.toml

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd rpg
```

2. Install uv if you don't have it already:
```bash
pip install uv
```

3. Create a virtual environment and install dependencies using uv:
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```

4. Set up environment variables:
```bash
# Create a .env file with:
OPENAI_API_KEY=your_api_key_here
SECRET_KEY=your_secret_key_here
```

## Running the Application

1. Start the Flask server:
```bash
python run.py
```

2. Open a web browser and navigate to:
```
http://localhost:5000
```

## Project Structure

```
rpg/
├── app/
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py
│   ├── models/
│   │   ├── character.py
│   │   ├── combat.py
│   │   ├── game_session.py
│   │   └── npc.py
│   ├── services/
│   │   ├── character_service.py
│   │   ├── game_master.py
│   │   └── game_state_service.py
│   ├── static/
│   ├── templates/
│   ├── utils/
│   └── __init__.py
├── instance/
├── .venv/        # uv creates .venv by default
├── .env
├── .gitignore
├── README.md
├── pyproject.toml # New dependency management file
└── run.py
```

## API Endpoints

### Character Management
- `POST /api/character/create` - Create a new character
- `GET /api/character/<character_id>` - Get character details
- `GET /api/inventory/<character_id>` - Get character inventory

### Game Session
- `POST /api/session/start` - Start a new game session
- `POST /api/action` - Process a player's action
- `GET /api/session/<session_id>` - Get session state

### Combat
- `POST /api/combat/start` - Start a combat encounter
- `POST /api/combat/action` - Process a combat action

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for GPT-4o-mini
- Flask framework
- All contributors and testers 