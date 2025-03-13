from flask import Flask
from flask_cors import CORS
import os
from dotenv import load_dotenv
from flaskext.markdown import Markdown

# Load environment variables
load_dotenv()


def create_app(test_config=None):
    # Create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    CORS(app)

    app.config.from_mapping(
        SECRET_KEY=os.environ.get("SECRET_KEY", "dev"),
        DATABASE=os.path.join(app.instance_path, "rpg.sqlite"),
        API_DEBUG=os.environ.get("API_DEBUG", "false").lower() == "true",
    )

    if test_config is None:
        # Load the instance config, if it exists, when not testing
        app.config.from_pyfile("config.py", silent=True)
    else:
        # Load the test config if passed in
        app.config.from_mapping(test_config)

    # Ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # Register blueprints
    from app.api import api_bp

    app.register_blueprint(api_bp)

    # Register main routes blueprint
    from app.routes import main

    app.register_blueprint(main)

    # Initialize Flask-Markdown
    Markdown(app)

    return app
