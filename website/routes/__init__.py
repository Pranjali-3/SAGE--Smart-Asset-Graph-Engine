import sys
import os
from flask import Flask

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "ai_core")
)

from website import db

def create_app():
    app = Flask(__name__)

    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///sage.db"
    app.config["SECRET_KEY"] = "dev"  # change later for production

    db.init_app(app)

    from .dashboard import dashboard_bp
    app.register_blueprint(dashboard_bp)

    with app.app_context():
        db.create_all()

    return app