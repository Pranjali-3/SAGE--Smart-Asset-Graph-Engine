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
    app.config["SECRET_KEY"] = "dev"
    app.config["UPLOAD_FOLDER"] = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "uploads"
    )
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    db.init_app(app)

    # Import models BEFORE db.create_all(), so SQLAlchemy knows the tables exist
    from . import models

    from routes.dashboard import dashboard_bp
    from routes.chat import chat_bp
    from routes.documents import documents_bp
    from routes.predict import predict_bp
    from routes.entities import entities_bp

    app.register_blueprint(dashboard_bp)
    app.register_blueprint(chat_bp)
    app.register_blueprint(documents_bp)
    app.register_blueprint(predict_bp)
    app.register_blueprint(entities_bp)

    with app.app_context():
        db.create_all()

    return app