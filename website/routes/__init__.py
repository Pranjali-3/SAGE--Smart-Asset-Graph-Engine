import sys
import os
from flask import Flask

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "ai_core")
)

from website.db_extension import db

def create_app():
    app = Flask(
    __name__,
    template_folder='../templates',
    static_folder='../static'
    )
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///sage.db"
    app.config["SECRET_KEY"] = "dev"
    app.config["UPLOAD_FOLDER"] = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "uploads"
    )
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    db.init_app(app)
    from website import models

    from .dashboard import dashboard_bp
    from .chat import chat_bp
    from .documents import documents_bp
    from .predict import predict_bp
    from .entities import entities_bp
    from .voice import voice_bp
    from .knowledge_graph import kg_bp

    app.register_blueprint(dashboard_bp)
    app.register_blueprint(chat_bp)
    app.register_blueprint(documents_bp)
    app.register_blueprint(predict_bp)
    app.register_blueprint(entities_bp)
    app.register_blueprint(voice_bp)
    app.register_blueprint(kg_bp)

    with app.app_context():
        db.create_all()

    return app