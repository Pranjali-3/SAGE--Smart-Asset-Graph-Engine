from datetime import datetime
from .db_extension import db


class Session(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    messages = db.relationship("Message", backref="session", lazy=True)


class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey("session.id"), nullable=False)
    role = db.Column(db.String(10))       
    content = db.Column(db.Text)
    sources = db.Column(db.Text)          
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)


class Document(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255))
    status = db.Column(db.String(20), default="pending")
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)