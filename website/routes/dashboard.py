from flask import Blueprint, render_template
from website.models import Document, Message

dashboard_bp = Blueprint("dashboard", __name__)


@dashboard_bp.route("/")
def dashboard():
    stats = {
        "documents": Document.query.count(),
        "messages": Message.query.count(),
    }
    return render_template("dashboard.html", stats=stats)