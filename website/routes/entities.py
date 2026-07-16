from flask import Blueprint, render_template, request
from ..services.ai_bridge import entity_report, graph_stats

entities_bp = Blueprint("entities", __name__)


@entities_bp.route("/entities", methods=["GET"])
def entities():
    entity_name = request.args.get("name", "").strip()
    report = None

    if entity_name:
        report = entity_report(entity_name)

    stats = graph_stats()

    return render_template("entities.html", report=report, stats=stats, query=entity_name)