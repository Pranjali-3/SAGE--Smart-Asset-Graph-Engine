from flask import Blueprint, render_template, request
from website.services.ai_bridge import entity_report, graph_stats
from ai_core.entity_extractor import extract_entities

entities_bp = Blueprint("entities", __name__)


@entities_bp.route("/entities", methods=["GET"])
def entities():
    query = request.args.get("name", "").strip()
    extracted = []
    report = None

    if query:
        # Extract entities from the query text
        extracted = extract_entities(query)

        # Also try knowledge graph lookup
        report = entity_report(query)

    stats = graph_stats()

    return render_template(
        "entities.html",
        report=report,
        stats=stats,
        query=query,
        extracted=extracted
    )