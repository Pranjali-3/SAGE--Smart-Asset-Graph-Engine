import json
import os
import tempfile
from flask import Blueprint, render_template, request, jsonify, send_file
from website.services.ai_bridge import get_kg

kg_bp = Blueprint("knowledge_graph", __name__)


@kg_bp.route("/knowledge-graph")
def knowledge_graph():
    kg = get_kg()
    stats = kg.graph_statistics()
    return render_template("knowledge_graph.html", stats=stats)


@kg_bp.route("/api/kg/graph-data")
def graph_data():
    """Return all nodes and edges for vis.js visualization."""
    kg = get_kg()
    graph = kg.entity_graph

    nodes = []
    for node in graph.nodes():
        in_deg = graph.in_degree(node)
        out_deg = graph.out_degree(node)
        total = in_deg + out_deg
        if total > 8:
            size = 45
        elif total > 4:
            size = 35
        elif total > 0:
            size = 28
        else:
            size = 20
        nodes.append({
            "id": node,
            "label": node,
            "size": size,
            "inDegree": in_deg,
            "outDegree": out_deg,
        })

    edges = []
    for u, v, data in graph.edges(data=True):
        edges.append({
            "from": u,
            "to": v,
            "label": data.get("relation", ""),
            "relationType": data.get("relation_type", ""),
            "confidence": data.get("confidence", 0),
        })

    return jsonify({"nodes": nodes, "edges": edges})


@kg_bp.route("/api/kg/search")
def search_entity():
    """Search for an entity and return its neighbors + relationships."""
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"error": "No query provided"}), 400

    kg = get_kg()
    entity = kg.find_entity(query)

    if entity is None:
        return jsonify({"error": f"Entity '{query}' not found"}), 404

    neighbors = kg.get_neighbors(entity)
    incoming = kg.get_incoming_relationships(entity)
    outgoing = kg.get_outgoing_relationships(entity)

    return jsonify({
        "entity": entity,
        "neighbors": neighbors,
        "incoming": incoming,
        "outgoing": outgoing,
    })


@kg_bp.route("/api/kg/entity/<entity_name>")
def entity_detail(entity_name):
    """Get full details for a specific entity."""
    kg = get_kg()
    entity = kg.find_entity(entity_name)

    if entity is None:
        return jsonify({"error": "Entity not found"}), 404

    neighbors = kg.get_neighbors(entity)
    incoming = kg.get_incoming_relationships(entity)
    outgoing = kg.get_outgoing_relationships(entity)
    status = kg.equipment_status(entity)
    root_causes = kg.find_root_causes(entity)

    return jsonify({
        "entity": entity,
        "status": status,
        "neighbors": neighbors,
        "incoming": incoming,
        "outgoing": outgoing,
        "rootCauses": root_causes,
    })


@kg_bp.route("/api/kg/stats")
def stats():
    """Return graph statistics."""
    kg = get_kg()
    return jsonify(kg.graph_statistics())


@kg_bp.route("/api/kg/export/<fmt>")
def export_graph(fmt):
    """Export graph as GraphML or GEXF."""
    kg = get_kg()

    if fmt == "graphml":
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".graphml", prefix="knowledge_graph_")
        tmp.close()
        kg.export_graphml(tmp.name)
        return send_file(tmp.name, as_attachment=True, download_name="knowledge_graph.graphml")
    elif fmt == "gexf":
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".gexf", prefix="knowledge_graph_")
        tmp.close()
        kg.export_gexf(tmp.name)
        return send_file(tmp.name, as_attachment=True, download_name="knowledge_graph.gexf")
    else:
        return jsonify({"error": "Invalid format"}), 400
