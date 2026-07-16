"""
knowledge_graph.py

Industrial Knowledge Graph Engine

Builds a graph from extracted relationships.

Author: Team AI
"""

from dataclasses import dataclass
from typing import List, Dict, Optional

import logging
import networkx as nx
import matplotlib.pyplot as plt

# Import relationship extractor

from relationship_extractor import (
    Relationship,
    extract_relationships
)

# -------------------------------------------------------
# Logging
# -------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

logger = logging.getLogger(__name__)


# ============================================================
# KNOWLEDGE GRAPH
# ============================================================

class KnowledgeGraph:

    """
    Directed Industrial Knowledge Graph
    """

    def __init__(self):

        logger.info("Initializing Knowledge Graph...")

        self.graph = nx.DiGraph()

        logger.info("Knowledge Graph initialized.")
        # ============================================================
    # ENTITY FUNCTIONS
    # ============================================================

    def add_entity(self, entity: str):

        """
        Add a node if it doesn't exist.
        """

        entity = entity.strip()

        if not entity:
            return

        if not self.graph.has_node(entity):

            self.graph.add_node(entity)

            logger.info(f"Added entity: {entity}")


    # ============================================================
    # RELATIONSHIP FUNCTIONS
    # ============================================================

    def add_relationship(self, relationship: Relationship):

        """
        Add relationship as graph edge.
        """

        self.add_entity(relationship.subject)
        self.add_entity(relationship.object)

        self.graph.add_edge(

            relationship.subject,

            relationship.object,

            relation=relationship.relation,

            confidence=relationship.confidence,

            relation_type=relationship.relation_type

        )

        logger.info(

            f"{relationship.subject} --"

            f"{relationship.relation}"

            f"--> {relationship.object}"

        )


    # ============================================================
    # BUILD GRAPH
    # ============================================================

    def build_from_relationships(

        self,

        relationships: List[Relationship]

    ):

        """
        Build graph from extracted relationships.
        """

        logger.info("Building Knowledge Graph...")

        for relationship in relationships:

            self.add_relationship(relationship)

        logger.info(

            f"Processed {len(relationships)} relationships."

        )

        logger.info(

            f"Graph contains "

            f"{self.graph.number_of_nodes()} nodes "

            f"and "

            f"{self.graph.number_of_edges()} edges."

        )
        # ============================================================
    # GRAPH QUERY FUNCTIONS
    # ============================================================

    def entity_exists(self, entity: str) -> bool:
        """
        Check whether an entity exists.
        """
        return self.graph.has_node(entity)


    def get_neighbors(self, entity: str):
        """
        Return all connected neighbors.
        """
        if not self.entity_exists(entity):
            return []

        neighbors = set()

        neighbors.update(self.graph.successors(entity))
        neighbors.update(self.graph.predecessors(entity))

        return sorted(list(neighbors))


    def get_children(self, entity: str):
        """
        Return outgoing neighbors.
        """
        if not self.entity_exists(entity):
            return []

        return sorted(list(self.graph.successors(entity)))


    def get_parents(self, entity: str):
        """
        Return incoming neighbors.
        """
        if not self.entity_exists(entity):
            return []

        return sorted(list(self.graph.predecessors(entity)))
        # ============================================================
    # RELATIONSHIP RETRIEVAL
    # ============================================================

    def get_outgoing_relationships(self, entity: str):
        """
        Return all outgoing relationships.
        """

        if not self.entity_exists(entity):
            return []

        relationships = []

        for _, target, data in self.graph.out_edges(entity, data=True):

            relationships.append({

                "subject": entity,

                "relation": data.get("relation"),

                "object": target,

                "confidence": data.get("confidence"),

                "relation_type": data.get("relation_type")

            })

        return relationships


    def get_incoming_relationships(self, entity: str):
        """
        Return all incoming relationships.
        """

        if not self.entity_exists(entity):
            return []

        relationships = []

        for source, _, data in self.graph.in_edges(entity, data=True):

            relationships.append({

                "subject": source,

                "relation": data.get("relation"),

                "object": entity,

                "confidence": data.get("confidence"),

                "relation_type": data.get("relation_type")

            })

        return relationships


    # ============================================================
    # GRAPH STATISTICS
    # ============================================================

    def node_count(self):
        return self.graph.number_of_nodes()


    def edge_count(self):
        return self.graph.number_of_edges()


    def graph_statistics(self):
        """
        Return graph summary.
        """

        return {

            "nodes": self.node_count(),

            "edges": self.edge_count(),

            "density": nx.density(self.graph),

            "connected_components": nx.number_weakly_connected_components(
                self.graph
            )

        }
        # ============================================================
    # PATH SEARCH
    # ============================================================

    def shortest_path(self, source: str, target: str):
        """
        Return shortest path between two entities.
        """

        if not self.entity_exists(source):
            return None

        if not self.entity_exists(target):
            return None

        try:

            return nx.shortest_path(

                self.graph,

                source,

                target

            )

        except nx.NetworkXNoPath:

            return None


    # ============================================================
    # CONNECTION EXPLANATION
    # ============================================================

    def explain_connection(self, source: str, target: str):
        """
        Explain how two entities are connected.
        """

        path = self.shortest_path(source, target)

        if path is None:

            return []

        explanation = []

        for i in range(len(path) - 1):

            edge = self.graph.get_edge_data(

                path[i],

                path[i + 1]

            )

            explanation.append({

                "subject": path[i],

                "relation": edge["relation"],

                "object": path[i + 1],

                "confidence": edge["confidence"]

            })

        return explanation
        # ============================================================
    # FAILURE CHAIN
    # ============================================================

    def failure_chain(self, equipment: str):
        """
        Return all incoming relationships
        that may explain a failure.
        """

        if not self.entity_exists(equipment):

            return []

        chain = []

        for parent in self.graph.predecessors(equipment):

            edge = self.graph.get_edge_data(

                parent,

                equipment

            )

            chain.append({

                "cause": parent,

                "relation": edge["relation"],

                "confidence": edge["confidence"]

            })

        return chain


    # ============================================================
    # ROOT CAUSE DETECTION
    # ============================================================

    def find_root_causes(self, equipment: str):
        """
        Return possible root causes.
        """

        causes = []

        for rel in self.failure_chain(equipment):

            if rel["relation"] in {

                "caused_failure",

                "failed",

                "failure",

                "overheated",

                "damaged",

                "leaking",

                "disconnected"

            }:

                causes.append(rel["cause"])

        return causes


    # ============================================================
    # EQUIPMENT STATUS
    # ============================================================

    def equipment_status(self, equipment: str):
        """
        Return current equipment status.
        """

        if not self.entity_exists(equipment):

            return None

        for _, obj, data in self.graph.out_edges(

            equipment,

            data=True

        ):

            if data["relation"] == "has_status":

                return obj

        return "unknown"


    # ============================================================
    # FAILURE REPORT
    # ============================================================

    def failure_report(self, equipment: str):
        """
        Complete report for one equipment.
        """

        return {

            "equipment": equipment,

            "status": self.equipment_status(

                equipment

            ),

            "root_causes": self.find_root_causes(

                equipment

            ),

            "incoming_relationships":

                self.get_incoming_relationships(

                    equipment

                ),

            "outgoing_relationships":

                self.get_outgoing_relationships(

                    equipment

                )

        }
        # ============================================================
    # EXPORT GRAPH
    # ============================================================

    def export_graphml(self, filename="knowledge_graph.graphml"):
        """
        Export graph to GraphML.
        """

        nx.write_graphml(self.graph, filename)

        logger.info(f"Graph exported to {filename}")


    def export_gexf(self, filename="knowledge_graph.gexf"):
        """
        Export graph for Gephi visualization.
        """

        nx.write_gexf(self.graph, filename)

        logger.info(f"Graph exported to {filename}")


    # ============================================================
    # IMPORT GRAPH
    # ============================================================

    def import_graphml(self, filename):

        self.graph = nx.read_graphml(filename)

        logger.info("Graph imported.")


    # ============================================================
    # VISUALIZATION
    # ============================================================

    def visualize(self):

        plt.figure(figsize=(12,8))

        pos = nx.spring_layout(
            self.graph,
            seed=42
        )

        nx.draw_networkx_nodes(
            self.graph,
            pos,
            node_size=1800
        )

        nx.draw_networkx_labels(
            self.graph,
            pos,
            font_size=8
        )

        nx.draw_networkx_edges(
            self.graph,
            pos,
            arrows=True,
            arrowsize=20
        )

        edge_labels = {

            (u,v):d["relation"]

            for u,v,d in self.graph.edges(data=True)

        }

        nx.draw_networkx_edge_labels(

            self.graph,

            pos,

            edge_labels=edge_labels,

            font_size=7

        )

        plt.axis("off")

        plt.show()

    def build_prediction_graph(self,prediction,reasons,recommendations):

        """
        Build graph from AI prediction results.
        """

        self.graph.clear()

        # --------------------------
        # Engine
        # --------------------------

        self.graph.add_node(
            "Engine",
            type="asset"
        )

        # --------------------------
        # Prediction
        # --------------------------

        status = prediction["Failure Status"]

        self.graph.add_node(
            status,
            type="prediction"
        )

        self.graph.add_edge(
            "Engine",
            status,
            relation="predicted_as"
        )

        # --------------------------
        # RUL
        # --------------------------

        rul = prediction["Remaining Useful Life"]

        rul_node = f"RUL={rul}"

        self.graph.add_node(
            rul_node,
            type="RUL"
        )

        self.graph.add_edge(
            "Engine",
            rul_node,
            relation="estimated_life"
        )

        # --------------------------
        # Important Sensors
        # --------------------------

        for reason in reasons:

            sensor = reason["feature"]

            self.graph.add_node(
                sensor,
                type="sensor"
            )

            self.graph.add_edge(
                "Engine",
                sensor,
                relation="monitored_by"
            )

            self.graph.add_edge(
                sensor,
                status,
                relation="influences"
            )

        # --------------------------
        # Recommendations
        # --------------------------

        for action in recommendations["actions"]:

            self.graph.add_node(
                action,
                type="recommendation"
            )

            self.graph.add_edge(
                status,
                action,
                relation="requires"
            )

        logger.info("Prediction Knowledge Graph built.")
    # ============================================================
# DEMO
# ============================================================

if __name__ == "__main__":

    prediction = {
        "Remaining Useful Life": 110,
        "Failure Status": "Warning"
    }

    reasons = [

        {"feature": "Fuel Flow", "importance": 0.46},

        {"feature": "Vibration", "importance": 0.08},

        {"feature": "Oil Temperature", "importance": 0.05}

    ]

    recommendations = {

        "actions": [

            "Inspect fuel injectors",

            "Check bearings",

            "Inspect lubrication system"

        ]

    }

    kg = KnowledgeGraph()

    kg.build_prediction_graph(
        prediction,
        reasons,
        recommendations
    )

    print()

    print("Nodes")

    print(kg.graph.nodes(data=True))

    print()

    print("Edges")

    for edge in kg.graph.edges(data=True):
        print(edge)

    kg.visualize()