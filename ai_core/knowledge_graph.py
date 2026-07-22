from dataclasses import dataclass
from typing import List, Dict, Optional

import logging
import networkx as nx
import matplotlib.pyplot as plt

# Import relationship extractor

from .relationship_extractor import (
    Relationship,
    extract_relationships
)
from .entity_extractor import extract_entities

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

        self.entity_graph = nx.DiGraph()
        self.prediction_graph = nx.DiGraph()

        logger.info("Knowledge Graph initialized.")
        # ============================================================
    # ENTITY FUNCTIONS
    # ============================================================

    def add_entity(self, entity: str):

        """
        Add a node if it doesn't exist.
        """

        entity = self.normalize_entity(entity)

        if not entity:
            return

        if not self.entity_graph.has_node(entity):

            self.entity_graph.add_node(entity)

            logger.info(f"Added entity: {entity}")


    # ============================================================
    # RELATIONSHIP FUNCTIONS
    # ============================================================
    def build_from_text(self, text):

        # Add entities
        entities = extract_entities(text)

        for entity in entities:
            self.add_entity(entity["text"])

        # Add relationships
        relationships = extract_relationships(text)

        self.build_from_relationships(relationships)

    def add_document(self, text):
        """
        Add one uploaded document into the graph.
        """

        logger.info("Adding uploaded document to Knowledge Graph...")

        self.build_from_text(text)

        self.export_graphml("knowledge_graph.graphml")

        logger.info("Knowledge Graph updated.")

    def normalize_entity(self, entity: str):
        """
        Normalize entity names for consistent searching.
        """

        entity = entity.strip().lower()

        entity = " ".join(entity.split())

        return entity

    def add_relationship(self, relationship: Relationship):

        """
        Add relationship as graph edge.
        """

        subject = self.normalize_entity(relationship.subject)
        object_ = self.normalize_entity(relationship.object)

        self.add_entity(subject)
        self.add_entity(object_)

        self.entity_graph.add_edge(
            subject,
            object_,
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

            f"{self.entity_graph.number_of_nodes()} nodes "

            f"and "

            f"{self.entity_graph.number_of_edges()} edges."

        )
        # ============================================================
    # GRAPH QUERY FUNCTIONS
    # ============================================================

    def entity_exists(self, entity):

        entity = self.normalize_entity(entity)

        if self.entity_graph.has_node(entity):
            return True

        for node in self.entity_graph.nodes():

            if entity in node:
                return True

        return False
    
    def find_entity(self, query):

        query = self.normalize_entity(query)

        if self.entity_graph.has_node(query):
            return query

        for node in self.entity_graph.nodes():

            if query in node:
                return node

        return None


    def get_neighbors(self, entity: str):
        """
        Return all connected neighbors.
        """
        if not self.entity_exists(entity):
            return []

        neighbors = set()

        neighbors.update(self.entity_graph.successors(entity))
        neighbors.update(self.entity_graph.predecessors(entity))

        return sorted(list(neighbors))


    def get_children(self, entity: str):
        """
        Return outgoing neighbors.
        """
        if not self.entity_exists(entity):
            return []

        return sorted(list(self.entity_graph.successors(entity)))


    def get_parents(self, entity: str):
        """
        Return incoming neighbors.
        """
        if not self.entity_exists(entity):
            return []

        return sorted(list(self.entity_graph.predecessors(entity)))
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

        for _, target, data in self.entity_graph.out_edges(entity, data=True):

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

        for source, _, data in self.entity_graph.in_edges(entity, data=True):

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
        return self.entity_graph.number_of_nodes()


    def edge_count(self):
        return self.entity_graph.number_of_edges()


    def graph_statistics(self):
        """
        Return graph summary.
        """

        return {

            "nodes": self.node_count(),

            "edges": self.edge_count(),

            "density": nx.density(self.entity_graph),

            "connected_components": nx.number_weakly_connected_components(
                self.entity_graph
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

                self.entity_graph,

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

            edge = self.entity_graph.get_edge_data(

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

        for parent in self.entity_graph.predecessors(equipment):

            edge = self.entity_graph.get_edge_data(

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

        for _, obj, data in self.entity_graph.out_edges(

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

        equipment = self.find_entity(equipment)

        if equipment is None:
            return None

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

        nx.write_graphml(self.entity_graph, filename)

        logger.info(f"Graph exported to {filename}")


    def export_gexf(self, filename="knowledge_graph.gexf"):
        """
        Export graph for Gephi visualization.
        """

        nx.write_gexf(self.entity_graph, filename)

        logger.info(f"Graph exported to {filename}")


    # ============================================================
    # IMPORT GRAPH
    # ============================================================

    def import_graphml(self, filename):

        self.entity_graph = nx.read_graphml(filename)

        logger.info("Graph imported.")


    # ============================================================
    # VISUALIZATION
    # ============================================================

    def visualize(self, graph_type="entity"):

        graph = (
            self.entity_graph
            if graph_type == "entity"
            else self.prediction_graph
        )

        plt.figure(figsize=(12,8))

        pos = nx.spring_layout(
            graph,
            seed=42
        )

        nx.draw_networkx_nodes(
            graph,
            pos,
            node_size=1800
        )

        nx.draw_networkx_labels(
            graph,
            pos,
            font_size=8
        )

        nx.draw_networkx_edges(
            graph,
            pos,
            arrows=True,
            arrowsize=20
        )

        edge_labels = {

            (u,v):d["relation"]

            for u,v,d in graph.edges(data=True)

        }

        nx.draw_networkx_edge_labels(

            graph,

            pos,

            edge_labels=edge_labels,

            font_size=7

        )

        plt.axis("off")

        plt.show()

    def build_prediction_graph(
    self,
    prediction,
    explanations,
    recommendations
):
        """
        Build an explainable prediction graph.
        """
        graph = self.prediction_graph

        graph.clear()


        status = prediction["Failure Status"]
        rul = prediction["Remaining Useful Life"]

        # ====================================================
        # Asset
        # ====================================================

        graph.add_node(
            "Engine",
            type="Asset"
        )

        # ====================================================
        # Prediction
        # ====================================================

        prediction_node = f"Failure={status}"

        graph.add_node(
            prediction_node,
            type="Prediction"
        )

        graph.add_edge(
            "Engine",
            prediction_node,
            relation="Predicted As"
        )

        # ====================================================
        # Remaining Useful Life
        # ====================================================

        rul_node = f"RUL={round(rul)} Cycles"

        graph.add_node(
            rul_node,
            type="RUL"
        )

        graph.add_edge(
            "Engine",
            rul_node,
            relation="Estimated Remaining Life"
        )

        # ====================================================
        # Risk Level
        # ====================================================

        if status == "Critical":
            risk = "High Risk"

        elif status == "Warning":
            risk = "Medium Risk"

        else:
            risk = "Low Risk"

        graph.add_node(
            risk,
            type="Risk"
        )

        graph.add_edge(
            prediction_node,
            risk,
            relation="Severity"
        )

        # ====================================================
        # Important Sensors
        # ====================================================

        for reason in explanations:

            sensor = reason["feature"]

            importance = reason["importance"]

            graph.add_node(
                sensor,
                type="Sensor",
                importance=importance
            )

            graph.add_edge(
                "Engine",
                sensor,
                relation="Monitored By"
            )

            graph.add_edge(
                sensor,
                prediction_node,
                relation= "Contributes To",
                weight = importance
            )

        # ====================================================
        # Maintenance Recommendations
        # ====================================================

        for action in recommendations["actions"]:

            graph.add_node(
                action,
                type="Recommendation"
            )

            graph.add_edge(
                prediction_node,
                action,
                relation="Recommended Action"
            )

        # ====================================================
        # Maintenance Priority
        # ====================================================

        priority = recommendations["priority"]

        graph.add_node(
            priority,
            type="Priority"
        )

        graph.add_edge(
            prediction_node,
            priority,
            relation="Maintenance Priority"
        )

        logger.info("Prediction Knowledge Graph built.")

if __name__ == "__main__":

    from predict import Predictor
    from explain_prediction import PredictionExplainer
    from recommendation import RecommendationEngine

    # -----------------------------
    # Load AI modules
    # -----------------------------

    predictor = Predictor()
    explainer = PredictionExplainer()
    recommender = RecommendationEngine()

    # -----------------------------
    # Example sensor input
    # (Later Flask API will send this)
    # -----------------------------

    sample = {}

    sample["setting1"] = 0.0
    sample["setting2"] = 0.0
    sample["setting3"] = 0.0

    for i in range(1, 22):
        sample[f"sensor_{i}"] = 0.5

    # -----------------------------
    # Make prediction
    # -----------------------------

    prediction = predictor.predict(sample)

    print("\nPrediction")
    print(prediction)

    # -----------------------------
    # Explain prediction
    # -----------------------------

    reasons = explainer.top_reasons()

    print("\nTop Reasons")

    for reason in reasons:
        print(reason)

    # -----------------------------
    # Generate recommendations
    # -----------------------------

    recommendations = recommender.recommend(prediction, reasons)


    print("\nRecommendations")
    print(recommendations)

    # -----------------------------
    # Build Knowledge Graph
    # -----------------------------

    kg = KnowledgeGraph()

    kg.build_prediction_graph(
        prediction,
        reasons,
        recommendations
    )

    print("\nNodes")
    print(kg.prediction_graph.nodes(data=True))

    print("\nEdges")

    for edge in kg.prediction_graph.edges(data=True):
        print(edge)

    kg.visualize("prediction")