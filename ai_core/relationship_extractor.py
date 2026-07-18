import logging
import spacy
import re

from dataclasses import dataclass
from typing import List, Optional

from .entity_extractor import extract_entities

# ==========================================================
# Logging
# ==========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

logging.info("Loading spaCy model...")

nlp = spacy.load("en_core_web_sm")

logging.info("spaCy model loaded.")

# ==========================================================
# Relationship Object
# ==========================================================

@dataclass
class Relationship:

    subject: str
    relation: str
    object: str

    confidence: float
    relation_type: str


# ==========================================================
# Industrial Verbs
# ==========================================================

INDUSTRIAL_VERBS = {

    "repair",
    "replace",
    "inspect",
    "monitor",
    "measure",
    "connect",
    "disconnect",
    "detect",
    "report",
    "trigger",
    "install",
    "remove",
    "shutdown",
    "operate",
    "control",
    "start",
    "stop",
    "drive",
    "activate",
    "deactivate",
    "check",
    "test",
    "service",
    "maintain",
    "follow",
    "contain",
    "record",
    "log",
    "measure",
    "inspect",
    "open",
    "close"

}

# ==========================================================
# Status Words
# ==========================================================

STATUS_WORDS = {

    "failed",
    "failure",
    "overheated",
    "damaged",
    "running",
    "stopped",
    "active",
    "inactive",
    "leaking",
    "corroded",
    "offline",
    "online",
    "replaced",
    "installed",
    "removed",
    "shutdown",
    "vibrating",
    "worn"

}

# ==========================================================
# Ignore These Entity Types
# ==========================================================

INVALID_OBJECT_TYPES = {

    "DATE",
    "TIME",
    "PERCENT",
    "MONEY",
    "QUANTITY",
    "ORDINAL",
    "CARDINAL"

}

# ==========================================================
# Ignore Partial Tokens
# ==========================================================

INVALID_ENTITY_TEXT = {

    "te",
    "sen",
    "pi",
    "pres",
    "press",
    "temp",
    "mot",
    "val",
    "pump",
    "sensor"

}

# ==========================================================
# Confidence Scores
# ==========================================================

CONFIDENCE = {

    "dependency":0.95,

    "entity_pair":0.90,

    "status":0.92,

    "heuristic":0.85

}

VALID_STATUS_TYPES = {
    "PUMP",
    "MOTOR",
    "VALVE",
    "PIPE",
    "PIPELINE",
    "ENGINE",
    "PRESSURE_SENSOR",
    "TEMPERATURE_SENSOR",
    "FLOW_SENSOR",
    "LEVEL_SENSOR",
    "SENSOR",
    "BEARING"
}

# ==========================================================
# Sentence Splitter
# ==========================================================

def split_sentences(text:str):

    doc = nlp(text)

    return [

        sent.text.strip()

        for sent in doc.sents

    ]


# ==========================================================
# Find Matching Entity
# ==========================================================

def match_entity(token, entities):
    """
    Match a spaCy token to the full extracted entity.

    Always return the complete entity name instead of
    partial words.
    """

    token_text = token.text.lower()

    for entity in entities:

        entity_text = entity["text"].lower()

        # Ignore invalid entities
        if entity_text in INVALID_ENTITY_TEXT:
            continue

        if len(entity_text) < 3:
            continue

        words = entity_text.split()

        # Exact full entity match
        if token_text == entity_text:
            return entity["text"]

        # Match equipment ID
        # Example:
        # token = "P-101"
        # entity = "Pump P-101"
        if token_text == words[-1]:
            return entity["text"]

        # Match last two words
        # Example:
        # token subtree = "Pressure Sensor PT-201"
        if len(words) >= 2:

            last_two = " ".join(words[-2:]).lower()

            if token_text == last_two:
                return entity["text"]

    return None

# ==========================================================
# Find Subject
# ==========================================================

def get_subject(verb, entities):

    """
    Find dependency subject.
    """

    for child in verb.children:

        if child.dep_ in (

            "nsubj",
            "nsubjpass"

        ):

            entity = match_entity(

                child,

                entities

            )

            if entity:

                return entity

    return None


# ==========================================================
# Find Object
# ==========================================================

def get_object(verb, entities):

    """
    Find dependency object.
    """

    for child in verb.children:

        if child.dep_ in (

            "dobj",
            "pobj",
            "attr",
            "oprd"

        ):

            entity = match_entity(

                child,

                entities

            )

            if entity:

                return entity

            return child.text

    return None


# ==========================================================
# Check Valid Verb
# ==========================================================

def is_valid_relation(token):

    if token.pos_ != "VERB":

        return False

    if token.lemma_.lower() not in INDUSTRIAL_VERBS:

        return False

    return True


# ==========================================================
# Status Detection
# ==========================================================

def detect_status(entity_text, sentence):

    sentence_lower = sentence.lower()

    for status in STATUS_WORDS:

        if status in sentence_lower:

            if entity_text.lower() in sentence_lower:

                return status

    return None

# ==========================================================
# Dependency Relationship Extraction
# ==========================================================

def dependency_relationships(sentence, entities):

    """
    Extract relationships using spaCy dependency parsing.
    """

    doc = nlp(sentence)

    relationships = []

    for token in doc:

        if not is_valid_relation(token):
            continue

        subject = get_subject(token, entities)
        obj = get_object(token, entities)

        if subject is None:
            continue

        if obj is None:
            continue

        relationships.append(

            Relationship(

                subject=subject,

                relation=token.lemma_.lower(),

                object=obj,

                confidence=CONFIDENCE["dependency"],

                relation_type="dependency"

            )

        )

    return relationships


# ==========================================================
# Rule-Based Relationship Extraction
# ==========================================================

def build_rule_relationships(sentence, entities):

    """
    Extract relationships using industrial rules.
    """

    sentence_lower = sentence.lower()

    relationships = []

    entity_names = [

        e["text"]

        for e in entities

        if e["label"] not in INVALID_OBJECT_TYPES

    ]

    # ------------------------------------------------------
    # connected_to
    # ------------------------------------------------------

    if "connected" in sentence_lower:

        if len(entity_names) >= 2:

            relationships.append(

                Relationship(

                    subject=entity_names[0],

                    relation="connected_to",

                    object=entity_names[1],

                    confidence=0.95,

                    relation_type="rule"

                )

            )

    # ------------------------------------------------------
    # repaired
    # ------------------------------------------------------

    if "repair" in sentence_lower or "repaired" in sentence_lower:

        doc = nlp(sentence)

        subject = None
        obj = None

        for token in doc:

            if token.lemma_ != "repair":
                continue

            subject = get_subject(token, entities)

            obj = get_object(token, entities)

        if subject and obj:

            relationships.append(

                Relationship(

                    subject,

                    "repaired",

                    obj,

                    0.95,

                    "rule"

                )

            )

    # ------------------------------------------------------
    # replaced
    # ------------------------------------------------------

    if "replace" in sentence_lower or "replaced" in sentence_lower:

        doc = nlp(sentence)

        subject = None
        obj = None

        for token in doc:

            if token.lemma_ != "replace":
                continue

            subject = get_subject(token, entities)

            obj = get_object(token, entities)

        if subject and obj:

            relationships.append(

                Relationship(

                    subject,

                    "replaced",

                    obj,

                    0.95,

                    "rule"

                )

            )

    # ------------------------------------------------------
    # detected
    # ------------------------------------------------------

    if "detect" in sentence_lower or "detected" in sentence_lower:

        doc = nlp(sentence)

        for token in doc:

            if token.lemma_ != "detect":

                continue

            subject = get_subject(token, entities)

            obj = get_object(token, entities)

            if subject and obj:

                relationships.append(

                    Relationship(

                        subject,

                        "detected",

                        obj,

                        0.95,

                        "rule"

                    )

                )

    return relationships

# ==========================================================
# Status Relationship Extraction
# ==========================================================

def build_status_relationships(sentence, entities):

    """
    Build has_status relationships.
    """

    relationships = []

    for entity in entities:

        status = detect_status(

            entity["text"],

            sentence

        )

        if status is None:

            continue

        relationships.append(

            Relationship(

                subject=entity["text"],

                relation="has_status",

                object=status,

                confidence=CONFIDENCE["status"],

                relation_type="status"

            )

        )

    return relationships


# ==========================================================
# Duplicate Removal
# ==========================================================

def remove_duplicates(relationships):

    unique = {}

    cleaned = []

    for rel in relationships:

        key = (

            rel.subject.lower(),

            rel.relation.lower(),

            rel.object.lower()

        )

        if key in unique:

            continue

        unique[key] = True

        cleaned.append(rel)

    return cleaned


# ==========================================================
# Validate Relationships
# ==========================================================

def validate_relationships(relationships):

    """
    Remove noisy relationships.
    """

    cleaned = []

    for rel in relationships:

        if len(rel.subject) < 3:
            continue

        if len(rel.object) < 3:
            continue

        if rel.subject.lower() in INVALID_ENTITY_TEXT:
            continue

        if rel.object.lower() in INVALID_ENTITY_TEXT:
            continue

        if rel.subject == rel.object:
            continue

        cleaned.append(rel)

    return cleaned

# ==========================================================
# Main Relationship Extraction Pipeline
# ==========================================================

def extract_relationships(text: str):
    """
    Complete relationship extraction pipeline.
    """

    # ---------------------------------------------------------
    # Extract entities
    # ---------------------------------------------------------

    entities = extract_entities(text)

    # ---------------------------------------------------------
    # Remove noisy / partial entities
    # ---------------------------------------------------------

    filtered_entities = []
    seen = set()

    for entity in entities:

        name = entity["text"].strip()

        if (
            len(name) < 3
            or name.endswith(" M")
            or name.endswith(" MTR")
            or name.endswith(" Sensor")
            or name == "Alarm"
        ):
            continue

        if name not in seen:
            seen.add(name)
            filtered_entities.append(entity)

    entities = filtered_entities

    # ---------------------------------------------------------
    # Split text into sentences
    # ---------------------------------------------------------

    sentences = split_sentences(text)

    all_relationships = []

    # ---------------------------------------------------------
    # Process each sentence
    # ---------------------------------------------------------

    for sentence in sentences:

        sentence_lower = sentence.lower()

        sentence_entities = []

        for entity in entities:

            if (
                entity["text"].lower() in sentence_lower
                and entity["label"] not in INVALID_OBJECT_TYPES
            ):
                sentence_entities.append(entity)

        if not sentence_entities:
            continue

        # -----------------------------------------------------
        # Dependency relationships
        # -----------------------------------------------------

        dependency_rels = dependency_relationships(
            sentence,
            sentence_entities
        )

        all_relationships.extend(dependency_rels)

        # -----------------------------------------------------
        # Rule relationships
        # -----------------------------------------------------

        rule_rels = build_rule_relationships(
            sentence,
            sentence_entities
        )

        all_relationships.extend(rule_rels)

        # -----------------------------------------------------
        # Status relationships
        # -----------------------------------------------------

        valid_status_entities = []

        for entity in sentence_entities:

            if entity["label"] in VALID_STATUS_TYPES:
                valid_status_entities.append(entity)

        status_rels = build_status_relationships(
            sentence,
            valid_status_entities
        )

        all_relationships.extend(status_rels)

    # ---------------------------------------------------------
    # Remove duplicate relationships
    # ---------------------------------------------------------

    all_relationships = remove_duplicates(
        all_relationships
    )

    # ---------------------------------------------------------
    # Validate relationships
    # ---------------------------------------------------------

    all_relationships = validate_relationships(
        all_relationships
    )

    return all_relationships

def print_relationships(relationships):

    """
    Print extracted relationships.
    """

    print("\nRelationships")

    print("-" * 70)

    if len(relationships) == 0:

        print("No relationships found.")

        return

    for i, rel in enumerate(

        relationships,

        start=1

    ):

        print(f"\nRelationship {i}")

        print(rel.subject)

        print("   │")

        print(

            f"   ├── {rel.relation}"

            f" ({rel.relation_type}, "

            f"{rel.confidence})"

        )

        print("   │")

        print(rel.object)

    print()

    print("=" * 70)

    print(

        f"Total Relationships : "

        f"{len(relationships)}"

    )

    print("=" * 70)

# ==========================================================
# Interactive Testing
# ==========================================================

def main():

    print("\nIndustrial Relationship Extractor")

    print("=" * 70)

    while True:

        text = input(

            "\nEnter text (or 'exit'): "

        ).strip()

        if text.lower() == "exit":

            print("\nRelationship Extractor Closed.")

            break

        if text == "":

            print("Please enter some text.")

            continue

        relationships = extract_relationships(text)

        print_relationships(relationships)


# ==========================================================
# Run
# ==========================================================

if __name__ == "__main__":

    main()

# ==========================================================
# Relationship to Dictionary
# (Useful later for Knowledge Graph / Neo4j)
# ==========================================================

def relationship_to_dict(rel):

    return {

        "subject": rel.subject,

        "relation": rel.relation,

        "object": rel.object,

        "confidence": rel.confidence,

        "relation_type": rel.relation_type

    }


# ==========================================================
# Convert All Relationships
# ==========================================================

def relationships_to_dict(relationships):

    return [

        relationship_to_dict(rel)

        for rel in relationships

    ]


# ==========================================================
# Sort Relationships
# ==========================================================

def sort_relationships(relationships):

    """
    Highest confidence first.
    """

    return sorted(

        relationships,

        key=lambda x: x.confidence,

        reverse=True

    )


# ==========================================================
# Utility
# ==========================================================

def count_relationships(relationships):

    print(

        f"\nExtracted "

        f"{len(relationships)} relationships."

    )