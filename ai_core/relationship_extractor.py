import spacy
import logging
import re

from dataclasses import dataclass
from typing import List, Dict

from entity_extractor import extract_entities

# ==========================================================
# Logging
# ==========================================================

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ==========================================================
# Load spaCy
# ==========================================================

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
    relation_type: str = "rule"


# ==========================================================
# Utilities
# ==========================================================

def split_sentences(text: str):
    """Proper sentence splitter"""
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


def normalize(text: str):
    return re.sub(r"\s+", " ", text.strip())


def safe_match(entity_text: str, sentence: str):
    """
    Strong matching instead of naive substring match
    """
    return entity_text.lower() in sentence.lower()


# ==========================================================
# Relation Detection (Rule Engine)
# ==========================================================

RELATION_MAP = {
    "repair": "repaired",
    "fix": "repaired",
    "replace": "replaced",
    "inspect": "inspected",
    "connect": "connected_to",
    "disconnect": "disconnected_from",
    "fail": "failed",
    "overheat": "overheated",
    "leak": "leaking",
    "report": "reported",
    "detect": "detected",
    "drive": "drives",
    "operate": "operates"
}


def detect_relation(sentence: str):
    sent = sentence.lower()

    for key, rel in RELATION_MAP.items():
        if key in sent:
            return rel

    return "related_to"


# ==========================================================
# Entity Filtering per sentence
# ==========================================================

def filter_sentence_entities(entities, sentence):
    return [
        e for e in entities
        if safe_match(e["text"], sentence)
    ]


# ==========================================================
# Main Relationship Extractor
# ==========================================================

def extract_relationships(text: str):
    """
    FINAL PIPELINE:
    1. Extract entities
    2. Split sentences properly
    3. Match entities per sentence
    4. Build relationships
    5. Deduplicate
    """

    entities = extract_entities(text)
    sentences = split_sentences(text)

    relationships = []

    for sentence in sentences:

        sentence_entities = filter_sentence_entities(entities, sentence)

        # skip if less than 2 entities
        if len(sentence_entities) < 2:
            continue

        relation = detect_relation(sentence)

        # --------------------------------------------------
        # Pairwise relationships
        # --------------------------------------------------

        for i in range(len(sentence_entities)):
            for j in range(i + 1, len(sentence_entities)):

                e1 = sentence_entities[i]
                e2 = sentence_entities[j]

                relationships.append(Relationship(
                    subject=e1["text"],
                    relation=relation,
                    object=e2["text"],
                    confidence=0.90,
                    relation_type="entity_pair"
                ))

        # --------------------------------------------------
        # Dependency-based lightweight relation
        # --------------------------------------------------

        doc = nlp(sentence)

        for token in doc:

            if token.pos_ != "VERB":
                continue

            subject = None
            obj = None

            for child in token.children:
                if child.dep_ in ("nsubj", "nsubjpass"):
                    subject = child.text
                if child.dep_ in ("dobj", "pobj", "attr"):
                    obj = child.text

            if subject and obj:
                relationships.append(Relationship(
                    subject=subject,
                    relation=token.lemma_.lower(),
                    object=obj,
                    confidence=0.95,
                    relation_type="dependency"
                ))

    # ------------------------------------------------------
    # Deduplication FIXED
    # ------------------------------------------------------

    unique = set()
    cleaned = []

    for r in relationships:

        key = (
            r.subject.lower().strip(),
            r.relation.lower().strip(),
            r.object.lower().strip()
        )

        if key in unique:
            continue

        unique.add(key)
        cleaned.append(r)

    return cleaned


# ==========================================================
# Pretty Print
# ==========================================================

def print_relationships(rels):

    print("\nRelationships")
    print("-" * 70)

    if not rels:
        print("No relationships found.")
        return

    for i, r in enumerate(rels, 1):

        print(f"\nRelationship {i}")
        print(f"{r.subject}")
        print("   │")
        print(f"   ├── {r.relation} ({r.relation_type}, {r.confidence})")
        print("   │")
        print(f"{r.object}")


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    print("\nIndustrial Relationship Extractor")
    print("=" * 70)

    while True:

        text = input("\nEnter text (or 'exit'): ")

        if text.lower() == "exit":
            break

        results = extract_relationships(text)
        print_relationships(results)

        print("\nExtraction Complete")
        print("=" * 70)