import spacy
import logging
import re

from typing import List, Dict
from transformers import pipeline

# ==========================================================
# Logging
# ==========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

# ==========================================================
# Load Models
# ==========================================================

logging.info("Loading spaCy model...")

nlp = spacy.load("en_core_web_sm")

logging.info("spaCy model loaded.")

logging.info("Loading BERT model...")

bert_ner = pipeline(
    "ner",
    model="dslim/bert-base-NER",
    aggregation_strategy="simple"
)

logging.info("BERT model loaded.")

# ==========================================================
# spaCy Entity Extraction
# ==========================================================

def extract_spacy_entities(text: str) -> List[Dict]:

    doc = nlp(text)

    entities = []

    for entity in doc.ents:

        entities.append({

            "text": entity.text,
            "label": entity.label_,
            "source": "spacy"

        })

    return entities


# ==========================================================
# BERT Entity Extraction
# ==========================================================

def extract_bert_entities(text: str) -> List[Dict]:

    results = bert_ner(text)

    entities = []

    for entity in results:

        entities.append({

            "text": entity["word"],
            "label": entity["entity_group"],
            "source": "bert"

        })

    return entities


# ==========================================================
# Regex Patterns
# ==========================================================

REGEX_PATTERNS = {

    "PUMP": r"\bP-\d+\b",

    "VALVE": r"\bVLV-\d+\b",

    "MOTOR": r"\bMTR-\d+\b",

    "PRESSURE_SENSOR": r"\bPT-\d+\b",

    "TEMPERATURE_SENSOR": r"\bTT-\d+\b",

    "SOP": r"\bSOP-\d+\b",

    "DOCUMENT": r"\bDOC-\d+(?:-\d+)?\b"

}


def extract_regex_entities(text: str) -> List[Dict]:

    entities = []

    for label, pattern in REGEX_PATTERNS.items():

        matches = re.findall(pattern, text)

        for match in matches:

            entities.append({

                "text": match,
                "label": label,
                "source": "regex"

            })

    return entities


# ==========================================================
# Label Mapping
# ==========================================================

LABEL_MAPPING = {

    "PER": "PERSON",
    "PERSON": "PERSON",

    "ORG": "ORGANIZATION",

    "LOC": "LOCATION",
    "GPE": "LOCATION",

    "DATE": "DATE",

    "NORP": "GROUP",

    "MISC": "MISC",

    "PUMP": "PUMP",
    "VALVE": "VALVE",
    "MOTOR": "MOTOR",
    "PRESSURE_SENSOR": "PRESSURE_SENSOR",
    "TEMPERATURE_SENSOR": "TEMPERATURE_SENSOR",
    "DOCUMENT": "DOCUMENT",
    "SOP": "SOP"

}

LABEL_PRIORITY = {

    "PUMP": 100,
    "VALVE": 100,
    "MOTOR": 100,
    "PRESSURE_SENSOR": 100,
    "TEMPERATURE_SENSOR": 100,
    "DOCUMENT": 100,
    "SOP": 100,

    "PERSON": 90,
    "ORGANIZATION": 90,
    "LOCATION": 90,
    "DATE": 90,

    "GROUP": 60,

    "MISC": 10

}


# ==========================================================
# Confidence Scores
# ==========================================================

SOURCE_WEIGHTS = {

    "regex": 1.00,
    "spacy": 0.90,
    "bert": 0.85

}


# ==========================================================
# Utility Functions
# ==========================================================

def normalize_label(label: str):

    return LABEL_MAPPING.get(label, label)


def normalize_text(text: str):

    text = text.strip()

    text = re.sub(r"\s*-\s*", "-", text)

    text = re.sub(r"\s+", " ", text)

    return text


# ==========================================================
# Merge Entities
# ==========================================================

def merge_entities(*entity_lists):

    merged = {}

    for entity_list in entity_lists:

        for entity in entity_list:

            text = normalize_text(entity["text"])
            label = normalize_label(entity["label"])
            source = entity["source"]

            key = text.lower()

            if key not in merged:

                merged[key] = {

                    "text": text,
                    "labels": [label],
                    "sources": [source]

                }

            else:

                if label not in merged[key]["labels"]:
                    merged[key]["labels"].append(label)

                if source not in merged[key]["sources"]:
                    merged[key]["sources"].append(source)

    return list(merged.values())


# ==========================================================
# Resolve Best Label
# ==========================================================

def resolve_best_label(labels):

    best = labels[0]

    for label in labels:

        if LABEL_PRIORITY.get(label, 0) > LABEL_PRIORITY.get(best, 0):

            best = label

    return best


# ==========================================================
# Confidence
# ==========================================================

def calculate_confidence(sources):

    score = 0

    for source in sources:

        score += SOURCE_WEIGHTS[source]

    score /= len(sources)

    return round(score, 2)


# ==========================================================
# Intelligent Filtering
# ==========================================================

BLACKLIST = {

    "refer",
    "valve",
    "pump",
    "motor",
    "sensor"

}


def is_valid_entity(entity):

    text = entity["text"]

    if len(text) <= 1:
        return False

    if re.search(r"\b[A-Z]$", text):
        return False

    if entity["label"] == "MISC" and len(text) < 5:
        return False

    if re.fullmatch(r"[-.,]+", text):
        return False

    return True


def filter_entities(entities):

    filtered = []

    for entity in entities:

        text = entity["text"].strip()

        if text.lower() in BLACKLIST:
            continue

        if is_valid_entity(entity):

            filtered.append(entity)

    return filtered


# ==========================================================
# Public Entity Extraction Function
# ==========================================================

def extract_entities(text: str):
    """
    Extract entities using spaCy, BERT and Regex.
    Returns the final cleaned entity list.
    """

    # Extract entities from all sources
    spacy_entities = extract_spacy_entities(text)
    bert_entities = extract_bert_entities(text)
    regex_entities = extract_regex_entities(text)

    # Merge entities
    merged_entities = merge_entities(
        spacy_entities,
        bert_entities,
        regex_entities
    )

    final_entities = []

    # Resolve labels and confidence
    for entity in merged_entities:

        final_entities.append({

            "text": entity["text"],

            "label": resolve_best_label(entity["labels"]),

            "sources": entity["sources"],

            "confidence": calculate_confidence(entity["sources"])

        })

    # Remove noisy entities
    final_entities = filter_entities(final_entities)

    return final_entities


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    text = """
John repaired Pump P-101
and Valve VLV-203
at Microsoft
on 12 June 2026
in Noida.

Pressure sensor PT-201 failed.

Motor MTR-05 overheated.

Refer SOP-001.

Document DOC-2025-01.
"""

    final_entities = extract_entities(text)

    print("\nFinal Entities")
    print("-" * 50)

    for entity in final_entities:
        print(entity)