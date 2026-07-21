import logging
import re

from typing import List, Dict

from .model_manager import models

# ==========================================================
# Logging Configuration
# ==========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

# ==========================================================
# Load Models from ModelManager
# ==========================================================

nlp = models.spacy

bert_ner = models.bert_pipeline

# ==========================================================
# spaCy Entity Extraction
# ==========================================================

def extract_spacy_entities(text: str) -> List[Dict]:
    """
    Extract entities using spaCy.
    """

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
    """
    Extract entities using BERT.
    """

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
# Industrial Regex Patterns
# ==========================================================

REGEX_PATTERNS = {

    # Equipment IDs

    "PUMP": r"\bP-\d+\b",

    "VALVE": r"\bVLV-\d+\b",

    "MOTOR": r"\bMTR-\d+\b",

    "PRESSURE_SENSOR": r"\bPT-\d+\b",

    "TEMPERATURE_SENSOR": r"\bTT-\d+\b",

    "FLOW_SENSOR": r"\bFT-\d+\b",

    "LEVEL_SENSOR": r"\bLT-\d+\b",

    "COMPRESSOR": r"\bCOMP-\d+\b",

    "ENGINE": r"\bENGINE-\d+\b",

    "TURBINE": r"\bTURB-\d+\b",

    "SOP": r"\bSOP-\d+\b",

    "DOCUMENT": r"\bDOC-\d+(?:-\d+)?\b",


    # Industrial Documents

    "DOCUMENT": r"\bDOC-\d+(?:-\d+)?\b",

    "SOP": r"\bSOP-\d+\b",

    "MANUAL": r"\bMAN-\d+\b",

    # NASA Files

    "NASA_DATASET": r"\bFD00[1-4]\b",

    # Failure Codes

    "FAULT_CODE": r"\bFC-\d+\b",

    # NASA C-MAPSS
    "ENGINE": r"\bEngine\s*\d+\b",

    "CYCLE": r"\bcycle\s*\d+\b",

    "SENSOR": r"\bS\d{1,2}\b",
    
    "SETTING": r"\bsetting\s*\d+\b"
}


# ==========================================================
# Regex Entity Extraction
# ==========================================================

def extract_regex_entities(text: str) ->List[Dict]:
    """
    Extract industrial entities using regex.
    """

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

    # spaCy / BERT labels

    "PER": "PERSON",

    "PERSON": "PERSON",

    "ORG": "ORGANIZATION",

    "LOC": "LOCATION",

    "GPE": "LOCATION",

    "DATE": "DATE",

    "TIME": "TIME",

    "NORP": "GROUP",

    "MISC": "MISC",

    # Industrial labels

    "PUMP": "PUMP",

    "VALVE": "VALVE",

    "MOTOR": "MOTOR",

    "PRESSURE_SENSOR": "PRESSURE_SENSOR",

    "TEMPERATURE_SENSOR": "TEMPERATURE_SENSOR",

    "FLOW_SENSOR": "FLOW_SENSOR",

    "LEVEL_SENSOR": "LEVEL_SENSOR",

    "COMPRESSOR": "COMPRESSOR",

    "ENGINE": "ENGINE",

    "CYCLE": "CYCLE",
    
    "SENSOR": "SENSOR",
    
    "SETTING": "SETTING",

    "TURBINE": "TURBINE",

    "DOCUMENT": "DOCUMENT",

    "SOP": "SOP",

    "MANUAL": "MANUAL",

    "FAULT_CODE": "FAULT_CODE",

    "NASA_DATASET": "NASA_DATASET"

}


# ==========================================================
# Label Priority
# ==========================================================

LABEL_PRIORITY = {

    "PUMP": 100,

    "VALVE": 100,

    "MOTOR": 100,

    "PRESSURE_SENSOR": 100,

    "TEMPERATURE_SENSOR": 100,

    "FLOW_SENSOR": 100,

    "LEVEL_SENSOR": 100,

    "COMPRESSOR": 100,

    "ENGINE": 100,

    "TURBINE": 100,

    "DOCUMENT": 100,

    "SOP": 100,

    "MANUAL": 100,

    "FAULT_CODE": 100,

    "NASA_DATASET": 100,

    "ENGINE": 100,
    
    "CYCLE": 100,
    
    "SENSOR": 100,
    
    "SETTING": 100,

    "PERSON": 90,

    "ORGANIZATION": 90,

    "LOCATION": 90,

    "DATE": 90,

    "TIME": 90,

    "GROUP": 60,

    "MISC": 10

}


# ==========================================================
# Confidence Weights
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
    """
    Convert different labels into a common format.
    """

    return LABEL_MAPPING.get(label, label)


def normalize_text(text: str):
    """
    Clean entity text.
    """

    text = text.strip()

    text = re.sub(r"\s*-\s*", "-", text)

    text = re.sub(r"\s+", " ", text)

    return text


# ==========================================================
# Merge Entities
# ==========================================================

def merge_entities(*entity_lists):
    """
    Merge entities extracted from Regex,
    spaCy and BERT.
    """

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
    """
    Select the highest priority label.
    """

    best = labels[0]

    for label in labels:

        if LABEL_PRIORITY.get(label, 0) > LABEL_PRIORITY.get(best, 0):

            best = label

    return best


# ==========================================================
# Confidence Calculation
# ==========================================================

def calculate_confidence(sources):
    """
    Calculate confidence from extraction sources.
    """

    score = 0

    for source in sources:

        score += SOURCE_WEIGHTS[source]

    score /= len(sources)

    return round(score, 2)


# ==========================================================
# Noise Filtering
# ==========================================================

BLACKLIST = {

    "refer",

    "section",

    "table",

    "figure",

    "fig",

    "sensor",

    "pump",

    "motor",

    "valve",

    "engine",

    "compressor",

    "system",

    "unit",

    "equipment"

}


# ==========================================================
# Entity Validation
# ==========================================================

def is_valid_entity(entity):
    """
    Remove low-quality entities.
    """

    text = entity["text"].strip()

    label = entity["label"]

    # Very short words

    if len(text) <= 1:

        return False

    # Punctuation only

    if re.fullmatch(r"[-.,:;()]+", text):

        return False

    # Single capital letter

    if re.fullmatch(r"[A-Z]", text):

        return False

    # Tiny miscellaneous words

    if label == "MISC" and len(text) < 5:

        return False

    # Pure numbers

    if text.isdigit():

        return False

    return True


# ==========================================================
# Final Filtering
# ==========================================================

def filter_entities(entities):
    """
    Remove duplicate and noisy entities.
    """

    filtered = []

    seen = set()

    for entity in entities:

        text = entity["text"].strip()

        key = text.lower()

        if key in BLACKLIST:

            continue

        if key in seen:

            continue

        if is_valid_entity(entity):

            filtered.append(entity)

            seen.add(key)

    return filtered

# ==========================================================
# Public Entity Extraction Function
# ==========================================================

def extract_entities(text: str):
    """
    Extract entities using Regex + spaCy + BERT.
    Returns a cleaned entity list.
    """

    # ------------------------------------------------------
    # Extract entities from all models
    # ------------------------------------------------------

    regex_entities = extract_regex_entities(text)

    spacy_entities = extract_spacy_entities(text)

    bert_entities = extract_bert_entities(text)

    # ------------------------------------------------------
    # Merge
    # ------------------------------------------------------

    merged_entities = merge_entities(
        regex_entities,
        spacy_entities,
        bert_entities
    )

    final_entities = []

    for entity in merged_entities:

        final_entities.append({

            "text": entity["text"],

            "label": resolve_best_label(
                entity["labels"]
            ),

            "sources": entity["sources"],

            "confidence": calculate_confidence(
                entity["sources"]
            )

        })

    # ------------------------------------------------------
    # Remove noisy entities
    # ------------------------------------------------------

    final_entities = filter_entities(final_entities)

    return final_entities


# ==========================================================
# Main (Testing)
# ==========================================================

if __name__ == "__main__":

    sample_text = """
    Engine 5 completed cycle 145.

    Sensor S12 detected abnormal vibration.

    Sensor S7 temperature increased.

    Operational setting 2 changed.

    Pump P-101 overheated.

    Valve VLV-203 leaked.

    Motor MTR-05 failed.

    Pressure sensor PT-201 malfunctioned.

    Refer SOP-001.

    Document DOC-2025-01.

    John repaired Pump P-101 at Microsoft
    on 12 June 2026 in Noida.
    """

    entities = extract_entities(sample_text)

    print("\nDetected Entities")
    print("=" * 70)

    for entity in entities:

        print(
            f"{entity['text']:25}"
            f"{entity['label']:25}"
            f"{entity['confidence']}"
        )