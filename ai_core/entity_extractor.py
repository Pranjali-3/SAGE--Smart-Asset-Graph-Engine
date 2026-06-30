import spacy
import json
import logging
from typing import List, Dict

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

logging.info("Loading spaCy model...")

nlp = spacy.load("en_core_web_sm")

logging.info("spaCy model loaded.")

def extract_spacy_entities(text):

    doc = nlp(text)

    entities = []

    for entity in doc.ents:

        entities.append({
            "text": entity.text,
            "label": entity.label_
        })

    return entities

text = """
John repaired Pump P-101 at Microsoft
on 12 June 2026 in Noida.
"""

entities = extract_spacy_entities(text)

for entity in entities:
    print(entity)