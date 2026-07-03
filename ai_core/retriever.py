from sentence_transformers import SentenceTransformer
import faiss
import pickle
import logging
import numpy as np
import re

from entity_extractor import extract_entities

# ==========================================================
# Logging Configuration
# ==========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

# ==========================================================
# Load Embedding Model
# ==========================================================

logging.info("Loading embedding model...")

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

logging.info("Embedding model loaded.")

# ==========================================================
# Load FAISS Index
# ==========================================================

def load_faiss_index(
    filename="data/faiss.index"
):
    """
    Load the FAISS vector database.
    """

    index = faiss.read_index(filename)

    logging.info("FAISS index loaded.")

    return index


# ==========================================================
# Load Stored Chunks
# ==========================================================

def load_chunks(
    filename="data/chunks.pkl"
):
    """
    Load stored document chunks.
    """

    with open(filename, "rb") as file:

        chunks = pickle.load(file)

    logging.info("Chunks loaded.")

    return chunks


# ==========================================================
# Embed User Query
# ==========================================================

def embed_query(
    query: str
):
    """
    Convert the user's query into an embedding.
    """

    embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True
    )

    return embedding.astype("float32")


# ==========================================================
# Search Vector Database
# ==========================================================

def search_index(
    index,
    query_embedding,
    top_k=10
):
    """
    Perform semantic similarity search
    using the FAISS vector database.
    """

    distances, indices = index.search(
        query_embedding,
        top_k
    )

    return distances, indices


# ==========================================================
# Retrieve Chunks
# ==========================================================

def retrieve_chunks(
    chunks,
    indices,
    distances
):
    """
    Retrieve chunks returned by FAISS
    together with their similarity score.
    """

    retrieved = []

    for idx, distance in zip(
        indices[0],
        distances[0]
    ):

        if idx == -1:
            continue

        if idx >= len(chunks):
            continue

        retrieved.append({

            "chunk": chunks[idx],

            "distance": float(distance),

            "index": int(idx)

        })

    return retrieved

# ==========================================================
# Relationship Extraction
# ==========================================================

# Common industrial action verbs

RELATION_PATTERNS = [

    "repair",
    "repaired",
    "replace",
    "replaced",
    "inspect",
    "inspected",
    "check",
    "checked",
    "connect",
    "connected",
    "disconnect",
    "disconnected",
    "install",
    "installed",
    "remove",
    "removed",
    "monitor",
    "monitored",
    "measure",
    "measured",
    "detect",
    "detected",
    "cause",
    "caused",
    "fail",
    "failed",
    "trigger",
    "triggered",
    "increase",
    "increased",
    "decrease",
    "decreased",
    "overheat",
    "overheated",
    "leak",
    "leaking",
    "leaked",
    "start",
    "started",
    "stop",
    "stopped",
    "shutdown",
    "opened",
    "closed",
    "activate",
    "activated"
]


# ==========================================================
# Extract Relationships
# ==========================================================

def extract_relationships(text):
    """
    Extract simple Subject-Relation-Object triples
    using entities and action words.
    """

    entities = extract_entities(text)

    relationships = []

    if len(entities) < 2:
        return relationships

    lower_text = text.lower()

    for verb in RELATION_PATTERNS:

        if verb in lower_text:

            relation = verb

            break

    else:

        relation = "related_to"

    for i in range(len(entities) - 1):

        relationships.append({

            "subject": entities[i]["text"],

            "relation": relation,

            "object": entities[i + 1]["text"]

        })

    return relationships


# ==========================================================
# Entity Match Score
# ==========================================================

def entity_score(
    query_entities,
    chunk_entities
):
    """
    Calculate score based on
    matching entities.
    """

    score = 0

    query_set = {

        entity["text"].lower()

        for entity in query_entities

    }

    for entity in chunk_entities:

        if entity["text"].lower() in query_set:

            score += 2

    return score


# ==========================================================
# Relationship Match Score
# ==========================================================

def relationship_score(
    query_relationships,
    chunk_relationships
):
    """
    Compare relationships extracted
    from query and chunk.
    """

    score = 0

    for query_relation in query_relationships:

        for chunk_relation in chunk_relationships:

            if (

                query_relation["relation"] ==

                chunk_relation["relation"]

            ):

                score += 2

            if (

                query_relation["subject"].lower()

                ==

                chunk_relation["subject"].lower()

            ):

                score += 1

            if (

                query_relation["object"].lower()

                ==

                chunk_relation["object"].lower()

            ):

                score += 1

    return score


# ==========================================================
# Combined Ranking Score
# ==========================================================

def compute_final_score(

    semantic_distance,

    entity_match,

    relationship_match

):
    """
    Lower FAISS distance means
    higher semantic similarity.
    """

    semantic_score = 1 / (

        1 + semantic_distance

    )

    final_score = (

        semantic_score

        + entity_match

        + relationship_match

    )

    return round(final_score, 4)

# ==========================================================
# Intelligent Re-ranking
# ==========================================================

def rerank_chunks(
    query,
    retrieved_chunks
):
    """
    Re-rank retrieved chunks using:
    1. Semantic similarity
    2. Entity matching
    3. Relationship matching
    """

    query_entities = extract_entities(query)

    query_relationships = extract_relationships(query)

    final_results = []

    for item in retrieved_chunks:

        chunk = item["chunk"]

        distance = item["distance"]

        chunk_entities = extract_entities(chunk)

        chunk_relationships = extract_relationships(chunk)

        entity_match = entity_score(
            query_entities,
            chunk_entities
        )

        relationship_match = relationship_score(
            query_relationships,
            chunk_relationships
        )

        final_score = compute_final_score(
            distance,
            entity_match,
            relationship_match
        )

        final_results.append({

            "chunk": chunk,

            "distance": distance,

            "entity_score": entity_match,

            "relationship_score": relationship_match,

            "final_score": final_score,

            "entities": chunk_entities,

            "relationships": chunk_relationships

        })

    final_results.sort(

        key=lambda x: x["final_score"],

        reverse=True

    )

    return final_results


# ==========================================================
# Retrieval Pipeline
# ==========================================================

def retrieve(
    query,
    top_k=10
):
    """
    Complete retrieval pipeline.
    """

    index = load_faiss_index()

    chunks = load_chunks()

    query_embedding = embed_query(query)

    distances, indices = search_index(

        index,

        query_embedding,

        top_k

    )

    retrieved_chunks = retrieve_chunks(

        chunks,

        indices,

        distances

    )

    ranked_results = rerank_chunks(

        query,

        retrieved_chunks

    )

    return ranked_results


# ==========================================================
# Display Results
# ==========================================================

def display_results(results):
    """
    Display ranked retrieval results.
    """

    print("\nFinal Retrieval Results")

    print("=" * 70)

    for rank, result in enumerate(

        results,

        start=1

    ):

        print(f"\nRank {rank}")

        print("-" * 70)

        print(f"Final Score        : {result['final_score']:.4f}")

        print(f"Semantic Distance  : {result['distance']:.4f}")

        print(f"Entity Score       : {result['entity_score']}")

        print(f"Relationship Score : {result['relationship_score']}")

        print("\nChunk")

        print(result["chunk"])

        print("\nEntities")

        for entity in result["entities"]:

            print(

                f"  {entity['text']}"

                f" ({entity['label']})"

            )

        print("\nRelationships")

        if result["relationships"]:

            for relation in result["relationships"]:

                print(

                    f"  {relation['subject']}"

                    f" --[{relation['relation']}]--> "

                    f"{relation['object']}"

                )

        else:

            print("  None")

# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    print("\nIndustrial Knowledge Retriever")
    print("=" * 70)

    while True:

        query = input(
            "\nEnter your query ('exit' to quit): "
        )

        if query.lower() == "exit":

            print("\nRetriever Closed.")

            break

        results = retrieve(
            query,
            top_k=10
        )

        display_results(results)

        print("\n" + "=" * 70)
        print("Retrieval Complete.")