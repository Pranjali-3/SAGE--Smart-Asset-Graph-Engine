from sentence_transformers import SentenceTransformer
import faiss
import pickle
import logging
import numpy as np

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

def load_faiss_index(filename="data/faiss.index"):
    """
    Load the FAISS vector database.
    """

    index = faiss.read_index(filename)

    logging.info("FAISS index loaded.")

    return index


# ==========================================================
# Load Stored Chunks
# ==========================================================

def load_chunks(filename="data/chunks.pkl"):
    """
    Load stored text chunks.
    """

    with open(filename, "rb") as file:

        chunks = pickle.load(file)

    logging.info("Chunks loaded.")

    return chunks


# ==========================================================
# Embed User Query
# ==========================================================

def embed_query(query: str):
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
    top_k=3
):
    """
    Search the FAISS vector database.
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
    indices
):
    """
    Retrieve chunks corresponding to
    the FAISS search results.
    """

    retrieved = []

    for idx in indices[0]:

        if idx != -1 and idx < len(chunks):

            retrieved.append(chunks[idx])

    return retrieved


# ==========================================================
# Entity-aware Re-ranking
# ==========================================================

def entity_filter(
    query: str,
    retrieved_chunks
):
    """
    Re-rank retrieved chunks using entities
    extracted from the query.
    """

    query_entities = extract_entities(query)

    if not query_entities:
        return retrieved_chunks

    query_entity_text = {

        entity["text"].lower()

        for entity in query_entities

    }

    scored_chunks = []

    for chunk in retrieved_chunks:

        chunk_entities = extract_entities(chunk)

        score = 0

        for entity in chunk_entities:

            if entity["text"].lower() in query_entity_text:

                score += 1

        scored_chunks.append(
            (score, chunk)
        )

    scored_chunks.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return [

        chunk

        for score, chunk in scored_chunks

    ]


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    print("\nIndustrial Knowledge Retriever")
    print("=" * 60)

    query = input("\nEnter your query: ")

    # ------------------------------------------------------
    # Load Vector Database
    # ------------------------------------------------------

    index = load_faiss_index()

    chunks = load_chunks()

    # ------------------------------------------------------
    # Convert Query to Embedding
    # ------------------------------------------------------

    query_embedding = embed_query(query)

    # ------------------------------------------------------
    # Semantic Search
    # ------------------------------------------------------

    distances, indices = search_index(
        index,
        query_embedding,
        top_k=3
    )

    # ------------------------------------------------------
    # Retrieve Chunks
    # ------------------------------------------------------

    retrieved_chunks = retrieve_chunks(
        chunks,
        indices
    )

    # ------------------------------------------------------
    # Entity-aware Re-ranking
    # ------------------------------------------------------

    reranked_chunks = entity_filter(
        query,
        retrieved_chunks
    )

    # ------------------------------------------------------
    # Display FAISS Results
    # ------------------------------------------------------

    print("\nFAISS Search Results")
    print("-" * 60)

    for rank, (idx, distance) in enumerate(
        zip(indices[0], distances[0]),
        start=1
    ):

        if idx == -1:
            continue

        print(f"\nRank {rank}")

        print(f"Distance : {distance:.4f}")

        print(chunks[idx])

    # ------------------------------------------------------
    # Display Final Re-ranked Results
    # ------------------------------------------------------

    print("\nEntity-aware Re-ranked Results")
    print("-" * 60)

    for i, chunk in enumerate(
        reranked_chunks,
        start=1
    ):

        print(f"\nChunk {i}")

        print(chunk)

    print("\nRetrieval Complete.")