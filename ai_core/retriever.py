from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import logging

# ==========================================================
# Logging
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

def create_embeddings(chunks):

    embeddings = embedding_model.encode(
        chunks,
        convert_to_numpy=True
    )

    return embeddings

def build_faiss_index(embeddings):

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    return index

def save_index(index, filename="data/faiss.index"):

    faiss.write_index(index, filename)

    logging.info("FAISS index saved.")

def save_chunks(chunks, filename="data/chunks.pkl"):

    with open(filename, "wb") as f:

        pickle.dump(chunks, f)

    logging.info("Chunks saved.")

if __name__ == "_main_":
    embeddings = create_embeddings(chunks)

    index = build_faiss_index(embeddings)

    save_index(index)

    save_chunks(chunks)

    print("\nVector Database Created Successfully!")