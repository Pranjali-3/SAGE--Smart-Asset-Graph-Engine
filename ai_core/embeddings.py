from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import logging
import spacy

# ==========================================================
# Logging Configuration
# ==========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

# ==========================================================
# Load spaCy Model
# ==========================================================

logging.info("Loading spaCy model...")

nlp = spacy.load("en_core_web_sm")

logging.info("spaCy model loaded.")

# ==========================================================
# Load Embedding Model
# ==========================================================

logging.info("Loading embedding model...")

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

logging.info("Embedding model loaded.")

# ==========================================================
# Sentence Splitter
# ==========================================================

def split_into_sentences(text: str):
    """
    Split text into complete sentences.
    """

    doc = nlp(text)

    return [
        sent.text.strip()
        for sent in doc.sents
    ]


# ==========================================================
# Chunk Builder
# ==========================================================

def build_chunks(
    text: str,
    max_chars: int = 150
):
    """
    Combine multiple sentences into chunks.
    """

    sentences = split_into_sentences(text)

    chunks = []

    current_chunk = ""

    for sentence in sentences:

        if len(current_chunk) + len(sentence) + 1 <= max_chars:

            current_chunk += sentence + " "

        else:

            chunks.append(current_chunk.strip())

            current_chunk = sentence + " "

    if current_chunk:

        chunks.append(current_chunk.strip())

    return chunks


# ==========================================================
# Generate Embeddings
# ==========================================================

def generate_embeddings(chunks):
    """
    Convert text chunks into embedding vectors.
    """

    embeddings = embedding_model.encode(
        chunks,
        convert_to_numpy=True
    )

    return embeddings.astype("float32")


# ==========================================================
# Build FAISS Index
# ==========================================================

def build_faiss_index(embeddings):
    """
    Create a FAISS vector database.
    """

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    return index


# ==========================================================
# Save FAISS Index
# ==========================================================

def save_faiss_index(
    index,
    filename="data/faiss.index"
):
    """
    Save FAISS index to disk.
    """

    faiss.write_index(index, filename)

    logging.info("FAISS index saved.")


# ==========================================================
# Save Chunks
# ==========================================================

def save_chunks(
    chunks,
    filename="data/chunks.pkl"
):
    """
    Save original chunks for retrieval.
    """

    with open(filename, "wb") as file:

        pickle.dump(chunks, file)

    logging.info("Chunks saved.")


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    text = """
    Pump P-101 is overheating.

    Pressure Sensor PT-201 has failed.

    Motor MTR-05 should be replaced.

    Valve VLV-203 is leaking.

    Operator John repaired Pump P-101 yesterday.
    """

    # ------------------------------------------------------
    # Build Chunks
    # ------------------------------------------------------

    chunks = build_chunks(text)

    print("\nChunks")
    print("-" * 60)

    for i, chunk in enumerate(chunks, start=1):

        print(f"\nChunk {i}")

        print(chunk)

    # ------------------------------------------------------
    # Generate Embeddings
    # ------------------------------------------------------

    embeddings = generate_embeddings(chunks)

    print("\nEmbeddings")
    print("-" * 60)

    for i, embedding in enumerate(embeddings, start=1):

        print(f"\nChunk {i}")

        print(f"Vector Length : {len(embedding)}")

        print("First 10 Values:")

        print(embedding[:10])

    # ------------------------------------------------------
    # Build FAISS Index
    # ------------------------------------------------------

    index = build_faiss_index(embeddings)

    print("\nFAISS Index")
    print("-" * 60)

    print("Number of vectors:", index.ntotal)

    # ------------------------------------------------------
    # Save Files
    # ------------------------------------------------------

    save_faiss_index(index)

    save_chunks(chunks)

    print("\nVector Database Created Successfully!")

    print("\nSaved Files")
    print("-" * 60)

    print("data/faiss.index")

    print("data/chunks.pkl")