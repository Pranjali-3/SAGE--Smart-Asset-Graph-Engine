from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import json
import logging
import spacy
import os

from .ingestion import ingest_document

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

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

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

        if sent.text.strip()

    ]


# ==========================================================
# Text Normalizer (Step 5)
# ==========================================================

def normalize_text(text):
    """
    Normalize text before chunking.
    Removes OCR artifacts and extra whitespace.
    """

    text = str(text)

    text = text.replace("\n", " ")

    text = " ".join(text.split())

    return text


# ==========================================================
# Chunk Builder with Overlap (Steps 2, 3)
# ==========================================================

def build_chunks(
    text: str,
    max_chars: int = 700,
    overlap_sentences: int = 2
):
    """
    Combine sentences into chunks with overlap.
    Overlap improves retrieval by maintaining context.
    """

    sentences = split_into_sentences(text)

    if not sentences:
        return []

    chunks = []

    current_chunk = []

    current_length = 0

    for sentence in sentences:

        sentence_len = len(sentence)

        if current_length + sentence_len + 1 <= max_chars:

            current_chunk.append(sentence)

            current_length += sentence_len + 1

        else:

            if current_chunk:

                chunks.append(" ".join(current_chunk))

            overlap_start = max(0, len(current_chunk) - overlap_sentences)

            current_chunk = current_chunk[overlap_start:] + [sentence]

            current_length = sum(len(s) + 1 for s in current_chunk)

    if current_chunk:

        chunks.append(" ".join(current_chunk))

    return chunks


# ==========================================================
# Generate Embeddings (Step 8)
# ==========================================================

def generate_embeddings(chunks):
    """
    Convert chunks into normalized embedding vectors.
    """

    embeddings = embedding_model.encode(

        chunks,

        convert_to_numpy=True,

        normalize_embeddings=True,

        show_progress_bar=True

    )

    return embeddings.astype("float32")


# ==========================================================
# Build FAISS Index (Step 9)
# ==========================================================

def build_faiss_index(
    embeddings
):
    """
    Build FAISS vector database using Inner Product
    for cosine similarity with normalized embeddings.
    """

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

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
    Save FAISS index.
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
    Save chunks with metadata.
    """

    with open(filename, "wb") as file:

        pickle.dump(chunks, file)

    logging.info("Chunks saved.")


# ==========================================================
# Save Metadata (Step 10)
# ==========================================================

def save_metadata(
    filename="data/metadata.json"
):
    """
    Save embedding metadata for future reference.
    """

    metadata = {

        "embedding_model": EMBEDDING_MODEL_NAME,

        "max_chars": 700,

        "overlap_sentences": 2,

        "normalize_embeddings": True,

        "faiss_metric": "inner_product"

    }

    with open(filename, "w") as file:

        json.dump(metadata, file, indent=2)

    logging.info("Metadata saved.")


# ==========================================================
# Garbage Filter (Step 12)
# ==========================================================

GARBAGE_PATTERNS = [
    "Figure ", "Figure\n",
    "Contents", "Page ",
    "References", "Table of",
    "Abstract", "Keywords",
    "Copyright", "Published by",
    "Author", "University"
]


def is_garbage_chunk(chunk):
    """
    Check if chunk is garbage/noise.
    """

    stripped = chunk.strip()

    if len(stripped.split()) < 10:
        return True

    for pattern in GARBAGE_PATTERNS:

        if stripped.startswith(pattern):
            return True

    return False


# ==========================================================
# Process Dataset Folder (Steps 4-7, 11-12)
# ==========================================================

def process_dataset(folder_path):
    """
    Read every supported document from the dataset
    and convert it into chunks.
    """

    all_chunks = []

    supported_extensions = (

        ".pdf",
        ".docx",
        ".xlsx",
        ".csv",
        ".png",
        ".jpg",
        ".jpeg",
        ".txt"

    )

    for file in sorted(os.listdir(folder_path)):

        path = os.path.join(folder_path, file)

        if not os.path.isfile(path):

            continue

        if not file.lower().endswith(supported_extensions):

            continue

        logging.info(f"Processing {file}")

        try:

            result = ingest_document(path)

            if result is None:

                continue

            # NASA TXT returns pre-chunked list of dicts
            if isinstance(result, list):

                for chunk_dict in result:

                    chunk_dict["source"] = file

                    chunk_dict["chunk_id"] = len(all_chunks)

                    all_chunks.append(chunk_dict)

                logging.info(
                    f"{len(result)} pre-chunked blocks added."
                )

                continue

            text = normalize_text(result)

            if len(text) == 0:

                continue

            document_chunks = build_chunks(text)

            logging.info(
                f"{len(document_chunks)} chunks created."
            )

            for chunk in document_chunks:

                if len(chunk) < 80:
                    continue

                if is_garbage_chunk(chunk):
                    continue

                all_chunks.append({

                    "text": chunk,

                    "source": file,

                    "chunk_id": len(all_chunks)

                })

        except Exception as e:

            logging.warning(

                f"Skipping {file}: {e}"

            )

    # Step 7: Remove duplicates
    unique = {}

    for chunk in all_chunks:

        unique[chunk["text"]] = chunk

    all_chunks = list(unique.values())

    return all_chunks

# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    print("\nIndustrial Knowledge Embedding Pipeline")
    print("=" * 70)

    # ------------------------------------------------------
    # Dataset Folder (Step 1)
    # ------------------------------------------------------

    dataset_folder = r"data\nasa\archive"

    if not os.path.exists(dataset_folder):

        raise FileNotFoundError(

            f"Dataset folder not found:\n{dataset_folder}"

        )

    # ------------------------------------------------------
    # Read Every Document
    # ------------------------------------------------------

    all_chunks = process_dataset(dataset_folder)

    print("\nDataset Summary")
    print("-" * 70)

    print(f"Total Chunks : {len(all_chunks)}")

    if len(all_chunks) == 0:

        raise ValueError("No chunks were created.")

    # ------------------------------------------------------
    # Chunk Statistics (Step 11)
    # ------------------------------------------------------

    lengths = [

        len(c["text"])

        for c in all_chunks

    ]

    print(f"Average chunk length: {sum(lengths)/len(lengths):.0f} chars")
    print(f"Min chunk length    : {min(lengths)} chars")
    print(f"Max chunk length    : {max(lengths)} chars")

    # ------------------------------------------------------
    # Preview Chunks
    # ------------------------------------------------------

    print("\nChunk Preview")
    print("-" * 70)

    preview = min(5, len(all_chunks))

    for i in range(preview):

        print(f"\nChunk {i+1}")

        print(f"Source : {all_chunks[i]['source']}")

        print(f"ID     : {all_chunks[i]['chunk_id']}")

        print(all_chunks[i]["text"][:250])

    # ------------------------------------------------------
    # Extract Text Only
    # ------------------------------------------------------

    chunk_texts = [

        chunk["text"]

        for chunk in all_chunks

    ]

    # ------------------------------------------------------
    # Generate Embeddings
    # ------------------------------------------------------

    print("\nGenerating Embeddings...")
    print("-" * 70)

    embeddings = generate_embeddings(chunk_texts)

    print("Embedding Shape :", embeddings.shape)

    print("Embedding Dimension :", embeddings.shape[1])

    # ------------------------------------------------------
    # Build FAISS Index
    # ------------------------------------------------------

    print("\nBuilding FAISS Index...")
    print("-" * 70)

    index = build_faiss_index(embeddings)

    print("Vectors Stored :", index.ntotal)

    # ------------------------------------------------------
    # Save Database
    # ------------------------------------------------------

    save_faiss_index(index)

    save_chunks(all_chunks)

    save_metadata()

    # ------------------------------------------------------
    # Finished
    # ------------------------------------------------------

    print("\nVector Database Created Successfully!")

    print("\nSaved Files")
    print("-" * 70)

    print("data/faiss.index")

    print("data/chunks.pkl")

    print("data/metadata.json")

    print("\nDatabase Statistics")
    print("-" * 70)

    print(f"Documents Processed : {len(set(chunk['source'] for chunk in all_chunks))}")

    print(f"Total Chunks        : {len(all_chunks)}")

    print(f"Avg Chunk Length     : {sum(lengths)/len(lengths):.0f} chars")

    print(f"Embedding Size      : {embeddings.shape[1]}")

    print(f"FAISS Metric        : Inner Product (cosine)")

    print(f"Vectors in FAISS    : {index.ntotal}")

    print(f"Embedding Model     : {EMBEDDING_MODEL_NAME}")

    print("\nEmbedding Pipeline Complete.")
