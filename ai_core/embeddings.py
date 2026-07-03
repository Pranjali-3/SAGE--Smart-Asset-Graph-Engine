from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import logging
import spacy
import os

from ingestion import ingest_document

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

        if sent.text.strip()

    ]


# ==========================================================
# Chunk Builder
# ==========================================================

def build_chunks(
    text: str,
    max_chars: int = 300
):
    """
    Combine sentences into chunks.
    """

    sentences = split_into_sentences(text)

    chunks = []

    current_chunk = ""

    for sentence in sentences:

        if len(current_chunk) + len(sentence) + 1 <= max_chars:

            current_chunk += sentence + " "

        else:

            if current_chunk:

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
    Convert chunks into embedding vectors.
    """

    embeddings = embedding_model.encode(

        chunks,

        convert_to_numpy=True,

        show_progress_bar=True

    )

    return embeddings.astype("float32")


# ==========================================================
# Build FAISS Index
# ==========================================================

def build_faiss_index(
    embeddings
):
    """
    Build FAISS vector database.
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
# Process Dataset Folder
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

            text = ingest_document(path)

            if text is None:

                continue

            text = str(text).strip()

            if len(text) == 0:

                continue

            document_chunks = build_chunks(text)

            logging.info(
                f"{len(document_chunks)} chunks created."
            )

            for chunk in document_chunks:

                all_chunks.append({

                    "text": chunk,

                    "source": file

                })

        except Exception as e:

            logging.warning(

                f"Skipping {file}: {e}"

            )

    return all_chunks

# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    print("\nIndustrial Knowledge Embedding Pipeline")
    print("=" * 70)

    # ------------------------------------------------------
    # Dataset Folder
    # ------------------------------------------------------

    dataset_folder = r"data\nasa\archive\CMaps"

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
    # Preview Chunks
    # ------------------------------------------------------

    print("\nChunk Preview")
    print("-" * 70)

    preview = min(5, len(all_chunks))

    for i in range(preview):

        print(f"\nChunk {i+1}")

        print(f"Source : {all_chunks[i]['source']}")

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

    # ------------------------------------------------------
    # Finished
    # ------------------------------------------------------

    print("\nVector Database Created Successfully!")

    print("\nSaved Files")
    print("-" * 70)

    print("data/faiss.index")

    print("data/chunks.pkl")

    print("\nDatabase Statistics")
    print("-" * 70)

    print(f"Documents Processed : {len(set(chunk['source'] for chunk in all_chunks))}")

    print(f"Total Chunks        : {len(all_chunks)}")

    print(f"Embedding Size      : {embeddings.shape[1]}")

    print(f"Vectors in FAISS    : {index.ntotal}")

    print("\nEmbedding Pipeline Complete.")