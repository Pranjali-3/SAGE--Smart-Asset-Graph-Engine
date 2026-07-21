import faiss
import pickle
import logging
import numpy as np
import re
from .model_manager import models
from .entity_extractor import extract_entities

# ==========================================================
# Logging Configuration
# ==========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

# ==========================================================
# Load Embedding Model from ModelManager
# ==========================================================

embedding_model = models.embedding_model

# ==========================================================
# Retriever Class
# ==========================================================

class Retriever:

    def __init__(self):
        self.index = self.load_faiss_index()
        self.chunks = self.load_chunks()

    # ==========================================================
    # Load FAISS Index
    # ==========================================================

    def load_faiss_index(
        self,
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
        self,
        filename="data/chunks.pkl"
    ):
        """
        Load document chunks stored during
        embedding generation.
        """

        with open(filename, "rb") as file:

            chunks = pickle.load(file)

        logging.info("Chunks loaded.")

        return chunks


    # ==========================================================
    # Embed User Query
    # ==========================================================

    def embed_query(
        self,
        query: str
    ):
        """
        Convert the user's query
        into an embedding vector.
        """

        embedding = embedding_model.encode(

            [query],

            convert_to_numpy=True

        )

        return embedding.astype("float32")


    # ==========================================================
    # Semantic Search
    # ==========================================================

    def search_index(
        self,
        index,
        query_embedding,
        top_k=20
    ):
        """
        Search the FAISS vector database
        for the most relevant chunks.
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
        self,
        chunks,
        indices,
        distances
    ):
        """
        Retrieve chunk text along with
        semantic distance.
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

                "index": idx,

                "distance": float(distance)

            })

        return retrieved

    # ==========================================================
    # Entity Match Score
    # ==========================================================

    def calculate_entity_score(
        self,
        query_entities,
        chunk_entities
    ):
        """
        Calculate score based on the number
        of matching entities.
        """

        if not query_entities:
            return 0

        query_set = {

            entity["text"].lower()

            for entity in query_entities

        }

        score = 0

        for entity in chunk_entities:

            if entity["text"].lower() in query_set:

                score += 2

        return score


    # ==========================================================
    # Keyword Overlap Score (Step 9)
    # ==========================================================

    def calculate_keyword_overlap(
    self,
    query,
    chunk_text
):

        query_words = set(
            re.findall(r"\w+", query.lower())
        )

        chunk_words = set(
            re.findall(r"\w+", chunk_text.lower())
        )

        stop_words = {
            "the","a","an","is","are","was","were",
            "of","to","in","on","for","with","by",
            "what","which","who","when","where",
            "how","does","do","did"
        }

        query_words -= stop_words
        chunk_words -= stop_words

        return len(query_words & chunk_words)

    # ==========================================================
    # Metadata Match Score (Step 9 - Improved)
    # ==========================================================

    def calculate_metadata_match(
        self,
        query,
        query_entities,
        chunk
    ):
        """
        Calculate score based on metadata matching (engine, status, dataset).
        Uses substring matching and also searches raw query text.
        """

        score = 0

        if not isinstance(chunk, dict):
            return 0

        query_lower = query.lower()

        # Check engine match - substring matching
        engine = str(chunk.get("engine", ""))
        if engine:
            # Direct engine number match in query
            if engine in query_lower:
                score += 3
            # Entity-based match
            for entity in query_entities:
                entity_text = entity["text"].lower()
                if engine in entity_text or entity_text in engine:
                    score += 3
                    break

        # Check dataset match
        if chunk.get("dataset"):
            dataset = chunk["dataset"].lower()
            if dataset in query_lower:
                score += 2
            for entity in query_entities:
                entity_text = entity["text"].lower()
                if dataset in entity_text:
                    score += 2
                    break

        # Check status match - raw query text
        if chunk.get("status"):
            status = chunk["status"].lower()
            if status in query_lower:
                score += 2

        return score

    # ==========================================================
    # Semantic Score
    # ==========================================================

    def calculate_semantic_score(
        self,
        distance
    ):
        """
        Convert FAISS distance into
        a similarity score.
        """

        return float(distance)


    # ==========================================================
    # Combined Score (Step 9)
    # ==========================================================

    def calculate_final_score(
        self,
        semantic_score,
        entity_score,
        keyword_score,
        metadata_score
    ):
        """
        Combine semantic similarity, entity matching, keyword overlap,
        and metadata matching scores.
        """

        return (
            0.45*semantic_score +

            0.25*keyword_score +

            0.20*metadata_score +

            0.10*entity_score
        )


    # ==========================================================
    # Entity-aware Re-ranking (Step 9 Enhanced)
    # ==========================================================

    def rerank_chunks(
        self,
        query,
        retrieved_chunks
    ):
        """
        Re-rank retrieved chunks using:
        - Semantic similarity
        - Entity overlap
        - Keyword overlap (normalized)
        - Metadata matching
        """

        query_entities = extract_entities(query)

        ranked_results = []

        for item in retrieved_chunks:

            chunk = item["chunk"]

            if isinstance(chunk, dict):
                chunk_text = chunk.get("text", "")
            else:
                chunk_text = str(chunk)

            chunk_entities = extract_entities(chunk_text)

            semantic_score = self.calculate_semantic_score(

                item["distance"]

            )

            entity_score = self.calculate_entity_score(

                query_entities,

                chunk_entities

            )

            keyword_score = self.calculate_keyword_overlap(

                query,

                chunk_text

            )

            # Normalize keyword score (max 5)
            keyword_score = min(keyword_score, 5) / 5

            metadata_score = self.calculate_metadata_match(

                query,

                query_entities,

                chunk

            )

            final_score = self.calculate_final_score(

                semantic_score,

                entity_score,

                keyword_score,

                metadata_score

            )

            ranked_results.append({

                "chunk": chunk,

                "index": item["index"],

                "distance": item["distance"],

                "semantic_score": round(
                    semantic_score,
                    4
                ),

                "entity_score": entity_score,

                "keyword_score": round(keyword_score, 4),

                "metadata_score": metadata_score,

                "final_score": round(
                    final_score,
                    4
                ),

                "entities": chunk_entities

            })

        ranked_results.sort(

            key=lambda x: x["final_score"],

            reverse=True

        )

        return ranked_results


    # ==========================================================
    # Remove Duplicate Chunks
    # ==========================================================

    def remove_duplicates(self, ranked_results):
        """
        Remove duplicate chunks while preserving ranking.
        """

        unique_chunks = set()
        filtered_results = []

        for result in ranked_results:

            # ---------------------------
            # Normalize chunk to string
            # ---------------------------
            chunk = result.get("chunk")

            if isinstance(chunk, dict):
                chunk_text = chunk.get("text", "")
            else:
                chunk_text = str(chunk)

            chunk_text = chunk_text.strip()

            # ---------------------------
            # Deduplication check
            # ---------------------------
            if chunk_text in unique_chunks:
                continue

            unique_chunks.add(chunk_text)

            # ---------------------------
            # Store cleaned result
            # ---------------------------
            filtered_results.append({
                **result,
                "chunk": chunk
            })

        return filtered_results

    # ==========================================================
    # Complete Retrieval Pipeline (Improved: FAISS 50 -> Rerank -> 8)
    # ==========================================================

    def retrieve(
        self,
        query: str,
        top_k=8,
        faiss_candidates=50
    ):
        """
        Complete retrieval pipeline.
        Retrieves 50 candidates from FAISS, reranks, returns best 8.
        """

        # Use pre-loaded data
        index = self.index
        chunks = self.chunks

        # Convert query into embedding
        query_embedding = self.embed_query(query)

        # Semantic search - get more candidates for reranking
        distances, indices = self.search_index(

            index,

            query_embedding,

            faiss_candidates

        )

        # Retrieve chunk text
        retrieved_chunks = self.retrieve_chunks(

            chunks,

            indices,

            distances

        )

        # Entity-aware ranking
        ranked_results = self.rerank_chunks(

            query,

            retrieved_chunks

        )

        # Remove duplicates
        ranked_results = self.remove_duplicates(

            ranked_results

        )

        # Return only top_k results
        return ranked_results[:top_k]


    # ==========================================================
    # Display Results
    # ==========================================================

    def display_results(
        self,
        ranked_results
    ):
        """
        Display retrieval results.
        """

        print("\nFinal Retrieval Results")

        print("=" * 70)

        if len(ranked_results) == 0:

            print("\nNo relevant chunks found.")

            return

        for rank, result in enumerate(

            ranked_results,

            start=1

        ):

            print(f"\nRank {rank}")

            print("-" * 70)

            print(

                f"Chunk Index      : {result['index']}"

            )

            print(

                f"Semantic Score   : {result['semantic_score']:.4f}"

            )

            print(

                f"Keyword Score    : {result['keyword_score']:.4f}"

            )

            print(

                f"Metadata Score   : {result['metadata_score']}"

            )

            print(

                f"Entity Score     : {result['entity_score']}"

            )

            print(

                f"Final Score      : {result['final_score']:.4f}"

            )

            print()

            chunk = result["chunk"]

            if isinstance(chunk, dict):

                print(f"Source          : {chunk.get('source')}")
                print(f"Dataset         : {chunk.get('dataset')}")
                print(f"Engine          : {chunk.get('engine')}")
                print(f"Status          : {chunk.get('status')}")
                print(f"RUL             : {chunk.get('rul')}")

                print("\nChunk")
                print(chunk.get("text"))

            else:

                print(chunk)

            print("\nEntities")

            if result["entities"]:

                for entity in result["entities"]:

                    print(

                        f"- {entity['text']} "

                        f"({entity['label']})"

                    )

            else:

                print("No entities found.")

            print("\n" + "=" * 70)


    # ==========================================================
    # Return Only Chunk Text
    # ==========================================================

    def get_context(
        self,
        ranked_results,
        max_chunks=8
    ):

        context = []

        for result in ranked_results[:max_chunks]:

            chunk = result["chunk"]

            if isinstance(chunk, dict):

                context.append(chunk["text"])

            else:

                context.append(str(chunk))

        return "\n\n".join(context)

# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    print("\nIndustrial Knowledge Retriever")
    print("=" * 70)

    retriever = Retriever()

    while True:

        query = input(
            "\nEnter your query ('exit' to quit): "
        )

        if query.lower() == "exit":

            print("\nRetriever Closed.")

            break

        # --------------------------------------------------
        # Retrieve Results (FAISS 50 -> Rerank -> 8)
        # --------------------------------------------------

        ranked_results = retriever.retrieve(

            query,

            top_k=8,

            faiss_candidates=50

        )

        # --------------------------------------------------
        # Display Results
        # --------------------------------------------------

        retriever.display_results(

            ranked_results

        )

        # --------------------------------------------------
        # Context for Downstream Modules
        # --------------------------------------------------

        context = retriever.get_context(

            ranked_results,

            max_chunks=8

        )

        print("\nContext Passed to Next Module")
        print("=" * 70)

        print(context)

        print("\nRetriever Finished.")
        print("=" * 70)
