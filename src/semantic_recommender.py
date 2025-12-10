import os
# --- Quiet down TensorFlow / TF-keras logs BEFORE anything imports it ---
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Hide INFO and WARNING from TF C++ backend

import logging
import json
import faiss
import numpy as np
import pandas as pd
from typing import List, Tuple

# Optional: further silence TF Python warnings if TF is present
try:
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    if hasattr(tf, "compat") and hasattr(tf.compat, "v1"):
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except Exception:
    # If TF isn't available or something goes weird, just ignore
    pass

from sentence_transformers import SentenceTransformer

# Configure logging
logger = logging.getLogger("clash_semantic")
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Define the model we'll use (small and fast)
MODEL_NAME = 'all-MiniLM-L6-v2'
INDEX_FILE = "faiss_card_index.idx"
MAPPING_FILE = "faiss_card_mapping.json"


class SemanticRecommender:
    """
    Manages building, loading, and searching a FAISS vector index 
    for card descriptions.
    """
    def __init__(self, model_name: str = MODEL_NAME):
        """
        Initializes the model. The index is not loaded until .load_index() is called.
        """
        try:
            logger.info(f"Loading SentenceTransformer model '{model_name}'...")
            self.model = SentenceTransformer(model_name)
            logger.info(f"SentenceTransformer model '{model_name}' loaded.")
        except Exception as e:
            logger.error(f"Could not load SentenceTransformer model: {e}")
            self.model = None

        self.index = None
        self.mapping = []

    def build_index(self, cards_df: pd.DataFrame, save_dir: str):
        """
        Builds the FAISS index from the 'cards_data.csv' DataFrame
        and saves it to the specified directory.
        """
        if self.model is None:
            logger.error("Model not loaded. Cannot build index.")
            return

        logger.info("Starting to build FAISS index...")

        # 1. Prepare texts. We'll combine name, type, and description for rich embeddings.
        cards_df['description'] = cards_df['description'].fillna("No description available.")

        # Create a "document" for each card
        def create_card_doc(row):
            return (
                f"Card: {row['card_name']}. "
                f"Type: {row['card_type']}. "
                f"Elixir: {row['elixir_cost']}. "
                f"Description: {row['description']}"
            )

        cards_df['doc_text'] = cards_df.apply(create_card_doc, axis=1)

        self.mapping = cards_df['card_name'].tolist()
        corpus = cards_df['doc_text'].tolist()

        # 2. Encode texts into vectors
        logger.info(f"Encoding {len(corpus)} card documents into vectors...")
        embeddings = self.model.encode(
            corpus,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        # 3. Build and save FAISS index
        d = embeddings.shape[1]  # Get embedding dimension
        self.index = faiss.IndexFlatL2(d)  # Using L2 distance
        self.index.add(embeddings.astype(np.float32))  # Add vectors to index

        os.makedirs(save_dir, exist_ok=True)
        index_path = os.path.join(save_dir, INDEX_FILE)
        mapping_path = os.path.join(save_dir, MAPPING_FILE)

        logger.info(f"Saving index to {index_path}...")
        faiss.write_index(self.index, index_path)

        logger.info(f"Saving mapping to {mapping_path}...")
        with open(mapping_path, 'w') as f:
            json.dump(self.mapping, f)

        logger.info("Index build complete and files saved.")

    def load_index(self, load_dir: str):
        """
        Loads the pre-built FAISS index and card mapping from a directory.
        """
        index_path = os.path.join(load_dir, INDEX_FILE)
        mapping_path = os.path.join(load_dir, MAPPING_FILE)

        if not os.path.exists(index_path) or not os.path.exists(mapping_path):
            raise FileNotFoundError(
                f"Index files not found in {load_dir}. "
                "Run `src/build_semantic_index.py` first."
            )

        logger.info(f"Loading FAISS index from {index_path}...")
        self.index = faiss.read_index(index_path)

        logger.info(f"Loading card mapping from {mapping_path}...")
        with open(mapping_path, 'r') as f:
            self.mapping = json.load(f)

        logger.info("Semantic index and mapping loaded successfully.")

    def search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """
        Searches the index for the k-nearest neighbors to the query.
        
        Returns:
            A list of tuples: (card_name, distance_score)
        """
        if self.model is None or self.index is None:
            logger.error("Model or index not loaded. Cannot perform search.")
            return []

        # 1. Encode the query
        query_vector = self.model.encode(
            [query],
            convert_to_numpy=True
        ).astype(np.float32)

        # 2. Search the index
        # D = distances (float), I = indices (int)
        D, I = self.index.search(query_vector, k)

        # 3. Map indices back to card names
        results = []
        for i, idx in enumerate(I[0]):
            if idx >= 0:  # -1 indicates an empty slot
                card_name = self.mapping[idx]
                distance = float(D[0][i])
                results.append((card_name, distance))

        return results


# ----------------------------------------------------
# Optional: simple CLI / smoke test when run directly
# ----------------------------------------------------
if __name__ == "__main__":
    logger.info("SemanticRecommender module invoked directly.")

    # Resolve project root and models dir (same as build_semantic_index.py)
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

    rec = SemanticRecommender()

    try:
        # Load index from models/, not data/
        rec.load_index(MODELS_DIR)
    except FileNotFoundError as e:
        logger.error(str(e))
        logger.info(
            "Index files not found in models directory. "
            "Build the index via `python src/build_semantic_index.py`."
        )
    else:
        # Tiny demo search so you see real output
        demo_query = "cheap air swarm support card"
        logger.info(f"Running demo search for query: {demo_query!r}")
        results = rec.search(demo_query, k=5)
        for name, dist in results:
            logger.info(f"  {name}  (distance={dist:.4f})")

    logger.info("semantic_recommender.py finished.")
