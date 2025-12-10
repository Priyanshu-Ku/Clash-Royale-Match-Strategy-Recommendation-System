"""
One-time script to build and save the FAISS index for semantic search.

This script loads the `cards_data.csv`, generates embeddings for all
card descriptions using the SemanticRecommender class, and saves
the index and mapping files to the `models/` directory.
"""

import os
import pandas as pd
import logging
import sys

# Add src to path to import our class + constants
try:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    SRC_DIR = os.path.join(PROJECT_ROOT, "src")
    if SRC_DIR not in sys.path:
        sys.path.append(SRC_DIR)
    from semantic_recommender import SemanticRecommender, INDEX_FILE, MAPPING_FILE
except ImportError:
    print("Error: Could not import SemanticRecommender. Make sure src/semantic_recommender.py exists.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- File Paths ---
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
CARDS_FILE = os.path.join(DATA_DIR, "cards_data.csv")

if __name__ == "__main__":
    logger.info("--- Starting Semantic Index Build ---")
    
    # 1. Create models directory if it doesn't exist
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # 2. Load card data
    try:
        cards_df = pd.read_csv(CARDS_FILE)
        logger.info(f"Loaded {len(cards_df)} cards from {CARDS_FILE}")
    except FileNotFoundError:
        logger.error(f"FATAL: {CARDS_FILE} not found. Cannot build index.")
        sys.exit(1)
        
    # 3. Initialize recommender and build index
    try:
        recommender = SemanticRecommender()
        recommender.build_index(cards_df, save_dir=MODELS_DIR)
        
        logger.info("\n--- Build Complete ---")
        logger.info(f"Files saved to {MODELS_DIR}:")
        logger.info(f"- {os.path.join(MODELS_DIR, INDEX_FILE)}")
        logger.info(f"- {os.path.join(MODELS_DIR, MAPPING_FILE)}")
        
        logger.info("\n--- Testing Search ---")
        test_query = "fast ground troop"
        results = recommender.search(test_query, k=3)
        logger.info(f"Test query: '{test_query}'")
        for card, dist in results:
            logger.info(f"  -> {card} (Distance: {dist:.4f})")
            
    except Exception as e:
        logger.error(f"An error occurred during index build: {e}", exc_info=True)
        sys.exit(1)
