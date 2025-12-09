"""Data preprocessing module for Clash Royale Match Strategy Recommendation System.

This module handles loading, cleaning, and preprocessing of multiple CSV datasets:
- clash_data.csv: Player profile data
- deck_features.csv: Deck composition and stats
- strategies_clean.csv: Strategy classifications and metrics
- match_history.csv: Battle outcomes and timestamps
- cards_data.csv: Card attributes and descriptions

The module performs cleaning operations including:
- Missing value imputation
- Categorical encoding
- Date parsing
- Text standardization
- Dataset merging
"""

import os
import json
import logging
from typing import Tuple, List, Optional
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ----------------------------
# File paths
# ----------------------------
DATA_DIR = "data"
CLASH_DATA_FILE = os.path.join(DATA_DIR, "clash_data.csv")
DECK_FEATURES_FILE = os.path.join(DATA_DIR, "deck_features.csv")
STRATEGIES_FILE = os.path.join(DATA_DIR, "strategies_clean.csv")
MATCH_HISTORY_FILE = os.path.join(DATA_DIR, "match_history.csv")
CARDS_FILE = os.path.join(DATA_DIR, "cards_data.csv")

# ----------------------------
# Load datasets
# ----------------------------
def load_datasets() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all required datasets from the data directory.
    
    Returns:
        Tuple[pd.DataFrame, ...]: Tuple containing DataFrames in order:
            (clash_data, deck_features, strategies, match_history, cards)
            
    Raises:
        FileNotFoundError: If any required dataset file is missing
        pd.errors.EmptyDataError: If any dataset is empty
    """
    try:
        clash_df = pd.read_csv(CLASH_DATA_FILE)
        deck_df = pd.read_csv(DECK_FEATURES_FILE)
        strategies_df = pd.read_csv(STRATEGIES_FILE)
        match_history_df = pd.read_csv(MATCH_HISTORY_FILE)
        cards_df = pd.read_csv(CARDS_FILE)
        
        # Validate that datasets are not empty
        for df, name in zip(
            [clash_df, deck_df, strategies_df, match_history_df, cards_df],
            ["clash_data", "deck_features", "strategies", "match_history", "cards"]
        ):
            if df.empty:
                raise pd.errors.EmptyDataError(f"Dataset {name} is empty")
        
        return clash_df, deck_df, strategies_df, match_history_df, cards_df
        
    except FileNotFoundError as e:
        logger.error(f"Failed to load datasets: {str(e)}")
        raise
    except pd.errors.EmptyDataError as e:
        logger.error(f"Empty dataset detected: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading datasets: {str(e)}")
        raise

# ----------------------------
# Clean clash_data.csv
# ----------------------------
def preprocess_clash_data(clash_df: pd.DataFrame) -> Tuple[pd.DataFrame, LabelEncoder]:
    """Preprocess player profile data from clash_data.csv.
    
    Args:
        clash_df: DataFrame containing player profiles
        
    Returns:
        Tuple containing:
        - Preprocessed DataFrame
        - LabelEncoder for arena encoding
        
    Raises:
        ValueError: If required columns are missing
    """
    required_cols = {'tag', 'trophies', 'explevel', 'arena', 'clan'}
    missing_cols = required_cols - set(clash_df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns in clash_data: {missing_cols}")

    # Create copy to avoid modifying original
    clash_df = clash_df.copy()
    
    # Standardize column names
    clash_df.columns = [c.lower().strip() for c in clash_df.columns]

    # Fill missing values
    clash_df['trophies'] = clash_df['trophies'].fillna(clash_df['trophies'].median())
    clash_df['explevel'] = clash_df['explevel'].fillna(clash_df['explevel'].median())
    clash_df['arena'] = clash_df['arena'].fillna('Unknown')
    clash_df['clan'] = clash_df['clan'].fillna('NoClan')

    # Optional: encode arena as numeric using LabelEncoder
    le_arena = LabelEncoder()
    clash_df['arena_encoded'] = le_arena.fit_transform(clash_df['arena'])

    return clash_df, le_arena

# ----------------------------
# Clean deck_features.csv
# ----------------------------
def preprocess_deck_features(deck_df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess deck composition data from deck_features.csv.
    
    Args:
        deck_df: DataFrame containing deck compositions
        
    Returns:
        Preprocessed DataFrame with standardized columns and parsed card lists
        
    Raises:
        ValueError: If required columns are missing or card list parsing fails
    """
    required_cols = {'playerTag', 'cards', 'averageElixir'}
    missing_cols = required_cols - set(deck_df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns in deck_features: {missing_cols}")
    
    # Create copy to avoid modifying original
    deck_df = deck_df.copy()
    
    deck_df.columns = [c.lower().strip() for c in deck_df.columns]

    # Fill missing values
    deck_df['averageelixir'] = deck_df['averageelixir'].fillna(deck_df['averageelixir'].median())

    # Convert cards column from string to list if stored as string
    try:
        if deck_df['cards'].dtype == object:
            deck_df['cards'] = deck_df['cards'].apply(lambda x: x.strip("[]").replace("'", "").split(", "))
    except Exception as e:
        logger.error("Failed to parse card lists")
        raise ValueError(f"Failed to parse card lists: {str(e)}")

    return deck_df

# ----------------------------
# Clean strategies_clean.csv
# ----------------------------
def preprocess_strategies(strategies_df):
    strategies_df.columns = [c.lower().strip() for c in strategies_df.columns]

    # Fill missing numeric values
    num_cols = ['avg_elixir', 'est_elixir_per_min', 'synergy', 'cohesion', 'mode_difficulty', 'winner_flag']
    for col in num_cols:
        if col in strategies_df.columns:
            strategies_df[col] = strategies_df[col].fillna(strategies_df[col].median())

    # Fill missing archetypes with 'Unknown'
    archetype_cols = [c for c in strategies_df.columns if 'player_archetype' in c or 'opponent_archetype' in c]
    for col in archetype_cols:
        strategies_df[col] = strategies_df[col].fillna('Unknown')

    # One-hot encode game modes
    game_mode_cols = [c for c in strategies_df.columns if 'game_mode' in c]
    strategies_df[game_mode_cols] = strategies_df[game_mode_cols].fillna(0)  # Replace NaN with 0

    return strategies_df

# ----------------------------
# Clean match_history.csv
# ----------------------------
def preprocess_match_history(match_history_df):
    match_history_df.columns = [c.lower().strip() for c in match_history_df.columns]

    # Convert winner column to binary
    match_history_df['winner'] = match_history_df['winner'].map({True:1, False:0})

    # Fill missing gameMode
    match_history_df['gamemode'] = match_history_df['gamemode'].fillna('Unknown')

    # Convert battleTime to datetime
    match_history_df['battletime'] = pd.to_datetime(match_history_df['battletime'], errors='coerce')

    return match_history_df

# ----------------------------
# Clean cards_data.csv (for LLM usage)
# ----------------------------
def preprocess_cards(cards_df):
    cards_df.columns = [c.lower().strip() for c in cards_df.columns]

    # Fill missing numeric values
    numeric_cols = ['elixir_cost','range','hit_speed','hitpoints','damage','dps','count']
    for col in numeric_cols:
        cards_df[col] = cards_df[col].fillna(0)

    # Fill missing categorical/text
    text_cols = ['card_type','rarity','targets','damage_type','speed','spawn_effect','special_ability','description']
    for col in text_cols:
        cards_df[col] = cards_df[col].fillna('Unknown')

    # Optional: Create LLM-friendly text column combining description + abilities
    cards_df['llm_text'] = cards_df.apply(lambda row: f"{row['card_name']} is a {row['rarity']} {row['card_type']} card with {row['description']}. Special ability: {row['special_ability']}", axis=1)

    return cards_df

# ----------------------------
# Merge datasets (without feature engineering)
# ----------------------------
def merge_datasets(
    clash_df: pd.DataFrame,
    deck_df: pd.DataFrame,
    strategies_df: pd.DataFrame,
    match_history_df: pd.DataFrame,
    cards_df: pd.DataFrame
) -> pd.DataFrame:
    """Merge all preprocessed datasets on player tag.
    
    Args:
        clash_df: Player profile data
        deck_df: Deck composition data
        strategies_df: Strategy classification data
        match_history_df: Battle history data
        cards_df: Card attribute data
        
    Returns:
        Merged DataFrame with all relevant features
        
    Note:
        - No feature engineering is performed in this step
        - cards_df is kept separate by default but available for future feature generation
    """
    # Validate input DataFrames
    if any(df.empty for df in [clash_df, deck_df, strategies_df, match_history_df, cards_df]):
        raise ValueError("One or more input DataFrames are empty")
    
    try:
        # Merge clash data with deck features on tag/playerTag
        deck_df_renamed = deck_df.rename(columns={'playertag':'tag'})
        merged_df = pd.merge(clash_df, deck_df_renamed, on='tag', how='left')

        # Merge strategies
        strategies_df_renamed = strategies_df.rename(columns={'playertag':'tag'})
        merged_df = pd.merge(merged_df, strategies_df_renamed, on='tag', how='left')

        # Log merge statistics
        logger.info(f"Final merged dataset shape: {merged_df.shape}")
        logger.info(f"Number of unique players: {merged_df['tag'].nunique()}")
        
        return merged_df
        
    except Exception as e:
        logger.error("Failed to merge datasets")
        raise ValueError(f"Dataset merge failed: {str(e)}")

# ----------------------------
# Main preprocessing function
# ----------------------------
def main(output_file: str = "preprocessed_data.csv") -> None:
    """Main function to run all preprocessing steps.
    
    Args:
        output_file: Name of output CSV file (default: "preprocessed_data.csv")
        
    Raises:
        Exception: If any preprocessing step fails
    """
    try:
        logger.info("Loading datasets...")
        clash_df, deck_df, strategies_df, match_history_df, cards_df = load_datasets()

        logger.info("Preprocessing clash_data.csv...")
        clash_df, le_arena = preprocess_clash_data(clash_df)

        logger.info("Preprocessing deck_features.csv...")
        deck_df = preprocess_deck_features(deck_df)

        logger.info("Preprocessing strategies_clean.csv...")
        strategies_df = preprocess_strategies(strategies_df)

        logger.info("Preprocessing match_history.csv...")
        match_history_df = preprocess_match_history(match_history_df)

        logger.info("Preprocessing cards_data.csv...")
        cards_df = preprocess_cards(cards_df)

        logger.info("Merging datasets...")
        preprocessed_df = merge_datasets(clash_df, deck_df, strategies_df, match_history_df, cards_df)

        # Save final preprocessed data
        output_path = os.path.join(DATA_DIR, output_file)
        preprocessed_df.to_csv(output_path, index=False)
        logger.info(f"Preprocessing complete. Saved as {output_file}")
        
        # Log some statistics
        logger.info("Dataset statistics:")
        logger.info(f"Total rows: {len(preprocessed_df)}")
        logger.info(f"Total columns: {len(preprocessed_df.columns)}")
        logger.info(f"Memory usage: {preprocessed_df.memory_usage().sum() / 1024**2:.2f} MB")
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
