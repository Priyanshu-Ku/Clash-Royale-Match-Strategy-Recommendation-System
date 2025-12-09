"""Feature engineering module for Clash Royale Match Strategy Recommendation System.

This module processes preprocessed data to generate ML-ready features including:
1.  Deck-level features (elixir cost, damage potential, defense strength)
2.  Counter metrics and win rates
3.  Mode-specific features (Attack/Defense)
4.  Card synergy and cohesion scores
5.  Deck embeddings for deep learning

Input:
- preprocessed_data.csv: Cleaned and merged dataset
- cards_data.csv: Card attributes
- match_history.csv: Battle outcomes

Output:
- attack_mode_features.csv: Features for attack mode strategies
- defense_mode_features.csv: Features for defense mode strategies
"""

import logging
import os
import ast
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# ----------------------------
# Configuration
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# File paths
DATA_DIR = "data"
PREPROCESSED_FILE = os.path.join(DATA_DIR, "preprocessed_data.csv")
CARDS_FILE = os.path.join(DATA_DIR, "cards_data.csv")
MATCH_HISTORY_FILE = os.path.join(DATA_DIR, "match_history.csv")
OUTPUT_ATTACK_FILE = os.path.join(DATA_DIR, "attack_mode_features.csv")
OUTPUT_DEFENSE_FILE = os.path.join(DATA_DIR, "defense_mode_features.csv")


# ----------------------------
# Helper Functions
# ----------------------------

def _get_card_row(card_table: pd.DataFrame, card_name: str) -> Optional[pd.Series]:
    """Safely get a card's data from the card table using fast index lookup."""
    if not isinstance(card_name, str):
        return None

    key = card_name.lower().strip()
    try:
        return card_table.loc[key]
    except KeyError:
        return None



def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all necessary datasets."""
    logger.info(f"Loading preprocessed data from {PREPROCESSED_FILE}")
    try:
        # low_memory=False avoids chunked, mixed-type inference
        df = pd.read_csv(PREPROCESSED_FILE, low_memory=False)
    except FileNotFoundError:
        logger.error(f"FATAL: Missing file {PREPROCESSED_FILE}. Run data_preprocessing.py first.")
        raise

    logger.info(f"Loading card data from {CARDS_FILE}")
    try:
        card_table = pd.read_csv(CARDS_FILE)
    except FileNotFoundError:
        logger.error(f"FATAL: Missing file {CARDS_FILE}.")
        raise
        
    logger.info(f"Loading match history from {MATCH_HISTORY_FILE}")
    try:
        match_history_df = pd.read_csv(MATCH_HISTORY_FILE)
    except FileNotFoundError:
        logger.error(f"FATAL: Missing file {MATCH_HISTORY_FILE}.")
        raise

    # Clean column names just in case
    df.columns = [c.lower().strip() for c in df.columns]
    card_table.columns = [c.lower().strip() for c in card_table.columns]
    match_history_df.columns = [c.lower().strip() for c in match_history_df.columns]

    # ---- Build fast lookup index for card_table ----
    if "card_name" not in card_table.columns:
        logger.error("FATAL: 'card_name' column missing from cards_data.csv")
        raise KeyError("card_name column missing in card_table")

    # Normalize card names once
    card_table["card_name_key"] = (
        card_table["card_name"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    # Set index to card_name_key for O(1) lookup
    card_table = card_table.set_index("card_name_key", drop=False)

    return df, card_table, match_history_df



# ----------------------------
# Heuristic Feature Functions (from Option 2)
# ----------------------------

def defense_strength(deck: Iterable[str], card_table: pd.DataFrame) -> float:
    """Estimate defensive capability of a deck. (Heuristic-based)"""
    score = 0.0
    for c in deck:
        r = _get_card_row(card_table, c)
        if r is None:
            continue
            
        hp = r.get("hitpoints", r.get("hp", np.nan))
        dps = r.get("dps", r.get("damage", np.nan))
        
        hp_val = float(hp) if not pd.isna(hp) else 0.0
        dps_val = float(dps) if not pd.isna(dps) else 0.0
        card_type = str(r.get("card_type", "")).lower()

        base = hp_val * 0.6 + dps_val * 0.4
        
        if card_type == "building":
            if "spawn" not in str(r.get("description", "")).lower() and "generate" not in str(r.get("description", "")).lower():
                 base *= 1.5
        
        if card_type == "troop" and hp_val > 1400:
            base *= 1.2
            
        score += base
    return float(score)


def synergy_score(deck: Iterable[str], card_table: pd.DataFrame) -> float:
    """Calculate a simple synergy score based on inferred role complementarity. (Heuristic-based)"""
    role_weights = {
        "wincon": 2.0, "support": 1.5, "tank": 1.2, "defense": 1.3,
        "cycle": 0.8, "spell": 1.0, "unknown": 0.1
    }
    counts: Dict[str, int] = {}
    roles_seen = []
    
    WINCON_NAMES = [
        "miner", "goblin drill", "skeleton barrel", "balloon", "graveyard",
        "x-bow", "mortar", "hog rider", "royal giant", "golem", "lava hound", "battle ram"
    ]

    for c in deck:
        r = _get_card_row(card_table, c)
        if r is None:
            continue
        
        key = "unknown"
        card_type = str(r.get("card_type", "")).lower()
        card_name = str(r.get("card_name", "")).lower()
        elixir = float(r.get("elixir_cost", 10))
        targets = str(r.get("targets", "")).lower()
        hp = float(r.get("hitpoints", 0))
        description = str(r.get("description", "")).lower()

        if card_type == "spell":
            key = "spell"
        elif card_type == "building":
            if "spawn" in description or "generate" in description:
                key = "support"
            elif any(wc in card_name for wc in ["x-bow", "mortar"]):
                key = "wincon"
            else:
                key = "defense"
        elif card_type == "troop":
            if "buildings" in targets or any(wc in card_name for wc in WINCON_NAMES):
                key = "wincon"
            elif hp > 1400:
                key = "tank"
            elif elixir <= 2:
                key = "cycle"
            else:
                key = "support"

        counts[key] = counts.get(key, 0) + 1
        roles_seen.append(key)

    score = 0.0
    for role, weight in role_weights.items():
        score += counts.get(role, 0) * weight

    total_cards = len(roles_seen) if roles_seen else 1
    max_role_count = max(counts.values()) if counts else 0
    diversity = len(counts) / total_cards
    
    score = score * (0.8 + 0.4 * diversity)
    
    if total_cards > 0 and (max_role_count / total_cards) > 0.6:
        score *= 0.7
        
    return float(score)


# ----------------------------
# Core Feature Functions
# ----------------------------

def compute_deck_stats(deck: List[str], card_table: pd.DataFrame) -> Dict[str, float]:
    """Compute comprehensive deck statistics."""
    stats = {
        'avg_elixir': 0.0, 'total_dps': 0.0, 'avg_dps': 0.0,
        'defense_strength': 0.0, 'offense_strength': 0.0,
        'spell_ratio': 0.0, 'troop_ratio': 0.0, 'building_ratio': 0.0,
        'avg_hitpoints': 0.0
    }
    
    valid_cards = []
    for card in deck:
        row = _get_card_row(card_table, card)
        if row is not None:
            valid_cards.append(row)
            
    if not valid_cards:
        return stats
        
    deck_df = pd.DataFrame(valid_cards)
    
    stats['avg_elixir'] = deck_df['elixir_cost'].mean()
    stats['total_dps'] = deck_df['dps'].fillna(0).sum()
    stats['avg_dps'] = deck_df['dps'].fillna(0).mean()
    
    total_cards = len(deck_df)
    if total_cards > 0:
        stats['spell_ratio'] = len(deck_df[deck_df['card_type'] == 'Spell']) / total_cards
        stats['troop_ratio'] = len(deck_df[deck_df['card_type'] == 'Troop']) / total_cards
        stats['building_ratio'] = len(deck_df[deck_df['card_type'] == 'Building']) / total_cards

    # Use the new heuristic-based defense strength
    stats['defense_strength'] = defense_strength(deck, card_table)
    
    # Offense strength (weighted sum of damage and offensive capabilities)
    stats['offense_strength'] = (
        deck_df['dps'].fillna(0).mean() * 0.7 +
        deck_df['damage'].fillna(0).mean() * 0.3
    )
    
    non_spell_df = deck_df[deck_df['card_type'] != 'Spell']
    if not non_spell_df.empty:
        stats['avg_hitpoints'] = non_spell_df['hitpoints'].fillna(0).mean()
    
    return stats


def compute_synergy_scores(
    deck: List[str],
    card_table: pd.DataFrame,
    match_history_df: pd.DataFrame
) -> Dict[str, float]:
    """Compute card synergy and cohesion scores for a deck."""
    
    # Use the new heuristic-based synergy score
    cohesion = synergy_score(deck, card_table)
    
    # Type synergy (good mix of troops, spells, buildings)
    types = []
    for card in deck:
        row = _get_card_row(card_table, card)
        if row is not None:
            types.append(row.get('card_type', 'unknown'))
    
    type_counts = pd.Series(types).value_counts()
    type_synergy = len(type_counts) / 3.0  # Perfect score if all 3 types present

    # Win synergy (historical performance of card combinations)
    win_rates = []
    # This is computationally expensive. For production, this should be pre-calculated.
    # For this script, we'll simplify it.
    # A full implementation would check all 28 pairs. We'll sample a few.
    if not match_history_df.empty:
        # Simplified: check win rate of the whole deck if it exists
        # A more complex version would check card pairs.
        # This part is a placeholder for a more complex co-occurrence calculation.
        win_synergy = 0.5 # Default
    else:
        win_synergy = 0.5

    return {
        'cohesion': cohesion,
        'type_synergy': type_synergy,
        'win_synergy': win_synergy
    }


def create_deck_embedding(
    deck: List[str],
    card_table: pd.DataFrame,
    embedding_size: int = 16
) -> np.ndarray:
    """Create a fixed-size numeric embedding for a deck by averaging card stats."""
    features = []
    for card in deck:
        row = _get_card_row(card_table, card)
        if row is not None:
            card_features = [
                row.get('elixir_cost', 0),
                row.get('dps', 0),
                row.get('hitpoints', 0),
                row.get('damage', 0),
                1 if row.get('card_type') == 'Troop' else 0,
                1 if row.get('card_type') == 'Spell' else 0,
                1 if row.get('card_type') == 'Building' else 0,
            ]
            features.append(card_features)
    
    if not features:
        return np.zeros(embedding_size)
    
    # Average the features of all cards in the deck
    avg_features = np.mean(features, axis=0)
    
    # Normalize
    avg_features = avg_features / (np.linalg.norm(avg_features) + 1e-8)
    
    # Pad or truncate to embedding size
    if len(avg_features) > embedding_size:
        deck_vector = avg_features[:embedding_size]
    else:
        deck_vector = np.pad(avg_features, (0, embedding_size - len(avg_features)))
        
    return deck_vector


def compute_counter_metrics(
    data_df: pd.DataFrame
) -> pd.DataFrame:
    """Compute deck archetype counter matrix."""
    
    # This function now receives a dataframe that has
    # reconstructed 'player_archetype' and 'opponent_archetype' columns.
    
    if 'player_archetype' in data_df.columns and 'opponent_archetype' in data_df.columns:
        logger.info("Computing archetype counter metrics...")
        
        # We also need a 'winner' flag. Let's use 'winner_flag' from preprocessed_data.csv
        if 'winner_flag' not in data_df.columns:
            logger.warning("No 'winner_flag' column found. Cannot compute counter metrics.")
            return pd.DataFrame()
            
        try:
            archetype_counters = pd.pivot_table(
                data_df,
                values='winner_flag',
                index='player_archetype',
                columns='opponent_archetype',
                aggfunc='mean'
            ).fillna(0.5)
            logger.info("Successfully computed archetype counter pivot table.")
            return archetype_counters
        except Exception as e:
            logger.warning(f"Failed to create pivot table for counter metrics: {e}")
            return pd.DataFrame()
    else:
        logger.warning("Could not find reconstructed 'player_archetype' or 'opponent_archetype' columns. Skipping counter metrics.")
        return pd.DataFrame()


# ----------------------------
# Main Pipeline
# ----------------------------

def generate_feature_matrix(
    df: pd.DataFrame,
    card_table: pd.DataFrame,
    match_history_df: pd.DataFrame,
    archetype_counters: pd.DataFrame,  # Pass counters in
    mode: str,
    embedding_size: int = 16
) -> pd.DataFrame:
    """Generate complete feature matrix for a specific mode."""

    features = []

    # Get list of all archetype columns for one-hot encoding
    archetype_cols = [c for c in df.columns if 'player_archetype_' in c or 'opponent_archetype_' in c]

    # Only keep columns we actually use in the loop – avoid giant object arrays
    base_cols = [
        'cards',
        'tag',
        'player_archetype',
        'opponent_archetype',
        'trophies',
        'explevel',
    ]
    used_cols = [c for c in (base_cols + archetype_cols) if c in df.columns]
    df_loop = df[used_cols].copy()

    # Local bindings (micro-optimizations, but cheap and clean)
    compute_deck_stats_local = compute_deck_stats
    compute_synergy_scores_local = compute_synergy_scores
    create_deck_embedding_local = create_deck_embedding
    features_append = features.append
    counters = archetype_counters

    # Fast row iteration
    for row in df_loop.itertuples(index=False):
        # row.<colname> for access, e.g. row.cards, row.tag, etc.

        cards_str = getattr(row, 'cards', None)
        if not isinstance(cards_str, str):
            logger.warning("Deck 'cards' field is not a string. Skipping row.")
            continue

        try:
            deck = ast.literal_eval(cards_str)
        except (ValueError, SyntaxError):
            # Can't parse deck, skip this row
            logger.warning("Failed to parse deck from 'cards' field. Skipping row.")
            continue

        # 1. Basic deck stats
        deck_stats = compute_deck_stats_local(deck, card_table)

        # 2. Synergy scores
        synergy = compute_synergy_scores_local(deck, card_table, match_history_df)

        # 3. Deck embedding
        embedding = create_deck_embedding_local(deck, card_table, embedding_size)

        # 4. Counter metrics (Using the new pivot table)
        player_arch = getattr(row, 'player_archetype', 'unknown') or 'unknown'
        opp_arch = getattr(row, 'opponent_archetype', 'unknown') or 'unknown'

        counter_score = 0.5  # Default
        if not counters.empty:
            if (player_arch in counters.index) and (opp_arch in counters.columns):
                counter_score = counters.loc[player_arch, opp_arch]
            elif player_arch in counters.index:
                # Fallback: average win rate for player's archetype
                counter_score = counters.loc[player_arch].mean()

        # 5. Combine all features
        feature_dict = {
            'player_tag': getattr(row, 'tag', None),
            'mode': mode,
            **deck_stats,
            **synergy,
            'counter_score': counter_score,
            **{f'embedding_{i}': v for i, v in enumerate(embedding)}
        }

        # Add player profile features
        feature_dict['trophies'] = getattr(row, 'trophies', 0) or 0
        feature_dict['explevel'] = getattr(row, 'explevel', 0) or 0

        # Add one-hot encoded archetypes
        for col in archetype_cols:
            if col in df_loop.columns:
                feature_dict[col] = getattr(row, col, 0) or 0
            else:
                feature_dict[col] = 0

        features_append(feature_dict)

    if not features:
        logger.warning(f"No features generated for mode: {mode}")
        return pd.DataFrame()

    feature_df = pd.DataFrame(features)

    # Scale numeric features
    numeric_features = [
        'avg_elixir', 'total_dps', 'avg_dps', 'defense_strength',
        'offense_strength', 'avg_hitpoints', 'cohesion', 'type_synergy',
        'win_synergy', 'counter_score', 'trophies', 'explevel'
    ]
    # Add embedding features to numeric list
    numeric_features.extend([f'embedding_{i}' for i in range(embedding_size)])

    # Filter out columns that don't exist
    numeric_features = [col for col in numeric_features if col in feature_df.columns]

    if not numeric_features:
        logger.error(f"No numeric features found to scale for mode: {mode}")
        return feature_df

    # Impute NaNs with median before scaling
    imputer = SimpleImputer(strategy='median')
    feature_df[numeric_features] = imputer.fit_transform(feature_df[numeric_features])

    scaler = StandardScaler()
    feature_df[numeric_features] = scaler.fit_transform(feature_df[numeric_features])

    return feature_df


def process_dataset(
    df: pd.DataFrame,
    card_table: pd.DataFrame,
    match_history_df: pd.DataFrame,
    embedding_size: int = 16
) -> None:
    """Process the preprocessed dataset and generate feature matrices."""
    
    # --- Reconstruct game_mode column ---
    game_mode_cols = [c for c in df.columns if c.startswith('game_mode_')]
    if not game_mode_cols:
        logger.error("FATAL: No 'game_mode_' columns found. Cannot determine game mode.")
        return
    logger.info("Reconstructing 'game_mode' column...")
    valid_game_mode_cols = [c for c in game_mode_cols if c != 'game_mode_nan']
    df['game_mode_encoded'] = df[valid_game_mode_cols].idxmax(axis=1)
    all_false_mask = (df[valid_game_mode_cols].sum(axis=1) == 0)
    df['game_mode'] = df['game_mode_encoded'].str.replace('game_mode_', '', regex=False)
    df.loc[all_false_mask, 'game_mode'] = 'Unknown'
    df['game_mode'] = df['game_mode'].replace('nan', 'Unknown').fillna('Unknown')
    
    # --- START OF FIX: Reconstruct archetype columns ---
    
    # Reconstruct Player Archetype
    player_arch_cols = [c for c in df.columns if c.startswith('player_archetype_')]
    if player_arch_cols:
        logger.info("Reconstructing 'player_archetype' column...")
        valid_player_cols = [c for c in player_arch_cols if c != 'player_archetype_nan']
        df['player_arch_encoded'] = df[valid_player_cols].idxmax(axis=1)
        player_false_mask = (df[valid_player_cols].sum(axis=1) == 0)
        df['player_archetype'] = df['player_arch_encoded'].str.replace('player_archetype_', '', regex=False)
        df.loc[player_false_mask, 'player_archetype'] = 'unknown'
        df['player_archetype'] = df['player_archetype'].replace('nan', 'unknown').fillna('unknown')
    else:
        logger.warning("No 'player_archetype_' columns found. Using 'unknown'.")
        df['player_archetype'] = 'unknown'

    # Reconstruct Opponent Archetype
    opp_arch_cols = [c for c in df.columns if c.startswith('opponent_archetype_')]
    if opp_arch_cols:
        logger.info("Reconstructing 'opponent_archetype' column...")
        valid_opp_cols = [c for c in opp_arch_cols if c != 'opponent_archetype_nan']
        df['opp_arch_encoded'] = df[valid_opp_cols].idxmax(axis=1)
        opp_false_mask = (df[valid_opp_cols].sum(axis=1) == 0)
        df['opponent_archetype'] = df['opp_arch_encoded'].str.replace('opponent_archetype_', '', regex=False)
        df.loc[opp_false_mask, 'opponent_archetype'] = 'unknown'
        df['opponent_archetype'] = df['opponent_archetype'].replace('nan', 'unknown').fillna('unknown')
    else:
        logger.warning("No 'opponent_archetype_' columns found. Using 'unknown'.")
        df['opponent_archetype'] = 'unknown'
        
    # --- END OF FIX ---

    # Split by mode
    attack_mask = df['game_mode'].str.contains(
        'ladder|challenge|ranked|showdown|competitive', 
        case=False
    )
    defense_mask = df['game_mode'].str.contains(
        'friendly|cw|war|boatbattle|team|draft|duel', 
        case=False
    )
    
    attack_df = df[attack_mask].copy()
    defense_df = df[defense_mask].copy()
    
    logger.info(f"Split dataset: {len(attack_df)} attack rows, {len(defense_df)} defense rows.")

    # --- Compute Counter Metrics *After* Splitting ---
    # This creates mode-specific counter tables
    attack_counters = compute_counter_metrics(attack_df)
    defense_counters = compute_counter_metrics(defense_df)

    if attack_df.empty:
        logger.warning("No data found for Attack mode.")
    else:
        logger.info("Computing features for attack mode...")
        attack_features = generate_feature_matrix(
            attack_df,
            card_table,
            match_history_df, # Pass full history for synergy (if needed)
            attack_counters,  # Pass the mode-specific counters
            mode='attack',
            embedding_size=embedding_size
        )
        if not attack_features.empty:
            attack_features.to_csv(OUTPUT_ATTACK_FILE, index=False)
            logger.info(f"Attack mode features saved to {OUTPUT_ATTACK_FILE}")
            logger.info(f"Attack features shape: {attack_features.shape}")
    
    if defense_df.empty:
        logger.warning("No data found for Defense mode.")
    else:
        logger.info("Computing features for defense mode...")
        defense_features = generate_feature_matrix(
            defense_df,
            card_table,
            match_history_df, # Pass full history for synergy (if needed)
            defense_counters, # Pass the mode-specific counters
            mode='defense',
            embedding_size=embedding_size
        )
        if not defense_features.empty:
            defense_features.to_csv(OUTPUT_DEFENSE_FILE, index=False)
            logger.info(f"Defense mode features saved to {OUTPUT_DEFENSE_FILE}")
            logger.info(f"Defense features shape: {defense_features.shape}")


def main():
    """Main execution function."""
    try:
        logger.info("Starting feature engineering process...")
        
        # 1. Load data
        df, card_table, match_history_df = load_data()
        
        # Ensure dataframes have same index for mode-splitting
        # We assume preprocessed_data.csv is the main driver
        # and match_history was merged into it.
        # For simplicity, we'll align match_history to df index.
        # A better way is to merge them on 'tag' and 'battletime'
        
        # For this script, we'll just pass the full match_history_df
        # and filter it inside generate_feature_matrix.
        # But `process_dataset` already filters. Let's align them.
        
        # This is tricky. `preprocessed_data.csv` is 3063 rows.
        # `match_history.csv` is 10k+ rows.
        # The 'game_mode' split should happen on the *source* of game_mode.
        # `preprocessed_data.csv` *has* 'game_mode' columns.
        
        # We will use `df` (from preprocessed_data.csv) as the main iterator
        # and pass the *full* `match_history_df` for metric calculations.
        
        # 2. Process dataset
        process_dataset(df, card_table, match_history_df)
        
        logger.info("\nFeature engineering completed successfully!")
        
    except FileNotFoundError:
        logger.error("A required data file was not found. Aborting.")
    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()

