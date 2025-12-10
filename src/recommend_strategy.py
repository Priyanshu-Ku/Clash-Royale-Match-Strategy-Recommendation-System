import logging
import os
import pandas as pd
import joblib
import numpy as np
import ast
from typing import List, Dict, Optional, Tuple

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
MODELS_DIR = "models"
CARDS_FILE = os.path.join(DATA_DIR, "cards_data.csv")
# --- Load BOTH models ---
WIN_SEEKER_MODEL_FILE = os.path.join(MODELS_DIR, "strategy_model_win_seeker.pkl")
LOSS_AVOIDER_MODEL_FILE = os.path.join(MODELS_DIR, "strategy_model_loss_avoider.pkl")
# --- Need feature names the model expects ---
# Load one of the feature files just to get column names reliably
try:
    ATTACK_FEATURES_EXAMPLE_FILE = os.path.join(DATA_DIR, "attack_mode_features.csv")
    FEATURES_DF_EXAMPLE = pd.read_csv(ATTACK_FEATURES_EXAMPLE_FILE, nrows=1)
    # Drop target and ID columns that are not features
    EXPECTED_FEATURES = list(FEATURES_DF_EXAMPLE.drop(columns=['player_tag', 'winner_flag'], errors='ignore').columns)
    # Store known archetype columns separately for easier one-hot encoding
    KNOWN_OPPONENT_ARCHETYPES = [col for col in EXPECTED_FEATURES if col.startswith('opponent_archetype_')]
    KNOWN_PLAYER_ARCHETYPES = [col for col in EXPECTED_FEATURES if col.startswith('player_archetype_')] # Needed if player archetype becomes an input later
    logger.info(f"Loaded expected feature names. Found {len(KNOWN_OPPONENT_ARCHETYPES)} opponent archetypes.")
except FileNotFoundError:
    logger.error(f"Cannot load example feature file '{ATTACK_FEATURES_EXAMPLE_FILE}' to get feature names. Recommendation will likely fail.")
    EXPECTED_FEATURES = []
    KNOWN_OPPONENT_ARCHETYPES = []
    KNOWN_PLAYER_ARCHETYPES = []
except Exception as e:
    logger.error(f"Error loading example feature file: {e}")
    EXPECTED_FEATURES = []
    KNOWN_OPPONENT_ARCHETYPES = []
    KNOWN_PLAYER_ARCHETYPES = []


# ----------------------------------------------------
# Helper Functions (Adapted from feature_engineering.py)
# ----------------------------------------------------

def load_card_table(path: str) -> Optional[pd.DataFrame]:
    """Load card attributes table from CSV file."""
    try:
        logger.info(f"Loading card table from {path}")
        df = pd.read_csv(path)
        # Ensure card names are lowercase for consistent matching
        if 'card_name' in df.columns:
            # --- Store original name before lowercasing for better matching ---
            df['card_name_original'] = df['card_name']
            df['card_name'] = df['card_name'].str.lower()
        else:
            logger.error("FATAL: 'card_name' column not found in cards data.")
            return None
        return df
    except FileNotFoundError:
        logger.error(f"FATAL: Card data file not found at {path}")
        return None
    except Exception as e:
        logger.error(f"FATAL: Error loading card data: {e}")
        return None

# --- UPDATED CARD LOOKUP LOGIC ---
def _get_card_row(card_table: pd.DataFrame, card_name: str) -> Optional[pd.Series]:
    """
    Safely get card data row using flexible matching.
    Tries exact match first, then checks if a card name in the table
    *starts with* the provided card_name (ignoring case and suffixes).
    """
    if not isinstance(card_name, str) or not card_name:
        return None

    card_name_lower = card_name.lower()

    try:
        # 1. Try exact match first (most reliable)
        exact_match = card_table[card_table["card_name"] == card_name_lower]
        if not exact_match.empty:
            # logger.debug(f"Exact match found for '{card_name}'")
            return exact_match.iloc[0]

        # 2. If no exact match, try startswith match
        # This handles cases like "Archers" vs "Archers (Evolution)"
        # We match against the original name to avoid lowercasing issues with startswith if not needed
        # but compare the lowercase input `card_name_lower`
        partial_match = card_table[card_table["card_name"].str.startswith(card_name_lower)]

        if not partial_match.empty:
            # If multiple partial matches (e.g., "Goblin" matching "Goblins", "Goblin Gang", "Goblin Barrel"),
            # prefer the shortest match as it's likely the base card.
            # Calculate length difference
            partial_match['len_diff'] = partial_match['card_name'].str.len() - len(card_name_lower)
            # Find the minimum length difference
            min_len_diff = partial_match['len_diff'].min()
            # Select rows with the minimum length difference
            best_partial_match = partial_match[partial_match['len_diff'] == min_len_diff]

            # If still multiple, log a warning and pick the first
            if len(best_partial_match) > 1:
                 logger.warning(f"Multiple partial matches found for '{card_name}'. Using first match: {best_partial_match.iloc[0]['card_name_original']}")

            logger.debug(f"Partial match found for '{card_name}': {best_partial_match.iloc[0]['card_name_original']}")
            return best_partial_match.iloc[0]

        # 3. If still no match
        logger.warning(f"Card '{card_name}' not found in card table using exact or partial match.")
        return None

    except Exception as e:
        logger.error(f"Error looking up card '{card_name}': {e}")
        return None
# --- END UPDATED CARD LOOKUP LOGIC ---


def calculate_deck_features(deck_list: List[str], card_table: pd.DataFrame) -> Dict[str, float]:
    """Calculate various numeric features for a given deck list."""
    features = {
        'avg_elixir': np.nan,
        'total_dps': 0.0,
        'avg_dps': 0.0,
        'defense_strength': 0.0,
        'offense_strength': 0.0,
        'spell_ratio': 0.0,
        'troop_ratio': 0.0,
        'building_ratio': 0.0,
        'avg_hitpoints': 0.0,
        'synergy_score': 0.0, # Placeholder, calculated separately if needed by models
        'cohesion': 0.0, # Placeholder, potentially complex
        'type_synergy': 0.0, # Placeholder
        'win_synergy': 0.0 # Placeholder
        # Add other simple features if present in EXPECTED_FEATURES
    }
    valid_cards_data = []
    elixirs = []
    dps_list = []
    damage_list = []
    hp_list = []
    types = {'spell': 0, 'troop': 0, 'building': 0}

    for card_name in deck_list:
        card_data = _get_card_row(card_table, card_name)
        if card_data is not None:
            valid_cards_data.append(card_data)
            elixirs.append(card_data.get('elixir_cost', np.nan))
            dps_list.append(card_data.get('dps', 0) or 0) # Ensure 0 if NaN/None
            damage_list.append(card_data.get('damage', 0) or 0)
            card_type = str(card_data.get('card_type', '')).lower()
            if card_type in types:
                types[card_type] += 1
            if card_type != 'spell':
                hp_list.append(card_data.get('hitpoints', 0) or 0)

    num_cards = len(valid_cards_data)
    if num_cards == 0:
        logger.warning("Deck list resulted in no valid cards found.")
        return features # Return defaults
    # --- Log if some cards were not found ---
    elif num_cards < len(deck_list):
         logger.warning(f"Processed {num_cards}/{len(deck_list)} cards from the input deck. Some cards were not found.")


    # Calculate features
    valid_elixirs = [e for e in elixirs if not pd.isna(e)]
    if valid_elixirs:
        features['avg_elixir'] = np.mean(valid_elixirs)

    features['total_dps'] = np.sum(dps_list)
    features['avg_dps'] = np.mean(dps_list) if dps_list else 0.0

    # Calculate ratios
    features['spell_ratio'] = types['spell'] / num_cards
    features['troop_ratio'] = types['troop'] / num_cards
    features['building_ratio'] = types['building'] / num_cards

    # Average HP (non-spells)
    features['avg_hitpoints'] = np.mean(hp_list) if hp_list else 0.0

    # Use the heuristic defense/offense calculations
    features['defense_strength'] = defense_strength_heuristic(deck_list, card_table)
    features['offense_strength'] = offense_strength_heuristic(deck_list, card_table)

    # Use heuristic synergy score
    features['synergy_score'] = synergy_score_heuristic(deck_list, card_table)
    # Add simple placeholders for cohesion etc. if needed by model features
    # These were complex/required match history in feature_eng, hard to replicate here
    # We might need to adjust the model training if these are highly important
    # For now, setting to average/neutral values or 0.
    features['cohesion'] = 0.5 # Assume average cohesion
    features['type_synergy'] = len([t for t, count in types.items() if count > 0]) / 3.0 # Basic type mix score
    features['win_synergy'] = 0.5 # Assume average win synergy

    return features

def create_deck_embedding(deck_list: List[str], card_table: pd.DataFrame, embedding_size: int = 16) -> np.ndarray:
    """Create a fixed-size numeric embedding for a deck (simplified version)."""
    card_feature_vectors = []
    default_card_vec = np.zeros(7) # elixir, dps, hp, damage, is_troop, is_spell, is_building

    for card_name in deck_list:
        card_data = _get_card_row(card_table, card_name)
        if card_data is not None:
            card_type = str(card_data.get('card_type', '')).lower()
            vec = [
                card_data.get('elixir_cost', 0) or 0,
                card_data.get('dps', 0) or 0,
                card_data.get('hitpoints', 0) or 0,
                card_data.get('damage', 0) or 0,
                1 if card_type == 'troop' else 0,
                1 if card_type == 'spell' else 0,
                1 if card_type == 'building' else 0,
            ]
            card_feature_vectors.append(vec)
        else:
             card_feature_vectors.append(default_card_vec) # Append zeros if card not found

    # Pad with default vectors if deck has less than 8 cards
    while len(card_feature_vectors) < 8:
        card_feature_vectors.append(default_card_vec)

    # If more than 8 (shouldn't happen but safe), truncate
    card_feature_vectors = card_feature_vectors[:8]

    # Combine: Use mean pooling for simplicity
    if not card_feature_vectors:
        deck_vector = np.zeros(7 * 8) # Fallback if absolutely no cards
    else:
        deck_vector = np.array(card_feature_vectors).flatten() # Flatten 8x7 features

    # Ensure fixed size (Project or Pad) - Simplified: Padding/Truncating flattened vector
    current_size = len(deck_vector)
    target_size = embedding_size # Use the size expected by the model

    # Find embedding feature names
    embedding_cols = [f for f in EXPECTED_FEATURES if f.startswith('embedding_')]
    if embedding_cols:
        target_size = len(embedding_cols)
    else:
        logger.warning(f"No 'embedding_' columns found in expected features. Using default size {embedding_size}.")
        target_size = embedding_size # Fallback


    if current_size > target_size:
        # Simple truncation
        final_embedding = deck_vector[:target_size]
    elif current_size < target_size:
        # Simple padding with zeros
        final_embedding = np.pad(deck_vector, (0, target_size - current_size))
    else:
        final_embedding = deck_vector

    # Normalize the final embedding
    norm = np.linalg.norm(final_embedding)
    if norm > 0:
         final_embedding = final_embedding / norm

    return final_embedding

# --- Heuristic functions copied/adapted from previous feature_engineering ---
def defense_strength_heuristic(deck: List[str], card_table: pd.DataFrame) -> float:
    """Estimate defensive capability of a deck."""
    score = 0.0
    for c in deck:
        r = _get_card_row(card_table, c)
        if r is None: continue
        hp = r.get("hitpoints", 0) or 0
        dps = r.get("dps", 0) or 0
        card_type = str(r.get("card_type", "")).lower()
        base = hp * 0.6 + dps * 0.4
        if card_type == "building" and "spawn" not in str(r.get("description", "")).lower() and "generate" not in str(r.get("description", "")).lower():
            base *= 1.5
        if card_type == "troop" and hp > 1400: base *= 1.2
        score += base
    return float(score)

def offense_strength_heuristic(deck: List[str], card_table: pd.DataFrame) -> float:
    """Estimate offensive capability."""
    score = 0.0
    for c in deck:
        r = _get_card_row(card_table, c)
        if r is None: continue
        dps = r.get("dps", 0) or 0
        damage = r.get("damage", 0) or 0
        # Simple weighted sum, prioritizing DPS
        score += dps * 0.7 + damage * 0.3
    return float(score)

def synergy_score_heuristic(deck: List[str], card_table: pd.DataFrame, role_weights: Optional[Dict[str, float]] = None) -> float:
    """Calculate a simple synergy score based on inferred role complementarity."""
    # (Using the same heuristic role inference as before)
    if role_weights is None: role_weights = {"wincon": 2.0, "support": 1.5, "tank": 1.2, "defense": 1.3, "cycle": 0.8, "spell": 1.0, "unknown": 0.1}
    counts: Dict[str, int] = {}
    roles_seen = []
    WINCON_NAMES = ["miner", "goblin drill", "skeleton barrel", "balloon", "graveyard", "x-bow", "mortar", "royal giant", "hog rider", "ram rider", "goblin giant", "giant", "golem", "elixir golem", "lava hound", "electro giant", "battle ram"]

    for c in deck:
        r = _get_card_row(card_table, c)
        if r is None: continue
        key = "unknown"
        card_type = str(r.get("card_type", "")).lower()
        # --- Use original name for matching wincons ---
        card_name_original = str(r.get("card_name_original", "")).lower() # Use original name field
        elixir = float(r.get("elixir_cost", 10))
        targets = str(r.get("targets", "")).lower()
        hp = float(r.get("hitpoints", 0) or 0)
        description = str(r.get("description", "")).lower()

        if card_type == "spell": key = "spell"
        elif card_type == "building":
            if "spawn" in description or "generate" in description or "goblin drill" in card_name_original: key = "support"
            elif any(wc in card_name_original for wc in ["x-bow", "mortar"]): key = "wincon"
            else: key = "defense"
        elif card_type == "troop":
             # Match against the cleaned original name
            if "buildings" in targets or any(wc in card_name_original for wc in WINCON_NAMES): key = "wincon"
            elif hp > 1400: key = "tank"
            elif elixir <= 2: key = "cycle"
            else: key = "support"

        counts[key] = counts.get(key, 0) + 1
        roles_seen.append(key)

    score = 0.0
    w = counts.get("wincon", 0)
    score += min(w, 2) * role_weights.get("wincon", 1.0)
    score += counts.get("support", 0) * role_weights.get("support", 1.0)
    score += counts.get("tank", 0) * role_weights.get("tank", 1.0)
    score += counts.get("defense", 0) * role_weights.get("defense", 1.0)
    total_cards = len(roles_seen) if roles_seen else 1
    max_role_count = max(counts.values()) if counts else 0
    diversity = len(counts) / total_cards
    score = score * (0.8 + 0.4 * diversity)
    if total_cards > 0 and (max_role_count / total_cards) > 0.6: score *= 0.7
    return float(score)

# ----------------------------
# Main Recommendation Function
# ----------------------------

def recommend_strategy(deck_list: List[str], opponent_archetype: str) -> Tuple[str, Dict[str, float]]:
    """
    Recommends a strategy based on deck, opponent, and model predictions.

    Args:
        deck_list: A list of 8 card names in the player's deck.
        opponent_archetype: The name of the opponent's deck archetype (e.g., 'Log Bait', 'Hog Cycle').

    Returns:
        A tuple containing:
        - recommendation (str): 'Aggressive Push', 'Defensive Counter', or 'Balanced Cycle'.
        - probabilities (dict): Predicted win probabilities for context.
            {
                'win_seeker_attack_win_prob': float,
                'win_seeker_defense_win_prob': float,
                'loss_avoider_attack_win_prob': float,
                'loss_avoider_defense_win_prob': float
            }
    """
    recommendation = "Undetermined"
    probabilities = {
        'win_seeker_attack_win_prob': 0.0,
        'win_seeker_defense_win_prob': 0.0,
        'loss_avoider_attack_win_prob': 0.0,
        'loss_avoider_defense_win_prob': 0.0
    }

    # 1. Load Models and Card Data
    try:
        win_seeker_model = joblib.load(WIN_SEEKER_MODEL_FILE)
        loss_avoider_model = joblib.load(LOSS_AVOIDER_MODEL_FILE)
    except FileNotFoundError:
        logger.error("FATAL: Trained model files not found. Run train_model.py first.")
        return "Error: Models not found", probabilities
    except Exception as e:
        logger.error(f"FATAL: Error loading models: {e}")
        return f"Error: {e}", probabilities

    card_table = load_card_table(CARDS_FILE)
    if card_table is None:
        return "Error: Card data not found", probabilities

    if not EXPECTED_FEATURES:
         return "Error: Could not determine model features", probabilities


    # 2. Calculate Deck Features
    logger.info(f"Calculating features for deck: {deck_list}")
    deck_features = calculate_deck_features(deck_list, card_table)
    deck_embedding = create_deck_embedding(deck_list, card_table)

    # 3. Prepare Feature Vectors for Attack (mode=1) and Defense (mode=0)
    base_features = {}
    # Populate with calculated deck features, ensuring keys match EXPECTED_FEATURES
    for feature_name in EXPECTED_FEATURES:
        if feature_name.startswith('embedding_'):
            try:
                index = int(feature_name.split('_')[1])
                if index < len(deck_embedding):
                    base_features[feature_name] = deck_embedding[index]
                else:
                    base_features[feature_name] = 0 # Pad if embedding shorter than expected
            except (IndexError, ValueError):
                base_features[feature_name] = 0 # Default if parsing fails
        elif feature_name.startswith('opponent_archetype_'):
            # Handle opponent archetype one-hot encoding
            # Construct the expected column name based on KNOWN_OPPONENT_ARCHETYPES
            clean_opponent_name = opponent_archetype.lower().replace(' ', '_').replace('-', '_') # Clean input name
            opponent_col_name = f"opponent_archetype_{clean_opponent_name}"

            # Check if the constructed name is a known column
            if opponent_col_name in KNOWN_OPPONENT_ARCHETYPES:
                base_features[feature_name] = 1 if feature_name == opponent_col_name else 0
            # Handle unknown case explicitly if the column exists
            elif 'opponent_archetype_unknown' in KNOWN_OPPONENT_ARCHETYPES:
                 base_features[feature_name] = 1 if feature_name == 'opponent_archetype_unknown' else 0
            # Fallback if archetype is truly unknown or column missing
            else:
                 base_features[feature_name] = 0 # Set all opponent archetype columns to 0 if unknown/missing

            # Explicitly handle 'nan' column if it exists from training
            if feature_name == 'opponent_archetype_nan' and 'opponent_archetype_nan' in KNOWN_OPPONENT_ARCHETYPES:
                 base_features[feature_name] = 0


        elif feature_name == 'mode':
            continue # Mode is set separately for attack/defense
        elif feature_name in deck_features:
            # Ensure NaN values from calculation are handled (e.g., avg_elixir for empty deck)
            base_features[feature_name] = deck_features[feature_name] if not pd.isna(deck_features[feature_name]) else 0.0
        # --- Handle missing features (e.g., trophies, explevel) ---
        # These were in the original feature files but not calculated from deck alone.
        # We need to provide *some* value. Using median/mean or 0 is common.
        # Let's use 0 for simplicity, or load defaults if available.
        elif feature_name == 'trophies':
            base_features[feature_name] = 7000 # Assume a reasonable default trophy count
        elif feature_name == 'explevel':
            base_features[feature_name] = 60 # Assume a reasonable default level
        elif feature_name == 'counter_score':
             base_features[feature_name] = 0.5 # Assume neutral counter score without match history context
        # Add other potential missing features here with defaults
        else:
            # Check if it's a player archetype column (not relevant for this prediction type)
            # Set all player archetype features to 0
            if feature_name.startswith('player_archetype_'):
                 base_features[feature_name] = 0
            else:
                logger.warning(f"Feature '{feature_name}' expected by model but not calculated/handled. Using default 0.")
                base_features[feature_name] = 0 # Default for any other missing feature


    # Create attack and defense versions
    attack_input_features = base_features.copy()
    attack_input_features['mode'] = 1
    defense_input_features = base_features.copy()
    defense_input_features['mode'] = 0

    # Convert to DataFrame with correct column order
    try:
        # --- Ensure all EXPECTED_FEATURES are present before indexing ---
        for feature in EXPECTED_FEATURES:
            if feature not in attack_input_features:
                logger.warning(f"Expected feature '{feature}' missing from generated features. Adding default 0.")
                attack_input_features[feature] = 0
                defense_input_features[feature] = 0

        attack_df = pd.DataFrame([attack_input_features])[EXPECTED_FEATURES]
        defense_df = pd.DataFrame([defense_input_features])[EXPECTED_FEATURES]
    except KeyError as ke:
        logger.error(f"FATAL: Mismatch between generated features and expected features. Missing: {ke}")
        # Log generated vs expected columns for debugging
        logger.debug(f"Generated keys count: {len(attack_input_features)}")
        logger.debug(f"Expected keys count: {len(EXPECTED_FEATURES)}")
        missing_expected = set(EXPECTED_FEATURES) - set(attack_input_features.keys())
        extra_generated = set(attack_input_features.keys()) - set(EXPECTED_FEATURES)
        logger.debug(f"Missing expected keys: {missing_expected}")
        logger.debug(f"Extra generated keys: {extra_generated}")

        return f"Error: Feature mismatch {ke}", probabilities
    except Exception as e:
         logger.error(f"FATAL: Error creating input DataFrame: {e}")
         return f"Error: {e}", probabilities


    # 4. Predict Probabilities
    try:
        # predict_proba returns [[P(Lose), P(Win]]]
        win_seeker_prob_attack = win_seeker_model.predict_proba(attack_df)[0][1]
        win_seeker_prob_defense = win_seeker_model.predict_proba(defense_df)[0][1]
        loss_avoider_prob_attack = loss_avoider_model.predict_proba(attack_df)[0][1]
        loss_avoider_prob_defense = loss_avoider_model.predict_proba(defense_df)[0][1]

        probabilities = {
            'win_seeker_attack_win_prob': float(win_seeker_prob_attack), # Cast numpy float
            'win_seeker_defense_win_prob': float(win_seeker_prob_defense),
            'loss_avoider_attack_win_prob': float(loss_avoider_prob_attack),
            'loss_avoider_defense_win_prob': float(loss_avoider_prob_defense)
        }
        logger.info(f"Predicted Probabilities: {probabilities}")

    except Exception as e:
        logger.error(f"Error during model prediction: {e}")
        # Add more detail if possible
        if hasattr(e, 'response'): logger.error(f"Prediction error details: {e.response}")
        return f"Error: Prediction failed {e}", probabilities

    # 5. Recommendation Logic
    avg_elixir_val = deck_features.get('avg_elixir') # Get calculated avg elixir
    if pd.isna(avg_elixir_val): avg_elixir_val = 4.0 # Default if calculation failed

    # Define thresholds (these may need tuning)
    prob_diff_threshold = 0.10 # How much higher should attack prob be for Aggressive?
    loss_avoid_threshold = 0.08 # How much lower should defense loss prob be for Defensive?
    cycle_elixir_threshold = 3.2 # Decks below this are likely cycle decks

    # Rule 1: Low Elixir -> Balanced Cycle
    if avg_elixir_val <= cycle_elixir_threshold:
        recommendation = "Balanced Cycle"
        logger.info(f"Recommendation: {recommendation} (Low Avg Elixir: {avg_elixir_val:.2f})")
    # Rule 2: Aggressive Push? (Win-Seeker favors Attack significantly)
    elif probabilities['win_seeker_attack_win_prob'] > probabilities['win_seeker_defense_win_prob'] + prob_diff_threshold:
        recommendation = "Aggressive Push"
        logger.info(f"Recommendation: {recommendation} (Win-Seeker Attack Prob {probabilities['win_seeker_attack_win_prob']:.2f} > Defense Prob {probabilities['win_seeker_defense_win_prob']:.2f} by > {prob_diff_threshold})")
    # Rule 3: Defensive Counter? (Loss-Avoider favors Defense for safety)
    # Compare P(Lose) = 1 - P(Win)
    elif (1 - probabilities['loss_avoider_defense_win_prob']) < (1 - probabilities['loss_avoider_attack_win_prob']) - loss_avoid_threshold:
        recommendation = "Defensive Counter"
        logger.info(f"Recommendation: {recommendation} (Loss-Avoider Defense Loss Prob {1-probabilities['loss_avoider_defense_win_prob']:.2f} < Attack Loss Prob {1-probabilities['loss_avoider_attack_win_prob']:.2f} by > {loss_avoid_threshold})")
    # Rule 4: Default to Balanced if signals are mixed or weak
    else:
        recommendation = "Balanced Cycle"
        logger.info(f"Recommendation: {recommendation} (Mixed signals or probabilities too close)")


    return recommendation, probabilities


# ----------------------------
# Example Usage
# ----------------------------
if __name__ == "__main__":
    # --- Example Input ---
    example_deck = [
        'Archers', 'Tesla', 'X-Bow', 'Knight', 'Fireball',
        'Skeletons', 'Electro Spirit', 'The Log'
    ]
    # Make sure archetype matches one used during training (check KNOWN_OPPONENT_ARCHETYPES)
    # Example: if 'opponent_archetype_log_bait' is a column name
    example_opponent = 'Log Bait' # Should be cleaned to 'log_bait'

    logger.info("--- Running Recommendation Example ---")
    recommendation, probs = recommend_strategy(example_deck, example_opponent)

    print("\n" + "="*30)
    print(f"Deck: {example_deck}")
    print(f"Opponent Archetype: {example_opponent}")
    print("-" * 30)
    print(f"Recommended Strategy: {recommendation}")
    print("-" * 30)
    print("Predicted Win Probabilities:")
    print(f"  Win-Seeker (Attack):  {probs.get('win_seeker_attack_win_prob', 0):.2%}")
    print(f"  Win-Seeker (Defense): {probs.get('win_seeker_defense_win_prob', 0):.2%}")
    print(f"  Loss-Avoider (Attack): {probs.get('loss_avoider_attack_win_prob', 0):.2%}")
    print(f"  Loss-Avoider (Defense):{probs.get('loss_avoider_defense_win_prob', 0):.2%}")
    print("=" * 30)

    # --- Example 2 ---
    example_deck_2 = [
        'Royal Giant', 'Royal Ghost', 'Skeletons', 'Fisherman', 'Hunter',
        'Barbarian Barrel', 'Electro Spirit', 'Lightning'
    ]
    example_opponent_2 = 'Hog Cycle' # Should be cleaned to 'hog_cycle'

    logger.info("\n--- Running Recommendation Example 2 ---")
    recommendation_2, probs_2 = recommend_strategy(example_deck_2, example_opponent_2)

    print("\n" + "="*30)
    print(f"Deck: {example_deck_2}")
    print(f"Opponent Archetype: {example_opponent_2}")
    print("-" * 30)
    print(f"Recommended Strategy: {recommendation_2}")
    print("-" * 30)
    print("Predicted Win Probabilities:")
    print(f"  Win-Seeker (Attack):  {probs_2.get('win_seeker_attack_win_prob', 0):.2%}")
    print(f"  Win-Seeker (Defense): {probs_2.get('win_seeker_defense_win_prob', 0):.2%}")
    print(f"  Loss-Avoider (Attack): {probs_2.get('loss_avoider_attack_win_prob', 0):.2%}")
    print(f"  Loss-Avoider (Defense):{probs_2.get('loss_avoider_defense_win_prob', 0):.2%}")
    print("=" * 30)



# ### How to Use:

# 1.  **Save:** Save this code as `src/recommend_strategy.py`.
# 2.  **Ensure Files:** Make sure `cards_data.csv`, `attack_mode_features.csv` (used for column names), `strategy_model_win_seeker.pkl`, and `strategy_model_loss_avoider.pkl` are in the correct `data/` and `models/` directories relative to where you run the script.
# 3.  **Run:** From your project root directory (where `src` and `data` are), execute:
#     ```sh
#     python src/recommend_strategy.py
