"""Utility functions for Clash Royale strategy prediction and related tasks.

Provides helpers for:
- Loading card data and ML models (with in-memory caching).
- Validating inputs (decks, archetypes).
- Calculating deck features (avg_elixir, heuristics, etc.) based on card attributes.
- Creating deck embeddings.
- Inferring deck archetypes heuristically.
- Preparing the exact feature vector for model prediction.
- Generating stable cache keys.
"""

import os
import sys
from typing import Dict, List, Optional, Tuple, Any, Set
import json
from pathlib import Path
# --- FIX: Import datetime module, not class ---
import datetime # Changed from 'from datetime import datetime, timedelta'
# --- End Fix ---
import logging
import hashlib
import joblib
import numpy as np
import pandas as pd
# Removed StandardScaler as scaling is pre-computed in features

# --- Project Root Setup ---
try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    PROJECT_ROOT = Path('.').resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
# --- End Project Root Setup ---

# Configure logging
logger = logging.getLogger("clash_utils")
if not logger.hasHandlers():
     handler = logging.StreamHandler()
     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
     handler.setFormatter(formatter)
     logger.addHandler(handler)
     logger.setLevel(logging.INFO)

# Type aliases
DeckFeatures = Dict[str, float]
ModelArtifact = Any
CardData = Dict[str, Any]

# --- Constants ---
PREDICTION_DECK_SIZE = 8
VALID_ARCHETYPES = {
    "beatdown", "hog cycle", "lavaloon", "log bait", "miner control",
    "royal giant", "x-bow", "siege", "spell cycle", "cycle", "bridge spam",
    "unknown"
}
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_CARDS_FILE = DEFAULT_DATA_DIR / "cards_data.csv"


# ===========================================
# Data and Model Loading (with Cache)
# ===========================================

_model_cache: Dict[str, Tuple[ModelArtifact, datetime.datetime]] = {}
_card_table_cache: Optional[pd.DataFrame] = None
_card_table_load_time: Optional[datetime.datetime] = None

def load_card_table(path: Path = DEFAULT_CARDS_FILE, max_age_seconds: int = 3600) -> Optional[pd.DataFrame]:
    """Load card attributes table from CSV file, with simple in-memory caching."""
    global _card_table_cache, _card_table_load_time

    # --- FIX: Use datetime.datetime.now() ---
    now = datetime.datetime.now()
    # --- End Fix ---

    # Check cache
    if _card_table_cache is not None and _card_table_load_time is not None:
        age = (now - _card_table_load_time).total_seconds()
        if age < max_age_seconds:
            logger.debug("Returning cached card table.")
            return _card_table_cache.copy()

    logger.info(f"Loading card table from {path}...")
    try:
        if not path.exists():
             logger.error(f"Card data file not found at {path}")
             # --- FIX: Re-raise FileNotFoundError ---
             raise FileNotFoundError(f"Card data file not found at {path}")
             # --- End Fix ---

        df = pd.read_csv(path)
        if 'card_name' not in df.columns:
            logger.error(f"'card_name' column not found in {path}")
            _card_table_cache = None; _card_table_load_time = None
            return None

        df['card_name_original'] = df['card_name']
        df['card_name'] = df['card_name'].astype(str).str.lower().str.strip()
        
        # --- FIX: Set cache variables BEFORE returning ---
        _card_table_cache = df
        _card_table_load_time = now
        # --- End Fix ---
        
        logger.info(f"Card table loaded successfully ({len(df)} rows).")
        return df.copy()

    except FileNotFoundError as e:
         _card_table_cache = None; _card_table_load_time = None
         logger.error(f"load_card_table FileNotFoundError: {e}")
         raise # Re-raise for test
    except Exception as e:
        logger.error(f"Error loading card data from {path}: {e}", exc_info=True)
        _card_table_cache = None; _card_table_load_time = None
        return None

def load_model(model_path: str, max_age_seconds: int = 3600) -> ModelArtifact:
    """Load a trained model artifact (.pkl/.joblib) from disk, using cache."""
    global _model_cache
    model_path_abs = str(Path(model_path).resolve())
    # --- FIX: Use datetime.datetime.now() ---
    now = datetime.datetime.now()
    # --- End Fix ---

    # Check cache
    if model_path_abs in _model_cache:
        model, loaded_at = _model_cache[model_path_abs]
        age = (now - loaded_at).total_seconds()
        if age < max_age_seconds:
            logger.debug(f"Returning cached model: {os.path.basename(model_path)}")
            return model
        else:
            logger.info(f"Model cache expired for: {os.path.basename(model_path)}")
            del _model_cache[model_path_abs]

    # Load from disk
    logger.info(f"Loading model from: {model_path}")
    if not os.path.exists(model_path):
        logger.error(f"Model file not found at {model_path}")
        raise FileNotFoundError(f"Model not found at {model_path}")

    try:
        model = joblib.load(model_path)
        _model_cache[model_path_abs] = (model, now)
        logger.info(f"Model loaded and cached: {os.path.basename(model_path)}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}", exc_info=True)
        if model_path_abs in _model_cache: del _model_cache[model_path_abs]
        raise ValueError(f"Failed to load model artifact from {model_path}: {e}")


# ===========================================
# Input Validation
# ===========================================

def _get_card_row(card_table: pd.DataFrame, card_name: str) -> Optional[CardData]:
    """Safely get card data row using flexible matching (exact then partial)."""
    if not isinstance(card_name, str) or not card_name: return None
    card_name_lower = card_name.lower().strip()
    try:
        exact_match = card_table[card_table["card_name"] == card_name_lower]
        if not exact_match.empty:
            return exact_match.iloc[0].to_dict()

        partial_match = card_table[card_table["card_name"].str.startswith(card_name_lower)]
        if not partial_match.empty:
            partial_match = partial_match.copy()
            partial_match['len_diff'] = partial_match['card_name'].str.len() - len(card_name_lower)
            min_len_diff = partial_match['len_diff'].min()
            best_partial_match = partial_match[partial_match['len_diff'] == min_len_diff]
            if len(best_partial_match) > 1: logger.warning(f"Multiple partial matches for '{card_name}'. Using first: {best_partial_match.iloc[0]['card_name_original']}")
            logger.debug(f"Partial match for '{card_name}': {best_partial_match.iloc[0]['card_name_original']}")
            return best_partial_match.iloc[0].to_dict()

        logger.warning(f"Card '{card_name}' not found.")
        return None
    except Exception as e:
         logger.error(f"Error looking up card '{card_name}': {e}", exc_info=True)
         return None

def validate_deck_cards(deck: List[str], card_table: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Checks deck size and if all cards can be matched in the card table."""
    if card_table is None or 'card_name' not in card_table.columns:
        logger.error("Card table unavailable for deck validation.")
        return False, deck

    missing_cards: List[str] = []
    if len(deck) != PREDICTION_DECK_SIZE:
        logger.warning(f"Deck size is {len(deck)}, but prediction requires {PREDICTION_DECK_SIZE}.")
        is_valid_size = False
    else:
        is_valid_size = True
    
    for card_name in deck:
        if _get_card_row(card_table, card_name) is None:
            missing_cards.append(card_name)

    all_cards_found = len(missing_cards) == 0
    
    if not all_cards_found: logger.warning(f"Deck validation unmatched cards: {missing_cards}")

    return (all_cards_found and is_valid_size), missing_cards


def validate_archetype(archetype: Optional[str]) -> str: # Always returns string
    """Validate and normalize an archetype string. Returns 'unknown' if invalid."""
    if archetype is None or not str(archetype).strip():
        return "unknown"
    normalized = str(archetype).lower().strip().replace('-', ' ').replace('_', ' ')
    variations = {"bridgespam": "bridge spam", "logbait": "log bait"}
    normalized = variations.get(normalized, normalized)
    if normalized not in VALID_ARCHETYPES:
        logger.warning(f"Input archetype '{archetype}' (norm: '{normalized}') not in VALID_ARCHETYPES. Using 'unknown'.")
        return "unknown"
    return normalized

# ===========================================
# Feature Calculation Helpers
# ===========================================

def defense_strength_heuristic(deck: List[str], card_table: pd.DataFrame) -> float:
    """Estimate defensive capability based on heuristics."""
    score = 0.0
    for c in deck:
        r = _get_card_row(card_table, c)
        if r is None: continue
        hp = r.get("hitpoints", 0) or 0.0
        dps = r.get("dps", 0) or 0.0
        card_type = str(r.get("card_type", "")).lower()
        base = hp * 0.6 + dps * 0.4
        if card_type == "building" and "spawn" not in str(r.get("description", "")).lower() and "generate" not in str(r.get("description", "")).lower():
            base *= 1.5
        if card_type == "troop" and hp > 1400:
             base *= 1.2
        score += base
    return float(score)

def offense_strength_heuristic(deck: List[str], card_table: pd.DataFrame) -> float:
    """Estimate offensive capability based on heuristics."""
    score = 0.0
    for c in deck:
        r = _get_card_row(card_table, c)
        if r is None: continue
        dps = r.get("dps", 0) or 0.0
        damage = r.get("damage", 0) or 0.0
        score += dps * 0.7 + damage * 0.3
    return float(score)

def synergy_score_heuristic(deck: List[str], card_table: pd.DataFrame, role_weights: Optional[Dict[str, float]] = None) -> float:
    """Calculate synergy score based on inferred roles and deck balance."""
    if role_weights is None: role_weights = {"wincon": 2.0, "support": 1.5, "tank": 1.2, "defense": 1.3, "cycle": 0.8, "spell": 1.0, "unknown": 0.1}
    counts: Dict[str, int] = {}; roles_seen = []
    WINCON_NAMES = ["miner", "goblin drill", "skeleton barrel", "balloon", "graveyard", "x-bow", "mortar", "royal giant", "hog rider", "ram rider", "goblin giant", "giant", "golem", "elixir golem", "lava hound", "electro giant", "battle ram"]
    for c in deck:
        r = _get_card_row(card_table, c);
        if r is None: continue
        key = "unknown"; card_type = str(r.get("card_type", "")).lower(); card_name_original = str(r.get("card_name_original", "")).lower()
        elixir = float(r.get("elixir_cost", 10.0)); targets = str(r.get("targets", "")).lower(); hp = float(r.get("hitpoints", 0.0) or 0.0); description = str(r.get("description", "")).lower()
        if card_type == "spell": key = "spell"
        elif card_type == "building":
            if "spawn" in description or "generate" in description or "goblin drill" in card_name_original: key = "support"
            elif any(wc in card_name_original for wc in ["x-bow", "mortar"]): key = "wincon"
            else: key = "defense"
        elif card_type == "troop":
            if "buildings" in targets or any(wc in card_name_original for wc in WINCON_NAMES): key = "wincon"
            # --- FIX: Include Knight/Valk in tank check ---
            elif hp >= 1400: key = "tank" # Changed from > to >=
            # --- End Fix ---
            elif elixir <= 2: key = "cycle"
            else: key = "support"
        counts[key] = counts.get(key, 0) + 1; roles_seen.append(key)
    score = 0.0; w = counts.get("wincon", 0); score += min(w, 2) * role_weights.get("wincon", 1.0); score += counts.get("support", 0) * role_weights.get("support", 1.0)
    score += counts.get("tank", 0) * role_weights.get("tank", 1.0); score += counts.get("defense", 0) * role_weights.get("defense", 1.0)
    total_cards = len(roles_seen) if roles_seen else 1; max_role_count = max(counts.values()) if counts else 0
    diversity = len(counts) / total_cards; score = score * (0.8 + 0.4 * diversity)
    if total_cards > 0 and (max_role_count / total_cards) > 0.6: score *= 0.7
    return float(score)

def calculate_deck_features(deck_list: List[str], card_table: pd.DataFrame) -> DeckFeatures:
    """Calculate core numeric features required by the prediction models."""
    default_features: DeckFeatures = {'avg_elixir': np.nan, 'total_dps': 0.0, 'avg_dps': 0.0, 'defense_strength': 0.0, 'offense_strength': 0.0, 'spell_ratio': 0.0, 'troop_ratio': 0.0, 'building_ratio': 0.0, 'avg_hitpoints': 0.0, 'synergy_score': 0.0, 'cohesion': 0.5, 'type_synergy': 0.0, 'win_synergy': 0.5}
    features = default_features.copy()
    if card_table is None: logger.error("calculate_deck_features: Card table missing."); return features

    valid_cards_data = []; elixirs = []; dps_list = []; damage_list = []; hp_list = []
    types_count = {'spell': 0, 'troop': 0, 'building': 0}; num_input_cards = len(deck_list)
    for card_name in deck_list:
        card_data = _get_card_row(card_table, card_name)
        if card_data is not None:
            valid_cards_data.append(card_data); elixirs.append(card_data.get('elixir_cost', np.nan)); dps_list.append(card_data.get('dps', 0) or 0.0); damage_list.append(card_data.get('damage', 0) or 0.0)
            card_type = str(card_data.get('card_type', '')).lower();
            if card_type in types_count: types_count[card_type] += 1
            if card_type != 'spell': hp_list.append(card_data.get('hitpoints', 0) or 0.0)
    num_valid_cards = len(valid_cards_data)
    if num_valid_cards == 0: logger.warning(f"calculate_deck_features: No valid cards for deck: {deck_list}"); return features
    if num_valid_cards < num_input_cards: logger.warning(f"Processed {num_valid_cards}/{num_input_cards} cards.")

    valid_elixirs = [e for e in elixirs if pd.notna(e)];
    if valid_elixirs: features['avg_elixir'] = np.mean(valid_elixirs)
    else: features['avg_elixir'] = np.nan

    features['total_dps'] = np.sum(dps_list); features['avg_dps'] = np.mean(dps_list) if dps_list else 0.0
    features['spell_ratio'] = types_count['spell'] / num_valid_cards; features['troop_ratio'] = types_count['troop'] / num_valid_cards; features['building_ratio'] = types_count['building'] / num_valid_cards
    features['avg_hitpoints'] = np.mean(hp_list) if hp_list else 0.0
    features['defense_strength'] = defense_strength_heuristic(deck_list, card_table); features['offense_strength'] = offense_strength_heuristic(deck_list, card_table)
    features['synergy_score'] = synergy_score_heuristic(deck_list, card_table)
    features['type_synergy'] = len([t for t, count in types_count.items() if count > 0]) / 3.0

    final_features = default_features.copy()
    final_features.update({k: float(v) for k, v in features.items() if pd.notna(v)})
    return final_features


def create_deck_embedding(deck_list: List[str], card_table: pd.DataFrame, embedding_size: int) -> np.ndarray:
    """Create a fixed-size numeric embedding vector for a deck."""
    card_feature_vectors = []; default_card_vec = np.zeros(7, dtype=np.float32)
    for card_name in deck_list:
        card_data = _get_card_row(card_table, card_name)
        if card_data is not None:
            card_type = str(card_data.get('card_type', '')).lower()
            vec = np.array([card_data.get('elixir_cost', 0) or 0, card_data.get('dps', 0) or 0, card_data.get('hitpoints', 0) or 0, card_data.get('damage', 0) or 0, 1 if card_type == 'troop' else 0, 1 if card_type == 'spell' else 0, 1 if card_type == 'building' else 0], dtype=np.float32)
            card_feature_vectors.append(vec)
        else: card_feature_vectors.append(default_card_vec)
    while len(card_feature_vectors) < 8: card_feature_vectors.append(default_card_vec)
    card_feature_vectors = card_feature_vectors[:8]
    if not card_feature_vectors: deck_vector = np.zeros(embedding_size, dtype=np.float32)
    else: deck_vector = np.array(card_feature_vectors, dtype=np.float32).flatten()
    current_size = len(deck_vector)
    if current_size > embedding_size: final_embedding = deck_vector[:embedding_size]
    elif current_size < embedding_size: final_embedding = np.pad(deck_vector, (0, embedding_size - current_size), constant_values=0.0).astype(np.float32)
    else: final_embedding = deck_vector
    norm = np.linalg.norm(final_embedding);
    if norm > 1e-6: final_embedding = final_embedding / norm
    else: final_embedding = np.zeros_like(final_embedding)
    return final_embedding.astype(np.float32)

def infer_archetype(deck: List[str], card_table: pd.DataFrame) -> str:
    """Infers a deck archetype string based on heuristics."""
    if card_table is None: logger.error("infer_archetype: Card table missing."); return "unknown"
    card_data_map = {row['card_name']: row.to_dict() for _, row in card_table.iterrows()}
    if not deck: return "unknown"
    win_conditions=0; high_hp_tanks=0; buildings=0; spells=0; cycle_cards=0; total_elixir=0; valid_cards=0
    WINCON_NAMES = ["giant", "royal giant", "golem", "lava hound", "balloon", "hog rider", "ram rider", "battle ram", "x-bow", "mortar", "goblin drill", "goblin giant", "electro giant", "elixir golem", "graveyard", "skeleton barrel", "miner"]
    for card_name in deck:
        card_info = card_data_map.get(card_name.lower().strip())
        if card_info:
            valid_cards+=1; elixir=card_info.get('elixir_cost'); card_type=card_info.get('card_type','').lower()
            hp=card_info.get('hitpoints',0); targets=str(card_info.get('targets','')).lower()
            if elixir is not None and pd.notna(elixir): total_elixir+=elixir;
            if elixir is not None and pd.notna(elixir) and elixir<=2: cycle_cards+=1
            if card_type=='building': buildings+=1
            elif card_type=='spell': spells+=1
            elif card_type=='troop':
                 clean_name=str(card_info.get('card_name_original',card_name)).lower().split('(')[0].strip()
                 if 'buildings' in targets or any(wc in clean_name for wc in WINCON_NAMES): win_conditions+=1
                 if hp and pd.notna(hp) and hp >= 1400: high_hp_tanks+=1 # FIX: >= 1400
    avg_elixir=(total_elixir/valid_cards) if valid_cards > 0 else 4.0
    deck_names_lower = [n.lower() for n in deck]
    if win_conditions==0 and spells>=4: return "spell cycle"
    if any(wc in n for wc in ["x-bow","mortar"] for n in deck_names_lower) and buildings>=2: return "siege"
    if any(wc in n for wc in ["lava hound","balloon"] for n in deck_names_lower): return "lavaloon"
    if any(wc in n for wc in ["golem","electro giant"] for n in deck_names_lower) and avg_elixir>=4.0: return "beatdown"
    if any("royal giant" in n for n in deck_names_lower): return "royal giant"
    if any("hog rider" in n for n in deck_names_lower) and avg_elixir<=3.5: return "hog cycle"
    if any("goblin barrel" in n for n in deck_names_lower) and cycle_cards>=3: return "log bait"
    if any("miner" in n for n in deck_names_lower) and avg_elixir<=3.8: return "miner control"
    # --- FIX: Relaxed beatdown elixir rule ---
    if win_conditions>=1 and high_hp_tanks>=1 and avg_elixir >= 3.6: # Changed from 3.8
         return "beatdown"
    # --- End Fix ---
    if avg_elixir<=3.2 and cycle_cards>=4: return "cycle"
    if win_conditions>=2 and any(br in n for br in ["battle ram","ram rider","bandit","royal ghost"] for n in deck_names_lower): return "bridge spam"
    return "unknown"


# ===========================================
# Prediction Input Preparation
# ===========================================

def prepare_prediction_input(
    deck_list: List[str],
    opponent_archetype: Optional[str],
    mode: str, # 'attack' or 'defense'
    card_table: pd.DataFrame,
    expected_features: List[str],
    known_opponent_archetypes: List[str]
) -> pd.DataFrame:
    """Prepares a single-row DataFrame for model prediction."""
    if card_table is None: raise ValueError("Card table required.")
    if not expected_features: raise ValueError("Expected features list required.")
    
    # --- FIX: Match error message from test ---
    if len(deck_list) != PREDICTION_DECK_SIZE:
         raise ValueError(f"Need {PREDICTION_DECK_SIZE} cards, got {len(deck_list)}.")
    # --- End Fix ---
         
    if mode not in ['attack', 'defense']: raise ValueError("Mode must be 'attack' or 'defense'.")

    # 1. Validate Deck (Log warnings but proceed, size is already checked)
    # validate_deck_cards already logs warnings
    validate_deck_cards(deck_list, card_table) 

    # 2. Validate Opponent Archetype
    valid_opponent_archetype = validate_archetype(opponent_archetype)

    # 3. Calculate Features & Embedding
    try:
        deck_features = calculate_deck_features(deck_list, card_table)
        embedding_cols = [f for f in expected_features if f.startswith('embedding_')]
        embedding_size = len(embedding_cols) if embedding_cols else 16
        if embedding_size == 0: logger.warning("No embedding features found in expected list, using default size 16.")
        deck_embedding = create_deck_embedding(deck_list, card_table, embedding_size=embedding_size)
    except Exception as e: raise ValueError(f"Feature calculation failed: {e}")

    # 4. Construct Feature Dictionary
    feature_dict: Dict[str, Any] = {}
    try:
        for feature_name in expected_features:
            if feature_name == 'mode': feature_dict[feature_name] = 1.0 if mode == 'attack' else 0.0
            elif feature_name.startswith('embedding_'):
                try: idx = int(feature_name.split('_')[1]); feature_dict[feature_name] = float(deck_embedding[idx]) if idx < len(deck_embedding) else 0.0
                except (IndexError, ValueError, TypeError): feature_dict[feature_name] = 0.0
            elif feature_name.startswith('opponent_archetype_'):
                current_arch_suffix = valid_opponent_archetype.replace(' ', '_')
                expected_suffix = feature_name.replace('opponent_archetype_', '')
                # Handle 'nan' column
                if expected_suffix == 'nan': feature_dict[feature_name] = 0.0 # Always 0 for 'nan' column
                elif expected_suffix == current_arch_suffix: feature_dict[feature_name] = 1.0
                else: feature_dict[feature_name] = 0.0
            elif feature_name in deck_features:
                val = deck_features[feature_name]; feature_dict[feature_name] = float(val) if pd.notna(val) else 0.0
            elif feature_name == 'trophies': feature_dict[feature_name] = 7000.0
            elif feature_name == 'explevel': feature_dict[feature_name] = 60.0
            elif feature_name == 'counter_score': feature_dict[feature_name] = 0.5
            elif feature_name.startswith('player_archetype_'): feature_dict[feature_name] = 0.0
            else: logger.warning(f"Unhandled feature '{feature_name}'. Setting to 0.0."); feature_dict[feature_name] = 0.0
    except Exception as e: raise ValueError(f"Feature mapping failed: {e}")

    # 5. Create DataFrame
    try:
        input_df = pd.DataFrame([feature_dict], columns=expected_features).fillna(0.0)
        if list(input_df.columns) != expected_features: raise ValueError("Column mismatch.")
        non_numeric = input_df.select_dtypes(exclude=np.number).columns
        if len(non_numeric) > 0:
             logger.warning(f"Non-numeric columns in final DF: {list(non_numeric)}. Forcing conversion.")
             for col in non_numeric: input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(0.0)
        input_df = input_df.astype(np.float32)
        return input_df
    except Exception as e: raise ValueError(f"DataFrame creation failed: {e}")


# ===========================================
# Cache Key Generation
# ===========================================

def generate_cache_key(prefix: str, **kwargs) -> str:
    """Generates a stable MD5 hash key."""
    try:
        payload = {"prefix": prefix}
        payload.update({k: sorted(v) if isinstance(v, list) else v for k, v in kwargs.items()})
        key_string = json.dumps(payload, sort_keys=True)
        return hashlib.md5(key_string.encode('utf-8')).hexdigest()
    except Exception as e:
        logger.error(f"Cache key generation error: {e}", exc_info=True)
        return f"{prefix}_error_{time.time()}_{np.random.rand()}"


# ===========================================
# Example Usage (if run directly)
# ===========================================
if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    logger.info("Running utils.py demo...")

    print("\n--- Loading Card Table ---")
    ct = load_card_table()
    if ct is not None:
        print(f"Loaded {len(ct)} cards.")
        print("\n--- Validating Archetypes ---")
        print(f"'Hog Cycle' -> {validate_archetype('Hog Cycle')}")
        print(f"None -> {validate_archetype(None)}")
        print(f"'Invalid' -> {validate_archetype('Invalid')}")

        print("\n--- Validating Deck ---")
        deck_ok = ['Knight', 'Archers', 'Arrows', 'Giant', 'Mini P.E.K.K.A', 'Musketeer', 'Goblin Barrel', 'Zap']
        deck_bad = ['Knight', 'Archers', 'BadCard', 'Giant', 'Mini P.E.K.K.A', 'Musketeer', 'Goblin Barrel', 'Zap']
        deck_short = ['Knight', 'Archers']
        valid, missing = validate_deck_cards(deck_ok, ct); print(f"Deck OK (8): Valid={valid}, Missing={missing}")
        valid, missing = validate_deck_cards(deck_bad, ct); print(f"Deck Bad (8): Valid={valid}, Missing={missing}")
        valid, missing = validate_deck_cards(deck_short, ct); print(f"Deck Short (2): Valid={valid}, Missing={missing}")

        print("\n--- Calculating Features ---")
        features = calculate_deck_features(deck_ok, ct)
        print("Features (sample):")
        if 'avg_elixir' in features: print(f"  Avg Elixir: {features.get('avg_elixir', 0):.2f}")
        if 'defense_strength' in features: print(f"  Defense Str: {features.get('defense_strength', 0):.0f}")
        if 'synergy_score' in features: print(f"  Synergy: {features.get('synergy_score', 0):.2f}")

        print("\n--- Preparing Prediction Input ---")
        try:
            feat_df = pd.read_csv(DEFAULT_DATA_DIR / "attack_mode_features.csv", nrows=1)
            model_features = list(feat_df.drop(columns=['player_tag', 'winner_flag'], errors='ignore').columns)
            opp_archs = [c for c in model_features if c.startswith('opponent_archetype_')]
            print(f"Loaded {len(model_features)} expected features.")

            input_df = prepare_prediction_input(deck_ok, "Log Bait", "attack", ct, model_features, opp_archs)
            print("Generated DataFrame (Attack vs Log Bait):")
            cols_to_show = ['mode', 'avg_elixir', 'opponent_archetype_log_bait', 'opponent_archetype_unknown']
            print(input_df[[c for c in cols_to_show if c in input_df.columns]].to_string())

            print("\nTesting short deck (expect ValueError):")
            try:
                 prepare_prediction_input(deck_short, "Log Bait", "attack", ct, model_features, opp_archs)
            except ValueError as e:
                 print(f"  Caught expected error: {e}")

        except FileNotFoundError: print("Skipping prediction input demo: attack_mode_features.csv not found.")
        except Exception as e: print(f"Error in prediction input demo: {e}", exc_info=True)
    else:
        print("Card table failed to load. Skipping dependent demos.")
    logger.info("utils.py demo finished.")

