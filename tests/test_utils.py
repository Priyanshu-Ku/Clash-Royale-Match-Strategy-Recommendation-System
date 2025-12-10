"""Tests for utility functions in src/utils.py"""

import pytest
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any # Added Tuple, Any
from datetime import datetime, timedelta # Added datetime imports for cache test
import joblib # For mocking model load
import time # For cache testing timing

# --- Standard Import (Assumes correct package structure) ---
# Ensure empty __init__.py exists in project root and src/
# Ensure NO __init__.py exists in tests/
# Ensure utils.py is a FILE in src/, not a folder
try:
    from src import utils
except ImportError as e:
    # Use pytest.fail for clear test failure message during collection/import
    pytest.fail(
        f"Failed to import 'from src import utils'. Check __init__.py files in root/src and ensure src/utils.py exists. "
        f"PYTHONPATH: {sys.path}. Error: {e}",
        pytrace=False
    )
# --- End Import ---


# --- Fixtures ---

@pytest.fixture(scope="module")
def mock_card_table() -> pd.DataFrame:
    """Provides a basic mock DataFrame for card data, simulating load_card_table preprocessing."""
    data = {
        'card_name_original': ['Knight', 'Archers', 'Arrows', 'Giant', 'Minions', 'Fireball', 'X-Bow', 'X-Bow (Lv 14)', 'Tesla'],
        'card_name': ['knight', 'archers', 'arrows', 'giant', 'minions', 'fireball', 'x-bow', 'x-bow (lv 14)', 'tesla'],
        'elixir_cost': [3, 3, 3, 5, 3, 4, 6, 6, 4],
        'card_type': ['troop', 'troop', 'spell', 'troop', 'troop', 'spell', 'building', 'building', 'building'],
        'hitpoints': [1400, 130, 0, 3300, 200, 0, 500, 550, 700],
        'dps': [120, 60, 0, 80, 80, 0, 25, 27, 100],
        'damage': [150, 70, 240, 150, 90, 570, 40, 40, 110],
        'targets': ['Ground', 'Air & Ground', 'Air & Ground', 'Buildings', 'Air & Ground', 'Air & Ground', 'Ground', 'Ground', 'Air & Ground'],
        'description': ['Tank', 'Support', 'Spell', 'Tank', 'Swarm', 'Spell', 'Siege', 'Siege', 'Defense Building'],
    }
    return pd.DataFrame(data)

@pytest.fixture
def valid_deck_8() -> List[str]:
    return ['Knight', 'Archers', 'Arrows', 'Giant', 'Minions', 'Fireball', 'X-Bow', 'Tesla']

@pytest.fixture
def mock_expected_features() -> List[str]:
    """Mock list of feature names the model expects."""
    return [
        'mode', 'avg_elixir', 'total_dps', 'avg_dps', 'defense_strength',
        'offense_strength', 'spell_ratio', 'troop_ratio', 'building_ratio',
        'avg_hitpoints', 'synergy_score', 'cohesion', 'type_synergy',
        'win_synergy', 'opponent_archetype_log_bait', 'opponent_archetype_hog_cycle',
        'opponent_archetype_beatdown', 'opponent_archetype_siege',
        'opponent_archetype_unknown', 'opponent_archetype_nan', 'trophies', 'explevel',
        'counter_score', 'player_archetype_beatdown', 'player_archetype_unknown',
        'embedding_0', 'embedding_1', 'embedding_2', 'embedding_3',
        'embedding_4', 'embedding_5', 'embedding_6', 'embedding_7',
        'embedding_8', 'embedding_9', 'embedding_10', 'embedding_11',
        'embedding_12', 'embedding_13', 'embedding_14', 'embedding_15'
    ]

@pytest.fixture
def mock_known_opponent_archetypes(mock_expected_features: List[str]) -> List[str]:
     return [f for f in mock_expected_features if f.startswith('opponent_archetype_')]


# --- Tests ---

def test_load_card_table_success(monkeypatch, tmp_path):
    csv_path = tmp_path / "test_cards.csv"
    mock_df_data = {'card_name': ['Knight', 'Archers'], 'elixir_cost': [3, 3]}
    pd.DataFrame(mock_df_data).to_csv(csv_path, index=False)
    # --- Access utils module variables directly ---
    monkeypatch.setattr(utils, '_card_table_cache', None)
    monkeypatch.setattr(utils, '_card_table_load_time', None)
    # --- Call function via module ---
    df = utils.load_card_table(csv_path)
    assert df is not None; assert len(df) == 2; assert 'card_name_original' in df.columns
    assert df['card_name'].tolist() == ['knight', 'archers']

def test_load_card_table_file_not_found(tmp_path, monkeypatch):
    monkeypatch.setattr(utils, '_card_table_cache', None)
    monkeypatch.setattr(utils, '_card_table_load_time', None)
    non_existent_path = tmp_path / "no_cards_here.csv"
    with pytest.raises(FileNotFoundError):
        utils.load_card_table(non_existent_path)

def test_load_card_table_cache(monkeypatch, tmp_path):
    csv_path = tmp_path / "cache_test_cards.csv"
    pd.DataFrame({'card_name': ['Knight'], 'elixir_cost': [3]}).to_csv(csv_path, index=False)
    monkeypatch.setattr(utils, '_card_table_cache', None)
    monkeypatch.setattr(utils, '_card_table_load_time', None)

    df1 = utils.load_card_table(csv_path)
    load_time1 = utils._card_table_load_time # Get time after load
    assert df1 is not None and load_time1 is not None

    time.sleep(0.1)
    df2 = utils.load_card_table(csv_path, max_age_seconds=60)
    load_time2 = utils._card_table_load_time # Get time again

    assert df2 is not None; assert len(df1) == len(df2)
    assert load_time1 == load_time2 # Should be same if cached

def test_load_card_table_cache_expiry(monkeypatch, tmp_path):
     csv_path = tmp_path / "expiry_test_cards.csv"
     pd.DataFrame({'card_name': ['Knight'], 'elixir_cost': [3]}).to_csv(csv_path, index=False)
     monkeypatch.setattr(utils, '_card_table_cache', None)
     monkeypatch.setattr(utils, '_card_table_load_time', None)

     df1 = utils.load_card_table(csv_path, max_age_seconds=1)
     load_time1 = utils._card_table_load_time

     time.sleep(1.1)

     df2 = utils.load_card_table(csv_path, max_age_seconds=1)
     load_time2 = utils._card_table_load_time

     assert df2 is not None
     assert load_time1 is not None and load_time2 is not None
     assert load_time2 > load_time1

# Test _get_card_row
def test_get_card_row_exact_match(mock_card_table):
    card_data = utils._get_card_row(mock_card_table, 'Knight'); assert isinstance(card_data, dict)
    assert card_data['card_name'] == 'knight'; assert card_data['elixir_cost'] == 3

def test_get_card_row_partial_match(mock_card_table):
    card_data = utils._get_card_row(mock_card_table, 'X-Bow'); assert card_data is not None
    assert card_data['card_name'] == 'x-bow'

def test_get_card_row_case_insensitive(mock_card_table):
    card_data = utils._get_card_row(mock_card_table, 'giant'); assert card_data is not None
    assert card_data['card_name'] == 'giant'

def test_get_card_row_not_found(mock_card_table):
    card_data = utils._get_card_row(mock_card_table, 'Pekka'); assert card_data is None

# Test validate_deck_cards
def test_validate_deck_cards_valid(mock_card_table, valid_deck_8):
    is_valid, missing = utils.validate_deck_cards(valid_deck_8, mock_card_table)
    assert is_valid is True; assert len(missing) == 0

def test_validate_deck_cards_invalid_card(mock_card_table):
    invalid_deck = ['Knight', 'Archers', 'Unknown', 'Giant', 'Minions', 'Fireball', 'X-Bow', 'Tesla']
    is_valid, missing = utils.validate_deck_cards(invalid_deck, mock_card_table)
    assert is_valid is False; assert missing == ['Unknown']

def test_validate_deck_cards_wrong_size_short(mock_card_table):
    short_deck = ['Knight', 'Archers']
    is_valid, missing = utils.validate_deck_cards(short_deck, mock_card_table)
    assert is_valid is False; assert len(missing) == 0

def test_validate_deck_cards_wrong_size_long(mock_card_table):
    long_deck = ['Knight'] * 9
    is_valid, missing = utils.validate_deck_cards(long_deck, mock_card_table)
    assert is_valid is False; assert len(missing) == 0

# Test validate_archetype
def test_validate_archetype():
    assert utils.validate_archetype("  Hog Cycle ") == "hog cycle"
    assert utils.validate_archetype("Beatdown") == "beatdown"
    # ... rest of validate_archetype tests ...
    assert utils.validate_archetype(None) == "unknown"
    assert utils.validate_archetype("INVALID ARCHETYPE") == "unknown"

# Test generate_cache_key
def test_generate_cache_key_consistency():
    deck = ['A', 'B', 'C']; opp = 'T1'
    key1 = utils.generate_cache_key("p", deck=deck, opp=opp)
    key2 = utils.generate_cache_key("p", deck=deck, opp=opp)
    assert key1 == key2 and len(key1) == 32

def test_generate_cache_key_order_invariant():
    deck1 = ['A', 'B', 'C']; deck2 = ['C', 'A', 'B']; opp = 'T1'
    key1 = utils.generate_cache_key("p", deck=deck1, opp=opp)
    key2 = utils.generate_cache_key("p", deck=deck2, opp=opp)
    assert key1 == key2

def test_generate_cache_key_different_inputs():
    deck1 = ['A', 'B', 'C']; deck2 = ['A', 'B', 'D']; opp1 = 'T1'; opp2 = 'T2'
    key1 = utils.generate_cache_key("p", deck=deck1, opp=opp1)
    key2 = utils.generate_cache_key("p", deck=deck2, opp=opp1)
    key3 = utils.generate_cache_key("p", deck=deck1, opp=opp2)
    key4 = utils.generate_cache_key("p2", deck=deck1, opp=opp1)
    assert key1 != key2; assert key1 != key3; assert key1 != key4

# --- Test prepare_prediction_input ---

def test_prepare_prediction_input_structure(
    mock_card_table, mock_expected_features, mock_known_opponent_archetypes, valid_deck_8
):
    try:
        df = utils.prepare_prediction_input(
            valid_deck_8, "Log Bait", "attack", mock_card_table,
            mock_expected_features, mock_known_opponent_archetypes
        )
        assert isinstance(df, pd.DataFrame); assert df.shape == (1, len(mock_expected_features))
        assert list(df.columns) == mock_expected_features; assert not df.isnull().values.any()
        assert all(df[col].dtype == np.float32 for col in df.columns)
    except Exception as e: pytest.fail(f"Unexpected exception: {e}")


def test_prepare_prediction_input_mode_encoding(
     mock_card_table, mock_expected_features, mock_known_opponent_archetypes, valid_deck_8
):
    df_attack = utils.prepare_prediction_input(valid_deck_8, "Log Bait", "attack", mock_card_table, mock_expected_features, mock_known_opponent_archetypes)
    df_defense = utils.prepare_prediction_input(valid_deck_8, "Log Bait", "defense", mock_card_table, mock_expected_features, mock_known_opponent_archetypes)
    assert 'mode' in df_attack.columns and df_attack['mode'].iloc[0] == 1.0
    assert 'mode' in df_defense.columns and df_defense['mode'].iloc[0] == 0.0

def test_prepare_prediction_input_archetype_encoding(
     mock_card_table, mock_expected_features, mock_known_opponent_archetypes, valid_deck_8
):
    assert 'opponent_archetype_log_bait' in mock_expected_features
    # ... rest of archetype encoding assertions ...
    df_logbait = utils.prepare_prediction_input(valid_deck_8, "Log Bait", "attack", mock_card_table, mock_expected_features, mock_known_opponent_archetypes)
    assert df_logbait['opponent_archetype_log_bait'].iloc[0] == 1.0
    assert df_logbait['opponent_archetype_unknown'].iloc[0] == 0.0

def test_prepare_prediction_input_deck_size_error(
     mock_card_table, mock_expected_features, mock_known_opponent_archetypes
):
     short_deck = ['Knight', 'Archers']
     with pytest.raises(ValueError, match=f"exactly {utils.PREDICTION_DECK_SIZE} cards"):
          utils.prepare_prediction_input(short_deck, "Log Bait", "attack", mock_card_table, mock_expected_features, mock_known_opponent_archetypes)

def test_prepare_prediction_input_default_values(
     mock_card_table, mock_expected_features, mock_known_opponent_archetypes, valid_deck_8
):
     df = utils.prepare_prediction_input(valid_deck_8, "Log Bait", "attack", mock_card_table, mock_expected_features, mock_known_opponent_archetypes)
     assert 'trophies' in df.columns and df['trophies'].iloc[0] == 7000.0
     # ... rest of default value assertions ...
     assert 'player_archetype_beatdown' in df.columns and df['player_archetype_beatdown'].iloc[0] == 0.0


# --- Tests for Heuristic Functions ---

def test_defense_strength_heuristic(mock_card_table):
    deck = ['Giant', 'Knight', 'Arrows', 'X-Bow']
    strength = utils.defense_strength_heuristic(deck, mock_card_table)
    assert isinstance(strength, float); assert strength > 0

def test_offense_strength_heuristic(mock_card_table):
    deck = ['Giant', 'Knight', 'Arrows', 'X-Bow']
    strength = utils.offense_strength_heuristic(deck, mock_card_table)
    assert isinstance(strength, float); assert strength > 0

def test_synergy_score_heuristic(mock_card_table, valid_deck_8):
    score = utils.synergy_score_heuristic(valid_deck_8, mock_card_table)
    assert isinstance(score, float)

def test_calculate_deck_features(mock_card_table, valid_deck_8):
    features = utils.calculate_deck_features(valid_deck_8, mock_card_table)
    assert isinstance(features, dict)
    # ... rest of calculate_deck_features assertions ...
    expected_avg_elixir = (3+3+3+5+3+4+6+4) / 8
    assert np.isclose(features['avg_elixir'], expected_avg_elixir)

def test_calculate_deck_features_invalid_card(mock_card_table):
     deck = ['Giant', 'Knight', 'Invalid', 'Fireball', 'Minions', 'Arrows', 'X-Bow', 'Tesla']
     features = utils.calculate_deck_features(deck, mock_card_table)
     assert isinstance(features, dict); assert pd.notna(features['avg_elixir'])
     # ... rest of calculate_deck_features_invalid_card assertions ...
     assert np.isclose(features['troop_ratio'], 3/7)

def test_create_deck_embedding(mock_card_table, valid_deck_8):
     embedding_size = 16
     embedding = utils.create_deck_embedding(valid_deck_8, mock_card_table, embedding_size)
     assert isinstance(embedding, np.ndarray); assert embedding.shape == (embedding_size,)
     # ... rest of create_deck_embedding assertions ...
     assert embedding.dtype == np.float32

def test_infer_archetype(mock_card_table):
     siege_deck = ['X-Bow', 'Tesla', 'Knight', 'Archers', 'Arrows', 'Fireball', 'Minions', 'Giant']
     beatdown_deck = ['Giant','Knight','Archers','Minions','Fireball','Arrows','Giant','Knight']
     cycle_deck = ['Knight', 'Archers', 'Arrows', 'Minions'] * 2

     assert utils.infer_archetype(siege_deck, mock_card_table) == "siege"
     assert utils.infer_archetype(beatdown_deck, mock_card_table) == "beatdown"
     assert utils.infer_archetype(cycle_deck, mock_card_table) == "cycle"


# --- Mock load_model fixture and tests ---
@pytest.fixture
def mock_load_model(monkeypatch):
    """Mocks joblib.load and clears utils._model_cache."""
    mock_model = {"predict_proba": lambda x: np.array([[0.2, 0.8]])}
    def mock_joblib_load(path):
        path_str = str(path)
        if "non_existent" in path_str: raise FileNotFoundError("Mock file not found")
        if "invalid" in path_str: raise ValueError("Mock invalid file")
        return mock_model
    monkeypatch.setattr(joblib, "load", mock_joblib_load)
    # --- Use imported global for clearing ---
    _model_cache.clear()
    return mock_model

def test_load_model_success(tmp_path, mock_load_model):
    model_path = tmp_path / "good_model.pkl"; model_path.touch()
    # --- Use imported function ---
    model = load_model(str(model_path))
    assert model == mock_load_model

def test_load_model_not_found(mock_load_model):
    with pytest.raises(FileNotFoundError):
        load_model("models/non_existent_model.pkl")

def test_load_model_invalid_file(tmp_path, mock_load_model):
     invalid_path = tmp_path / "invalid_model.pkl"; invalid_path.touch()
     with pytest.raises(ValueError, match="Failed to load model artifact: Mock invalid file"):
          load_model(str(invalid_path))

def test_load_model_cache(tmp_path, mock_load_model, monkeypatch):
     model_path = tmp_path / "cached_model.pkl"; model_path.touch()
     start_time = datetime.now(); call_count = {'now': 0}
     def mock_datetime_now():
         call_count['now'] += 1
         return start_time + timedelta(seconds=call_count['now'] * 10)

     # --- Patch datetime directly in the utils module ---
     monkeypatch.setattr(utils.datetime, "now", mock_datetime_now)

     # Load once
     model1 = load_model(str(model_path), max_age_seconds=60)
     # --- Use imported global ---
     assert str(model_path.resolve()) in _model_cache
     model1_cache_time = _model_cache[str(model_path.resolve())][1]

     # Load again immediately
     model2 = load_model(str(model_path), max_age_seconds=60)
     assert model1 is model2
     assert _model_cache[str(model_path.resolve())][1] == model1_cache_time

     # Simulate expiry
     start_time += timedelta(seconds=100)
     model3 = load_model(str(model_path), max_age_seconds=60)
     model3_cache_time = _model_cache[str(model_path.resolve())][1]
     assert model3_cache_time > model1_cache_time

     # Load again within new TTL
     model4 = load_model(str(model_path), max_age_seconds=60)
     assert model3 is model4

