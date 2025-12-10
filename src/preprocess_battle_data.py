# src/preprocess_battle_data.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
import joblib

from src.utils_old.deck_utils import parse_cards_field, identify_archetype, average_elixir_from_cards
from src.utils_old.synergy import compute_pairwise_synergy, deck_cohesion_score
from src.utils_old.elixir_utils import compute_elixir_features_from_deck, elixir_trend_from_actions

DATA_DIR = "data"
OUTPUT_CLEAN = os.path.join(DATA_DIR, "strategies_clean.csv")
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "winprob_model.joblib")
PREPROC_PATH = os.path.join(MODEL_DIR, "preprocessor.joblib")

os.makedirs(MODEL_DIR, exist_ok=True)

# Load raw files (robust to missing)
clash_path = os.path.join(DATA_DIR, "clash_data.csv")
decks_path = os.path.join(DATA_DIR, "deck_features.csv")
matches_path = os.path.join(DATA_DIR, "match_history.csv")

print("Loading files...")
players_df = pd.read_csv(clash_path) if os.path.exists(clash_path) else pd.DataFrame()
decks_df = pd.read_csv(decks_path) if os.path.exists(decks_path) else pd.DataFrame()
matches_df = pd.read_csv(matches_path) if os.path.exists(matches_path) else pd.DataFrame()

# ensure consistent column names
matches_df.columns = [c.strip() for c in matches_df.columns]
decks_df.columns = [c.strip() for c in decks_df.columns]
players_df.columns = [c.strip() for c in players_df.columns]

# create a map from playerTag -> deck & avg elixir if available
deck_map = {}
for _, row in decks_df.iterrows():
    tag = row.get("playerTag") or row.get("player_tag") or row.get("player")
    cards_raw = row.get("cards", "")
    cards = parse_cards_field(cards_raw)
    avg_elixir = row.get("averageElixir", None)
    if avg_elixir is None:
        avg_elixir = average_elixir_from_cards(cards)
    deck_map[tag] = {"cards": cards, "avg_elixir": float(avg_elixir)}

# helper to get opponent deck from deck_map, else empty
def get_deck_for_tag(tag):
    if pd.isna(tag):
        return []
    return deck_map.get(tag, {}).get("cards", [])

# Build enriched matches dataset
rows = []
print("Processing matches and enriching features...")
for _, m in tqdm(matches_df.iterrows(), total=len(matches_df)):
    try:
        player_tag = m.get("playerTag") or m.get("player_tag") or m.get("player_tag")
        opponent_tag = m.get("opponent") or m.get("opponent_tag") or m.get("opponentTag") or m.get("opponent")
        # parse player cards: prefer deck_features entry; else maybe matches contains cards columns
        player_cards = deck_map.get(player_tag, {}).get("cards", [])
        # if deck missing, try any in matches row (fields like 'player_deck')
        if not player_cards:
            if "player_deck" in m.index:
                player_cards = parse_cards_field(m["player_deck"])
        opponent_cards = get_deck_for_tag(opponent_tag)
        if not opponent_cards:
            if "opponent_deck" in m.index:
                opponent_cards = parse_cards_field(m["opponent_deck"])

        # archetypes
        player_arch = identify_archetype(player_cards)
        opponent_arch = identify_archetype(opponent_cards)

        # average elixir (fallback)
        avg_elixir = deck_map.get(player_tag, {}).get("avg_elixir", None)
        if avg_elixir is None:
            avg_elixir = average_elixir_from_cards(player_cards)

        # elixir features (best-effort)
        elix_feats = compute_elixir_features_from_deck(player_cards)

        # synergy/cohesion
        synergy = compute_pairwise_synergy(player_cards)
        cohesion = deck_cohesion_score(player_cards)

        # mode difficulty heuristic
        gamemode = m.get("game_mode") or m.get("gameMode") or m.get("gameMode", "")
        gm = str(gamemode).lower()
        if "challenge" in gm or "tournament" in gm:
            mode_diff = 0.9
        elif "ranked" in gm or "ladder" in gm:
            mode_diff = 0.7
        else:
            mode_diff = 0.5

        # winner flag (1 = player won)
        winner_field = m.get("winner") or m.get("result") or m.get("outcome") or m.get("winner_flag")
        if isinstance(winner_field, str):
            w = 1 if winner_field.lower().startswith("player") or winner_field.lower().startswith("win") else 0
        else:
            w = int(winner_field) if pd.notna(winner_field) else None

        rows.append({
            "playerTag": player_tag,
            "player_archetype": player_arch,
            "opponent_archetype": opponent_arch,
            "avg_elixir": float(avg_elixir) if avg_elixir is not None else 0.0,
            "est_elixir_per_min": float(elix_feats.get("est_elixir_per_min", 0.0)),
            "synergy": float(synergy),
            "cohesion": float(cohesion),
            "mode_difficulty": float(mode_diff),
            "game_mode": gamemode,
            "winner_flag": w
        })
    except Exception as e:
        # skip this match with log
        print("Warning, skipping a match due to:", e)
        continue

enriched = pd.DataFrame(rows)
print(f"Enriched matches: {len(enriched)} rows")

# Drop rows without label
enriched = enriched[enriched["winner_flag"].notna()]
if enriched.empty:
    print("No labeled matches available to train model. Exiting after saving enriched CSV.")
    enriched.to_csv(OUTPUT_CLEAN, index=False)
    exit(0)

# One-hot encode archetypes and game_mode for modeling
enriched = pd.get_dummies(enriched, columns=["player_archetype", "opponent_archetype", "game_mode"], dummy_na=True)

# Fill NaNs
enriched.fillna(0, inplace=True)

# Save intermediate cleaned dataset
enriched.to_csv(OUTPUT_CLEAN, index=False)
print(f"Saved enriched strategies dataset to {OUTPUT_CLEAN}")

# Prepare X/y
X = enriched.drop(columns=["playerTag", "winner_flag"])
y = enriched["winner_flag"].astype(int)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Simple standard scaler + logistic regression pipeline (we'll save model and a simple preprocessor)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipeline = Pipeline([
    ("scale", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000))
])

print("Training win-probability logistic regression model...")
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:,1]
print("Accuracy:", accuracy_score(y_test, y_pred))
try:
    print("AUC:", roc_auc_score(y_test, y_prob))
except Exception:
    pass
print(classification_report(y_test, y_pred))

# Save model and (simple) preprocessor (pipeline)
joblib.dump(pipeline, MODEL_PATH)
print(f"Saved win-probability model to {MODEL_PATH}")

# Save a lightweight preprocessor (here, columns used)
preproc_meta = {"feature_columns": X.columns.tolist()}
joblib.dump(preproc_meta, PREPROC_PATH)
print(f"Saved preprocessor metadata to {PREPROC_PATH}")

print("All done.")
