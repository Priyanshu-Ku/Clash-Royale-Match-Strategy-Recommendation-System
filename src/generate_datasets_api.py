import pandas as pd
from tqdm import tqdm
from utils_old.api_helpers import get_player_data, get_battle_log

# ===============================
# Step 1: Read player tags from file
# ===============================
PLAYER_TAGS_FILE = "data/player_tags.txt"

try:
    with open(PLAYER_TAGS_FILE, "r") as f:
        PLAYER_TAGS = [line.strip() for line in f.readlines() if line.strip()]
except FileNotFoundError:
    print(f"❌ File not found: {PLAYER_TAGS_FILE}")
    exit(1)

print(f"📊 Fetching data for {len(PLAYER_TAGS)} players via Supercell API...")

# ===============================
# Step 2: Initialize lists for datasets
# ===============================
clash_data = []
deck_features = []
match_history = []

# ===============================
# Step 3: Loop through player tags
# ===============================
for tag in tqdm(PLAYER_TAGS):
    try:
        player = get_player_data(tag)
        if not player:
            print(f"⚠️ Failed to fetch player #{tag}")
            continue

        # --- Player overview ---
        clash_data.append({
            "tag": player.get("tag", ""),
            "name": player.get("name", "Unknown"),
            "trophies": player.get("trophies", 0),
            "explevel": player.get("expLevel", 0),
            "arena": player.get("arena", {}).get("name", "Unknown"),
            "clan": player.get("clan", {}).get("name", "No Clan"),
        })

        # --- Current Deck features ---
        current_deck = player.get("currentDeck", [])
        deck_features.append({
            "playerTag": player.get("tag", ""),
            "cards": [card.get("name") for card in current_deck],
            "averageElixir": (
                sum(card.get("elixirCost", 0) for card in current_deck) / len(current_deck)
                if current_deck else 0
            ),
        })

        # --- Battle history ---
        battles = get_battle_log(tag)
        if battles:
            for b in battles:
                try:
                    match_history.append({
                        "playerTag": player.get("tag", ""),
                        "opponent": b.get("opponent", [{}])[0].get("name", "Unknown"),
                        "winner": "player" if b.get("team", [{}])[0].get("crowns", 0) >
                                            b.get("opponent", [{}])[0].get("crowns", 0) else "opponent",
                        "battleTime": b.get("battleTime", ""),
                        "gameMode": b.get("gameMode", {}).get("name", "Unknown"),
                    })
                except Exception as e:
                    print(f"⚠️ Error processing a battle for {tag}: {e}")

    except Exception as e:
        print(f"⚠️ Exception fetching player {tag}: {e}")

# ===============================
# Step 4: Save datasets to CSV
# ===============================
pd.DataFrame(clash_data).to_csv("data/clash_data.csv", index=False)
pd.DataFrame(deck_features).to_csv("data/deck_features.csv", index=False)
pd.DataFrame(match_history).to_csv("data/match_history.csv", index=False)

print("✅ Data fetching complete! Saved to data/ folder.")