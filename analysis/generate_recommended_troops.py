import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
import ast
from collections import Counter

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set plot style
sns.set_theme(style="whitegrid")

# --- File Paths ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "analysis")

# Input files
PREPROCESSED_FILE = os.path.join(DATA_DIR, "preprocessed_data.csv")
CARDS_FILE = os.path.join(DATA_DIR, "cards_data.csv")

# Output plot file
RECOMMENDED_TROOPS_FILE = os.path.join(OUTPUT_DIR, "recommended_troop_lists.png")

# ----------------------------
# Helper Function
# ----------------------------
def get_mode_type(row):
    """Classifies a row as 'Attack', 'Defense', or 'Other' based on game_mode columns."""
    # Define keywords for Attack modes
    attack_cols = [
        'game_mode_ladder', 'game_mode_challenge_allcards_eventdeck_noset',
        'game_mode_ranked1v1_newarena', 'game_mode_ranked1v1_newarena2',
        'game_mode_draft_competitive', 'game_mode_showdown_friendly'
    ]
    # Define keywords for Defense/Misc modes
    defense_cols = [
        'game_mode_7xelixir_friendly', 'game_mode_cw_battle_1v1', 'game_mode_cw_duel_1v1',
        'game_mode_clanwar_boatbattle', 'game_mode_duel_1v1_friendly', 'game_mode_friendly',
        'game_mode_teamvsteam', 'game_mode_teamvsteam_tripleelixir_friendly',
        'game_mode_touchdown_draft'
    ]
    
    for col in attack_cols:
        if col in row and row[col] == 1:
            return 'Attack'
            
    for col in defense_cols:
        if col in row and row[col] == 1:
            return 'Defense'
            
    return 'Other'

# ----------------------------
# Main Script
# ----------------------------
if __name__ == "__main__":
    logger.info("--- Starting Recommended Troop List Generation ---")
    
    try:
        # 1. Load Data
        logger.info(f"Loading {PREPROCESSED_FILE}...")
        df = pd.read_csv(PREPROCESSED_FILE)
        
        logger.info(f"Loading {CARDS_FILE} to filter for troops...")
        df_cards = pd.read_csv(CARDS_FILE, usecols=['card_name', 'card_type'])
        
        # Get a set of all cards that are 'Troop'
        troop_cards = set(df_cards[df_cards['card_type'] == 'Troop']['card_name'])
        logger.info(f"Found {len(troop_cards)} unique troops.")

    except FileNotFoundError as e:
        logger.error(f"\nERROR: File not found. Make sure all files are in the correct directories.")
        logger.error(f"Missing file: {e.filename}")
        exit()
    except Exception as e:
        logger.error(f"\nAn unexpected error occurred during loading: {e}", exc_info=True)
        exit()

    try:
        # 2. Parse card lists and determine mode
        logger.info("Parsing 'cards' column and determining modes...")
        df['card_list'] = df['cards'].apply(ast.literal_eval)
        df['mode_type'] = df.apply(get_mode_type, axis=1)

        # 3. Filter for wins only
        df_wins = df[df['winner_flag'] == 1].copy()
        logger.info(f"Filtered to {len(df_wins)} winning decks.")

        # 4. Separate lists by mode and count troops
        attack_troop_counts = Counter()
        defense_troop_counts = Counter()

        for _, row in df_wins.iterrows():
            # Filter the deck to only include troops
            deck_troops = [card for card in row['card_list'] if card in troop_cards]
            
            if row['mode_type'] == 'Attack':
                attack_troop_counts.update(deck_troops)
            elif row['mode_type'] == 'Defense':
                defense_troop_counts.update(deck_troops)

        # 5. Get the Top 15 for each
        df_attack_troops = pd.DataFrame(attack_troop_counts.most_common(15), 
                                        columns=['Troop', 'Frequency']).sort_values('Frequency', ascending=False)
        df_defense_troops = pd.DataFrame(defense_troop_counts.most_common(15), 
                                         columns=['Troop', 'Frequency']).sort_values('Frequency', ascending=False)
        
        logger.info(f"Top Attack Troops: \n{df_attack_troops.head(3).to_string()}")
        logger.info(f"Top Defense Troops: \n{df_defense_troops.head(3).to_string()}")

        # 6. Plot the visualizations
        logger.info(f"Generating plot... saving to {RECOMMENDED_TROOPS_FILE}")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        fig.suptitle('Top 15 Most Successful Troops by Mode (Based on Wins)', fontsize=20)

        # Plot 1: Attack Troops
        sns.barplot(data=df_attack_troops, x='Frequency', y='Troop', ax=ax1, palette='Reds_r')
        ax1.set_title('Attack Mode Wins', fontsize=16)
        ax1.set_xlabel('Count in Winning Decks', fontsize=12)
        ax1.set_ylabel('Troop Card', fontsize=12)

        # Plot 2: Defense Troops
        sns.barplot(data=df_defense_troops, x='Frequency', y='Troop', ax=ax2, palette='Blues_r')
        ax2.set_title('Defense Mode Wins', fontsize=16)
        ax2.set_xlabel('Count in Winning Decks', fontsize=12)
        ax2.set_ylabel('') # No label, shared from left
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle
        
        # Save the figure
        plt.savefig(RECOMMENDED_TROOPS_FILE)
        
        logger.info("--- Recommended Troop List Generation Complete ---")

    except Exception as e:
        logger.error(f"\nAn unexpected error occurred during processing: {e}", exc_info=True)