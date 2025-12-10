import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
import ast
from itertools import combinations
from collections import defaultdict

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
SYNERGY_HEATMAP_FILE = os.path.join(OUTPUT_DIR, "synergy_heatmap.png")

# ----------------------------
# Main Script
# ----------------------------
if __name__ == "__main__":
    logger.info("--- Starting Synergy Heatmap Generation ---")
    
    try:
        # 1. Load Data
        logger.info(f"Loading {PREPROCESSED_FILE}...")
        df = pd.read_csv(PREPROCESSED_FILE, usecols=['cards', 'winner_flag'])
        
        logger.info(f"Loading {CARDS_FILE} to filter for core cards...")
        df_cards = pd.read_csv(CARDS_FILE, usecols=['card_name', 'card_type'])
        
        # We only want to see synergy between Troops and Buildings, not Spells
        core_cards = set(df_cards[df_cards['card_type'].isin(['Troop', 'Building'])]['card_name'])
        logger.info(f"Filtered to {len(core_cards)} core cards (Troops/Buildings).")

    except FileNotFoundError as e:
        logger.error(f"\nERROR: File not found. Make sure all files are in the correct directories.")
        logger.error(f"Missing file: {e.filename}")
        exit()
    except Exception as e:
        logger.error(f"\nAn unexpected error occurred during loading: {e}", exc_info=True)
        exit()

    try:
        # 2. Parse card lists
        logger.info("Parsing 'cards' column...")
        # Use ast.literal_eval to safely parse the string representation of a list
        df['card_list'] = df['cards'].apply(ast.literal_eval)
        
        # Filter lists to only include our core cards
        df['card_list'] = df['card_list'].apply(lambda cardlist: [card for card in cardlist if card in core_cards])

        # 3. Find Top 20 most popular core cards to build the heatmap
        all_cards_list = [card for deck in df['card_list'] for card in deck]
        top_20_cards = pd.Series(all_cards_list).value_counts().nlargest(20).index.tolist()
        logger.info(f"Top 20 core cards selected for heatmap: {top_20_cards}")

        # 4. Calculate win rates for card pairs
        pair_wins = defaultdict(int)
        pair_counts = defaultdict(int)

        logger.info("Calculating win rates for card pairs... (This may take a moment)")
        for _, row in df.iterrows():
            # Only use pairs from our Top 20 list
            deck_cards = set(row['card_list']) & set(top_20_cards)
            
            # Create all unique combinations of 2 cards from this deck
            for card_a, card_b in combinations(sorted(deck_cards), 2):
                pair = (card_a, card_b)
                pair_counts[pair] += 1
                if row['winner_flag'] == 1:
                    pair_wins[pair] += 1

        # 5. Create DataFrame from the pair data
        heatmap_data = []
        for pair, count in pair_counts.items():
            if count > 10:  # Only include pairs that appeared at least 10 times
                win_rate = pair_wins[pair] / count
                heatmap_data.append({'card_a': pair[0], 'card_b': pair[1], 'win_rate': win_rate})
                # Add the reverse pair for a full matrix
                heatmap_data.append({'card_a': pair[1], 'card_b': pair[0], 'win_rate': win_rate})

        df_heatmap = pd.DataFrame(heatmap_data)

        if df_heatmap.empty:
            logger.error("No card pairs found with sufficient data. Aborting plot generation.")
            exit()
            
        # 6. Pivot data for heatmap
        logger.info("Pivoting data for heatmap...")
        df_pivot = df_heatmap.pivot_table(index='card_a', columns='card_b', values='win_rate', aggfunc='mean')
        
        # Ensure it only includes our Top 20 list
        df_pivot = df_pivot.reindex(index=top_20_cards, columns=top_20_cards)

        # 7. Plot the heatmap
        logger.info(f"Generating plot... saving to {SYNERGY_HEATMAP_FILE}")
        plt.figure(figsize=(16, 12))
        sns.heatmap(
            df_pivot, 
            annot=True, 
            fmt=".1%",  # Format as percentage
            cmap="viridis", 
            linewidths=.5,
            cbar_kws={'label': 'Win Rate'}
        )
        plt.title('Deck Synergy Heatmap (Win Rate for Core Card Pairs)', fontsize=18)
        plt.xlabel('Card B', fontsize=14)
        plt.ylabel('Card A', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(SYNERGY_HEATMAP_FILE)
        
        logger.info("--- Synergy Heatmap Generation Complete ---")

    except Exception as e:
        logger.error(f"\nAn unexpected error occurred during processing: {e}", exc_info=True)