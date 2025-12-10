import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
import ast # We need this to parse the card lists, even if just for archetype

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

# Output plot file
ELIXIR_BOXPLOT_FILE = os.path.join(OUTPUT_DIR, "elixir_usage_by_archetype.png")

# ----------------------------
# Main Script
# ----------------------------
if __name__ == "__main__":
    logger.info("--- Starting Elixir Usage Box Plot Generation ---")
    
    try:
        # 1. Load Data
        logger.info(f"Loading {PREPROCESSED_FILE}...")
        df = pd.read_csv(PREPROCESSED_FILE)

    except FileNotFoundError as e:
        logger.error(f"\nERROR: File not found. Make sure all files are in the correct directories.")
        logger.error(f"Missing file: {e.filename}")
        exit()
    except Exception as e:
        logger.error(f"\nAn unexpected error occurred during loading: {e}", exc_info=True)
        exit()

    try:
        # 2. Reconstruct Archetype column
        logger.info("Reconstructing 'archetype' column...")
        arch_cols = [col for col in df.columns if col.startswith('player_archetype_') and col != 'player_archetype_nan']
        if not arch_cols:
            logger.error("No 'player_archetype_' columns found. Cannot create plot.")
            exit()
            
        df['archetype'] = df[arch_cols].idxmax(axis=1).str.replace('player_archetype_', '', regex=False)
        
        # Filter out 'Unknown' or 'nan' archetypes for a cleaner plot
        df_plot = df[~df['archetype'].isin(['Unknown', 'nan'])]
        logger.info(f"Filtered to {len(df_plot)} rows with known archetypes.")

        # 3. Plot the box plot
        logger.info(f"Generating plot... saving to {ELIXIR_BOXPLOT_FILE}")
        
        plt.figure(figsize=(14, 8))
        sns.boxplot(
            data=df_plot,
            x='archetype',
            y='avg_elixir',
            palette='coolwarm'
        )
        
        plt.title('Average Elixir Usage by Archetype', fontsize=18)
        plt.xlabel('Player Archetype', fontsize=14)
        plt.ylabel('Average Elixir', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(ELIXIR_BOXPLOT_FILE)
        
        logger.info("--- Elixir Box Plot Generation Complete ---")

    except Exception as e:
        logger.error(f"\nAn unexpected error occurred during processing: {e}", exc_info=True)