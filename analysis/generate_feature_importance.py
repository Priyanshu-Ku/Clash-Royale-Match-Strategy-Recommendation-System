import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import xgboost # This is required to load the models
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set plot style
sns.set_theme(style="whitegrid")

# --- Configuration ---
# Assume this script is in 'analysis/' and the root is one level up
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "analysis") # Save plots in the same dir

# Input files
WIN_SEEKER_MODEL_FILE = os.path.join(MODELS_DIR, "strategy_model_win_seeker.pkl")
LOSS_AVOIDER_MODEL_FILE = os.path.join(MODELS_DIR, "strategy_model_loss_avoider.pkl")
ATTACK_FEATURES_FILE = os.path.join(DATA_DIR, "attack_mode_features.csv")
DEFENSE_FEATURES_FILE = os.path.join(DATA_DIR, "defense_mode_features.csv")

# Output plot files
WIN_SEEKER_PLOT_FILE = os.path.join(OUTPUT_DIR, "win_seeker_feature_importance.png")
LOSS_AVOIDER_PLOT_FILE = os.path.join(OUTPUT_DIR, "loss_avoider_feature_importance.png")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

logger.info("--- Starting Feature Importance Analysis ---")

try:
    # --- 1. Load Models ---
    logger.info(f"Loading models from {MODELS_DIR}...")
    model_win_seeker = joblib.load(WIN_SEEKER_MODEL_FILE)
    model_loss_avoider = joblib.load(LOSS_AVOIDER_MODEL_FILE)
    logger.info("Models loaded successfully.")

    # --- 2. Load Feature Names ---
    # We must get the feature list in the *exact* order the model was trained on.
    logger.info(f"Loading feature names from {ATTACK_FEATURES_FILE} and {DEFENSE_FEATURES_FILE}...")
    df_attack = pd.read_csv(ATTACK_FEATURES_FILE, index_col=False)
    df_defense = pd.read_csv(DEFENSE_FEATURES_FILE, index_col=False)
    
    df_features = pd.concat([df_attack, df_defense], ignore_index=True)
    
    # The feature set 'X' used for training was df_features.drop(columns=['player_tag'], errors='ignore')
    X = df_features.drop(columns=['player_tag'], errors='ignore')
    feature_names = X.columns.tolist()
    logger.info(f"Successfully loaded {len(feature_names)} feature names.")

    # --- 3. Analyze Win-Seeker Model ---
    logger.info("Analyzing 'Win-Seeker' model...")
    importances_win = model_win_seeker.feature_importances_
    df_imp_win = pd.DataFrame({
        'feature': feature_names, 
        'importance': importances_win
    }).sort_values(by='importance', ascending=False).head(20)

    plt.figure(figsize=(10, 8))
    sns.barplot(data=df_imp_win, x='importance', y='feature', palette='viridis')
    plt.title('Win-Seeker Model: Top 20 Feature Importances', fontsize=16)
    plt.xlabel('Importance (Gini)', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    plt.savefig(WIN_SEEKER_PLOT_FILE)
    logger.info(f"Saved Win-Seeker importance plot to {WIN_SEEKER_PLOT_FILE}")

    # --- 4. Analyze Loss-Avoider Model ---
    logger.info("Analyzing 'Loss-Avoider' model...")
    importances_loss = model_loss_avoider.feature_importances_
    df_imp_loss = pd.DataFrame({
        'feature': feature_names, 
        'importance': importances_loss
    }).sort_values(by='importance', ascending=False).head(20)

    plt.figure(figsize=(10, 8))
    sns.barplot(data=df_imp_loss, x='importance', y='feature', palette='plasma')
    plt.title('Loss-Avoider Model: Top 20 Feature Importances', fontsize=16)
    plt.xlabel('Importance (Gini)', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    plt.savefig(LOSS_AVOIDER_PLOT_FILE)
    logger.info(f"Saved Loss-Avoider importance plot to {LOSS_AVOIDER_PLOT_FILE}")

    logger.info("\n--- Analysis Complete ---")
    logger.info("Top 5 Features for Win-Seeker:")
    print(df_imp_win.head(5).to_string(index=False))
    
    logger.info("\nTop 5 Features for Loss-Avoider:")
    print(df_imp_loss.head(5).to_string(index=False))

except FileNotFoundError as e:
    logger.error(f"\nERROR: File not found. Make sure all files are in the correct directories.")
    logger.error(f"Missing file: {e.filename}")
except Exception as e:
    logger.error(f"\nAn unexpected error occurred: {e}", exc_info=True)