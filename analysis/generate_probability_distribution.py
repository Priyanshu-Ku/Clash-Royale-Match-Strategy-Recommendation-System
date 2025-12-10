import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import xgboost # Required to load the models
import logging
from typing import Tuple

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

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
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "analysis")

# Input files
WIN_SEEKER_MODEL_FILE = os.path.join(MODELS_DIR, "strategy_model_win_seeker.pkl")
LOSS_AVOIDER_MODEL_FILE = os.path.join(MODELS_DIR, "strategy_model_loss_avoider.pkl")
ATTACK_FEATURES_FILE = os.path.join(DATA_DIR, "attack_mode_features.csv")
DEFENSE_FEATURES_FILE = os.path.join(DATA_DIR, "defense_mode_features.csv")
PREPROCESSED_DATA_FILE = os.path.join(DATA_DIR, "preprocessed_data.csv")

# Output plot file
PROBABILITY_PLOT_FILE = os.path.join(OUTPUT_DIR, "probability_distribution.png")

# ----------------------------
# Data Loading Function (Copied from roc_curve script)
# ----------------------------
def load_and_split_data() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Loads the feature (X) and target (Y) data, cleans them the same way
    as in train_model.py / generate_roc_curve.py, and returns the TEST split.
    """
    logger.info("Loading data sources...")
    try:
        df_attack = pd.read_csv(ATTACK_FEATURES_FILE, low_memory=False)
        df_defense = pd.read_csv(DEFENSE_FEATURES_FILE, low_memory=False)

        # Load 'winner_flag' and coerce to numeric
        df_preprocessed = pd.read_csv(
            PREPROCESSED_DATA_FILE,
            usecols=['winner_flag'],
            low_memory=False
        )
        Y = pd.to_numeric(df_preprocessed['winner_flag'], errors='coerce')

    except FileNotFoundError as e:
        logger.error(f"Error loading data: {e}. Make sure all feature files exist.")
        raise

    # Combine X features
    df_features = pd.concat([df_attack, df_defense], ignore_index=True)
    X = df_features.drop(columns=['player_tag'], errors='ignore')

    logger.info(f"Loaded {len(X)} feature rows. Loaded {len(Y)} labels.")

    # Fix length mismatch
    if len(X) != len(Y):
        logger.warning(f"Feature count ({len(X)}) and label count ({len(Y)}) mismatch!")
        min_len = min(len(X), len(Y))
        X = X.head(min_len)
        Y = Y.head(min_len)
        logger.info(f"Truncated data to {min_len} samples.")

    # Convert 'mode' column to numeric
    if 'mode' in X.columns:
        logger.info("Converting 'mode' column to numeric (1=attack, 0=defense)...")
        X['mode'] = X['mode'].apply(
            lambda v: 1 if str(v).strip().lower() == 'attack' else 0
        )
    else:
        logger.error("Critical error: 'mode' column not found in features. Aborting.")
        raise ValueError("Missing 'mode' column in feature set.")

    # Clean object/bool feature columns (e.g. archetype flags)
    object_cols = X.select_dtypes(include=['object', 'bool']).columns.tolist()
    if object_cols:
        logger.info(f"Cleaning non-numeric cols: {object_cols}")
        for col in object_cols:
            if col.startswith('player_archetype_') or col.startswith('opponent_archetype_'):
                # Treat True/1/yes as 1, everything else as 0
                X[col] = X[col].map(
                    lambda v: 1 if str(v).strip().lower() in ('1', 'true', 'yes') else 0
                )
            else:
                tmp = pd.to_numeric(X[col], errors='coerce')
                if tmp.isna().sum() < len(tmp) * 0.5:
                    X[col] = tmp.fillna(0)
                else:
                    X[col] = X[col].astype('category').cat.codes

    # Drop NaN labels before splitting
    nan_count = Y.isna().sum()
    if nan_count > 0:
        logger.warning(f"Found {nan_count} NaN labels. Dropping those rows.")
        mask = Y.notna()
        X = X[mask].reset_index(drop=True)
        Y = Y[mask].reset_index(drop=True)
        logger.info(f"After dropping NaNs: {len(Y)} samples remain.")

    logger.info(f"Final label distribution:\n{Y.value_counts().to_string()}")

    # Split to get test set (same params as training)
    _, X_test, _, Y_test = train_test_split(
        X,
        Y,
        test_size=0.25,
        random_state=42,
        stratify=Y
    )

    logger.info(f"Data split. Using test set with {len(Y_test)} samples.")
    return X_test, Y_test

# ----------------------------
# Main Script
# ----------------------------
if __name__ == "__main__":
    logger.info("--- Starting Probability Distribution Plot Generation ---")
    
    try:
        # 1. Load Data
        X_test, Y_test = load_and_split_data()

        # 2. Load Models
        logger.info("Loading models...")
        model_win_seeker = joblib.load(WIN_SEEKER_MODEL_FILE)
        model_loss_avoider = joblib.load(LOSS_AVOIDER_MODEL_FILE)
        logger.info("Models loaded.")

        # 3. Get Predicted Probabilities
        logger.info("Calculating predicted probabilities...")
        probs_win_seeker = model_win_seeker.predict_proba(X_test)[:, 1]
        probs_loss_avoider = model_loss_avoider.predict_proba(X_test)[:, 1]

        # 4. Create a DataFrame for plotting
        Y_test_int = Y_test.astype(int)

        df_plot = pd.DataFrame({
            'True Label': Y_test_int.map({0: 'Lose (0)', 1: 'Win (1)'}),
            'Win-Seeker Prob.': probs_win_seeker,
            'Loss-Avoider Prob.': probs_loss_avoider
        })


        # 5. Plot the distributions
        logger.info(f"Generating plot... saving to {PROBABILITY_PLOT_FILE}")
        
        # Create a 2x1 figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

        # --- Plot 1: Win-Seeker ---
        sns.kdeplot(data=df_plot, x='Win-Seeker Prob.', hue='True Label', 
                    fill=True, common_norm=False, palette="Set1",
                    ax=ax1)
        ax1.set_title('Win-Seeker (Imbalanced) Probability Distribution', fontsize=16)
        ax1.set_xlabel('Predicted Win Probability', fontsize=12)
        ax1.set_ylabel('Density', fontsize=12)
        ax1.axvline(0.5, color='k', linestyle='--', label='Decision Threshold (0.5)')
        ax1.legend()

        # --- Plot 2: Loss-Avoider ---
        sns.kdeplot(data=df_plot, x='Loss-Avoider Prob.', hue='True Label', 
                    fill=True, common_norm=False, palette="Set2",
                    ax=ax2)
        ax2.set_title('Loss-Avoider (SMOTE-Balanced) Probability Distribution', fontsize=16)
        ax2.set_xlabel('Predicted Win Probability', fontsize=12)
        ax2.set_ylabel('Density', fontsize=12)
        ax2.axvline(0.5, color='k', linestyle='--', label='Decision Threshold (0.5)')
        ax2.legend()

        plt.tight_layout()
        
        # Save the figure
        plt.savefig(PROBABILITY_PLOT_FILE)
        
        logger.info("--- Plot Generation Complete ---")

    except FileNotFoundError as e:
        logger.error(f"\nERROR: File not found. Make sure all files are in the correct directories.")
        logger.error(f"Missing file: {e.filename}")
    except Exception as e:
        logger.error(f"\nAn unexpected error occurred: {e}", exc_info=True)