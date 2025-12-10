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
ROC_CURVE_PLOT_FILE = os.path.join(OUTPUT_DIR, "roc_curve_comparison.png")

# ----------------------------
# Data Loading Function
# ----------------------------
def load_and_split_data() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Loads features + labels and applies the SAME cleaning pipeline used for training.
    Returns ONLY the test split for ROC evaluation.
    """
    logger.info("Loading data sources...")
    try:
        df_attack = pd.read_csv(ATTACK_FEATURES_FILE, low_memory=False)
        df_defense = pd.read_csv(DEFENSE_FEATURES_FILE, low_memory=False)

        df_preprocessed = pd.read_csv(
            PREPROCESSED_DATA_FILE, 
            usecols=['winner_flag'], 
            low_memory=False
        )

        # Convert winner_flag to numeric (invalid -> NaN)
        Y = pd.to_numeric(df_preprocessed['winner_flag'], errors='coerce')

    except FileNotFoundError as e:
        logger.error(f"Error loading data: {e}. Make sure all feature files exist.")
        raise

    # Combine attack + defense features
    df_features = pd.concat([df_attack, df_defense], ignore_index=True)
    X = df_features.drop(columns=['player_tag'], errors='ignore')

    logger.info(f"Loaded {len(X)} feature rows. Loaded {len(Y)} labels.")

    # ----- Fix length mismatch -----
    if len(X) != len(Y):
        logger.warning(f"Feature count ({len(X)}) and label count ({len(Y)}) mismatch!")
        min_len = min(len(X), len(Y))
        X = X.head(min_len)
        Y = Y.head(min_len)
        logger.info(f"Truncated data to {min_len} samples.")

    # ----- Convert 'mode' column -----
    if "mode" not in X.columns:
        raise ValueError("Critical error: Missing 'mode' column in features.")

    logger.info("Converting 'mode' column to numeric (1=attack, 0=defense)...")
    X["mode"] = X["mode"].apply(
        lambda v: 1 if str(v).strip().lower() == "attack" else 0
    )

    # ----- Clean object dtype columns (archetype booleans, stray strings) -----
    object_cols = X.select_dtypes(include=["object", "bool"]).columns.tolist()
    if object_cols:
        logger.info(
            f"Cleaning non-numeric cols: {object_cols}"
        )
        for col in object_cols:
            # Archetype booleans ("True", "False", 1, 0)
            if col.startswith("player_archetype_") or col.startswith("opponent_archetype_"):
                X[col] = X[col].map(
                    lambda v: 1 if str(v).strip().lower() in ("1", "true", "yes") else 0
                )
            else:
                # Try numeric, fallback to category codes
                tmp = pd.to_numeric(X[col], errors="coerce")
                if tmp.isna().sum() < len(tmp) * 0.5:
                    X[col] = tmp.fillna(0)
                else:
                    X[col] = X[col].astype("category").cat.codes

    # ----- Drop NaN labels BEFORE splitting -----
    nan_count = Y.isna().sum()
    if nan_count > 0:
        logger.warning(f"Found {nan_count} NaN labels. Dropping those rows.")
        mask = Y.notna()
        X = X[mask].reset_index(drop=True)
        Y = Y[mask].reset_index(drop=True)
        logger.info(f"After dropping NaNs: {len(Y)} samples remain.")

    # Final sanity check
    logger.info(f"Final label distribution:\n{Y.value_counts().to_string()}")

    # ----- Train-test split using same parameters as training -----
    _, X_test, _, Y_test = train_test_split(
        X,
        Y,
        test_size=0.25,
        random_state=42,
        stratify=Y
    )

    logger.info(f"Data split complete. Test set size: {len(Y_test)}")
    logger.info(
        f"Test class distribution:\n{Y_test.value_counts(normalize=True).to_string()}"
    )

    return X_test, Y_test


# ----------------------------
# Main Script
# ----------------------------
if __name__ == "__main__":
    logger.info("--- Starting ROC Curve Generation ---")
    
    try:
        # 1. Load Data
        X_test, Y_test = load_and_split_data()

        # 2. Load Models
        logger.info("Loading models...")
        model_win_seeker = joblib.load(WIN_SEEKER_MODEL_FILE)
        model_loss_avoider = joblib.load(LOSS_AVOIDER_MODEL_FILE)
        logger.info("Models loaded.")

        # 3. Get Predicted Probabilities (for the 'Win' class [1])
        logger.info("Calculating predicted probabilities...")
        probs_win_seeker = model_win_seeker.predict_proba(X_test)[:, 1]
        probs_loss_avoider = model_loss_avoider.predict_proba(X_test)[:, 1]

        # 4. Calculate ROC Curve & AUC
        logger.info("Calculating ROC & AUC...")
        fpr_ws, tpr_ws, _ = roc_curve(Y_test, probs_win_seeker)
        auc_ws = auc(fpr_ws, tpr_ws)

        fpr_la, tpr_la, _ = roc_curve(Y_test, probs_loss_avoider)
        auc_la = auc(fpr_la, tpr_la)

        logger.info(f"Win-Seeker AUC: {auc_ws:.4f}")
        logger.info(f"Loss-Avoider AUC: {auc_la:.4f}")

        # 5. Plot the ROC Curves
        logger.info(f"Generating plot... saving to {ROC_CURVE_PLOT_FILE}")
        plt.figure(figsize=(10, 8))
        
        # Plot Win-Seeker
        plt.plot(fpr_ws, tpr_ws, color='blue', lw=2, 
                 label=f'Win-Seeker (AUC = {auc_ws:.4f})')
        
        # Plot Loss-Avoider
        plt.plot(fpr_la, tpr_la, color='green', lw=2, 
                 label=f'Loss-Avoider (AUC = {auc_la:.4f})')
        
        # Plot the "No Skill" line
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='No Skill (AUC = 0.50)')
        
        # Customize the plot
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.title('ROC Curve Comparison: Win-Seeker vs. Loss-Avoider', fontsize=16)
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(True)
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(ROC_CURVE_PLOT_FILE)
        
        logger.info("--- ROC Curve Generation Complete ---")

    except FileNotFoundError as e:
        logger.error(f"\nERROR: File not found. Make sure all files are in the correct directories.")
        logger.error(f"Missing file: {e.filename}")
    except Exception as e:
        logger.error(f"\nAn unexpected error occurred: {e}", exc_info=True)