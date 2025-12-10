"""
Trains TWO robust classification models using SMOTE to handle class imbalance,
predicting match outcomes (win/lose).

This script fixes previous data alignment and evaluation issues.

1. "Win-Seeker" (strategy_model_win_seeker.pkl):
   - Trained on the original, IMBALANCED data.
   - This model will be biased towards predicting "Win" and is
     used to identify high-probability win scenarios.

2. "Loss-Avoider" (strategy_model_loss_avoider.pkl):
   - Trained on BALANCED (SMOTE-resampled) data.
   - This model is better at identifying "Losses" and is
     used for a risk-averse, balanced strategy.

Both models are evaluated on the same, clean, imbalanced test set
to get realistic performance metrics, including AUC.

Inputs:
- data/attack_mode_features.csv
- data/defense_mode_features.csv
- data/preprocessed_data.csv

Outputs:
- models/strategy_model_win_seeker.pkl (NEW)
- models/strategy_model_loss_avoider.pkl (NEW)
"""

import logging
import os
import pandas as pd
import joblib
import numpy as np
import sys
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# ----------------------------
# Configuration
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# File paths
DATA_DIR = "data"
MODELS_DIR = "models"
ATTACK_FEATURES_FILE = os.path.join(DATA_DIR, "attack_mode_features.csv")
DEFENSE_FEATURES_FILE = os.path.join(DATA_DIR, "defense_mode_features.csv")
PREPROCESSED_DATA_FILE = os.path.join(DATA_DIR, "preprocessed_data.csv")

WIN_SEEKER_MODEL_FILE = os.path.join(MODELS_DIR, "strategy_model_win_seeker.pkl")
LOSS_AVOIDER_MODEL_FILE = os.path.join(MODELS_DIR, "strategy_model_loss_avoider.pkl")

os.makedirs(MODELS_DIR, exist_ok=True)

# ----------------------------
# Data Loading
# ----------------------------
def load_and_align_data() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Loads feature (X) and target (Y) data, fixing alignment, dtype, and NaN issues.
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

        # Force to numeric; invalid -> NaN
        df_preprocessed['winner_flag'] = pd.to_numeric(
            df_preprocessed['winner_flag'],
            errors='coerce'
        )
        Y = df_preprocessed['winner_flag']

    except FileNotFoundError as e:
        logger.error(f"Error loading data: {e}. Make sure all feature files exist.")
        sys.exit(1)

    # Combine X features
    X_features = pd.concat([df_attack, df_defense], ignore_index=True)
    X = X_features.drop(columns=['player_tag'], errors='ignore')

    logger.info(f"Loaded {len(X)} feature rows. Loaded {len(Y)} labels.")

    # Align X and Y
    if len(X) != len(Y):
        logger.warning(f"Feature count ({len(X)}) and label count ({len(Y)}) mismatch!")
        min_len = min(len(X), len(Y))
        X = X.head(min_len)
        Y = Y.head(min_len)
        logger.info(f"Truncated data to {min_len} samples.")

    # Convert 'mode' to numeric
    if 'mode' in X.columns:
        logger.info("Converting 'mode' column to numeric (1=attack, 0=defense)...")
        X['mode'] = X['mode'].apply(lambda x: 1 if str(x).lower() == 'attack' else 0)
    else:
        logger.error("Critical error: 'mode' column not found in features. Aborting.")
        sys.exit(1)

    # ---------- NEW BLOCK: fix object columns ----------
    object_cols = X.select_dtypes(include=['object']).columns.tolist()
    if object_cols:
        logger.info(
            f"Found {len(object_cols)} object dtype feature columns. "
            f"Converting them to numeric/categorical: {object_cols}"
        )

        for col in object_cols:
            # Special handling for one-hot archetype flags
            if col.startswith('player_archetype_'):
                # Map typical true-ish values to 1, everything else to 0
                X[col] = X[col].map(
                    lambda v: 1
                    if str(v).strip().lower() in ('1', 'true', 'yes')
                    else 0
                )
            else:
                # Try numeric conversion first
                tmp = pd.to_numeric(X[col], errors='coerce')
                # If conversion produced many NaNs, fallback to category codes
                if tmp.isna().sum() < len(tmp) * 0.5:
                    X[col] = tmp.fillna(0)
                else:
                    X[col] = X[col].astype('category').cat.codes
    # ---------------------------------------------------

    # Drop NaN labels
    nan_labels = Y.isna().sum()
    if nan_labels > 0:
        logger.warning(f"Found {nan_labels} NaN labels in target 'winner_flag'. Dropping those rows...")
        valid_mask = Y.notna()
        X = X[valid_mask].reset_index(drop=True)
        Y = Y[valid_mask].reset_index(drop=True)
        logger.info(f"After dropping NaN labels: {len(Y)} samples remain.")

    # Sanity check: need at least 2 classes
    unique_classes = Y.nunique()
    if unique_classes < 2:
        logger.error(f"Target has only {unique_classes} unique class after cleaning. Cannot train a classifier.")
        logger.error(f"Class distribution:\n{Y.value_counts(dropna=False)}")
        sys.exit(1)

    logger.info(f"Final label distribution:\n{Y.value_counts().to_string()}")

    return X, Y

# ----------------------------
# Main Training Script
# ----------------------------
if __name__ == "__main__":
    logger.info("--- Starting Model Retraining Script ---")

    X, Y = load_and_align_data()

    if X.empty or Y.empty:
        logger.error("No data loaded. Aborting training.")
        sys.exit(1)

    logger.info("Splitting data into train and test sets (25% test size)...")
    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.25,
        random_state=42,
        stratify=Y
    )

    logger.info(f"Train set size: {len(Y_train)}, Test set size: {len(Y_test)}")
    logger.info(f"Test set class distribution:\n{Y_test.value_counts(normalize=True).to_string()}")

    # ==========================================================
    # Model 1: Win-Seeker (Trained on original, imbalanced data)
    # ==========================================================
    logger.info("\n--- Training 'Win-Seeker' Model ---")
    logger.info("Training on original, imbalanced data to optimize for finding wins.")

    win_seeker_model = XGBClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    win_seeker_model.fit(X_train, Y_train)

    logger.info("'Win-Seeker' training complete.")
    logger.info("Saving 'Win-Seeker' model...")
    joblib.dump(win_seeker_model, WIN_SEEKER_MODEL_FILE)

    # ==========================================================
    # Model 2: Loss-Avoider (Trained on SMOTE-balanced data)
    # ==========================================================
    logger.info("\n--- Training 'Loss-Avoider' Model ---")
    logger.info("Applying SMOTE to training data to create a balanced set...")

    smote = SMOTE(random_state=42)
    X_train_resampled, Y_train_resampled = smote.fit_resample(X_train, Y_train)

    logger.info(f"Original train set shape: {Y_train.value_counts().to_dict()}")
    logger.info(f"Resampled train set shape: {Y_train_resampled.value_counts().to_dict()}")

    loss_avoider_model = XGBClassifier(
        n_estimators=100,
        random_state=42,
        scale_pos_weight=1,
        n_jobs=-1
    )

    loss_avoider_model.fit(X_train_resampled, Y_train_resampled)

    logger.info("'Loss-Avoider' training complete.")
    logger.info("Saving 'Loss-Avoider' model...")
    joblib.dump(loss_avoider_model, LOSS_AVOIDER_MODEL_FILE)

    # ==========================================================
    # FINAL EVALUATION (On the clean, imbalanced X_test)
    # ==========================================================
    logger.info("\n" + "=" * 50)
    logger.info("--- Final Model Evaluation on Clean Test Set ---")
    logger.info("=" * 50)

    # Evaluate Win-Seeker
    Y_pred_ws = win_seeker_model.predict(X_test)
    Y_prob_ws = win_seeker_model.predict_proba(X_test)[:, 1]
    auc_ws = roc_auc_score(Y_test, Y_prob_ws)
    report_ws = classification_report(Y_test, Y_pred_ws, target_names=['Lose (0)', 'Win (1)'])

    logger.info("\n--- 'Win-Seeker' (Imbalanced) Performance ---")
    logger.info(f"Test Set AUC: {auc_ws:.4f}")
    logger.info(f"Test Set Classification Report:\n{report_ws}")

    # Evaluate Loss-Avoider
    Y_pred_la = loss_avoider_model.predict(X_test)
    Y_prob_la = loss_avoider_model.predict_proba(X_test)[:, 1]
    auc_la = roc_auc_score(Y_test, Y_prob_la)
    report_la = classification_report(Y_test, Y_pred_la, target_names=['Lose (0)', 'Win (1)'])

    logger.info("\n--- 'Loss-Avoider' (SMOTE-Balanced) Performance ---")
    logger.info(f"Test Set AUC: {auc_la:.4f}")
    logger.info(f"Test Set Classification Report:\n{report_la}")

    logger.info("\n--- Training Script Finished ---")
