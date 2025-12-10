"""
Trains a Feedforward Neural Network (FNN) "challenger" model 
to predict match outcomes (win/lose).

This model uses a multi-input architecture to handle deck embeddings and
statistical features separately.

Inputs:
- data/attack_mode_features.csv
- data/defense_mode_features.csv

Outputs:
- models/strategy_model_fnn.keras
(The new challenger model)
"""

import logging
import os
import sys  # Import sys
import pandas as pd
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, recall_score
from typing import Tuple, List
from sklearn.utils.class_weight import compute_class_weight


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
FNN_MODEL_FILE = os.path.join(MODELS_DIR, "strategy_model_fnn.keras")  # Use .keras extension

os.makedirs(MODELS_DIR, exist_ok=True)

# ----------------------------
# Data Loading and Prep
# ----------------------------
def load_and_prep_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Loads and combines attack/defense features; cleans labels and key columns."""
    logger.info("Loading data sources...")
    try:
        # low_memory=False to avoid mixed-type issues
        df_attack = pd.read_csv(ATTACK_FEATURES_FILE, low_memory=False)
        df_defense = pd.read_csv(DEFENSE_FEATURES_FILE, low_memory=False)

        df_preprocessed = pd.read_csv(
            PREPROCESSED_DATA_FILE,
            usecols=['winner_flag'],
            low_memory=False
        )

        # Coerce winner_flag to numeric; invalid → NaN
        df_preprocessed['winner_flag'] = pd.to_numeric(
            df_preprocessed['winner_flag'],
            errors='coerce'
        )
        y = df_preprocessed['winner_flag']

    except FileNotFoundError as e:
        logger.error(f"Error loading data: {e}. Make sure all feature files exist.")
        raise

    # Combine X features
    df_features = pd.concat([df_attack, df_defense], ignore_index=True)

    logger.info(f"Loaded {len(df_features)} feature rows. Loaded {len(y)} labels.")

    X = df_features.drop(columns=['player_tag'], errors='ignore')

    # Align lengths
    if len(X) != len(y):
        logger.warning(f"Feature count ({len(X)}) and label count ({len(y)}) mismatch!")
        min_len = min(len(X), len(y))
        X = X.head(min_len)
        y = y.head(min_len)
        logger.info(f"Truncated data to {min_len} samples.")

    # Convert 'mode' column from string ("attack"/"defense") to numeric (1/0)
    if 'mode' in X.columns:
        logger.info("Converting 'mode' column to numeric (1=attack, 0=defense)...")
        X['mode'] = X['mode'].apply(lambda x: 1 if str(x).lower() == 'attack' else 0)
    else:
        logger.error("Critical error: 'mode' column not found in features. Aborting.")
        raise ValueError("Missing 'mode' column in feature set.")

    # Drop NaN labels
    nan_labels = y.isna().sum()
    if nan_labels > 0:
        logger.warning(f"Found {nan_labels} NaN labels in target 'winner_flag'. Dropping those rows...")
        mask = y.notna()
        X = X[mask].reset_index(drop=True)
        y = y[mask].reset_index(drop=True)
        logger.info(f"After dropping NaN labels: {len(y)} samples remain.")

    # Optional: ensure binary int labels
    y = y.astype(int)

    logger.info(f"Final label distribution:\n{y.value_counts().to_string()}")

    return X, y


def create_fnn_model(embedding_dim: int, num_stats: int) -> Model:
    """Builds the multi-input Keras Functional API model."""

    # Input 1: Embeddings
    embedding_input = Input(shape=(embedding_dim,), name="embedding_input")
    x_emb = Dense(8, activation="relu")(embedding_input)
    x_emb = BatchNormalization()(x_emb)

    # Input 2: Stats
    stats_input = Input(shape=(num_stats,), name="stats_input")
    x_stats = Dense(32, activation="relu")(stats_input)
    x_stats = BatchNormalization()(x_stats)

    # Concatenate features
    concatenated = Concatenate()([x_emb, x_stats])

    # Head of the network
    x = Dense(64, activation="relu")(concatenated)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.2)(x)

    # Output layer
    output = Dense(1, activation="sigmoid", name="output")(x)

    # Create the model
    model = Model(inputs=[embedding_input, stats_input], outputs=output)

    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Recall(name='recall')]
    )

    return model


# ----------------------------
# Main Training
# ----------------------------
if __name__ == "__main__":
    logger.info("--- Starting FNN Challenger Model Training ---")

    try:
        X, y = load_and_prep_data()
    except Exception as e:
        logger.error(f"Failed to load data. Aborting. Error: {e}")
        sys.exit(1)

    # --- Split X into Embedding and Stats ---
    embedding_cols = [col for col in X.columns if col.startswith('embedding_')]
    stats_cols = [col for col in X.columns if not col.startswith('embedding_')]

    if not embedding_cols:
        logger.error("No 'embedding_' columns found. Cannot train FNN. Aborting.")
        sys.exit(1)

    logger.info(f"Found {len(embedding_cols)} embedding features.")
    logger.info(f"Found {len(stats_cols)} statistical features.")

    # Clean non-numeric stats columns BEFORE casting to float32
    object_stats = X[stats_cols].select_dtypes(include=['object', 'bool']).columns.tolist()
    if object_stats:
        logger.info(f"Cleaning non-numeric stats columns: {object_stats}")
        for col in object_stats:
            series = X[col]
            # Handle boolean-ish values
            X[col] = series.map(
                lambda v: 1.0
                if str(v).strip().lower() in ('true', '1', 'yes')
                else 0.0
            )

    # Now cast to float32 safely
    X_embeddings = X[embedding_cols].astype(np.float32)
    X_stats = X[stats_cols].astype(np.float32)

    # --- Train/Test Split ---
    X_emb_train, X_emb_test, X_stats_train, X_stats_test, y_train, y_test = train_test_split(
        X_embeddings,
        X_stats,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y  # Important for imbalanced data
    )

    logger.info(f"Training on {len(y_train)} samples, validating on {len(y_test)} samples.")
    
    # --- Class weights to fight imbalance ---
    classes = np.array([0, 1])
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=y_train
    )
    class_weight_dict = {int(c): w for c, w in zip(classes, class_weights)}
    logger.info(f"Using class weights: {class_weight_dict}")

    # --- Build Model ---
    model = create_fnn_model(
        embedding_dim=len(embedding_cols),
        num_stats=len(stats_cols)
    )
    model.summary()  # Print model architecture

    # --- Train Model ---
    logger.info("Starting model training...")
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    history = model.fit(
        [X_emb_train, X_stats_train],
        y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1,
        class_weight=class_weight_dict,  # <-- THIS
    )


    logger.info("Training complete.")

    # --- Evaluate Model ---
    logger.info("Evaluating FNN model on the test set...")

    # Predict probabilities
    y_pred_prob = model.predict([X_emb_test, X_stats_test])
    # Convert probabilities to binary classes (0 or 1)
    y_pred = (y_pred_prob > 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Lose (0)', 'Win (1)'])

    logger.info(f"--- FNN Challenger Model Performance ---")
    logger.info(f"Test Set Accuracy: {accuracy * 100:.2f}%")
    logger.info(f"Test Set Classification Report:\n{report}")

    # --- Compare to XGBoost (from your train_model.py logs) ---
    logger.info("--- Comparison ---")
    logger.info("Compare the Recall for 'Lose (0)' and 'Win (1)' above to your")
    logger.info("XGBoost 'Win-Seeker' and 'Loss-Avoider' models to see which is better!")

    # --- Save Model ---
    logger.info(f"Saving FNN model to {FNN_MODEL_FILE}...")
    model.save(FNN_MODEL_FILE)
    logger.info(f"Model saved successfully.")

    logger.info("--- FNN Training Script Finished ---")
