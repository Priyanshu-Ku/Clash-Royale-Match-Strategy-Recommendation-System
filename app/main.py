import logging
import os
import sys
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Dict, Tuple, Optional, Any
import pandas as pd
import joblib
import numpy as np
import time
import hashlib
import json
import google.generativeai as genai
from app.cache import TtlCache  # <-- This will import our updated class
from dotenv import load_dotenv

# --- Logging Setup (Top) ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Add src directory to path ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# --- Load Environment Variables ---
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
logger.info("Loaded environment variables from .env file.")

# --- Import helpers ---
try:
    from src.recommend_strategy import (
        load_card_table, calculate_deck_features, create_deck_embedding,
        _get_card_row, defense_strength_heuristic, offense_strength_heuristic,
        synergy_score_heuristic
    )
    from src.utils import infer_archetype
    from src.semantic_recommender import SemanticRecommender
    logger.info("Successfully imported all helper functions.")
except ImportError as e:
    logger.error(f"Error importing from src: {e}", exc_info=True)
    # Define dummy functions...
    class SemanticRecommender:
        def __init__(self): logger.error("Dummy SemanticRecommender class used.")
        def load_index(self, d): pass
        def search(self, q, k): return []
    def load_card_table(path: str): logger.error("Dummy load_card_table used."); return None
    def calculate_deck_features(deck_list: List[str], card_table: pd.DataFrame): logger.error("Dummy calc features."); return {}
    def create_deck_embedding(deck_list: List[str], card_table: pd.DataFrame, embedding_size: int = 16): logger.error("Dummy embedding."); return np.array([])
    def infer_archetype(deck: List[str], card_table: pd.DataFrame): logger.error("Dummy infer_archetype."); return "unknown"
    # (Other dummies omitted for brevity)

# --- Global variables ---
EXPECTED_FEATURES = []
KNOWN_OPPONENT_ARCHETYPES = []

# ----------------------------
# Pydantic Models
# ----------------------------
class DeckRequest(BaseModel):
    player_deck: List[str] = Field(..., min_length=8, max_length=8)
    opponent_deck: Optional[List[str]] = Field(None, min_length=8, max_length=8)
    opponent_archetype: Optional[str] = Field(None)
    player_trophies: float = Field(7000.0)
    player_explevel: float = Field(60.0)

class RecommendationResponse(BaseModel):
    recommended_strategy: str
    confidence_score: float
    probabilities: Dict[str, float]

class OptimizeResponse(BaseModel):
    original_recommendation: str
    original_synergy: float
    original_avg_win_prob: float
    best_swap: Dict[str, Any]
    top_swaps: List[Dict[str, Any]]

class CardInfo(BaseModel):
    name: str
    elixir: Optional[float] = None
    type: Optional[str] = None
    attributes: Dict[str, Any] = Field(default_factory=dict)

class CardsResponse(BaseModel):
    cards: List[CardInfo]
    total_cards: int

class HealthResponse(BaseModel):
    status: str
    card_data: bool
    win_seeker_model: bool
    loss_avoider_model: bool
    semantic_index: bool
    gemini_model: bool
    total_cards: Optional[int] = None
    details: Optional[Dict[str, str]] = None

class SemanticRequest(BaseModel):
    query: str
    top_k: int = Field(5, ge=1, le=50)

class SemanticResult(BaseModel):
    card_name: str
    distance: float

class SemanticResponse(BaseModel):
    results: List[SemanticResult]
    query_time_ms: float
    total_cards_indexed: int

class ExplainRequest(BaseModel):
    player_deck: List[str]
    opponent_archetype: str
    recommended_strategy: str
    probabilities: Dict[str, float]

class ExplainResponse(BaseModel):
    explanation: str
    
class CardTipsRequest(BaseModel):
    card_name: str
    
class CardTipsResponse(BaseModel):
    tips: str

# --- UPDATED: Cache Stat Models ---
class CacheStatsResponse(BaseModel):
    hits: int
    misses: int
    total_requests: int
    hit_rate: float
    current_items: int
    expired_items_cleared: int
    last_cleanup: Optional[str] = None

class CacheInfoResponse(BaseModel):
    gemini_llm_cache: CacheStatsResponse
# --- END UPDATED ---

# ----------------------------
# FastAPI App Initialization
# ----------------------------
app = FastAPI(
    title="Clash Royale Strategy Recommender API",
    description="API for recommending optimal strategy, deck optimization, semantic search, and AI explanations.",
    version="2.0.5", # Final version
)

# --- Global state ---
app.state.win_seeker_model = None
app.state.loss_avoider_model = None
app.state.card_table = None
app.state.semantic_recommender = None
app.state.gemini_model = None
app.state.gemini_cache = None

# ----------------------------
# Load Models and Data on Startup
# ----------------------------
@app.on_event("startup")
async def load_resources():
    """Load models, data, and initialize API clients on startup."""
    global EXPECTED_FEATURES, KNOWN_OPPONENT_ARCHETYPES
    logger.info("API starting up. Loading models and card data...")

    # --- Initialize Cache (Uses updated class) ---
    app.state.gemini_cache = TtlCache(ttl_seconds=3600)
    logger.info("In-memory cache initialized.")

    # --- Initialize Gemini API ---
    try:
        gemini_api_key = os.environ.get("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found. Make sure it is set in your .env file.")
        genai.configure(api_key=gemini_api_key)
        
        # This is your working model name
        app.state.gemini_model = genai.GenerativeModel('models/gemini-2.5-pro')

        logger.info("Google Gemini API configured successfully with 'models/gemini-2.5-pro'.")
    except Exception as e:
        logger.error(f"FATAL: Failed to configure Gemini API: {e}")
        app.state.gemini_model = None

    # (The rest of your load_resources function is unchanged)
    data_dir = os.path.join(PROJECT_ROOT, "data")
    models_dir = os.path.join(PROJECT_ROOT, "models")
    cards_file = os.path.join(data_dir, "cards_data.csv")
    win_seeker_file = os.path.join(models_dir, "strategy_model_win_seeker.pkl")
    loss_avoider_file = os.path.join(models_dir, "strategy_model_loss_avoider.pkl")
    features_example_file = os.path.join(data_dir, "attack_mode_features.csv")

    app.state.card_table = load_card_table(cards_file)
    if app.state.card_table is not None:
        if 'card_name' in app.state.card_table.columns and not app.state.card_table.index.name == 'card_name':
             app.state.card_table.set_index('card_name', inplace=True, drop=False)
        logger.info(f"Card table successfully loaded ({len(app.state.card_table)} rows).")
    else:
        logger.error("FATAL: Failed to load card table on startup.")

    try:
        features_df_example = pd.read_csv(features_example_file, nrows=1)
        EXPECTED_FEATURES = list(features_df_example.drop(columns=['player_tag', 'winner_flag'], errors='ignore').columns)
        KNOWN_OPPONENT_ARCHETYPES = [col for col in EXPECTED_FEATURES if col.startswith('opponent_archetype_')]
        logger.info(f"Loaded expected features list ({len(EXPECTED_FEATURES)}).")
    except Exception as e:
         logger.error(f"FATAL: Failed to load expected features: {e}", exc_info=True)

    try:
        app.state.win_seeker_model = joblib.load(win_seeker_file)
        logger.info(f"Win-Seeker model loaded from {win_seeker_file}.")
    except Exception as e:
        logger.error(f"FATAL: Error loading Win-Seeker model: {e}", exc_info=True)

    try:
        app.state.loss_avoider_model = joblib.load(loss_avoider_file)
        logger.info(f"Loss-Avoider model loaded from {loss_avoider_file}.")
    except Exception as e:
        logger.error(f"FATAL: Error loading Loss-Avoider model: {e}", exc_info=True)

    try:
        app.state.semantic_recommender = SemanticRecommender()
        app.state.semantic_recommender.load_index(models_dir)
        logger.info("Semantic index loaded successfully.")
    except Exception as e:
        logger.warning(f"Failed to load semantic index: {e}. /semantic endpoint will be unavailable.")
        app.state.semantic_recommender = None

    if not app.state.win_seeker_model or not app.state.loss_avoider_model or app.state.card_table is None or not EXPECTED_FEATURES:
        logger.error("!!! API startup failed: Critical ML/Data resources missing. Check logs. !!!")
    else:
        logger.info("All essential ML/Data resources loaded successfully.")

# ----------------------------
# Endpoints
# ----------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    cards_ok = app.state.card_table is not None and not app.state.card_table.empty
    ws_model_ok = app.state.win_seeker_model is not None
    la_model_ok = app.state.loss_avoider_model is not None
    features_ok = bool(EXPECTED_FEATURES)
    semantic_ok = app.state.semantic_recommender is not None and app.state.semantic_recommender.index is not None
    gemini_ok = app.state.gemini_model is not None
    overall_ok = cards_ok and ws_model_ok and la_model_ok and features_ok and semantic_ok and gemini_ok
    status = "healthy" if overall_ok else "unhealthy"
    details = {}
    if not cards_ok: details["card_data"] = "Card data failed to load or is empty."
    if not ws_model_ok: details["win_seeker_model"] = "Win-Seeker model failed to load."
    if not la_model_ok: details["loss_avoider_model"] = "Loss-Avoider model failed to load."
    if not features_ok: details["feature_definitions"] = "Expected feature list failed to load."
    if not semantic_ok: details["semantic_index"] = "Semantic index failed to load."
    if not gemini_ok: details["gemini_model"] = "Gemini API client failed to initialize. Check GEMINI_API_KEY in .env file."
    return HealthResponse(
        status=status, card_data=cards_ok, win_seeker_model=ws_model_ok,
        loss_avoider_model=la_model_ok, semantic_index=semantic_ok,
        gemini_model=gemini_ok, total_cards=len(app.state.card_table) if cards_ok else None,
        details=details or None
    )

@app.post("/explain/strategy", response_model=ExplainResponse, tags=["AI Explanation"])
async def explain_strategy(request: ExplainRequest):
    if not app.state.gemini_model:
        raise HTTPException(status_code=503, detail="Gemini API is not available.")
    key_dict = {
        "deck": sorted(request.player_deck),
        "opponent": request.opponent_archetype,
        "recommendation": request.recommended_strategy
    }
    cache_key = hashlib.md5(json.dumps(key_dict, sort_keys=True).encode("utf-8")).hexdigest()
    
    # --- Cache `get` call now tracks stats ---
    cached = app.state.gemini_cache.get(cache_key)
    if cached:
        logger.info(f"Returning cached explanation for key: {cache_key}")
        return ExplainResponse(explanation=cached)
    
    logger.info(f"Generating new explanation for key: {cache_key}")
    prompt = f"""
    You are an expert Clash Royale analyst. Your goal is to explain *why* a specific strategy was recommended.
    Be concise, insightful, and use bullet points.

    **Context:**
    * **My Deck:** {', '.join(request.player_deck)}
    * **Opponent's Archetype:** {request.opponent_archetype}
    * **Our AI's Recommendation:** **{request.recommended_strategy}**

    **Model Data (Win Probabilities):**
    * **Win-Seeker (Optimistic):**
        * Attack Mode: {request.probabilities.get('win_seeker_attack_win_prob', 0):.1%}
        * Defense Mode: {request.probabilities.get('win_seeker_defense_win_prob', 0):.1%}
    * **Loss-Avoider (Balanced):**
        * Attack Mode: {request.probabilities.get('loss_avoider_attack_win_prob', 0):.1%}
        * Defense Mode: {request.probabilities.get('loss_avoider_defense_win_prob', 0):.1%}

    **Task:**
    Based on the recommendation and the model data, explain *why* this is the best strategy.
    For example:
    - If "Aggressive Push" was chosen, it's likely because the Win-Seeker model found a high win-rate in Attack mode.
    - If "Defensive Counter" was chosen, it's likely because the Loss-Avoider model saw a high risk in attacking.
    - If "Balanced Cycle" was chosen, it's likely the models saw no clear advantage or the deck is a fast cycle deck.
    
    Provide your analysis.
    """
    try:
        response = await app.state.gemini_model.generate_content_async(prompt)
        explanation = response.text if response.parts else "The model returned an empty response."
        app.state.gemini_cache.set(cache_key, explanation)
        return ExplainResponse(explanation=explanation)
    except Exception as e:
        logger.error(f"Error generating content from Gemini: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating AI explanation: {e}")

@app.post("/explain/card_tips", response_model=CardTipsResponse, tags=["AI Explanation"])
async def get_card_tips(request: CardTipsRequest):
    if not app.state.gemini_model:
        raise HTTPException(status_code=503, detail="Gemini API is not available.")
        
    card_name = request.card_name
    cache_key = f"card_tips_{card_name.lower().replace(' ', '_')}"
    
    # --- Cache `get` call now tracks stats ---
    cached = app.state.gemini_cache.get(cache_key)
    if cached:
        logger.info(f"Returning cached card tips for: {card_name}")
        return CardTipsResponse(tips=cached)
        
    logger.info(f"Generating new card tips for: {card_name}")
    
    prompt = f"""
    You are an expert Clash Royale analyst.
    Give me 3-4 concise, actionable, bullet-pointed tips for how to use the card "{card_name}".
    Focus on mechanics, common interactions, and optimal placement.
    
    Example for "Hog Rider":
    * **Kite & Counter:** Use the Hog Rider's speed to pull enemy troops (like P.E.K.K.A) to the center.
    * **Pig Push:** Place a fast troop (like Ice Spirit) directly behind the Hog Rider at the bridge to push it faster.
    * **Predictive Log:** Send a Log just *after* your Hog Rider crosses the bridge to hit their Skeleton Army or Goblin Gang.
    
    Give me the tips for "{card_name}".
    """
    
    try:
        response = await app.state.gemini_model.generate_content_async(prompt)
        tips = response.text if response.parts else "The model returned an empty response."
        app.state.gemini_cache.set(cache_key, tips)
        return CardTipsResponse(tips=tips)
    except Exception as e:
        logger.error(f"Error generating card tips from Gemini: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating AI tips: {e}")

@app.get("/cards", response_model=CardsResponse, tags=["Cards"])
async def list_cards():
    if app.state.card_table is None:
        raise HTTPException(status_code=503, detail="Card data not available.")
    cards_list: List[CardInfo] = []
    try:
        for _, row in app.state.card_table.iterrows():
            name = row.get("card_name", "Unknown Card")
            elixir = row.get("elixir_cost")
            card_type = row.get("card_type")
            attrs = {}
            for col in ['rarity', 'targets', 'damage_type', 'range', 'hit_speed', 'hitpoints', 'damage', 'dps', 'speed', 'count']:
                if col in row and pd.notna(row[col]):
                    attrs[col] = row[col]
            card_info = CardInfo(name=str(name), elixir=float(elixir) if pd.notna(elixir) else None, type=str(card_type) if pd.notna(card_type) else None, attributes=attrs)
            cards_list.append(card_info)
        response = CardsResponse(cards=cards_list, total_cards=len(cards_list))
        return response
    except Exception as e:
        logger.error(f"Error processing card list: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving card list.")

@app.post("/recommend", response_model=RecommendationResponse, tags=["Strategy"])
async def recommend_endpoint(request: DeckRequest):
    logger.info(f"Received recommendation request for deck: {request.player_deck} vs {request.opponent_archetype or 'Unknown Opponent'}")
    if not app.state.win_seeker_model or not app.state.loss_avoider_model or app.state.card_table is None or not EXPECTED_FEATURES:
        raise HTTPException(status_code=503, detail="Service not ready: Models or data unavailable.")
    try:
        logic_result = get_recommendation_logic(
            deck_list=request.player_deck,
            opponent_archetype=request.opponent_archetype or "unknown",
            win_seeker_model=app.state.win_seeker_model,
            loss_avoider_model=app.state.loss_avoider_model,
            card_table=app.state.card_table,
            expected_features=EXPECTED_FEATURES,
            known_opponent_archetypes=KNOWN_OPPONENT_ARCHETYPES,
            player_trophies=request.player_trophies,
            player_explevel=request.player_explevel
        )
        recommendation = logic_result["recommendation"]
        probabilities = logic_result["probabilities"]
        if recommendation.startswith("Error"):
             logger.error(f"Recommendation logic failed internally: {recommendation}")
             raise HTTPException(status_code=500, detail="Failed to generate recommendation.")
        confidence = 0.0
        if recommendation == "Aggressive Push":
            confidence = probabilities.get('win_seeker_attack_win_prob', 0.0)
        elif recommendation == "Defensive Counter":
            confidence = probabilities.get('loss_avoider_defense_win_prob', 0.0)
        elif recommendation == "Balanced Cycle":
             confidence = max(probabilities.get('loss_avoider_attack_win_prob', 0.0), probabilities.get('loss_avoider_defense_win_prob', 0.0))
        response = RecommendationResponse(recommended_strategy=recommendation, confidence_score=confidence, probabilities=probabilities)
        return response
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.exception(f"Unexpected error during recommendation: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")

@app.post("/recommend/optimize", response_model=OptimizeResponse, tags=["Strategy"])
async def recommend_optimize(request: DeckRequest):
    player_deck = request.player_deck
    card_table = app.state.card_table
    if card_table is None:
         raise HTTPException(status_code=503, detail="Card data not available for optimizer.")
    for card in player_deck:
        if card not in card_table.index:
            logger.error(f"Card not found in index: '{card}'")
            raise HTTPException(status_code=400, detail=f"Card not found: {card}")
    opponent_archetype = request.opponent_archetype
    if not opponent_archetype:
        if request.opponent_deck and len(request.opponent_deck) == 8:
            try:
                opponent_archetype = infer_archetype(request.opponent_deck, card_table) 
            except Exception as e:
                logger.warning(f"Could not infer opponent archetype, defaulting to unknown. Error: {e}")
                opponent_archetype = "unknown"
        else:
            opponent_archetype = "unknown"
    try:
        baseline_logic_result = get_recommendation_logic(
            deck_list=player_deck, opponent_archetype=opponent_archetype,
            win_seeker_model=app.state.win_seeker_model, loss_avoider_model=app.state.loss_avoider_model,
            card_table=card_table, expected_features=EXPECTED_FEATURES,
            known_opponent_archetypes=KNOWN_OPPONENT_ARCHETYPES, player_trophies=request.player_trophies,
            player_explevel=request.player_explevel
        )
        baseline_rec = baseline_logic_result["recommendation"]
        baseline_features = baseline_logic_result["features"]
        baseline_synergy = baseline_features.get("synergy_score", 0)
        baseline_avg_win_prob = (baseline_logic_result["probabilities"]["win_seeker_attack_win_prob"] + 
                                 baseline_logic_result["probabilities"]["win_seeker_defense_win_prob"]) / 2
    except Exception as e:
        logger.error(f"Error getting baseline recommendation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error calculating baseline logic: {e}")
    all_cards = list(card_table.index)
    baseline_spell_ratio = baseline_features.get("spell_ratio", 0.25)
    baseline_building_ratio = baseline_features.get("building_ratio", 0.0)
    candidate_cards = []
    for card_name in all_cards:
        if card_name in player_deck:
            continue
        try:
            card_type = card_table.loc[card_name, 'card_type']
        except KeyError:
            logger.warning(f"Skipping candidate {card_name}: not found in card_table index.")
            continue
        if card_type == 'Spell' and baseline_spell_ratio >= 0.375:
            continue
        if card_type == 'Building' and baseline_building_ratio >= 0.25:
            continue
        candidate_cards.append(card_name)
    logger.info(f"Original deck. Prob: {baseline_avg_win_prob:.2f}, Synergy: {baseline_synergy:.2f}")
    logger.info(f"Pruned swap candidates from {len(all_cards)} to {len(candidate_cards)}.")
    swaps = []
    for i, card_out in enumerate(player_deck):
        for card_in in candidate_cards:
            new_deck = player_deck.copy()
            new_deck[i] = card_in
            try:
                swap_logic_result = get_recommendation_logic(
                    deck_list=new_deck, opponent_archetype=opponent_archetype,
                    win_seeker_model=app.state.win_seeker_model, loss_avoider_model=app.state.loss_avoider_model,
                    card_table=card_table, expected_features=EXPECTED_FEATURES,
                    known_opponent_archetypes=KNOWN_OPPONENT_ARCHETYPES, player_trophies=request.player_trophies,
                    player_explevel=request.player_explevel
                )
                new_features = swap_logic_result["features"]
                new_synergy = new_features.get("synergy_score", 0)
                new_avg_win_prob = (swap_logic_result["probabilities"]["win_seeker_attack_win_prob"] + 
                                     swap_logic_result["probabilities"]["win_seeker_defense_win_prob"]) / 2
                prob_gain = new_avg_win_prob - baseline_avg_win_prob
                synergy_gain = new_synergy - baseline_synergy
                score = (0.7 * prob_gain) + (0.3 * synergy_gain)
                if score > 0.005: 
                    swaps.append({
                        "card_out": card_out, "card_in": card_in,
                        "new_recommendation": swap_logic_result["recommendation"],
                        "new_avg_win_prob": new_avg_win_prob, "prob_gain": prob_gain,
                        "new_synergy_score": new_synergy, "score": score
                    })
            except Exception as e:
                logger.warning(f"Skipping swap {card_out}->{card_in} due to error: {e}")
                continue
    if not swaps:
        return OptimizeResponse(
            original_recommendation=baseline_rec, original_synergy=baseline_synergy,
            original_avg_win_prob=baseline_avg_win_prob,
            best_swap={"detail": "No beneficial swaps found. Your deck is already well-optimized for this matchup!"},
            top_swaps=[]
        )
    top_swaps = sorted(swaps, key=lambda x: x["score"], reverse=True)
    best_swap = top_swaps[0]
    return OptimizeResponse(
        original_recommendation=baseline_rec, original_synergy=baseline_synergy,
        original_avg_win_prob=baseline_avg_win_prob,
        best_swap=best_swap, top_swaps=top_swaps[:10]
    )

@app.post("/semantic", response_model=SemanticResponse, tags=["Semantic Search"])
async def get_similar_strategies(request: SemanticRequest):
    if not app.state.semantic_recommender or not app.state.semantic_recommender.index:
        logger.warning("Semantic endpoint called but index is not loaded.")
        raise HTTPException(status_code=503, detail="Semantic search feature is not loaded or available.")
    try:
        start_time = time.time()
        results = app.state.semantic_recommender.search(request.query, k=request.top_k)
        query_time = (time.time() - start_time) * 1000
        response_results = [SemanticResult(card_name=card_name, distance=float(distance)) for card_name, distance in results]
        return SemanticResponse(
            results=response_results, query_time_ms=query_time,
            total_cards_indexed=len(app.state.semantic_recommender.mapping)
        )
    except Exception as e:
        logger.error(f"Error during semantic search: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error processing semantic query.")

# --- UPDATED: /stats Endpoint ---
@app.get("/stats", response_model=CacheInfoResponse, tags=["System"])
async def get_cache_stats():
    """
    (Implemented) Get cache statistics for the running API.
    """
    if not app.state.gemini_cache:
        raise HTTPException(status_code=503, detail="Cache is not initialized.")
    
    logger.info("Fetching cache stats.")
    
    # Get stats from our cache object
    stats_dict = app.state.gemini_cache.get_stats()
    
    # Pack them into the Pydantic models
    gemini_stats = CacheStatsResponse(
        hits=stats_dict.get("hits", 0),
        misses=stats_dict.get("misses", 0),
        total_requests=stats_dict.get("total_requests", 0),
        hit_rate=stats_dict.get("hit_rate", 0.0),
        current_items=stats_dict.get("current_items", 0),
        expired_items_cleared=stats_dict.get("expired_items_cleared", 0),
        last_cleanup=stats_dict.get("last_cleanup")
    )
    
    return CacheInfoResponse(gemini_llm_cache=gemini_stats)
# --- END UPDATED ---


# --- Internal Logic Function ---
def get_recommendation_logic(
    deck_list: List[str],
    opponent_archetype: str,
    win_seeker_model,
    loss_avoider_model,
    card_table: pd.DataFrame,
    expected_features: List[str],
    known_opponent_archetypes: List[str],
    player_trophies: float = 7000.0,
    player_explevel: float = 60.0
    ) -> Dict[str, Any]: 

    _recommendation = "Undetermined"
    _probabilities = {
        'win_seeker_attack_win_prob': 0.0, 'win_seeker_defense_win_prob': 0.0,
        'loss_avoider_attack_win_prob': 0.0, 'loss_avoider_defense_win_prob': 0.0
    }
    _features = {} 
    if card_table is None or not expected_features:
        logger.error("API Logic Error: Card table or expected features missing.")
        return {"recommendation": "Error: Missing config", "probabilities": _probabilities, "features": _features}
    try:
        deck_features = calculate_deck_features(deck_list, card_table)
        _features = deck_features.copy() 
        embedding_cols = [f for f in expected_features if f.startswith('embedding_')]
        embedding_size = len(embedding_cols) if embedding_cols else 16
        deck_embedding = create_deck_embedding(deck_list, card_table, embedding_size=embedding_size)
    except Exception as calc_e:
        logger.error(f"API Logic Error calculating features: {calc_e}", exc_info=True)
        return {"recommendation": "Error: Feature calculation failed", "probabilities": _probabilities, "features": _features}
    base_features = {}
    try:
        for feature_name in expected_features:
            if feature_name.startswith('embedding_'):
                try:
                    index = int(feature_name.split('_')[1])
                    base_features[feature_name] = deck_embedding[index] if index < len(deck_embedding) else 0.0
                except (IndexError, ValueError):
                    base_features[feature_name] = 0.0
            elif feature_name.startswith('opponent_archetype_'):
                clean_opponent_name = opponent_archetype.lower().replace(' ', '_').replace('-', '_')
                opponent_col_name = f"opponent_archetype_{clean_opponent_name}"
                is_known = feature_name == opponent_col_name and opponent_col_name in known_opponent_archetypes
                is_unknown = (clean_opponent_name == 'unknown' or opponent_col_name not in known_opponent_archetypes) and feature_name == 'opponent_archetype_unknown'
                base_features[feature_name] = 1.0 if (is_known or is_unknown) else 0.0
            elif feature_name in deck_features:
                val = deck_features[feature_name]
                base_features[feature_name] = val if not pd.isna(val) else 0.0
            elif feature_name == 'trophies':
                base_features[feature_name] = player_trophies
            elif feature_name == 'explevel':
                base_features[feature_name] = player_explevel
            elif feature_name == 'mode':
                continue
            elif feature_name == 'counter_score':
                base_features[feature_name] = 0.5
            elif feature_name.startswith('player_archetype_'):
                base_features[feature_name] = 0.0
            else:
                base_features[feature_name] = 0.0
        _features.update(base_features)
    except Exception as prep_e:
        logger.error(f"API Logic Error preparing features: {prep_e}", exc_info=True)
        return {"recommendation": "Error: Feature preparation failed", "probabilities": _probabilities, "features": _features}
    attack_input = base_features.copy(); attack_input['mode'] = 1.0
    defense_input = base_features.copy(); defense_input['mode'] = 0.0
    try:
        attack_df = pd.DataFrame([attack_input]).reindex(columns=expected_features).fillna(0.0)
        defense_df = pd.DataFrame([defense_input]).reindex(columns=expected_features).fillna(0.0)
    except Exception as df_e:
        logger.error(f"API Logic Error creating DataFrame: {df_e}. Columns expected: {expected_features}", exc_info=True)
        return {"recommendation": "Error: DataFrame creation failed", "probabilities": _probabilities, "features": _features}
    try:
        _probabilities['win_seeker_attack_win_prob'] = float(win_seeker_model.predict_proba(attack_df)[0][1])
        _probabilities['win_seeker_defense_win_prob'] = float(win_seeker_model.predict_proba(defense_df)[0][1])
        _probabilities['loss_avoider_attack_win_prob'] = float(loss_avoider_model.predict_proba(attack_df)[0][1])
        _probabilities['loss_avoider_defense_win_prob'] = float(loss_avoider_model.predict_proba(defense_df)[0][1])
        logger.info(f"API Logic Predicted Probabilities: {_probabilities}")
    except Exception as pred_e:
        logger.error(f"API Logic Error during prediction: {pred_e}", exc_info=True)
        return {"recommendation": "Error: Prediction failed", "probabilities": _probabilities, "features": _features}
    avg_elixir = deck_features.get('avg_elixir', 4.0)
    if pd.isna(avg_elixir):
        avg_elixir = 4.0
    attack_score = (0.6 * _probabilities['win_seeker_attack_win_prob'] + 0.4 * (1 - _probabilities['loss_avoider_attack_win_prob']))
    defense_score = (0.6 * _probabilities['win_seeker_defense_win_prob'] + 0.4 * (1 - _probabilities['loss_avoider_defense_win_prob']))
    if avg_elixir <= 3.0: style_bias = "cycle"
    elif avg_elixir >= 4.1: style_bias = "heavy"
    else: style_bias = "balanced"
    confidence_gap = abs(attack_score - defense_score)
    margin_threshold = 0.07
    if style_bias == "cycle" and confidence_gap < 0.1:
        _recommendation = "Balanced Cycle"
    elif attack_score > defense_score + margin_threshold:
        _recommendation = "Aggressive Push"
    elif defense_score > attack_score + margin_threshold:
        _recommendation = "Defensive Counter"
    else:
        _recommendation = "Balanced Cycle"
    logger.info(f"Internal Logic Recommendation: {_recommendation}")
    return {"recommendation": _recommendation, "probabilities": _probabilities, "features": _features}

# ----------------------------
# Uvicorn entry point
# ----------------------------
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting API server with Uvicorn...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

'''
**Reasoning for the Fix:**

The `os.path.join(base_directory, filename)` function correctly combines parts of a path.

* `data_dir` already holds `C:\...\Match Strategy Recommendation System\data`
* `models_dir` already holds `C:\...\Match Strategy Recommendation System\models`

When you previously had `os.path.join(data_dir, "data/cards_data.csv")`, it was essentially doing `os.path.join("...\\data", "data/cards_data.csv")`, resulting in the incorrect `...\\data\\data/cards_data.csv`.

By changing it to `os.path.join(data_dir, "cards_data.csv")`, it correctly becomes `...\\data\\cards_data.csv`.

**Action:**

1.  Replace the content of your `app/main.py` with the corrected code above.
2.  **Re-run the `uvicorn` command:**
    ```sh
    uvicorn app.main:app --reload
    

**Explanation:**

1.  **Imports & Path:** It imports necessary libraries and adds the `src` directory to the Python path so it can import functions from `recommend_strategy.py`.
2.  **Pydantic Models:** `DeckInput` defines the expected JSON input (`deck` list, `opponent_archetype` string). `RecommendationResponse` defines the JSON output structure.
3.  **FastAPI App:** Initializes the FastAPI app.
4.  **`app.state`:** Used to store loaded models and data, making them accessible across requests without reloading.
5.  **`@app.on_event("startup")`:** This function runs once when the API starts. It loads the card data, expected features list, and both trained models into `app.state`. Error handling is included.
6.  **`@app.post("/recommend")`:** Defines the main endpoint.
    * It takes the `DeckInput` JSON as input.
    * Performs validation checks (models loaded, deck size).
    * **Calls `get_recommendation_logic`:** This function (re-implemented within `main.py` for simplicity, reusing helpers) takes the input deck/opponent and the loaded resources (`app.state`) to perform feature calculation, prediction using both models, and applies the recommendation rules.
    * Calculates a single `confidence_score` based on which rule triggered the recommendation.
    * Returns the `RecommendationResponse`.
7.  **`get_recommendation_logic` (Internal):** This function now resides within `main.py` and mirrors the core logic of your original `recommend_strategy.py`, ensuring it uses the loaded models and data correctly within the API context. It calculates features, prepares input vectors, gets probabilities from both models, and applies the rules.
8.  **Uvicorn Entry Point:** The `if __name__ == "__main__":` block allows you to run the API directly using `python app/main.py`, although the standard way is using `uvicorn app.main:app --reload`.

**To Run This API:**

1.  **Install FastAPI and Uvicorn:**
    ```sh
    pip install fastapi "uvicorn[standard]"
    ```
2.  **Ensure File Paths:** Make sure the paths in the script (`DATA_DIR`, `MODELS_DIR`) correctly point to your data and models relative to the project root directory where you will run the command.
3.  **Run from Project Root:** Open your terminal in the `Match Strategy Recommendation System` directory (the one containing `app`, `src`, `data`, etc.) and run:
    ```sh
    uvicorn app.main:app --reload
    
'''