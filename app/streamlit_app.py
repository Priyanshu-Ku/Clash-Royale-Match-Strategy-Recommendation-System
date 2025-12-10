import streamlit as st
import requests
import pandas as pd
import os
import numpy as np
from typing import List, Dict, Optional, Any
import json
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# --- Configuration ---
API_BASE_URL = "http://127.0.0.1:8000"
CARDS_ENDPOINT = f"{API_BASE_URL}/cards"
RECOMMEND_ENDPOINT = f"{API_BASE_URL}/recommend"
OPTIMIZE_ENDPOINT = f"{API_BASE_URL}/recommend/optimize"
SEMANTIC_ENDPOINT = f"{API_BASE_URL}/semantic"
# --- NEW ENDPOINTS ---
EXPLAIN_STRATEGY_ENDPOINT = f"{API_BASE_URL}/explain/strategy"
EXPLAIN_TIPS_ENDPOINT = f"{API_BASE_URL}/explain/card_tips"

# --- Project Directory Setup ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
ANALYSIS_DIR = os.path.join(PROJECT_ROOT, "analysis")

# --- Data Loading (Cached) ---
@st.cache_data(ttl=300)
def get_card_data() -> Dict[str, Dict[str, Any]]:
    # (This function is unchanged)
    card_data_map = {}
    try:
        response = requests.get(CARDS_ENDPOINT, timeout=10)
        response.raise_for_status()
        data = response.json()
        cards = data.get('cards', [])
        if not cards:
            st.error("Failed to parse card data from API response.")
            return {"Error fetching cards": {}}
        for card in cards:
            name = card.get('name')
            if name:
                card_data_map[name] = {
                    'elixir': card.get('elixir'),
                    'type': card.get('type'),
                    'attributes': card.get('attributes', {})
                }
        return dict(sorted(card_data_map.items()))
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to API to fetch cards: {e}")
        return {"Error connecting to API": {}}
    except Exception as e:
        st.error(f"An unexpected error occurred while fetching cards: {e}")
        return {"Error processing card data": {}}

@st.cache_data(ttl=600)
def load_preprocessed_data():
    # (This function is unchanged)
    PREPROCESSED_PATH = os.path.join(DATA_DIR, "preprocessed_data.csv")
    MATCH_HISTORY_PATH = os.path.join(DATA_DIR, "match_history.csv")
    try:
        df = pd.read_csv(PREPROCESSED_PATH)
    except FileNotFoundError:
        st.error(f"`preprocessed_data.csv` not found. Looked in: {PREPROCESSED_PATH}")
        return None, None, None
    arch_cols = [col for col in df.columns if col.startswith('player_archetype_') and col != 'player_archetype_nan']
    if not arch_cols:
        df['archetype'] = 'Unknown'
    else:
        df['archetype'] = df[arch_cols].idxmax(axis=1).str.replace('player_archetype_', '', regex=False)
    try:
        df['cards_list'] = df['cards'].apply(lambda x: ast.literal_eval(x) if (isinstance(x, str) and x.startswith('[')) else [])
        df_exploded = df.explode('cards_list').rename(columns={'cards_list': 'card_name'})
        df_exploded['winner_flag'] = pd.to_numeric(df_exploded['winner_flag'], errors='coerce').fillna(0)
    except Exception as e:
        st.error(f"Error parsing 'cards' column for analysis: {e}")
        return None, None, None
    return df, df_exploded, MATCH_HISTORY_PATH

@st.cache_data(ttl=600)
def load_match_history_data(match_history_path: str):
    # (This function is unchanged)
    if not match_history_path:
        return None
    try:
        df = pd.read_csv(match_history_path)
    except FileNotFoundError:
        st.error(f"`match_history.csv` not found. Looked in: {match_history_path}")
        return None
    try:
        df['battleTime'] = pd.to_datetime(df['battleTime'])
        df['winner_binary'] = (df['winner'] == 'player').astype(int)
        attack_keywords = 'ladder|challenge|ranked|showdown|competitive'
        defense_keywords = 'friendly|cw|war|boatbattle|team|draft|duel'
        df['mode_type'] = 'Other'
        df.loc[df['gameMode'].str.contains(attack_keywords, case=False, na=False), 'mode_type'] = 'Attack'
        df.loc[df['gameMode'].str.contains(defense_keywords, case=False, na=False), 'mode_type'] = 'Defense'
        return df[df['mode_type'].isin(['Attack', 'Defense'])]
    except Exception as e:
        st.error(f"Error processing `match_history.csv`: {e}")
        return None

# --- Recommender Helper Functions ---

def calculate_average_elixir(deck: List[str], card_data_map: Dict[str, Dict[str, Any]]) -> Optional[float]:
    # (This function is unchanged)
    elixirs = []
    found_cards = 0
    for card_name in deck:
        card_info = card_data_map.get(card_name)
        if card_info and card_info.get('elixir') is not None and pd.notna(card_info['elixir']):
            elixirs.append(card_info['elixir'])
            found_cards += 1
    if not elixirs: return None
    if deck and found_cards < len(deck):
         st.sidebar.warning(f"Calculated avg elixir based on {found_cards}/{len(deck)} cards.")
    return np.mean(elixirs)

def infer_archetype(deck: List[str], card_data_map: Dict[str, Dict[str, Any]]) -> str:
    # (This function is unchanged)
    if not deck: return "Unknown"
    win_conditions = 0; high_hp_tanks = 0; buildings = 0; spells = 0; cycle_cards = 0
    total_elixir = 0; valid_cards = 0
    WINCON_NAMES_HEURISTIC = ["giant", "royal giant", "golem", "lava hound", "balloon", "hog rider", "ram rider", "battle ram", "x-bow", "mortar", "goblin drill", "goblin giant", "electro giant", "elixir golem", "graveyard", "skeleton barrel", "miner"]
    for card_name in deck:
        card_info = card_data_map.get(card_name)
        if card_info:
            valid_cards += 1; elixir = card_info.get('elixir'); card_type = card_info.get('type', '').lower()
            attrs = card_info.get('attributes', {}); hp = pd.to_numeric(attrs.get('hitpoints'), errors='coerce'); hp = 0.0 if pd.isna(hp) else hp
            targets = str(attrs.get('targets', '')).lower()
            if elixir is not None and pd.notna(elixir):
                total_elixir += elixir;
                if elixir <= 2: cycle_cards += 1
            if card_type == 'building': buildings += 1
            elif card_type == 'spell': spells += 1
            elif card_type == 'troop':
                 clean_card_name = card_name.lower().split('(')[0].strip()
                 if 'buildings' in targets or any(wc in clean_card_name for wc in WINCON_NAMES_HEURISTIC): win_conditions += 1
                 if hp > 1400: high_hp_tanks += 1
    avg_elixir = (total_elixir / valid_cards) if valid_cards > 0 else 4.0
    deck_names_lower = [n.lower() for n in deck]
    if win_conditions == 0 and spells >= 4: return "Spell Cycle"
    if any(wc in n for wc in ["x-bow", "mortar"] for n in deck_names_lower) and buildings >= 2: return "Siege"
    if any(wc in n for wc in ["lava hound", "balloon"] for n in deck_names_lower): return "LavaLoon"
    if any(wc in n for wc in ["golem", "electro giant"] for n in deck_names_lower) and avg_elixir >= 4.0: return "Beatdown"
    if any("royal giant" in n for n in deck_names_lower): return "Royal Giant"
    if any("hog rider" in n for n in deck_names_lower) and avg_elixir <= 3.5: return "Hog Cycle"
    if any("goblin barrel" in n for n in deck_names_lower) and cycle_cards >= 3: return "Log Bait"
    if any("miner" in n for n in deck_names_lower) and avg_elixir <= 3.8: return "Miner Control"
    if win_conditions >= 1 and high_hp_tanks >= 1 and avg_elixir >= 3.8: return "Beatdown"
    if avg_elixir <= 3.2 and cycle_cards >= 4: return "Cycle"
    if win_conditions >= 2 and any(br in n for br in ["battle ram", "ram rider", "bandit", "royal ghost"] for n in deck_names_lower): return "Bridge Spam"
    return "Unknown"

def get_recommendation(
    player_deck: List[str], 
    opponent_deck: Optional[List[str]], 
    opponent_archetype_inferred: str
) -> Optional[Dict]:
    # (This function is unchanged)
    payload = {
        "player_deck": player_deck,
        "opponent_deck": opponent_deck if opponent_deck else None,
        "opponent_archetype": opponent_archetype_inferred,
        "player_trophies": 7000.0,
        "player_explevel": 60.0
    }
    try:
        response = requests.post(RECOMMEND_ENDPOINT, json=payload, timeout=30)
        response.raise_for_status(); return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Connection Error: {e}")
    return None

def get_optimizer_results(
    player_deck: List[str],
    opponent_deck: Optional[List[str]],
    opponent_archetype: str
) -> Optional[Dict]:
    # (This function is unchanged)
    payload = {
        "player_deck": player_deck,
        "opponent_deck": opponent_deck if opponent_deck else None,
        "opponent_archetype": opponent_archetype,
        "player_trophies": 7000.0,
        "player_explevel": 60.0
    }
    try:
        response = requests.post(OPTIMIZE_ENDPOINT, json=payload, timeout=120)
        response.raise_for_status(); 
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Optimizer API Error: {e}")
    return None

def search_similar_strategies(query: str, top_k: int = 5) -> Optional[List[Dict]]:
    # (This function is unchanged)
    if not query: 
        st.warning("Please enter a search query.")
        return None
    payload = {"query": query, "top_k": top_k}
    try:
        response = requests.post(SEMANTIC_ENDPOINT, json=payload, timeout=20)
        response.raise_for_status()
        return response.json().get('results', [])
    except requests.exceptions.RequestException as e: 
        st.error(f"Semantic Search API Error: {e}")
    return None

# --- NEW: REFACTORED LLM FUNCTIONS ---

def get_llm_explanation(
    player_deck: List[str], 
    opponent_archetype: str, 
    recommended_strategy: str, 
    probabilities: Dict[str, float]
) -> str:
    """Gets AI explanation from the FastAPI backend."""
    payload = {
        "player_deck": player_deck,
        "opponent_archetype": opponent_archetype,
        "recommended_strategy": recommended_strategy,
        "probabilities": probabilities
    }
    try:
        response = requests.post(EXPLAIN_STRATEGY_ENDPOINT, json=payload, timeout=60)
        response.raise_for_status()
        return response.json().get("explanation", "Error: No explanation received.")
    except requests.exceptions.RequestException as e:
        st.error(f"AI Explanation API Error: {e}")
        return "Could not connect to the AI explanation service."

def get_card_tips(card_name: str) -> str:
    """Gets AI card tips from the FastAPI backend."""
    payload = {"card_name": card_name}
    try:
        response = requests.post(EXPLAIN_TIPS_ENDPOINT, json=payload, timeout=60)
        response.raise_for_status()
        return response.json().get("tips", "Error: No tips received.")
    except requests.exceptions.RequestException as e:
        st.error(f"AI Card Tips API Error: {e}")
        return f"Could not get tips for {card_name}."

# --- Streamlit App UI ---
st.set_page_config(page_title="Clash Royale Strategy Recommender", layout="wide")
# (CSS Markdown is unchanged)
st.markdown("""<style>... (omitted for brevity) ...</style>""", unsafe_allow_html=True)
st.title("👑 Clash Royale Strategy Recommender")

# --- Load Card Data (Needed for both tabs) ---
with st.spinner("Loading available cards from API..."):
    card_data_map = get_card_data()
available_card_names = list(card_data_map.keys())
if not available_card_names or "Error" in available_card_names[0]:
    st.error("Could not load card list. Ensure the backend API is running: " + API_BASE_URL)
    st.stop()

# --- Define Tabs: Recommender and Analysis ---
tab1, tab2 = st.tabs(["**Recommender**", "**Data Analysis (EDA)**"])

# ==================================================================
# --- TAB 1: RECOMMENDER ---
# ==================================================================
with tab1:
    # (Input Section is unchanged)
    st.caption("Select your deck and the opponent's deck to get ML-powered strategy recommendations!")
    col_deck, col_opponent = st.columns(2)
    with col_deck:
        st.header("🛡️ Your Deck")
        selected_cards = st.multiselect(
            "Select exactly 8 cards:", options=available_card_names, max_selections=8,
            key="player_deck_select", help="Choose the 8 cards in your deck."
        )
        avg_elixir_player = calculate_average_elixir(selected_cards, card_data_map)
        avg_elixir_player_display = f"{avg_elixir_player:.2f}" if avg_elixir_player is not None else "--"
        st.metric(label="Average Elixir Cost", value=avg_elixir_player_display)
    with col_opponent:
        st.header("⚔️ Opponent's Deck")
        opponent_deck_cards = st.multiselect(
            "Select opponent's 8 cards (Optional):", options=available_card_names, max_selections=8,
            key="opponent_deck_select", help="Select opponent's cards if known. Helps infer their archetype."
        )
        avg_elixir_opponent = calculate_average_elixir(opponent_deck_cards, card_data_map)
        avg_elixir_opponent_display = f"{avg_elixir_opponent:.2f}" if avg_elixir_opponent is not None else "--"
        st.metric(label="Opponent Avg Elixir", value=avg_elixir_opponent_display)

    # (Recommendation Trigger is unchanged)
    st.divider()
    button_disabled = len(selected_cards) != 8
    recommend_button = st.button( "Analyze Matchup & Recommend Strategy", type="primary", use_container_width=True, disabled=button_disabled, key="recommend_button_main")

    if recommend_button: st.session_state['recommend_clicked'] = True
    else: st.session_state.setdefault('recommend_clicked', False)
    if 'last_recommendation' not in st.session_state: st.session_state['last_recommendation'] = None
    if 'has_recommendation' not in st.session_state: st.session_state['has_recommendation'] = False

    if recommend_button and not button_disabled:
        inferred_opponent_archetype = "Unknown"
        opponent_archetype_display = "Unknown (Deck not specified)"
        opponent_deck_payload = opponent_deck_cards if len(opponent_deck_cards) == 8 else None
        if opponent_deck_payload:
            with st.spinner("Inferring opponent archetype..."):
                inferred_opponent_archetype = infer_archetype(opponent_deck_payload, card_data_map)
                opponent_archetype_display = f"{inferred_opponent_archetype} (Inferred)"
                st.info(f"Inferred Opponent Archetype: **{inferred_opponent_archetype}**")
        elif opponent_deck_cards:
             st.warning("Cannot infer opponent archetype (need 8 cards). Using 'Unknown'.")
             opponent_archetype_display = "Unknown (Partial deck selected)"
        with st.spinner(f"Analyzing your deck vs '{inferred_opponent_archetype}'..."):
            recommendation_data = get_recommendation(
                player_deck=selected_cards, 
                opponent_deck=opponent_deck_payload,
                opponent_archetype_inferred=inferred_opponent_archetype
            )
            st.session_state['last_recommendation'] = recommendation_data
            st.session_state['last_opponent_archetype'] = inferred_opponent_archetype
            st.session_state['last_opponent_display'] = opponent_archetype_display
            st.session_state['last_player_deck'] = selected_cards
            st.session_state['last_opponent_deck'] = opponent_deck_payload
            st.session_state['has_recommendation'] = recommendation_data is not None
            st.session_state['optimizer_results'] = None

    # (Display Results section is unchanged)
    if st.session_state['has_recommendation'] and st.session_state['last_recommendation']:
        recommendation_data = st.session_state['last_recommendation']
        opponent_archetype_display = st.session_state.get('last_opponent_display', "Unknown")
        st.header("📊 Recommendation & Analysis")
        strategy = recommendation_data.get('recommended_strategy', 'N/A')
        confidence = recommendation_data.get('confidence_score', 0.0)
        probabilities = recommendation_data.get('probabilities', {})
        
        st.subheader("🏆 Recommended Strategy")
        rec_col, conf_col = st.columns([3, 1])
        with rec_col:
            if strategy == "Aggressive Push": st.success(f"**{strategy}**")
            elif strategy == "Defensive Counter": st.info(f"**{strategy}**")
            elif strategy == "Balanced Cycle": st.warning(f"**{strategy}**")
            else: st.error(f"**{strategy}**")
            st.caption(f"Against Opponent: **{opponent_archetype_display}**")
        with conf_col: st.metric(label="Confidence", value=f"{confidence:.1%}")
        
        st.subheader("📈 Win Probability Analysis")
        if probabilities:
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Win-Seeker (Optimistic - AUC: 0.59):**")
                ws_attack = probabilities.get('win_seeker_attack_win_prob', 0); ws_defense = probabilities.get('win_seeker_defense_win_prob', 0)
                st.progress(ws_attack, text=f"Attack: {ws_attack:.1%}"); st.progress(ws_defense, text=f"Defense: {ws_defense:.1%}")
            with col_b:
                st.markdown("**Loss-Avoider (Balanced - AUC: 0.58):**")
                la_attack = probabilities.get('loss_avoider_attack_win_prob', 0); la_defense = probabilities.get('loss_avoider_defense_win_prob', 0)
                st.progress(la_attack, text=f"Attack: {la_attack:.1%}"); st.progress(la_defense, text=f"Defense: {la_defense:.1%}")

        # --- Advanced Insights Section ---
        st.divider()
        st.subheader("💡 Advanced Insights & Tools")
        current_deck = st.session_state.get('last_player_deck', [])
        current_opponent_deck = st.session_state.get('last_opponent_deck', None)
        current_opponent_arch = st.session_state.get('last_opponent_archetype', 'Unknown')
        current_strategy = strategy
        current_probabilities = probabilities # Get current probabilities

        # (Optimizer Expander is unchanged)
        with st.expander("🔄 Deck Optimizer"):
            st.markdown("Find the best single-card swap to improve your deck's performance in this matchup.")
            if 'optimizer_results' not in st.session_state:
                st.session_state['optimizer_results'] = None
            if st.button("Find Best Card Swap", key="optimizer_btn"):
                with st.spinner("Running optimizer... This may take up to 2 minutes."):
                    results = get_optimizer_results(
                        player_deck=current_deck,
                        opponent_deck=current_opponent_deck,
                        opponent_archetype=current_opponent_arch
                    )
                    st.session_state['optimizer_results'] = results
            if st.session_state['optimizer_results']:
                # (Optimizer display logic is unchanged)
                results = st.session_state['optimizer_results']
                best_swap = results.get('best_swap', {})
                if "detail" in best_swap:
                    st.info(best_swap["detail"])
                elif "card_out" in best_swap:
                    st.subheader("🔥 Best Swap Found")
                    orig_prob = results.get('original_avg_win_prob', 0)
                    new_prob = best_swap.get('new_avg_win_prob', 0)
                    prob_gain = best_swap.get('prob_gain', 0)
                    orig_synergy = results.get('original_synergy', 0)
                    new_synergy = best_swap.get('new_synergy_score', 0)
                    c1, c2, c3 = st.columns(3)
                    with c1: st.metric(label="Swap This Card", value=f"❌ {best_swap.get('card_out')}")
                    with c2: st.metric(label="For This Card", value=f"✅ {best_swap.get('card_in')}")
                    with c3: st.metric(label="New Strategy", value=best_swap.get('new_recommendation', 'N/A'))
                    st.markdown("#### Impact of Swap")
                    c4, c5 = st.columns(2)
                    with c4: st.metric(label="Avg Win Probability", value=f"{new_prob:.1%}", delta=f"{prob_gain * 100:+.1f} pts")
                    with c5: st.metric(label="Deck Synergy Score", value=f"{new_synergy:.2f}", delta=f"{new_synergy - orig_synergy:+.2f}")
                    st.markdown("---")
                    st.subheader("Top 10 Recommended Swaps")
                    top_swaps = results.get('top_swaps', [])
                    if top_swaps:
                        df_swaps = pd.DataFrame(top_swaps)
                        st.dataframe(df_swaps[['card_out', 'card_in', 'score', 'new_avg_win_prob', 'prob_gain', 'new_synergy_score'
                        ]].style.format({'score': '{:.3f}', 'new_avg_win_prob': '{:.1%}', 'prob_gain': '{:+.1%}', 'new_synergy_score': '{:.2f}'
                        }), use_container_width=True)

        # --- THIS SECTION IS UPDATED ---
        with st.expander("🤖 Get AI Explanation"):
            st.markdown("Understand *why* this strategy is recommended.")
            if st.button("Generate Explanation (LLM)", key="explain_btn"):
                with st.spinner("Generating explanation via API..."):
                    # Call the new function
                    explanation = get_llm_explanation(
                        player_deck=current_deck,
                        opponent_archetype=current_opponent_arch,
                        recommended_strategy=current_strategy,
                        probabilities=current_probabilities
                    )
                    st.markdown(explanation) # Display result from API
            else:
                st.caption("Click the button to get an AI-generated explanation from the backend.")

        # (Semantic Search Expander is unchanged)
        with st.expander("🔍 Search Similar Cards"):
            st.markdown("Find cards based on a text description.")
            semantic_query = st.text_input("Describe a card:", placeholder="e.g., fast cycle deck to counter pekka", key="semantic_input")
            if st.button("Search Similar", key="semantic_btn"):
                if semantic_query:
                    with st.spinner("Searching semantic index via API..."):
                        search_results = search_similar_strategies(semantic_query)
                        if search_results is not None:
                            if search_results:
                                st.markdown("**Similar Cards Found:**")
                                for result in search_results[:5]:
                                    st.markdown(f"- {result.get('card_name', 'N/A')} *(Distance: {result.get('distance', 0):.2f})*")
                            else:
                                st.info("No similar cards found.")
                else:
                    st.warning("Please enter a search query.")
            else:
                st.caption("Enter a description and click search.")

        # --- THIS SECTION IS UPDATED ---
        with st.expander("🃏 Card-Specific Tips"):
            st.markdown("Get tips for using your cards in this matchup.")
            
            # Allow user to select a card from their deck
            tip_card = st.selectbox(
                "Select a card from your deck for tips:",
                options=current_deck,
                key="card_tip_select"
            )
            
            if st.button(f"Generate Tips for {tip_card}", key="tips_btn"):
                 with st.spinner(f"Generating tips for {tip_card} via API..."):
                     # Call the new function
                     tips = get_card_tips(tip_card)
                     st.markdown(tips) # Display result from API
            else:
                st.caption("Select a card and click the button to get AI-generated tips.")

    # (Button disabled warning is unchanged)
    elif button_disabled and st.session_state.get('recommend_clicked', False):
       st.warning("Please select exactly 8 cards for *Your Deck*.")


# ==================================================================
# --- TAB 2: DATA ANALYSIS (EDA) ---
# ==================================================================
with tab2:
    # (This entire tab is unchanged from the last version)
    st.header("📈 Exploratory Data Analysis (EDA)")
    st.caption("Visualizing trends from the `preprocessed_data.csv` and `match_history.csv` datasets.")
    data_load_state = st.info("Loading analysis data...")
    df_preprocessed, df_exploded, match_history_path = load_preprocessed_data()
    df_matches = load_match_history_data(match_history_path)
    data_load_state.empty()

    if df_preprocessed is None or df_exploded is None or df_matches is None:
        st.error("Failed to load one or more datasets required for analysis. Please check file paths and data integrity.")
    else:
        st.success(f"Loaded {len(df_preprocessed)} preprocessed records, {len(df_matches)} match records, and {len(df_exploded)} card instances.")
        st.divider()
        
        # --- Popular Cards ---
        st.subheader("🏅 Top 20 Most Popular Cards")
        st.markdown("Shows how frequently each card appears in decks.")
        try:
            pick_rates = df_exploded['card_name'].value_counts().nlargest(20)
            fig1, ax1 = plt.subplots(figsize=(10, 5))
            sns.barplot(x=pick_rates.values, y=pick_rates.index, ax=ax1, palette="coolwarm")
            ax1.set_xlabel("Frequency of Appearance"); ax1.set_ylabel("Card Name"); ax1.set_title("Top 20 Most Used Cards")
            st.pyplot(fig1)
        except Exception as e:
            st.error(f"Error generating card popularity chart: {e}")
        st.divider()

                # --- THIS IS THE UPDATED BLOCK ---
        st.subheader("⚔️ Player Performance by Archetype")
        st.markdown("Compares trophy distributions across archetypes.")
        try:
            if 'archetype' in df_preprocessed.columns:
                # Create the figure and axes
                fig2, ax2 = plt.subplots(figsize=(12, 7))
                
                # Normalize names
                df_preprocessed['archetype'] = df_preprocessed['archetype'].str.strip().str.title()
                
                # Filter valid data
                df_plot_trophies = df_preprocessed[
                    (~df_preprocessed['archetype'].isin(['Unknown', 'Nan'])) &
                    (df_preprocessed['trophies'] > 0)
                ].copy()
                
                # Add small jitter to make overlapping values visible
                df_plot_trophies['trophies_jitter'] = (
                    df_plot_trophies['trophies'] + np.random.uniform(-50, 50, len(df_plot_trophies))
                )

                # Sort archetypes by median trophy value
                order = (
                    df_plot_trophies.groupby('archetype')['trophies']
                    .median()
                    .sort_values(ascending=False)
                    .index
                )

                # --- Boxplot ---
                sns.boxplot(
                    data=df_plot_trophies,
                    x='archetype',
                    y='trophies_jitter',
                    order=order,
                    palette='coolwarm',
                    fliersize=0,  # hide outlier markers
                    ax=ax2 # Pass the axis
                )

                # --- Swarmplot (individual players) ---
                sns.swarmplot(
                    data=df_plot_trophies,
                    x='archetype',
                    y='trophies_jitter',
                    order=order,
                    color='black',
                    alpha=0.6,
                    size=3,
                    ax=ax2 # Pass the axis
                )

                # --- Add median labels ---
                medians = df_plot_trophies.groupby('archetype')['trophies'].median()
                for i, arch in enumerate(order):
                    median_val = medians[arch]
                    # Use ax2.text instead of plt.text
                    ax2.text(
                        i, median_val + 20,
                        f"{int(median_val)}",
                        ha='center',
                        va='bottom',
                        fontsize=9,
                        fontweight='bold',
                        color='black'
                    )
                
                ax2.set_title('Player Trophy Distribution by Archetype (Sorted by Median)', fontsize=16)
                ax2.set_xlabel('Archetype', fontsize=12)
                ax2.set_ylabel('Trophies', fontsize=12)
                ax2.tick_params(axis='x', rotation=45) # Use the object-oriented way
                fig2.tight_layout() # Use the figure object
                st.pyplot(fig2) # Pass the figure to streamlit

            else:
                st.warning("Skipping Plot: 'archetype' column not created.")
            
        except Exception as e:
            st.error(f"Error generating trophy distribution plot: {e}")
        st.divider()
        
        # --- Player Performance by Archetype (Violin Plot) ---
        st.subheader("🎻 Player Performance by Archetype (Trophy Distribution)")
        st.markdown("Displays how player trophies are distributed across archetypes, highlighting performance consistency and variability.")

        try:
            if 'archetype' in df_preprocessed.columns:
                # --- 1️⃣ Filter data for valid archetypes and trophies ---
                df_plot_trophies = df_preprocessed[
                    (~df_preprocessed['archetype'].isin(['Unknown', 'nan'])) &
                    (df_preprocessed['trophies'] > 0)
                ]

                if not df_plot_trophies.empty:
                    # --- 2️⃣ Create the violin plot ---
                    fig6, ax6 = plt.subplots(figsize=(12, 7))
                    sns.violinplot(
                        data=df_plot_trophies,
                        x='archetype',
                        y='trophies',
                        palette='coolwarm',
                        inner='quartile',  # Shows median and quartiles
                        ax=ax6
                    )

                    # --- 3️⃣ Customize plot aesthetics ---
                    ax6.set_title('Player Trophy Distribution by Archetype (Violin Plot)', fontsize=16)
                    ax6.set_xlabel('Inferred Archetype', fontsize=12)
                    ax6.set_ylabel('Trophies', fontsize=12)
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()

                    # --- 4️⃣ Render in Streamlit ---
                    st.pyplot(fig6)

                else:
                    st.info("Skipping Plot 2: No valid data for player trophies and archetypes.")

            else:
                st.info("Skipping Plot 2: 'archetype' column not found in df_prep.")

        except Exception as e:
            st.error(f"Error generating player performance violin plot: {e}")

        st.divider()

        # --- END OF UPDATED BLOCK ---

        # --- NEW: Elixir Usage by Archetype ---
        st.subheader("💧 Average Elixir Usage by Archetype")
        st.markdown("Compares average elixir cost across different archetypes.")
        try:
            df_plot_elixir = df_preprocessed[
                (df_preprocessed['archetype'] != 'Unknown')
                & (df_preprocessed['archetype'] != 'nan')
            ]

            fig5, ax5 = plt.subplots(figsize=(12, 6))
            sns.boxplot(
                data=df_plot_elixir,
                x='archetype',
                y='avg_elixir', # Use the avg_elixir column
                ax=ax5,
                palette='coolwarm'
            )
            ax5.set_title("Average Elixir Usage by Archetype")
            ax5.set_xlabel("Archetype")
            ax5.set_ylabel("Average Elixir")
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig5)
        except Exception as e:
            st.error(f"Error generating elixir usage plot: {e}")
        st.divider()
        # --- END OF NEW SECTION ---
        
        # --- Win Rate Over Time ---
        st.subheader("📊 Win Rate Over Time (Attack vs. Defense Modes)")
        st.markdown("Analyzes daily average win rates from match history data.")

        try:
            # Ensure data is sorted
            df_matches = df_matches.sort_values('battleTime')

            # Ensure datetime
            df_matches['battleTime'] = pd.to_datetime(df_matches['battleTime'], errors='coerce')

            # Filter non-empty matches
            df_match_filtered = df_matches.dropna(subset=['battleTime', 'winner_binary', 'mode_type'])

            if not df_match_filtered.empty:
                # --- Resample win rate per day per mode ---
                df_time_series = (
                    df_match_filtered
                    .set_index('battleTime')
                    .groupby('mode_type')['winner_binary']
                    .resample('D')
                    .mean()
                    .reset_index()
                )

                # --- Fill missing days for continuous lines ---
                df_time_series = (
                    df_time_series
                    .groupby('mode_type')
                    .apply(lambda g: g.set_index('battleTime').asfreq('D').fillna(method='ffill'))
                    .drop(columns='mode_type', errors='ignore')
                    .reset_index()
                )

                # --- Add 3-day rolling average for smoother trends ---
                df_time_series['winrate_smooth'] = (
                    df_time_series
                    .groupby('mode_type')['winner_binary']
                    .transform(lambda x: x.rolling(3, min_periods=1).mean())
                )

                # --- Plot ---
                fig3, ax3 = plt.subplots(figsize=(12, 7))
                sns.lineplot(
                    data=df_time_series,
                    x='battleTime',
                    y='winrate_smooth',
                    hue='mode_type',
                    palette={'Attack': 'red', 'Defense': 'blue'},
                    marker='o',
                    ax=ax3
                )

                ax3.set_title("Win Rate Over Time (Attack vs. Defense Modes)", fontsize=16)
                ax3.set_xlabel("Date", fontsize=12)
                ax3.set_ylabel("Average Win Rate (Player)", fontsize=12)
                plt.xticks(rotation=30)
                ax3.legend(title="Mode Type")
                plt.tight_layout()

                # --- Display in Streamlit ---
                st.pyplot(fig3)

            else:
                st.info("Skipping Plot 3: Processed match history is empty.")

        except Exception as e:
            st.error(f"Error generating win rate time-series: {e}")

        st.divider()

        
        # --- Archetype Win Rate ---
        st.subheader("🏆 Average Win Rate by Archetype")
        st.markdown("Shows which archetypes generally perform better in matches.")
        try:
            PREPROCESSED_PATH = os.path.join(DATA_DIR, "preprocessed_data.csv")
            CARDS_PATH = os.path.join(DATA_DIR, "cards_data.csv")
            df_prep = pd.read_csv(PREPROCESSED_PATH)
            df_cards = pd.read_csv(CARDS_PATH)
            df_archetype_win = (df_preprocessed.groupby('archetype')['winner_flag'].mean().sort_values(ascending=False).reset_index().head(15))
            fig4, ax4 = plt.subplots(figsize=(10, 5))
            sns.barplot(data=df_archetype_win, x='winner_flag', y='archetype', ax=ax4, palette='crest')
            ax4.set_xlabel("Average Win Rate"); ax4.set_ylabel("Archetype"); ax4.set_title("Top 15 Archetypes by Win Rate")
            st.pyplot(fig4)
        except Exception as e:
            st.error(f"Error generating archetype win rate chart: {e}")
        st.divider()
        
        # --- Troop Win Rate by Archetype ---
        st.subheader("🧱 Troop Win Rate by Archetype")
        st.markdown("Shows how different troop cards perform across various player archetypes based on average win rates.")

        try:
            import re

            # --- Step 1️⃣: Reattach archetype info to exploded match data ---
            if 'tag' in df_prep.columns and 'archetype' in df_prep.columns:
                df_exploded = df_exploded.merge(
                    df_prep[['tag', 'archetype']],
                    on='tag',
                    how='left'
                )
                st.success("✅ Archetype column successfully added to df_exploded.")
            # else:
            #     st.warning("⚠️ Missing 'tag' or 'archetype' column in df_prep.")

            # --- Step 2️⃣: Proceed only if required data exists ---
            if not df_exploded.empty and 'archetype' in df_exploded.columns and not df_cards.empty:

                # --- Clean card names for safer joins ---
                def clean_name(name):
                    if isinstance(name, str):
                        return re.sub(r'[^a-z0-9 ]', '', name.lower().strip())
                    return ''
        
                df_cards['card_name_clean'] = df_cards['card_name'].apply(clean_name)
                df_exploded['card_name_clean'] = df_exploded['card_name'].apply(clean_name)

                # --- Map card types ---
                card_types = (
                    df_cards[['card_name_clean', 'card_type']]
                    .drop_duplicates()
                    .set_index('card_name_clean')
                )
                df_exploded['card_type'] = df_exploded['card_name_clean'].map(card_types['card_type']).fillna('Unknown')

                # --- Filter to troops only ---
                df_troops = df_exploded[df_exploded['card_type'].str.lower() == 'troop']

                if not df_troops.empty:
                    # --- Pivot table: Troop (rows) vs Archetype (columns) ---
                    win_rate_pivot = df_troops.pivot_table(
                        index='card_name',
                        columns='archetype',
                        values='winner_flag',
                        aggfunc='mean'
                    )

                    # --- Keep only most common cards/archetypes for clarity ---
                    common_cards = df_troops['card_name'].value_counts().nlargest(25).index
                    common_archetypes = df_troops['archetype'].value_counts().nlargest(6).index

                    win_rate_pivot_filtered = win_rate_pivot.loc[
                        win_rate_pivot.index.isin(common_cards),
                        win_rate_pivot.columns.isin(common_archetypes)
                    ]

                    if win_rate_pivot_filtered.empty:
                        st.info("No data available for heatmap after filtering.")
                    else:
                        # --- Convert win rates to percentage ---
                        win_rate_pivot_filtered = win_rate_pivot_filtered * 100

                        # --- Plot heatmap ---
                        fig4, ax4 = plt.subplots(figsize=(12, 14))
                        sns.heatmap(
                            win_rate_pivot_filtered,
                            annot=True,
                            fmt=".1f",
                            cmap="RdYlGn",
                            linewidth=0.5,
                            cbar_kws={'label': 'Average Win Rate (%)'},
                            ax=ax4
                        )

                        ax4.set_title('Troop Win Rate (%) vs. Player Archetype', fontsize=16)
                        ax4.set_xlabel('Player Archetype', fontsize=12)
                        ax4.set_ylabel('Troop Card', fontsize=12)
                        plt.tight_layout()

                        st.pyplot(fig4)

                else:
                    st.info("Skipping Plot 4: No troop data found after filtering.")

            else:
                st.info("Skipping Plot 4: Missing necessary data for heatmap.")

        except Exception as e:
            st.error(f"Error generating troop win rate heatmap: {e}")

        st.divider()
        
        # --- Top 3 Troops per Archetype (Average Win Rate %) ---
        st.subheader("🔥 Top 3 Troops per Archetype (Average Win Rate %)")
        st.markdown("Highlights the top-performing troop cards across different player archetypes, based on average win rates.")

        try:
            import re

            if not df_exploded.empty and 'archetype' in df_exploded.columns and not df_cards.empty:

                # --- 1️⃣ Clean card names consistently ---
                def clean_name(name):
                    if isinstance(name, str):
                        return re.sub(r'[^a-z0-9 ]', '', name.lower().strip())
                    return ''
        
                df_cards['card_name_clean'] = df_cards['card_name'].apply(clean_name)
                df_exploded['card_name_clean'] = df_exploded['card_name'].apply(clean_name)

                # --- 2️⃣ Map card types ---
                card_types = (
                    df_cards[['card_name_clean', 'card_type']]
                    .drop_duplicates()
                    .set_index('card_name_clean')
                )
                df_exploded['card_type'] = df_exploded['card_name_clean'].map(card_types['card_type']).fillna('Unknown')

                # --- 3️⃣ Filter troops only ---
                df_troops = df_exploded[df_exploded['card_type'].str.lower() == 'troop']

                if not df_troops.empty:
                    # --- 4️⃣ Pivot for average win rate ---
                    win_rate_pivot = df_troops.pivot_table(
                        index='card_name',
                        columns='archetype',
                        values='winner_flag',
                        aggfunc='mean'
                    )

                    # --- 5️⃣ Limit to top cards/archetypes ---
                    common_cards = df_troops['card_name'].value_counts().nlargest(40).index
                    common_archetypes = df_troops['archetype'].value_counts().nlargest(6).index

                    win_rate_pivot_filtered = win_rate_pivot.loc[
                        win_rate_pivot.index.isin(common_cards),
                        win_rate_pivot.columns.isin(common_archetypes)
                    ] * 100  # convert to %

                    if win_rate_pivot_filtered.empty:
                        st.info("No data available for heatmap after filtering.")
                    else:
                        # --- 6️⃣ Sort by overall performance ---
                        card_order = (
                            win_rate_pivot_filtered
                            .max(axis=1)
                            .sort_values(ascending=False)
                            .index
                        )
                        win_rate_sorted = win_rate_pivot_filtered.loc[card_order]

                        # --- 7️⃣ Identify top 3 troops per archetype ---
                        top_cells = set()
                        for col in win_rate_sorted.columns:
                            top3 = win_rate_sorted[col].nlargest(3).index
                            for idx in top3:
                                top_cells.add((idx, col))

                        # --- 8️⃣ Plot heatmap ---
                        fig5, ax5 = plt.subplots(figsize=(12, 14))
                        sns.heatmap(
                            win_rate_sorted,
                            annot=True,
                            fmt=".1f",
                            cmap="RdYlGn",
                            linewidth=0.5,
                            cbar_kws={'label': 'Average Win Rate (%)'},
                            ax=ax5
                        )

                        # --- 9️⃣ Highlight Top 3 cells with black boxes ---
                        for i, card in enumerate(win_rate_sorted.index):
                            for j, arch in enumerate(win_rate_sorted.columns):
                                if (card, arch) in top_cells:
                                    ax5.add_patch(plt.Rectangle(
                                        (j, i), 1, 1,
                                        fill=False,
                                        edgecolor='black',
                                        lw=2.5
                                    ))

                        ax5.set_title('🔥 Top 3 Troops per Archetype (Average Win Rate %)', fontsize=16)
                        ax5.set_xlabel('Player Archetype', fontsize=12)
                        ax5.set_ylabel('Troop Card (Sorted by Peak Performance)', fontsize=12)
                        plt.tight_layout()

                        st.pyplot(fig5)

                else:
                    st.info("Skipping Plot 5: No troop data found after filtering.")

            else:
                st.info("Skipping Plot 5: Missing necessary data for heatmap.")

        except Exception as e:
            st.error(f"Error generating top troop heatmap: {e}")

        st.divider()


        # --- Model Performance Section ---
        st.header("🔬 Model Performance & Insights")
        st.caption("Visualizations generated by our `analysis/` scripts, showing how our models work.")
        def load_analysis_image(file_name):
            path = os.path.join(ANALYSIS_DIR, file_name)
            if os.path.exists(path):
                try:
                    return Image.open(path)
                except Exception as e:
                    st.error(f"Error loading image {file_name}: {e}")
                    return None
            else:
                st.warning(f"Plot not found: `{file_name}`. Please run the corresponding `analysis/` script.")
                return None

        st.subheader("🧠 Model Feature Importance")
        st.markdown("This shows what features our models learned are most important for predicting a win.")
        img_fi_ws = load_analysis_image("win_seeker_feature_importance.png")
        img_fi_la = load_analysis_image("loss_avoider_feature_importance.png")
        if img_fi_ws and img_fi_la:
            col1, col2 = st.columns(2)
            with col1:
                st.image(img_fi_ws, caption="Win-Seeker (Optimistic) Model Importance", use_container_width=True)
            with col2:
                st.image(img_fi_la, caption="Loss-Avoider (Balanced) Model Importance", use_container_width=True)
        st.divider()

        st.subheader("📈 Model Performance Validation")
        st.markdown("These plots show how well our models separate 'Win' vs. 'Lose' predictions.")
        img_roc = load_analysis_image("roc_curve_comparison.png")
        img_prob = load_analysis_image("probability_distribution.png")
        if img_roc:
            st.image(img_roc, caption="ROC Curve Comparison (AUC = Area Under Curve)", use_container_width=True)
            st.markdown("Our models (blue, green) are **above the 'No Skill' line (AUC: 0.5)**, proving they are predictive. The 'Win-Seeker' (AUC 0.59) and 'Loss-Avoider' (AUC 0.58) show similar, positive predictive power.")
        if img_prob:
            st.image(img_prob, caption="Predicted Probability Distributions", use_container_width=True)
            st.markdown("This shows *how* the models predict. The **(Top)** 'Win-Seeker' is biased, pushing 'Win' predictions to 1.0. The **(Bottom)** 'Loss-Avoider' shows a clearer separation, with the 'Lose' (blue) peak to the left and the 'Win' (red) peak to the right.")
        st.divider()

        st.subheader("🤝 Card Synergy & Top Troops")
        st.markdown("Final analysis of which card combinations and specific troops lead to the most wins.")
        img_syn = load_analysis_image("synergy_heatmap.png")
        img_troops = load_analysis_image("recommended_troop_lists.png")
        if img_syn:
            st.image(img_syn, caption="Core Card Synergy Heatmap", use_container_width=True)
        if img_troops:
            st.image(img_troops, caption="Top 15 Most Successful Troops (by Mode)", use_container_width=True)
        st.success("✅ Model Evaluation Complete!")

# --- Sidebar Info ---
# st.sidebar.header("About")
# st.sidebar.info("ML-powered Clash Royale strategy recommendations and EDA.")
# st.sidebar.header("Backend Status")
# st.sidebar.markdown(f"API URL: `{API_BASE_URL}`")
# try:
#     health_resp = requests.get(f"{API_BASE_URL}/health", timeout=5)
#     if health_resp.status_code == 200:
#         health_data = health_resp.json()
#         status = health_data.get("status", "unknown")
#         if status == "healthy": 
#             st.sidebar.success(f"API Status: {status.capitalize()}")
#         else: 
#             st.sidebar.error(f"API Status: {status.capitalize()}")
#             st.sidebar.json(health_data.get("details"), expanded=False)
#     else:
#         st.sidebar.error(f"API Status: Error ({health_resp.status_code})")
# except requests.exceptions.RequestException:
#     st.sidebar.error("API Status: Unreachable")



# **How to Run:**

# 1.  **Save the code** as `app/streamlit_app.py`.
# 2.  **Ensure FastAPI is running:** Keep the `uvicorn app.main:app --reload` command running in its own terminal.
# 3.  **Open a NEW terminal** in your project's root directory (`Match Strategy Recommendation System`).
# 4.  **Activate your venv** in the new terminal (`.\venv\Scripts\activate`).
# 5.  **Run Streamlit:**
#     ```sh
#     streamlit run app/streamlit_app.py
    
