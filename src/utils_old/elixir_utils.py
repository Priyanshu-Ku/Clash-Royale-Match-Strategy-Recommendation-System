# src/utils/elixir_utils.py
from typing import List, Dict, Any
from datetime import datetime
from src.utils_old.deck_utils import average_elixir_from_cards, parse_cards_field

def compute_elixir_features_from_deck(cards: List[str], elixir_map: Dict[str,int]=None) -> Dict[str, float]:
    """If we only have deck composition, produce approximate elixir features."""
    avg_elixir = average_elixir_from_cards(cards, elixir_map)
    # estimated elixir per minute: approximate cycles per minute ~ 20 / avg_elixir
    # This is a heuristic fallback, not precise without action logs
    est_elixir_per_min = max(0.0, 10.0 / max(1.0, avg_elixir))  # crude heuristic
    return {"avg_elixir": avg_elixir, "est_elixir_per_min": est_elixir_per_min}


def elixir_trend_from_actions(actions: List[Dict[str,Any]]) -> Dict[str,float]:
    """
    If detailed action timeline exists (list of events with timestamps and elixir spent),
    compute per-minute elixir spent over first minute, mid minute, late minute etc.
    The shape of 'actions' depends on source; this is a best-effort parser.
    """
    if not actions:
        return {"elixir_per_min_0_1": 0.0, "elixir_per_min_1_2": 0.0, "elixir_total_3min": 0.0}
    # assume actions have 'time' in seconds and 'elixir' spent
    # group into minute bins
    bins = {0:0.0, 1:0.0, 2:0.0}
    for a in actions:
        t = a.get("time", None)
        e = a.get("elixir", 0.0)
        if t is None:
            continue
        if t < 60:
            bins[0] += e
        elif t < 120:
            bins[1] += e
        else:
            bins[2] += e
    return {"elixir_per_min_0_1": bins[0], "elixir_per_min_1_2": bins[1], "elixir_total_3min": bins[0]+bins[1]+bins[2]}
