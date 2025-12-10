# src/utils/synergy.py
from typing import List
from collections import Counter
from src.utils_old.deck_utils import normalize_name, card_roles

# simple pairwise synergy heuristics:
# +1 if good pair (tank + support), +1 if win_condition + spell (good), -1 if two heavy win conditions (conflict)
PAIR_SYNERGY = {
    ("tank", "support"): 1.5,
    ("tank", "spell"): 1.0,
    ("win_condition", "support"): 1.5,
    ("win_condition", "spell"): 1.2,
    ("air", "spell"): 0.8,
    ("win_condition", "win_condition"): -1.0,
    ("cycle", "win_condition"): 0.5,
}


def compute_pairwise_synergy(cards: List[str]) -> float:
    roles = card_roles(cards)
    cnt = 0.0
    total_pairs = 0
    # evaluate all unordered pairs
    n = len(roles)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = roles[i], roles[j]
            total_pairs += 1
            # check both orientations
            score = PAIR_SYNERGY.get((a, b)) or PAIR_SYNERGY.get((b, a)) or 0.0
            cnt += score
    if total_pairs == 0:
        return 0.0
    # normalized score
    return cnt / total_pairs


def deck_cohesion_score(cards: List[str]) -> float:
    """A simple cohesion metric: how many card roles are repeated (helps synergy)"""
    roles = card_roles(cards)
    c = Counter(roles)
    # if many roles concentrated in few keys -> high cohesion
    most_common = c.most_common(1)
    if not most_common:
        return 0.0
    top_count = most_common[0][1]
    return top_count / max(1, len(cards))
