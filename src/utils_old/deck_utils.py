# src/utils/deck_utils.py
import ast
from typing import List, Optional, Dict

# Basic card role map for simple archetype/synergy heuristics.
# Expand this map as needed.
CARD_ROLE = {
    "royal giant": "win_condition",
    "royal ghost": "win_condition",
    "hog rider": "win_condition",
    "lava hound": "win_condition",
    "miner": "win_condition",
    "golem": "tank",
    "giant": "tank",
    "pekka": "tank",
    "baby dragon": "air",
    "mega minion": "air",
    "minions": "air",
    "skeletons": "cycle",
    "ice spirit": "cycle",
    "electro spirit": "cycle",
    "log": "spell",
    "zap": "spell",
    "fireball": "spell",
    "poison": "spell",
    "lightning": "spell",
    "princess": "support",
    "musketeer": "support",
    "wizard": "support",
    "hunter": "support",
    "fisherman": "control",
    "barbarian barrel": "utility",
    "goblin barrel": "win_condition",
    # add more mappings as you see fit...
}

# small elixir default map for fallback; you can load full mapping from data/cards_elixir.json if available
FALLBACK_ELIXIR = {
    "royal giant": 6, "royal ghost": 3, "skeletons": 1, "fisherman": 3,
    "hunter": 4, "barbarian barrel": 2, "electro spirit": 1, "lightning": 6,
    "ice spirit": 1, "zap":2, "fireball":4, "princess":3, "musketeer":4,
    "baby dragon":4, "mega minion":3, "golem":8, "hog rider":4, "miner":3
}


def parse_cards_field(x) -> List[str]:
    """Safely parse a cards column that may be string representation of list or comma-separated."""
    if x is None:
        return []
    if isinstance(x, list):
        return x
    s = str(x)
    s = s.strip()
    # if it's a Python list string: "['A','B']"
    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                return [str(i).strip() for i in parsed]
        except Exception:
            pass
    # if comma-separated
    if "," in s:
        return [c.strip() for c in s.split(",") if c.strip()]
    # single card name
    if s:
        return [s]
    return []


def normalize_name(name: str) -> str:
    if name is None:
        return ""
    return name.strip().lower()


def average_elixir_from_cards(cards: List[str], elixir_map: Optional[Dict[str, int]] = None) -> float:
    if not cards:
        return 0.0
    elixir_map = elixir_map or FALLBACK_ELIXIR
    vals = []
    for c in cards:
        n = normalize_name(c)
        v = elixir_map.get(n)
        if v is None:
            # try approximate by removing punctuation
            v = elixir_map.get(n.replace("'", "").replace(".", ""), None)
        vals.append(v if v is not None else 4)  # default 4
    return sum(vals) / len(vals)


def identify_archetype(cards: List[str]) -> str:
    """Simple rule-based archetype identifier. Expand rules as needed."""
    names = [normalize_name(c) for c in cards]
    nset = set(names)
    # High priority checks
    if any("lava hound" in c for c in names) or "balloon" in nset:
        return "LavaLoon"
    if "royal giant" in nset or "royalghost" in nset or "royal ghost" in nset:
        return "Royal Giant"
    if "golem" in nset or "golem" in " ".join(names):
        return "Beatdown"
    if "hog rider" in nset:
        return "Hog Cycle"
    if "miner" in nset and "poison" in nset:
        return "Miner Control"
    if "x-bow" in nset or "xbow" in nset:
        return "X-Bow"
    # log bait
    if "log" in nset or "goblin barrel" in nset or "princess" in nset:
        if "goblin barrel" in nset or "princess" in nset:
            return "Log Bait"
    # default fallback
    return "Unknown"


def card_roles(cards: List[str]) -> List[str]:
    """Return list of roles for cards"""
    return [CARD_ROLE.get(normalize_name(c), "unknown") for c in cards]
