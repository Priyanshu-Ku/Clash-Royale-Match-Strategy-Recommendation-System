import requests
import time
import os
from dotenv import load_dotenv

load_dotenv()
API_TOKEN = os.getenv("SUPERCELL_TOKEN")
BASE_URL = "https://api.clashroyale.com/v1"

HEADERS = {"Authorization": f"Bearer {API_TOKEN}"}


def get_player_data(player_tag):
    """Fetch player profile data."""
    url = f"{BASE_URL}/players/%23{player_tag}"  # %23 = #
    resp = requests.get(url, headers=HEADERS)
    if resp.status_code == 200:
        return resp.json()
    else:
        print(f"⚠️ Failed to fetch player {player_tag}: {resp.status_code}")
        return None


def get_battle_log(player_tag):
    """Fetch recent battles for a player."""
    url = f"{BASE_URL}/players/%23{player_tag}/battlelog"
    resp = requests.get(url, headers=HEADERS)
    if resp.status_code == 200:
        return resp.json()
    else:
        print(f"⚠️ Failed to fetch battles for {player_tag}: {resp.status_code}")
        return None
