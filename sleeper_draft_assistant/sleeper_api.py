import requests
from typing import Dict, Any, List, Optional

BASE = "https://api.sleeper.app"

def _get(path: str) -> Any:
    url = f"{BASE}{path}"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return r.json()

def get_user(user_name: str) -> Dict[str, Any]:
    """Get user object by username (returns id, name, etc.)."""
    return _get(f"/v1/user/{user_name}")

def get_user_leagues(user_id: str, season: str) -> List[Dict[str, Any]]:
    return _get(f"/v1/user/{user_id}/leagues/nfl/{season}")

def get_league(league_id: str) -> Dict[str, Any]:
    return _get(f"/v1/league/{league_id}")

def get_league_users(league_id: str) -> List[Dict[str, Any]]:
    return _get(f"/v1/league/{league_id}/users")

def get_league_rosters(league_id: str) -> List[Dict[str, Any]]:
    return _get(f"/v1/league/{league_id}/rosters")

def get_league_drafts(league_id: str) -> List[Dict[str, Any]]:
    return _get(f"/v1/league/{league_id}/drafts")

def get_draft(draft_id: str) -> Dict[str, Any]:
    return _get(f"/v1/draft/{draft_id}")

def get_draft_picks(draft_id: str) -> List[Dict[str, Any]]:
    return _get(f"/v1/draft/{draft_id}/picks")

def get_players() -> Dict[str, Any]:
    """All NFL players keyed by Sleeper player_id."""
    return _get("/v1/players/nfl")
