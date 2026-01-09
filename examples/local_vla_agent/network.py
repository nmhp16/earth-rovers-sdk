import aiohttp
from typing import Dict, Any, Optional

try:
    from config import logger, CONTROL_ENDPOINT, START_MISSION_ENDPOINT, INTERVENTION_START_ENDPOINT
    from schema import Action
except ImportError:
    from .config import logger, CONTROL_ENDPOINT, START_MISSION_ENDPOINT, INTERVENTION_START_ENDPOINT
    from .schema import Action


async def http_get_json(session: aiohttp.ClientSession, url: str) -> Optional[Dict[str, Any]]:
    """
    Perform an async GET request and return JSON response.
    
    Args:
        session: aiohttp client session
        url: API endpoint URL
        
    Returns:
        Parsed JSON dict, or None on error
    """
    try:
        async with session.get(url) as resp:
            if resp.status != 200:
                txt = await resp.text()
                logger.warning(f"GET {url} -> {resp.status}: {txt[:200]}")
                return None
            return await resp.json()
    except Exception as e:
        logger.error(f"GET {url} failed: {e}")
        return None


async def http_post_json(session: aiohttp.ClientSession, url: str, payload: Dict[str, Any]) -> bool:
    """
    Perform an async POST request with JSON payload.
    
    Args:
        session: aiohttp client session
        url: API endpoint URL
        payload: JSON-serializable dict
        
    Returns:
        True if status 200, False otherwise
    """
    try:
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                txt = await resp.text()
                logger.warning(f"POST {url} -> {resp.status}: {txt[:200]}")
                return False
            return True
    except Exception as e:
        logger.error(f"POST {url} failed: {e}")
        return False


async def send_stop(session: aiohttp.ClientSession, lamp: int = 0) -> None:
    """
    Send an immediate stop command to the rover.
    
    Args:
        session: aiohttp client session
        lamp: Lamp state to maintain during stop
    """
    await http_post_json(session, CONTROL_ENDPOINT, Action(0.0, 0.0, lamp=lamp).to_payload())


async def start_mission_if_enabled(session: aiohttp.ClientSession, mission_mode: bool) -> None:
    """
    Start a mission if mission mode is enabled.
    
    Args:
        session: aiohttp client session
        mission_mode: Whether mission mode is enabled
    """
    if not mission_mode:
        return
    
    logger.info("MISSION_MODE=1 -> calling /start-mission")
    ok = await http_post_json(session, START_MISSION_ENDPOINT, payload={})
    if ok:
        logger.info("Mission started (200).")
    else:
        logger.warning("Mission start failed (bot may be unavailable).")


async def intervention_start(session: aiohttp.ClientSession, auto_intervention: bool) -> None:
    """
    Request human intervention when auto-intervention is enabled.
    
    Args:
        session: aiohttp client session
        auto_intervention: Whether auto-intervention is enabled
    """
    if not auto_intervention:
        return
    
    logger.warning("AUTO_INTERVENTION: starting intervention...")
    await http_post_json(session, INTERVENTION_START_ENDPOINT, payload={})
