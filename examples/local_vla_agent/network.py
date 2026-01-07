import aiohttp
from typing import Dict, Any, Optional
from config import logger, CONTROL_ENDPOINT, START_MISSION_ENDPOINT, INTERVENTION_START_ENDPOINT
from schema import Action

async def http_get_json(session: aiohttp.ClientSession, url: str) -> Optional[Dict[str, Any]]:
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
    await http_post_json(session, CONTROL_ENDPOINT, Action(0.0, 0.0, lamp=lamp).to_payload())


async def start_mission_if_enabled(session: aiohttp.ClientSession, mission_mode: bool) -> None:
    if not mission_mode:
        return
    logger.info("MISSION_MODE=1 -> calling /start-mission")
    ok = await http_post_json(session, START_MISSION_ENDPOINT, payload={})
    if ok:
        logger.info("Mission started (200).")
    else:
        logger.warning("Mission start failed (bot may be unavailable).")


async def intervention_start(session: aiohttp.ClientSession, auto_intervention: bool) -> None:
    if not auto_intervention:
        return
    logger.warning("AUTO_INTERVENTION: starting intervention...")
    await http_post_json(session, INTERVENTION_START_ENDPOINT, payload={})
