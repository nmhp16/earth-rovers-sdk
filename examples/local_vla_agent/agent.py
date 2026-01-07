import asyncio
import json
import logging
import time
from typing import Optional, Dict, Any

import aiohttp

try:
    from config import (
        logger, DATA_ENDPOINT, V2_SCREENSHOT_ENDPOINT, CONTROL_ENDPOINT,
        HTTP_TIMEOUT_S, LOOP_DT, STALE_S,
        FAIL_STREAK_TO_INTERVENE, DANGER_STREAK_TO_INTERVENE,
        MISSION_MODE, AUTO_INTERVENTION
    )
    from utils import extract_first_json_object, is_timestamp_stale
    from schema import Action
    from safety import validate_action, safety_override
    from vision import VisionSystem
    from brain import BrainSystem
    from network import http_get_json, http_post_json, send_stop, start_mission_if_enabled, intervention_start
except ImportError:
    # Fallback to package-relative imports when executed as a module
    from .config import (
        logger, DATA_ENDPOINT, V2_SCREENSHOT_ENDPOINT, CONTROL_ENDPOINT,
        HTTP_TIMEOUT_S, LOOP_DT, STALE_S,
        FAIL_STREAK_TO_INTERVENE, DANGER_STREAK_TO_INTERVENE,
        MISSION_MODE, AUTO_INTERVENTION
    )
    from .utils import extract_first_json_object, is_timestamp_stale
    from .schema import Action
    from .safety import validate_action, safety_override
    from .vision import VisionSystem
    from .brain import BrainSystem
    from .network import http_get_json, http_post_json, send_stop, start_mission_if_enabled, intervention_start


# -------------------- Main loop --------------------
async def main():
    vision = VisionSystem()
    brain = BrainSystem()

    timeout = aiohttp.ClientTimeout(total=HTTP_TIMEOUT_S)
    last_action = Action(0.0, 0.0, 0)

    danger_streak = 0
    fail_streak = 0
    intervention_active = False

    logger.info("=== VLA AGENT STARTED ===")
    async with aiohttp.ClientSession(timeout=timeout) as session:
        await start_mission_if_enabled(session, MISSION_MODE)

        while True:
            tick_start = time.time()
            now = time.time()

            telemetry = await http_get_json(session, DATA_ENDPOINT)
            frames = await http_get_json(session, V2_SCREENSHOT_ENDPOINT)

            if telemetry is None or frames is None:
                fail_streak += 1
                logger.warning(f"Missing data (fail_streak={fail_streak}) -> STOP")
                await send_stop(session, lamp=last_action.lamp)

                if AUTO_INTERVENTION and (fail_streak >= FAIL_STREAK_TO_INTERVENE) and not intervention_active:
                    await intervention_start(session, AUTO_INTERVENTION)
                    intervention_active = True

                await asyncio.sleep(LOOP_DT)
                continue

            # staleness check (uses timestamps described in the docs)
            if is_timestamp_stale(telemetry.get("timestamp"), now, STALE_S) or is_timestamp_stale(frames.get("timestamp"), now, STALE_S):
                fail_streak += 1
                logger.warning(f"Stale timestamp (fail_streak={fail_streak}) -> STOP")
                await send_stop(session, lamp=last_action.lamp)

                if AUTO_INTERVENTION and (fail_streak >= FAIL_STREAK_TO_INTERVENE) and not intervention_active:
                    await intervention_start(session, AUTO_INTERVENTION)
                    intervention_active = True

                await asyncio.sleep(LOOP_DT)
                continue

            fail_streak = 0  # we got fresh data

            front_b64 = frames.get("front_frame")
            rear_b64 = frames.get("rear_frame")

            # caption in background threads (BLIP is heavy)
            if front_b64:
                cap_front = await asyncio.to_thread(vision.caption_b64, front_b64)
            else:
                cap_front = "no front frame"

            if rear_b64:
                cap_rear = await asyncio.to_thread(vision.caption_b64, rear_b64)
            else:
                cap_rear = "no rear frame"

            logger.info(f"👀 Front: '{cap_front}' | Rear: '{cap_rear}'")

            # LLM decision (heavy) in background thread
            raw = await asyncio.to_thread(brain.decide, telemetry, cap_front, cap_rear)

            # parse JSON robustly
            action_obj: Optional[Dict[str, Any]] = None
            try:
                action_obj = json.loads(raw)
            except json.JSONDecodeError:
                action_obj = extract_first_json_object(raw)

            if not isinstance(action_obj, dict):
                logger.error(f"LLM invalid output -> STOP. Raw: {raw!r}")
                await send_stop(session, lamp=last_action.lamp)
                fail_streak += 1

                if AUTO_INTERVENTION and (fail_streak >= FAIL_STREAK_TO_INTERVENE) and not intervention_active:
                    await intervention_start(session, AUTO_INTERVENTION)
                    intervention_active = True

                await asyncio.sleep(LOOP_DT)
                continue

            # validate + safety override
            try:
                action = validate_action(action_obj)
                action, danger = safety_override(action, telemetry, cap_front, cap_rear)
            except Exception as e:
                logger.error(f"Validation failed -> STOP: {e}")
                await send_stop(session, lamp=last_action.lamp)
                fail_streak += 1
                await asyncio.sleep(LOOP_DT)
                continue

            if danger:
                danger_streak += 1
            else:
                danger_streak = 0

            if AUTO_INTERVENTION and (danger_streak >= DANGER_STREAK_TO_INTERVENE) and not intervention_active:
                await intervention_start(session, AUTO_INTERVENTION)
                intervention_active = True

            logger.info(f"Action: linear={action.linear:.2f}, angular={action.angular:.2f}, lamp={action.lamp}")

            ok = await http_post_json(session, CONTROL_ENDPOINT, action.to_payload())
            if not ok:
                logger.warning("Control send failed -> STOP")
                await send_stop(session, lamp=action.lamp)
                fail_streak += 1
            else:
                last_action = action

            # stabilize loop rate
            elapsed = time.time() - tick_start
            await asyncio.sleep(max(0.0, LOOP_DT - elapsed))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down (Ctrl+C).")
