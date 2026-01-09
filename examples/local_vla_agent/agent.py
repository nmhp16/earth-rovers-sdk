import asyncio
import json
import time
from collections import deque
from enum import Enum
from typing import Optional, Dict, Any

import aiohttp

try:
    from config import (
        logger, DATA_ENDPOINT, V2_SCREENSHOT_ENDPOINT, CONTROL_ENDPOINT,
        HTTP_TIMEOUT_S, LOOP_DT, STALE_S,
        FAIL_STREAK_TO_INTERVENE, DANGER_STREAK_TO_INTERVENE,
        MISSION_MODE, AUTO_INTERVENTION,
        MAX_LINEAR, MAX_ANGULAR,
        ACTION_SMOOTHING_ALPHA, MAX_DELTA_LINEAR_PER_SEC, MAX_DELTA_ANGULAR_PER_SEC,
        DEADBAND_LINEAR, DEADBAND_ANGULAR,
        CAPTURE_HISTORY, CAPTURE_AGREEMENT, CAPTURE_SIMILARITY_THRESHOLD,
        STALE_RETRY, STALE_RETRY_DELAY, STALE_BACKOFF_MAX_SLEEP,
        LLM_HISTORY_LEN
    )
    from utils import (
        extract_first_json_object, is_timestamp_stale,
        stable_caption_from_history, smooth_action
    )
    from schema import Action
    from safety import validate_action, safety_override
    from vision import VisionSystem
    from brain import BrainSystem, extract_json_from_text
    from network import (
        http_get_json, http_post_json, send_stop,
        start_mission_if_enabled, intervention_start
    )
except ImportError:
    from .config import (
        logger, DATA_ENDPOINT, V2_SCREENSHOT_ENDPOINT, CONTROL_ENDPOINT,
        HTTP_TIMEOUT_S, LOOP_DT, STALE_S,
        FAIL_STREAK_TO_INTERVENE, DANGER_STREAK_TO_INTERVENE,
        MISSION_MODE, AUTO_INTERVENTION,
        MAX_LINEAR, MAX_ANGULAR,
        ACTION_SMOOTHING_ALPHA, MAX_DELTA_LINEAR_PER_SEC, MAX_DELTA_ANGULAR_PER_SEC,
        DEADBAND_LINEAR, DEADBAND_ANGULAR,
        CAPTURE_HISTORY, CAPTURE_AGREEMENT, CAPTURE_SIMILARITY_THRESHOLD,
        STALE_RETRY, STALE_RETRY_DELAY, STALE_BACKOFF_MAX_SLEEP,
        LLM_HISTORY_LEN
    )
    from .utils import (
        extract_first_json_object, is_timestamp_stale,
        stable_caption_from_history, smooth_action
    )
    from .schema import Action
    from .safety import validate_action, safety_override
    from .vision import VisionSystem
    from .brain import BrainSystem, extract_json_from_text
    from .network import (
        http_get_json, http_post_json, send_stop,
        start_mission_if_enabled, intervention_start
    )


# ==============================================================================
# STUCK RECOVERY STATE MACHINE
# ==============================================================================

class StuckRecoveryState(Enum):
    """States for the stuck recovery state machine."""
    NORMAL = 0           # Not stuck, normal operation
    BACKING_UP = 1       # Step 1: Backing up
    ROTATE_RIGHT = 2     # Step 2: Rotating 90° right
    CHECK_RIGHT = 3      # Step 3: Checking if can move forward
    ROTATE_LEFT = 4      # Step 4: Rotating 180° left (if right didn't work)
    CHECK_LEFT = 5       # Step 5: Checking if can move forward
    GIVING_UP = 6        # Final: Stop and wait for help


class StuckRecoveryManager:
    """
    Manages the stuck recovery sequence:
    
    1. If stuck detected → backup for N ticks
    2. Rotate 90° right
    3. Check if path is clear → if yes, resume normal
    4. If still blocked → rotate 180° left
    5. Check if path is clear → if yes, resume normal
    6. If still blocked → stop and give up
    """
    
    # Duration in ticks for each maneuver (at 2 Hz, 1 tick = 0.5s)
    BACKUP_TICKS = 4      # 2 seconds of backup
    ROTATE_TICKS = 3      # 1.5 seconds of rotation (~90°)
    CHECK_TICKS = 2       # 1 second to check
    STUCK_THRESHOLD = 6   # 3 seconds stuck before recovery starts
    
    def __init__(self):
        self.state = StuckRecoveryState.NORMAL
        self.state_ticks = 0
        self.stuck_counter = 0
        self.gave_up = False
    
    def update(self, is_stuck: bool, depth_front: str) -> Optional[Action]:
        """
        Update the recovery state machine.
        
        Args:
            is_stuck: True if rover is currently stuck (commanding move but not moving)
            depth_front: Depth analysis string from front camera
            
        Returns:
            Action to execute, or None if normal operation should continue
        """
        # Track how long we've been stuck
        if is_stuck and self.state == StuckRecoveryState.NORMAL:
            self.stuck_counter += 1
        elif not is_stuck and self.state == StuckRecoveryState.NORMAL:
            self.stuck_counter = 0
            self.gave_up = False
        
        # Start recovery sequence if stuck for too long
        if self.stuck_counter >= self.STUCK_THRESHOLD and self.state == StuckRecoveryState.NORMAL:
            logger.warning(f"Stuck detected (counter={self.stuck_counter}). Starting recovery sequence.")
            self.state = StuckRecoveryState.BACKING_UP
            self.state_ticks = 0
            self.stuck_counter = 0
        
        # Process current state
        if self.state == StuckRecoveryState.NORMAL:
            return None  # Normal operation
        
        self.state_ticks += 1
        
        if self.state == StuckRecoveryState.BACKING_UP:
            if self.state_ticks >= self.BACKUP_TICKS:
                logger.info("Backup complete. Rotating 90° right...")
                self.state = StuckRecoveryState.ROTATE_RIGHT
                self.state_ticks = 0
            return Action(linear=-0.3, angular=0.0, lamp=1)
        
        elif self.state == StuckRecoveryState.ROTATE_RIGHT:
            if self.state_ticks >= self.ROTATE_TICKS:
                logger.info("Right rotation complete. Checking path...")
                self.state = StuckRecoveryState.CHECK_RIGHT
                self.state_ticks = 0
            # Negative angular = turn right
            return Action(linear=0.0, angular=-0.7, lamp=1)
        
        elif self.state == StuckRecoveryState.CHECK_RIGHT:
            if self.state_ticks >= self.CHECK_TICKS:
                # Check if path is clear based on depth
                if self._is_path_clear(depth_front):
                    logger.info("Path clear after right turn! Resuming normal operation.")
                    self.state = StuckRecoveryState.NORMAL
                    self.state_ticks = 0
                    return None
                else:
                    logger.info("Path still blocked. Rotating 180° left...")
                    self.state = StuckRecoveryState.ROTATE_LEFT
                    self.state_ticks = 0
            # Small forward creep while checking
            return Action(linear=0.1, angular=0.0, lamp=1)
        
        elif self.state == StuckRecoveryState.ROTATE_LEFT:
            # 180° left = ~twice the rotation time
            if self.state_ticks >= self.ROTATE_TICKS * 2:
                logger.info("Left rotation complete. Checking path...")
                self.state = StuckRecoveryState.CHECK_LEFT
                self.state_ticks = 0
            # Positive angular = turn left
            return Action(linear=0.0, angular=0.7, lamp=1)
        
        elif self.state == StuckRecoveryState.CHECK_LEFT:
            if self.state_ticks >= self.CHECK_TICKS:
                if self._is_path_clear(depth_front):
                    logger.info("Path clear after left turn! Resuming normal operation.")
                    self.state = StuckRecoveryState.NORMAL
                    self.state_ticks = 0
                    return None
                else:
                    logger.warning("All paths blocked. Giving up - need manual intervention.")
                    self.state = StuckRecoveryState.GIVING_UP
                    self.state_ticks = 0
                    self.gave_up = True
            return Action(linear=0.1, angular=0.0, lamp=1)
        
        elif self.state == StuckRecoveryState.GIVING_UP:
            # Stay stopped until something changes
            if self.state_ticks > 20:  # After 10 seconds, try again
                logger.info("Retry timeout reached. Attempting recovery again...")
                self.state = StuckRecoveryState.BACKING_UP
                self.state_ticks = 0
            return Action(linear=0.0, angular=0.0, lamp=1)
        
        return None
    
    def _is_path_clear(self, depth_front: str) -> bool:
        """Check if the front path is clear based on depth analysis."""
        if not depth_front:
            return False
        depth_lower = depth_front.lower()
        # Path is clear if no "obstacle ahead" detected
        return "obstacle ahead" not in depth_lower and "path clear" in depth_lower
    
    def reset(self):
        """Reset the recovery state machine."""
        self.state = StuckRecoveryState.NORMAL
        self.state_ticks = 0
        self.stuck_counter = 0


# ==============================================================================
# MAIN CONTROL LOOP
# ==============================================================================

async def main():
    """
    Main control loop for the VLA agent.
    
    Continuously:
    1. Fetches sensor data and camera frames
    2. Generates vision captions and depth analysis
    3. Consults LLM brain for navigation decisions
    4. Applies safety overrides and smoothing
    5. Sends motor commands to rover
    """
    # Initialize subsystems
    vision = VisionSystem()
    brain = BrainSystem()
    stuck_recovery = StuckRecoveryManager()

    # HTTP client setup
    timeout = aiohttp.ClientTimeout(total=HTTP_TIMEOUT_S)
    last_action = Action(0.0, 0.0, 0)

    # Tracking variables
    danger_streak = 0
    fail_streak = 0
    intervention_active = False

    # Caption history for stability (reduces noise from frame-to-frame variation)
    front_hist = deque(maxlen=CAPTURE_HISTORY)
    rear_hist = deque(maxlen=CAPTURE_HISTORY)

    # Action history for LLM context
    llm_history = deque(maxlen=LLM_HISTORY_LEN)

    logger.info("=== VLA AGENT STARTED ===")
    
    async with aiohttp.ClientSession(timeout=timeout) as session:
        # Optionally start a mission
        await start_mission_if_enabled(session, MISSION_MODE)

        while True:
            tick_start = time.time()
            now = time.time()

            # ------------------------------------------------------------------
            # STEP 1: Fetch telemetry and camera frames
            # ------------------------------------------------------------------
            telemetry = await http_get_json(session, DATA_ENDPOINT)
            frames = await http_get_json(session, V2_SCREENSHOT_ENDPOINT)

            if telemetry is None or frames is None:
                fail_streak += 1
                logger.warning(f"Missing data (fail_streak={fail_streak}) -> STOP")
                await send_stop(session, lamp=last_action.lamp)

                if AUTO_INTERVENTION and fail_streak >= FAIL_STREAK_TO_INTERVENE and not intervention_active:
                    await intervention_start(session, AUTO_INTERVENTION)
                    intervention_active = True

                backoff_sleep = min(LOOP_DT * (1 + fail_streak * 0.25), STALE_BACKOFF_MAX_SLEEP)
                await asyncio.sleep(backoff_sleep)
                continue

            # ------------------------------------------------------------------
            # STEP 2: Check for stale data (timestamps too old)
            # ------------------------------------------------------------------
            stale_tele = is_timestamp_stale(telemetry.get("timestamp"), now, STALE_S)
            stale_frames = is_timestamp_stale(frames.get("timestamp"), now, STALE_S)
            
            if stale_tele or stale_frames:
                retried_ok = False
                for i in range(STALE_RETRY):
                    await asyncio.sleep(STALE_RETRY_DELAY)
                    telemetry = await http_get_json(session, DATA_ENDPOINT)
                    frames = await http_get_json(session, V2_SCREENSHOT_ENDPOINT)
                    now = time.time()
                    
                    if (telemetry is not None and frames is not None and
                        not is_timestamp_stale(telemetry.get("timestamp"), now, STALE_S) and
                        not is_timestamp_stale(frames.get("timestamp"), now, STALE_S)):
                        retried_ok = True
                        break
                
                if not retried_ok:
                    fail_streak += 1
                    logger.warning(f"Stale timestamp (fail_streak={fail_streak}) -> STOP")
                    await send_stop(session, lamp=last_action.lamp)

                    if AUTO_INTERVENTION and fail_streak >= FAIL_STREAK_TO_INTERVENE and not intervention_active:
                        await intervention_start(session, AUTO_INTERVENTION)
                        intervention_active = True

                    backoff_sleep = min(LOOP_DT * (1 + fail_streak * 0.25), STALE_BACKOFF_MAX_SLEEP)
                    await asyncio.sleep(backoff_sleep)
                    continue

            fail_streak = 0  # Fresh data received

            # ------------------------------------------------------------------
            # STEP 3: Generate vision captions and depth analysis
            # ------------------------------------------------------------------
            front_b64 = frames.get("front_frame")
            rear_b64 = frames.get("rear_frame")

            if front_b64:
                t_cap = asyncio.to_thread(vision.caption_b64, front_b64)
                t_depth = asyncio.to_thread(vision.analyze_depth_b64, front_b64)
                cap_front, depth_front = await asyncio.gather(t_cap, t_depth)
            else:
                cap_front = "no front frame"
                depth_front = "unavailable"

            if rear_b64:
                t_cap_rear = asyncio.to_thread(vision.caption_b64, rear_b64)
                t_depth_rear = asyncio.to_thread(vision.analyze_depth_b64, rear_b64)
                cap_rear, depth_rear = await asyncio.gather(t_cap_rear, t_depth_rear)
            else:
                cap_rear = "no rear frame"
                depth_rear = "unavailable"

            # Update caption history and get stable captions
            front_hist.append(cap_front)
            rear_hist.append(cap_rear)

            f_caption, f_stable = stable_caption_from_history(
                list(front_hist), CAPTURE_AGREEMENT, CAPTURE_SIMILARITY_THRESHOLD
            )
            r_caption, r_stable = stable_caption_from_history(
                list(rear_hist), CAPTURE_AGREEMENT, CAPTURE_SIMILARITY_THRESHOLD
            )

            cap_for_brain_front = f"{f_caption} (stable)" if f_stable else f"{cap_front} (uncertain)"
            cap_for_brain_rear = f"{r_caption} (stable)" if r_stable else f"{cap_rear} (uncertain)"

            logger.info(f"Front: '{cap_front}' | Depth: {depth_front}")
            logger.info(f"Rear: '{cap_rear}' | Depth: {depth_rear}")

            # ------------------------------------------------------------------
            # STEP 4: Check for stuck recovery override
            # ------------------------------------------------------------------
            current_speed = float(telemetry.get("speed") or 0.0)
            is_trying_to_move = abs(last_action.linear) > 0.15
            is_actually_moving = abs(current_speed) >= 0.04
            is_stuck = is_trying_to_move and not is_actually_moving

            recovery_action = stuck_recovery.update(is_stuck, depth_front)
            
            if recovery_action is not None:
                # Override with recovery action
                action = recovery_action
                logger.info(
                    f"Recovery mode ({stuck_recovery.state.name}): "
                    f"linear={action.linear:.2f}, angular={action.angular:.2f}"
                )
            else:
                # ------------------------------------------------------------------
                # STEP 5: Normal operation - consult LLM brain
                # ------------------------------------------------------------------
                if "no front frame" in cap_front and "no rear frame" in cap_rear:
                    logger.warning("Blind (no frames) -> Forcing STOP")
                    raw = '{"linear": 0.0, "angular": 0.0, "lamp": 0}'
                else:
                    raw = await asyncio.to_thread(
                        brain.decide,
                        telemetry,
                        cap_front,
                        cap_rear,
                        list(llm_history),
                        stuck_recovery.stuck_counter,
                        depth_front,
                        depth_rear
                    )

                # Parse LLM response
                action_obj: Optional[Dict[str, Any]] = None
                try:
                    action_obj = json.loads(raw)
                except json.JSONDecodeError:
                    try:
                        cleaned = extract_json_from_text(raw)
                        action_obj = json.loads(cleaned)
                    except Exception:
                        action_obj = extract_first_json_object(raw)

                if not isinstance(action_obj, dict):
                    logger.error(f"LLM invalid output -> STOP. Raw: {raw!r}")
                    await send_stop(session, lamp=last_action.lamp)
                    fail_streak += 1

                    if AUTO_INTERVENTION and fail_streak >= FAIL_STREAK_TO_INTERVENE and not intervention_active:
                        await intervention_start(session, AUTO_INTERVENTION)
                        intervention_active = True

                    await asyncio.sleep(LOOP_DT)
                    continue

                # Validate and apply safety overrides
                try:
                    action = validate_action(action_obj)
                    action, danger = safety_override(
                        action, telemetry, cap_for_brain_front, cap_for_brain_rear
                    )
                except Exception as e:
                    logger.error(f"Validation failed -> STOP: {e}")
                    await send_stop(session, lamp=last_action.lamp)
                    fail_streak += 1
                    await asyncio.sleep(LOOP_DT)
                    continue

                # Track danger streaks for intervention
                if danger:
                    danger_streak += 1
                else:
                    danger_streak = 0

                if AUTO_INTERVENTION and danger_streak >= DANGER_STREAK_TO_INTERVENE and not intervention_active:
                    await intervention_start(session, AUTO_INTERVENTION)
                    intervention_active = True

            # ------------------------------------------------------------------
            # STEP 6: Apply smoothing and send command
            # ------------------------------------------------------------------
            raw_linear = action.linear
            raw_angular = action.angular

            s_lin, s_ang = smooth_action(
                new_linear=raw_linear,
                new_angular=raw_angular,
                last_linear=last_action.linear,
                last_angular=last_action.angular,
                alpha=ACTION_SMOOTHING_ALPHA,
                max_delta_linear_per_sec=MAX_DELTA_LINEAR_PER_SEC,
                max_delta_angular_per_sec=MAX_DELTA_ANGULAR_PER_SEC,
                dt=LOOP_DT,
                deadband_linear=DEADBAND_LINEAR,
                deadband_angular=DEADBAND_ANGULAR,
                max_linear=MAX_LINEAR,
                max_angular=MAX_ANGULAR,
            )

            action.linear = s_lin
            action.angular = s_ang

            # Update LLM history
            llm_history.append({
                "action": {"linear": action.linear, "angular": action.angular},
                "front": f_caption if f_stable else cap_front,
                "stuck": stuck_recovery.stuck_counter
            })

            logger.info(
                f"Action: raw lin={raw_linear:.2f}, ang={raw_angular:.2f} -> "
                f"smoothed lin={action.linear:.2f}, ang={action.angular:.2f}, lamp={action.lamp}"
            )

            # Send command to rover
            ok = await http_post_json(session, CONTROL_ENDPOINT, action.to_payload())
            if not ok:
                logger.warning("Control send failed -> STOP")
                await send_stop(session, lamp=action.lamp)
                fail_streak += 1
            else:
                last_action = action

            # Maintain loop rate
            elapsed = time.time() - tick_start
            await asyncio.sleep(max(0.0, LOOP_DT - elapsed))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down (Ctrl+C).")

