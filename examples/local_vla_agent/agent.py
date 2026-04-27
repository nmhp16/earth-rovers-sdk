"""Decoupled VLA agent: perception, brain, control, and watchdog run as
independent async workers communicating through a shared snapshot.

Workers:
- perception_worker: telemetry+frames in parallel, then captions/depth/objects
  in parallel, then advect+update the polar occupancy grid.
- brain_worker: builds the rich prompt (spatial map + detected objects +
  mission bearing + outcome-tagged history) and queries the LLM at BRAIN_HZ.
- control_worker: smooths target_action and POSTs to /control at CONTROL_HZ
  independent of perception/brain latency.
- watchdog_worker: emergency-stop if any worker stops emitting heartbeats.

Optional sinks:
- JSONL_LOG_PATH: structured per-tick records for offline tuning.
"""

import asyncio
import json
import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List

import aiohttp

try:
    from config import (
        logger, DATA_ENDPOINT, V2_SCREENSHOT_ENDPOINT, CONTROL_ENDPOINT,
        HTTP_TIMEOUT_S, STALE_S,
        CONTROL_DT, BRAIN_DT, PERCEPTION_DT_MIN, DECISION_MAX_AGE_S,
        FAIL_STREAK_TO_INTERVENE, DANGER_STREAK_TO_INTERVENE,
        MISSION_MODE, AUTO_INTERVENTION,
        MISSION_TARGET_LAT, MISSION_TARGET_LON, MISSION_REACHED_M,
        MAX_LINEAR, MAX_ANGULAR,
        ACTION_SMOOTHING_ALPHA, MAX_DELTA_LINEAR_PER_SEC, MAX_DELTA_ANGULAR_PER_SEC,
        MAX_DELTA_LINEAR_PER_SEC_RECOVERY, MAX_DELTA_ANGULAR_PER_SEC_RECOVERY,
        DEADBAND_LINEAR, DEADBAND_ANGULAR,
        CAPTURE_HISTORY, CAPTURE_AGREEMENT, CAPTURE_SIMILARITY_THRESHOLD,
        STALE_RETRY, STALE_RETRY_DELAY, STALE_BACKOFF_MAX_SLEEP,
        LLM_HISTORY_LEN,
        ROTATE_RIGHT_DEG, ROTATE_LEFT_DEG, ANGULAR_STUCK_THRESHOLD, ANGULAR_MIN_DEG_PER_TICK,
        OBSTACLE_AHEAD_THRESHOLD, SKIP_REAR_VISION_WHEN_FORWARD, REAR_VISION_INTERVAL,
        CAPTION_TICK_STRIDE,
        DETECTION_TICK_STRIDE,
        CAMERA_HFOV_DEG, CAMERA_VFOV_DEG, CAMERA_MOUNT_HEIGHT_M, CAMERA_PITCH_DEG,
        TELEMETRY_SPEED_TO_MPS, SAFE_FORWARD_CLEARANCE_M,
        WATCHDOG_TIMEOUT_S, WATCHDOG_CHECK_DT,
        JSONL_LOG_PATH,
        RECOVERY_GPS_RADIUS_M, RECOVERY_REPEAT_WINDOW_S, RECOVERY_ESCALATE_AFTER,
    )
    from utils import (
        extract_first_json_object, is_timestamp_stale,
        stable_caption_from_history, smooth_action, orientation_delta,
        normalize_orientation_to_deg, haversine_m, bearing_deg, relative_bearing_deg,
    )
    from schema import Action
    from safety import validate_action, safety_override
    from vision import VisionSystem, DepthReading, DEPTH_UNAVAILABLE, DetectedObject
    from brain import BrainSystem, extract_json_from_text
    from spatial import PolarOccupancyGrid
    from network import (
        http_get_json, http_post_json, send_stop,
        start_mission_if_enabled, intervention_start,
    )
except ImportError:
    from .config import (
        logger, DATA_ENDPOINT, V2_SCREENSHOT_ENDPOINT, CONTROL_ENDPOINT,
        HTTP_TIMEOUT_S, STALE_S,
        CONTROL_DT, BRAIN_DT, PERCEPTION_DT_MIN, DECISION_MAX_AGE_S,
        FAIL_STREAK_TO_INTERVENE, DANGER_STREAK_TO_INTERVENE,
        MISSION_MODE, AUTO_INTERVENTION,
        MISSION_TARGET_LAT, MISSION_TARGET_LON, MISSION_REACHED_M,
        MAX_LINEAR, MAX_ANGULAR,
        ACTION_SMOOTHING_ALPHA, MAX_DELTA_LINEAR_PER_SEC, MAX_DELTA_ANGULAR_PER_SEC,
        MAX_DELTA_LINEAR_PER_SEC_RECOVERY, MAX_DELTA_ANGULAR_PER_SEC_RECOVERY,
        DEADBAND_LINEAR, DEADBAND_ANGULAR,
        CAPTURE_HISTORY, CAPTURE_AGREEMENT, CAPTURE_SIMILARITY_THRESHOLD,
        STALE_RETRY, STALE_RETRY_DELAY, STALE_BACKOFF_MAX_SLEEP,
        LLM_HISTORY_LEN,
        ROTATE_RIGHT_DEG, ROTATE_LEFT_DEG, ANGULAR_STUCK_THRESHOLD, ANGULAR_MIN_DEG_PER_TICK,
        OBSTACLE_AHEAD_THRESHOLD, SKIP_REAR_VISION_WHEN_FORWARD, REAR_VISION_INTERVAL,
        CAPTION_TICK_STRIDE,
        DETECTION_TICK_STRIDE,
        CAMERA_HFOV_DEG, CAMERA_VFOV_DEG, CAMERA_MOUNT_HEIGHT_M, CAMERA_PITCH_DEG,
        TELEMETRY_SPEED_TO_MPS, SAFE_FORWARD_CLEARANCE_M,
        WATCHDOG_TIMEOUT_S, WATCHDOG_CHECK_DT,
        JSONL_LOG_PATH,
        RECOVERY_GPS_RADIUS_M, RECOVERY_REPEAT_WINDOW_S, RECOVERY_ESCALATE_AFTER,
    )
    from .utils import (
        extract_first_json_object, is_timestamp_stale,
        stable_caption_from_history, smooth_action, orientation_delta,
        normalize_orientation_to_deg, haversine_m, bearing_deg, relative_bearing_deg,
    )
    from .schema import Action
    from .safety import validate_action, safety_override
    from .vision import VisionSystem, DepthReading, DEPTH_UNAVAILABLE, DetectedObject
    from .brain import BrainSystem, extract_json_from_text
    from .spatial import PolarOccupancyGrid
    from .network import (
        http_get_json, http_post_json, send_stop,
        start_mission_if_enabled, intervention_start,
    )


# ==============================================================================
# STUCK RECOVERY STATE MACHINE (with GPS-based escalation)
# ==============================================================================

class StuckRecoveryState(Enum):
    NORMAL = 0
    BACKING_UP = 1
    ROTATE_RIGHT = 2
    CHECK_RIGHT = 3
    ROTATE_LEFT = 4
    CHECK_LEFT = 5
    GIVING_UP = 6


class StuckRecoveryManager:
    """Drives the unstuck sequence: backup -> rotate-right -> rotate-left -> give-up.

    Diversification: track GPS coordinates where recovery triggers. If we
    re-trigger near the same spot within a time window, escalate
    (longer backup, lamp on, faster intervention request).
    """

    BACKUP_TICKS = 4
    ROTATE_TICKS = 3
    CHECK_TICKS = 2
    STUCK_THRESHOLD = 6

    def __init__(self):
        self.state = StuckRecoveryState.NORMAL
        self.state_ticks = 0
        self.stuck_counter = 0
        self.gave_up = False
        self.prev_orientation: Optional[float] = None
        self.cum_rot_deg = 0.0
        # Diversification state
        self.recent_triggers: deque = deque(maxlen=8)  # (ts, lat, lon)
        self.escalation_level = 0  # 0=normal, 1=longer backup, 2=request intervention

    # ------------------------------------------------------------------
    def _record_trigger(self, ts: float, lat: Optional[float], lon: Optional[float]):
        if lat is None or lon is None:
            self.recent_triggers.append((ts, None, None))
            self.escalation_level = 0
            return
        # Count how many recent triggers fall within RECOVERY_GPS_RADIUS_M
        nearby = 0
        cutoff_ts = ts - RECOVERY_REPEAT_WINDOW_S
        for (ots, olat, olon) in self.recent_triggers:
            if olat is None or olon is None or ots < cutoff_ts:
                continue
            if haversine_m(lat, lon, olat, olon) <= RECOVERY_GPS_RADIUS_M:
                nearby += 1
        self.recent_triggers.append((ts, lat, lon))
        if nearby >= RECOVERY_ESCALATE_AFTER:
            self.escalation_level = min(2, self.escalation_level + 1)
            logger.warning(
                f"Repeat-stuck near same GPS ({nearby} times in window) "
                f"-> escalation_level={self.escalation_level}"
            )
        else:
            self.escalation_level = 0

    def update(
        self,
        is_stuck: bool,
        depth_front: DepthReading,
        orientation: Optional[float],
        ts: float = 0.0,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
    ) -> Optional[Action]:
        if is_stuck and self.state == StuckRecoveryState.NORMAL:
            self.stuck_counter += 1
        elif not is_stuck and self.state == StuckRecoveryState.NORMAL:
            self.stuck_counter = 0
            self.gave_up = False

        if self.stuck_counter >= self.STUCK_THRESHOLD and self.state == StuckRecoveryState.NORMAL:
            logger.warning(f"Stuck detected (counter={self.stuck_counter}). Starting recovery.")
            self._record_trigger(ts, lat, lon)
            self.state = StuckRecoveryState.BACKING_UP
            self.state_ticks = 0
            self.stuck_counter = 0
            self.prev_orientation = orientation
            self.cum_rot_deg = 0.0

        if self.state == StuckRecoveryState.NORMAL:
            self.prev_orientation = orientation
            return None

        if orientation is not None and self.prev_orientation is not None:
            d = abs(orientation_delta(self.prev_orientation, orientation))
            if d > 0.0:
                self.cum_rot_deg += d
        self.prev_orientation = orientation

        self.state_ticks += 1

        # Escalation: longer backup
        backup_ticks = self.BACKUP_TICKS + (self.escalation_level * 2)
        rotate_left_ticks = self.ROTATE_TICKS * 2 + (self.escalation_level * 2)

        if self.state == StuckRecoveryState.BACKING_UP:
            if self.state_ticks >= backup_ticks:
                logger.info("Backup complete. Rotating right (orientation-aware)...")
                self.state = StuckRecoveryState.ROTATE_RIGHT
                self.state_ticks = 0
                self.cum_rot_deg = 0.0
            return Action(linear=-0.3, angular=0.0, lamp=1)

        if self.state == StuckRecoveryState.ROTATE_RIGHT:
            if self.cum_rot_deg >= ROTATE_RIGHT_DEG or self.state_ticks >= self.ROTATE_TICKS:
                logger.info("Right rotation complete. Checking path...")
                self.state = StuckRecoveryState.CHECK_RIGHT
                self.state_ticks = 0
                self.cum_rot_deg = 0.0
            return Action(linear=0.0, angular=-0.6, lamp=1)

        if self.state == StuckRecoveryState.CHECK_RIGHT:
            if self.state_ticks >= self.CHECK_TICKS:
                if self._is_path_clear(depth_front):
                    logger.info("Path clear after right turn. Resuming.")
                    self.state = StuckRecoveryState.NORMAL
                    self.state_ticks = 0
                    self.cum_rot_deg = 0.0
                    return None
                logger.info("Still blocked. Rotating left (longer)...")
                self.state = StuckRecoveryState.ROTATE_LEFT
                self.state_ticks = 0
                self.cum_rot_deg = 0.0
            return Action(linear=0.1, angular=0.0, lamp=1)

        if self.state == StuckRecoveryState.ROTATE_LEFT:
            if self.cum_rot_deg >= ROTATE_LEFT_DEG or self.state_ticks >= rotate_left_ticks:
                logger.info("Left rotation complete. Checking path...")
                self.state = StuckRecoveryState.CHECK_LEFT
                self.state_ticks = 0
                self.cum_rot_deg = 0.0
            return Action(linear=0.0, angular=0.6, lamp=1)

        if self.state == StuckRecoveryState.CHECK_LEFT:
            if self.state_ticks >= self.CHECK_TICKS:
                if self._is_path_clear(depth_front):
                    logger.info("Path clear after left turn. Resuming.")
                    self.state = StuckRecoveryState.NORMAL
                    self.state_ticks = 0
                    self.cum_rot_deg = 0.0
                    return None
                logger.warning("All paths blocked. Giving up - manual help needed.")
                self.state = StuckRecoveryState.GIVING_UP
                self.state_ticks = 0
                self.gave_up = True
            return Action(linear=0.1, angular=0.0, lamp=1)

        if self.state == StuckRecoveryState.GIVING_UP:
            if self.state_ticks > 20:
                logger.info("Retry timeout. Attempting recovery again...")
                self.state = StuckRecoveryState.BACKING_UP
                self.state_ticks = 0
                self.cum_rot_deg = 0.0
            return Action(linear=0.0, angular=0.0, lamp=1)

        return None

    def _is_path_clear(self, depth: DepthReading) -> bool:
        if depth.error or depth.summary in ("unavailable", "depth unavailable", "depth error"):
            return False
        return not depth.is_blocked_ahead()

    def reset(self):
        self.state = StuckRecoveryState.NORMAL
        self.state_ticks = 0
        self.stuck_counter = 0


# ==============================================================================
# SHARED STATE
# ==============================================================================

@dataclass
class Snapshot:
    # Sensor inputs
    telemetry: Optional[Dict[str, Any]] = None
    perception_ts: float = 0.0
    front_caption_raw: str = "no front frame"
    rear_caption_raw: str = "no rear frame"
    front_caption_stable: str = ""
    rear_caption_stable: str = ""
    front_caption_is_stable: bool = False
    rear_caption_is_stable: bool = False
    front_depth: DepthReading = field(default_factory=lambda: DEPTH_UNAVAILABLE)
    rear_depth: DepthReading = field(default_factory=lambda: DEPTH_UNAVAILABLE)
    detected_objects: List[DetectedObject] = field(default_factory=list)
    perception_ok: bool = False

    # Brain output
    target_action: Action = field(default_factory=lambda: Action(0.0, 0.0, 0))
    decision_ts: float = 0.0
    is_recovery: bool = False
    recovery_state_name: str = "NORMAL"
    danger: bool = False

    # Actuator state
    last_action: Action = field(default_factory=lambda: Action(0.0, 0.0, 0))
    last_orientation: Optional[float] = None
    prev_clearance_m: Optional[float] = None  # for outcome tagging

    # Streaks
    fail_streak: int = 0
    danger_streak: int = 0
    intervention_active: bool = False

    # Heartbeats (monotonic timestamps)
    heartbeat_perception: float = 0.0
    heartbeat_brain: float = 0.0
    heartbeat_control: float = 0.0


# ==============================================================================
# JSONL SINK
# ==============================================================================

class JsonlSink:
    """Append-only JSON-lines sink for structured per-tick logs.

    Disabled (no-op) when JSONL_LOG_PATH is empty.
    """
    def __init__(self, path: str):
        self.path = path
        self.enabled = bool(path)
        if self.enabled:
            try:
                # Open in append mode; create directory if needed
                import os
                d = os.path.dirname(path)
                if d:
                    os.makedirs(d, exist_ok=True)
                self.fh = open(path, "a", buffering=1)
                logger.info(f"JSONL sink: {path}")
            except Exception as e:
                logger.error(f"JSONL sink failed to open {path}: {e}")
                self.enabled = False

    def write(self, kind: str, **fields):
        if not self.enabled:
            return
        try:
            payload = {"ts": time.time(), "kind": kind, **fields}
            self.fh.write(json.dumps(payload, default=str) + "\n")
        except Exception as e:
            logger.error(f"JSONL write failed: {e}")

    def close(self):
        if self.enabled:
            try:
                self.fh.close()
            except Exception:
                pass


# ==============================================================================
# WORKERS
# ==============================================================================

async def perception_worker(
    session: aiohttp.ClientSession,
    vision: VisionSystem,
    state: Snapshot,
    grid: PolarOccupancyGrid,
    front_hist: deque,
    rear_hist: deque,
    sink: JsonlSink,
    shutdown: asyncio.Event,
):
    """Telemetry+frames in parallel, vision in parallel, advect+update grid."""
    rear_skip_counter = 0
    caption_tick = 0
    detection_tick = 0
    last_orientation_for_grid: Optional[float] = None
    last_perception_ts_for_grid: float = 0.0

    while not shutdown.is_set():
        tick_start = time.time()
        state.heartbeat_perception = tick_start

        # --- Fetch telemetry + frames in parallel ---
        telemetry, frames = await asyncio.gather(
            http_get_json(session, DATA_ENDPOINT),
            http_get_json(session, V2_SCREENSHOT_ENDPOINT),
        )
        now = time.time()

        if telemetry is None or frames is None:
            state.fail_streak += 1
            state.perception_ok = False
            logger.warning(f"Missing data (fail_streak={state.fail_streak})")
            await _maybe_intervene(session, state)
            backoff = min(PERCEPTION_DT_MIN * (1 + state.fail_streak * 0.25), STALE_BACKOFF_MAX_SLEEP)
            await asyncio.sleep(backoff)
            continue

        if (is_timestamp_stale(telemetry.get("timestamp"), now, STALE_S) or
                is_timestamp_stale(frames.get("timestamp"), now, STALE_S)):
            retried_ok = False
            for _ in range(STALE_RETRY):
                await asyncio.sleep(STALE_RETRY_DELAY)
                telemetry, frames = await asyncio.gather(
                    http_get_json(session, DATA_ENDPOINT),
                    http_get_json(session, V2_SCREENSHOT_ENDPOINT),
                )
                now = time.time()
                if (telemetry is not None and frames is not None and
                        not is_timestamp_stale(telemetry.get("timestamp"), now, STALE_S) and
                        not is_timestamp_stale(frames.get("timestamp"), now, STALE_S)):
                    retried_ok = True
                    break
            if not retried_ok:
                state.fail_streak += 1
                state.perception_ok = False
                logger.warning(f"Stale data (fail_streak={state.fail_streak})")
                await _maybe_intervene(session, state)
                backoff = min(PERCEPTION_DT_MIN * (1 + state.fail_streak * 0.25), STALE_BACKOFF_MAX_SLEEP)
                await asyncio.sleep(backoff)
                continue

        state.fail_streak = 0

        # --- Decide which heavy stages to run this tick ---
        going_backward = state.last_action.linear < -0.05
        rear_skip_counter += 1
        run_rear = (
            (not SKIP_REAR_VISION_WHEN_FORWARD)
            or going_backward
            or (rear_skip_counter % REAR_VISION_INTERVAL == 0)
        )

        caption_tick += 1
        run_caption = (caption_tick % max(1, CAPTION_TICK_STRIDE)) == 1
        detection_tick += 1
        run_detection = (
            vision.detector is not None
            and (detection_tick % max(1, DETECTION_TICK_STRIDE)) == 1
        )

        front_b64 = frames.get("front_frame")
        rear_b64 = frames.get("rear_frame") if run_rear else None

        # --- Run all enabled vision stages in parallel ---
        tasks: List[asyncio.Task] = []
        keys: List[str] = []
        if front_b64:
            tasks.append(asyncio.to_thread(vision.analyze_depth_b64, front_b64))
            keys.append("depth_front")
            if run_caption:
                tasks.append(asyncio.to_thread(vision.caption_b64, front_b64))
                keys.append("cap_front")
        if rear_b64:
            tasks.append(asyncio.to_thread(vision.analyze_depth_b64, rear_b64))
            keys.append("depth_rear")
            if run_caption:
                tasks.append(asyncio.to_thread(vision.caption_b64, rear_b64))
                keys.append("cap_rear")

        results = await asyncio.gather(*tasks) if tasks else []
        result_map = dict(zip(keys, results))

        cap_front = result_map.get("cap_front", state.front_caption_raw if not run_caption else "no front frame")
        depth_front: DepthReading = result_map.get("depth_front", DEPTH_UNAVAILABLE)
        if run_rear:
            cap_rear = result_map.get("cap_rear", state.rear_caption_raw if not run_caption else "no rear frame")
            depth_rear: DepthReading = result_map.get("depth_rear", DEPTH_UNAVAILABLE)
            rear_hist.append(cap_rear)
        else:
            cap_rear = state.rear_caption_raw
            depth_rear = state.rear_depth

        if run_caption:
            front_hist.append(cap_front)

        # --- Object detection (front frame only, on stride) ---
        objects: List[DetectedObject] = state.detected_objects
        if run_detection and front_b64:
            try:
                objects = await asyncio.to_thread(vision.detect_b64, front_b64, depth_front)
            except Exception as e:
                logger.error(f"Detection failed: {e}")
                objects = []

        # --- Stable captions ---
        f_caption, f_stable = stable_caption_from_history(
            list(front_hist), CAPTURE_AGREEMENT, CAPTURE_SIMILARITY_THRESHOLD
        )
        r_caption, r_stable = stable_caption_from_history(
            list(rear_hist), CAPTURE_AGREEMENT, CAPTURE_SIMILARITY_THRESHOLD
        )

        # --- Advect + update polar occupancy grid ---
        cur_orient = normalize_orientation_to_deg(telemetry.get("orientation"))
        speed_raw = float(telemetry.get("speed") or 0.0)
        speed_mps = speed_raw * TELEMETRY_SPEED_TO_MPS
        if last_orientation_for_grid is not None and last_perception_ts_for_grid > 0 and cur_orient is not None:
            dt = now - last_perception_ts_for_grid
            d_orient = orientation_delta(last_orientation_for_grid, cur_orient)
            grid.advect(d_orient, speed_mps, dt)
        grid.update_from_depth(
            depth_map=depth_front.depth_map,
            is_metric=depth_front.is_metric,
            hfov_deg=CAMERA_HFOV_DEG,
            vfov_deg=CAMERA_VFOV_DEG,
            mount_height_m=CAMERA_MOUNT_HEIGHT_M,
            camera_pitch_deg=CAMERA_PITCH_DEG,
        )
        last_orientation_for_grid = cur_orient
        last_perception_ts_for_grid = now

        # --- Publish snapshot ---
        state.telemetry = telemetry
        state.perception_ts = now
        state.front_caption_raw = cap_front
        state.rear_caption_raw = cap_rear
        state.front_caption_stable = f_caption
        state.rear_caption_stable = r_caption
        state.front_caption_is_stable = f_stable
        state.rear_caption_is_stable = r_stable
        state.front_depth = depth_front
        state.rear_depth = depth_rear
        state.detected_objects = objects
        state.perception_ok = True

        logger.info(f"Front: '{cap_front}' | {depth_front.summary}")
        if run_rear:
            logger.info(f"Rear:  '{cap_rear}' | {depth_rear.summary}")
        if run_detection:
            logger.info(f"Detected: {[f'{o.label}@{o.bearing_deg:+.0f}°' for o in objects]}")

        sink.write(
            "perception",
            cap_front=cap_front, cap_rear=cap_rear,
            front_depth_summary=depth_front.summary,
            rear_depth_summary=depth_rear.summary,
            objects=[{"label": o.label, "bearing": o.bearing_deg, "distance_m": o.distance_m}
                     for o in objects],
            telemetry={k: telemetry.get(k) for k in
                       ("battery", "speed", "orientation", "latitude", "longitude")},
            grid_summary=grid.to_compact_summary(),
        )

        elapsed = time.time() - tick_start
        await asyncio.sleep(max(0.0, PERCEPTION_DT_MIN - elapsed))


async def brain_worker(
    session: aiohttp.ClientSession,
    brain: BrainSystem,
    stuck_recovery: StuckRecoveryManager,
    state: Snapshot,
    grid: PolarOccupancyGrid,
    llm_history: deque,
    sink: JsonlSink,
    shutdown: asyncio.Event,
):
    """Decide target action at BRAIN_HZ from the latest perception snapshot."""
    last_brain_action: Optional[Action] = None
    last_brain_clearance: Optional[float] = None

    while not shutdown.is_set():
        tick_start = time.time()
        state.heartbeat_brain = tick_start

        if not state.perception_ok or state.telemetry is None:
            await asyncio.sleep(BRAIN_DT)
            continue

        telemetry = state.telemetry
        cap_front_raw = state.front_caption_raw
        cap_rear_raw = state.rear_caption_raw
        cap_for_brain_front = (
            f"{state.front_caption_stable} (stable)"
            if state.front_caption_is_stable
            else f"{cap_front_raw} (uncertain)"
        )
        cap_for_brain_rear = (
            f"{state.rear_caption_stable} (stable)"
            if state.rear_caption_is_stable
            else f"{cap_rear_raw} (uncertain)"
        )
        depth_front = state.front_depth
        depth_rear = state.rear_depth

        # --- Stuck detection ---
        current_speed = float(telemetry.get("speed") or 0.0)
        current_orientation = telemetry.get("orientation")

        is_trying_to_move = abs(state.last_action.linear) > 0.15
        is_actually_moving = abs(current_speed) >= 0.04
        linear_stuck = is_trying_to_move and not is_actually_moving

        rotation_expected = abs(state.last_action.angular) > ANGULAR_STUCK_THRESHOLD
        orientation_changed = False
        if (rotation_expected
                and state.last_orientation is not None
                and current_orientation is not None):
            delta_deg = abs(orientation_delta(state.last_orientation, current_orientation))
            orientation_changed = delta_deg >= ANGULAR_MIN_DEG_PER_TICK
        angular_stuck = rotation_expected and not orientation_changed

        is_stuck = linear_stuck or angular_stuck

        recovery_action = stuck_recovery.update(
            is_stuck, depth_front, current_orientation,
            ts=time.time(),
            lat=telemetry.get("latitude"),
            lon=telemetry.get("longitude"),
        )
        state.last_orientation = current_orientation

        # --- Mission target (bearing/distance) when configured ---
        mission_target = _compute_mission_target(telemetry)

        # --- Recovery vs LLM ---
        if recovery_action is not None:
            try:
                action, danger = safety_override(
                    recovery_action, telemetry,
                    cap_for_brain_front, cap_for_brain_rear,
                )
            except Exception as e:
                logger.error(f"Safety override on recovery failed: {e}")
                action, danger = recovery_action, False
            logger.info(
                f"Recovery {stuck_recovery.state.name}: lin={action.linear:.2f}, "
                f"ang={action.angular:.2f} (escalation={stuck_recovery.escalation_level})"
            )
            state.target_action = action
            state.is_recovery = True
            state.recovery_state_name = stuck_recovery.state.name
            state.danger = danger
            state.decision_ts = time.time()
            sink.write("decision", source="recovery", state=stuck_recovery.state.name,
                       action={"linear": action.linear, "angular": action.angular, "lamp": action.lamp})
            # Don't pollute LLM history with recovery actions
        else:
            # --- Build action-outcome history for the LLM prompt ---
            outcome = _outcome_string(state, last_brain_action, last_brain_clearance)
            if last_brain_action is not None and llm_history:
                # Tag the most recent history entry with its outcome
                llm_history[-1]["outcome"] = outcome

            # --- Spatial / objects / clearance / cliff for the prompt ---
            spatial_summary = grid.to_compact_summary()
            object_lines = [o.to_brain_line() for o in state.detected_objects]
            clearance_m = depth_front.clearance_ahead_m()

            if "no front frame" in cap_front_raw and "no rear frame" in cap_rear_raw:
                logger.warning("Blind (no frames) -> Forcing STOP")
                action = Action(0.0, 0.0, 0)
                danger = False
                raw = ""
            else:
                raw = await asyncio.to_thread(
                    brain.decide,
                    telemetry,
                    cap_for_brain_front,
                    cap_for_brain_rear,
                    list(llm_history),
                    depth_front.summary,
                    depth_rear.summary,
                    mission_target,
                    spatial_summary,
                    object_lines,
                    depth_front.cliff,
                    clearance_m,
                )

                action_obj: Optional[Dict[str, Any]] = None
                try:
                    action_obj = json.loads(raw)
                except json.JSONDecodeError:
                    try:
                        action_obj = json.loads(extract_json_from_text(raw))
                    except Exception:
                        action_obj = extract_first_json_object(raw)

                if not isinstance(action_obj, dict):
                    logger.error(f"LLM invalid output -> STOP. Raw: {raw!r}")
                    action = Action(0.0, 0.0, 0)
                    danger = False
                    state.fail_streak += 1
                    await _maybe_intervene(session, state)
                else:
                    try:
                        action = validate_action(action_obj)
                        action, danger = safety_override(
                            action, telemetry,
                            cap_for_brain_front, cap_for_brain_rear,
                        )
                    except Exception as e:
                        logger.error(f"Validation failed -> STOP: {e}")
                        action = Action(0.0, 0.0, 0)
                        danger = False
                        state.fail_streak += 1

            if danger:
                state.danger_streak += 1
            else:
                state.danger_streak = 0

            if (AUTO_INTERVENTION
                    and state.danger_streak >= DANGER_STREAK_TO_INTERVENE
                    and not state.intervention_active):
                await intervention_start(session, AUTO_INTERVENTION)
                state.intervention_active = True

            llm_history.append({
                "action": {"linear": action.linear, "angular": action.angular},
                "front": (state.front_caption_stable
                          if state.front_caption_is_stable
                          else cap_front_raw),
                "outcome": "",  # filled in next tick
                "stuck": stuck_recovery.stuck_counter,
                "clearance_m": clearance_m,
            })

            state.target_action = action
            state.is_recovery = False
            state.recovery_state_name = stuck_recovery.state.name
            state.danger = danger
            state.decision_ts = time.time()

            sink.write("decision",
                       source="brain",
                       action={"linear": action.linear, "angular": action.angular, "lamp": action.lamp},
                       danger=danger,
                       clearance_m=clearance_m,
                       cliff=depth_front.cliff,
                       mission=mission_target,
                       raw=raw)

            last_brain_action = action
            last_brain_clearance = clearance_m

        elapsed = time.time() - tick_start
        if elapsed > BRAIN_DT * 1.5:
            logger.warning(f"Brain tick slow: {elapsed:.2f}s (budget {BRAIN_DT:.2f}s)")
        await asyncio.sleep(max(0.0, BRAIN_DT - elapsed))


async def control_worker(
    session: aiohttp.ClientSession,
    state: Snapshot,
    sink: JsonlSink,
    shutdown: asyncio.Event,
):
    """Smooth target_action and POST to /control at CONTROL_HZ."""
    while not shutdown.is_set():
        tick_start = time.time()
        state.heartbeat_control = tick_start
        now = tick_start

        target = state.target_action
        last = state.last_action
        decision_age = now - state.decision_ts if state.decision_ts > 0 else float("inf")

        if state.decision_ts == 0.0:
            await asyncio.sleep(CONTROL_DT)
            continue

        if decision_age > DECISION_MAX_AGE_S:
            logger.warning(f"Decision stale ({decision_age:.1f}s) -> safe stop")
            await send_stop(session, lamp=last.lamp)
            state.last_action = Action(0.0, 0.0, last.lamp)
            sink.write("control", reason="stale", action={"linear": 0.0, "angular": 0.0, "lamp": last.lamp})
            await asyncio.sleep(CONTROL_DT)
            continue

        s_lin, s_ang = smooth_action(
            new_linear=target.linear,
            new_angular=target.angular,
            last_linear=last.linear,
            last_angular=last.angular,
            alpha=ACTION_SMOOTHING_ALPHA,
            max_delta_linear_per_sec=MAX_DELTA_LINEAR_PER_SEC,
            max_delta_angular_per_sec=MAX_DELTA_ANGULAR_PER_SEC,
            dt=CONTROL_DT,
            deadband_linear=DEADBAND_LINEAR,
            deadband_angular=DEADBAND_ANGULAR,
            max_linear=MAX_LINEAR,
            max_angular=MAX_ANGULAR,
            recovery_max_delta_linear_per_sec=MAX_DELTA_LINEAR_PER_SEC_RECOVERY,
            recovery_max_delta_angular_per_sec=MAX_DELTA_ANGULAR_PER_SEC_RECOVERY,
            is_recovery=state.is_recovery,
        )

        applied = Action(linear=s_lin, angular=s_ang, lamp=target.lamp)
        ok = await http_post_json(session, CONTROL_ENDPOINT, applied.to_payload())
        if ok:
            state.last_action = applied
        else:
            await send_stop(session, lamp=applied.lamp)
            state.last_action = Action(0.0, 0.0, applied.lamp)
            state.fail_streak += 1
            logger.warning("Control send failed -> STOP")

        sink.write("control", reason="ok" if ok else "fail",
                   action={"linear": applied.linear, "angular": applied.angular, "lamp": applied.lamp})

        elapsed = time.time() - tick_start
        await asyncio.sleep(max(0.0, CONTROL_DT - elapsed))


async def watchdog_worker(
    session: aiohttp.ClientSession,
    state: Snapshot,
    shutdown: asyncio.Event,
):
    """Emergency-stop if any worker stops emitting heartbeats."""
    while not shutdown.is_set():
        await asyncio.sleep(WATCHDOG_CHECK_DT)
        now = time.time()
        # Only check workers that have started at least once
        for name, hb in (
            ("perception", state.heartbeat_perception),
            ("brain", state.heartbeat_brain),
            ("control", state.heartbeat_control),
        ):
            if hb > 0 and now - hb > WATCHDOG_TIMEOUT_S:
                logger.error(
                    f"WATCHDOG: {name} silent for {now - hb:.1f}s "
                    f"(>{WATCHDOG_TIMEOUT_S:.1f}s). Emergency stop."
                )
                try:
                    await send_stop(session, lamp=state.last_action.lamp)
                except Exception as e:
                    logger.error(f"Watchdog stop failed: {e}")


# ==============================================================================
# HELPERS
# ==============================================================================

def _compute_mission_target(telemetry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return {bearing_deg, distance_m, ...} when MISSION_MODE and target are set."""
    if not MISSION_MODE or MISSION_TARGET_LAT is None or MISSION_TARGET_LON is None:
        return None
    lat = telemetry.get("latitude")
    lon = telemetry.get("longitude")
    if lat is None or lon is None:
        return None
    try:
        lat = float(lat); lon = float(lon)
    except Exception:
        return None
    distance = haversine_m(lat, lon, MISSION_TARGET_LAT, MISSION_TARGET_LON)
    if distance < MISSION_REACHED_M:
        return {"bearing_deg": 0.0, "distance_m": distance, "reached": True,
                "target_lat": MISSION_TARGET_LAT, "target_lon": MISSION_TARGET_LON}
    target_bearing = bearing_deg(lat, lon, MISSION_TARGET_LAT, MISSION_TARGET_LON)
    heading = normalize_orientation_to_deg(telemetry.get("orientation"))
    if heading is None:
        # Without heading we can't compute a relative bearing
        return {"bearing_deg": 0.0, "distance_m": distance, "reached": False,
                "target_lat": MISSION_TARGET_LAT, "target_lon": MISSION_TARGET_LON}
    rel = relative_bearing_deg(target_bearing, heading)
    return {
        "bearing_deg": rel,
        "distance_m": distance,
        "reached": False,
        "target_lat": MISSION_TARGET_LAT,
        "target_lon": MISSION_TARGET_LON,
    }


def _outcome_string(
    state: Snapshot,
    last_action: Optional[Action],
    last_clearance: Optional[float],
) -> str:
    """Build a short outcome tag for the LLM history.

    Compares the previous tick's commanded motion to actual motion and
    clearance change. Helps the LLM detect "I bumped the same obstacle".
    """
    if last_action is None:
        return ""
    speed = float((state.telemetry or {}).get("speed") or 0.0)
    parts = []
    cur_clearance = state.front_depth.clearance_ahead_m()
    if abs(last_action.linear) > 0.15 and abs(speed) < 0.04:
        parts.append("blocked")
    elif abs(last_action.linear) > 0.15:
        parts.append("moving")
    if last_clearance is not None and cur_clearance is not None:
        if cur_clearance < last_clearance - 0.3:
            parts.append(f"closer ({last_clearance:.1f}->{cur_clearance:.1f}m)")
        elif cur_clearance > last_clearance + 0.3:
            parts.append(f"farther ({last_clearance:.1f}->{cur_clearance:.1f}m)")
    return ", ".join(parts) if parts else "no change"


async def _maybe_intervene(session: aiohttp.ClientSession, state: Snapshot):
    if (AUTO_INTERVENTION
            and state.fail_streak >= FAIL_STREAK_TO_INTERVENE
            and not state.intervention_active):
        await intervention_start(session, AUTO_INTERVENTION)
        state.intervention_active = True


# ==============================================================================
# MAIN
# ==============================================================================

async def main():
    vision = VisionSystem()
    brain = BrainSystem()
    stuck_recovery = StuckRecoveryManager()
    grid = PolarOccupancyGrid()

    state = Snapshot()
    front_hist: deque = deque(maxlen=CAPTURE_HISTORY)
    rear_hist: deque = deque(maxlen=CAPTURE_HISTORY)
    llm_history: deque = deque(maxlen=LLM_HISTORY_LEN)

    shutdown = asyncio.Event()
    sink = JsonlSink(JSONL_LOG_PATH)

    timeout = aiohttp.ClientTimeout(total=HTTP_TIMEOUT_S)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            await start_mission_if_enabled(session, MISSION_MODE)

            logger.info("=== VLA AGENT STARTED ===")
            logger.info(
                f"Rates: control={1.0/CONTROL_DT:.1f}Hz, brain={1.0/BRAIN_DT:.1f}Hz, "
                f"perception<= {1.0/PERCEPTION_DT_MIN:.1f}Hz"
            )
            logger.info(
                f"Spatial: grid {grid.n_angular} bins, edges={grid.distance_edges}, "
                f"detection={'on' if vision.detector else 'off'}, "
                f"depth={'metric' if vision.depth_is_metric else 'relative'}"
            )

            await asyncio.gather(
                perception_worker(session, vision, state, grid, front_hist, rear_hist, sink, shutdown),
                brain_worker(session, brain, stuck_recovery, state, grid, llm_history, sink, shutdown),
                control_worker(session, state, sink, shutdown),
                watchdog_worker(session, state, shutdown),
            )
    finally:
        sink.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down (Ctrl+C).")
