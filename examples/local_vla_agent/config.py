import os
import logging

# ==============================================================================
# LOGGING SETUP
# ==============================================================================
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s - [VLA/%(levelname)s] - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("vla")


# ==============================================================================
# NETWORK CONFIGURATION
# ==============================================================================
# Base URL for the Earth Rovers SDK server
SDK_URL = os.getenv("SDK_URL", "http://127.0.0.1:8000").rstrip("/")

# API endpoints
CONTROL_ENDPOINT = f"{SDK_URL}/control"
DATA_ENDPOINT = f"{SDK_URL}/data"
V2_SCREENSHOT_ENDPOINT = f"{SDK_URL}/screenshot"

START_MISSION_ENDPOINT = f"{SDK_URL}/start-mission"
INTERVENTION_START_ENDPOINT = f"{SDK_URL}/interventions/start"
INTERVENTION_END_ENDPOINT = f"{SDK_URL}/interventions/end"

# HTTP request timeout in seconds
# Increase default timeout to handle slower networks / page fetches
HTTP_TIMEOUT_S = float(os.getenv("HTTP_TIMEOUT_S", "5.0"))


# ==============================================================================
# CONTROL LOOP TIMING
# ==============================================================================
# Three independent loops in the decoupled architecture:
#   - Control loop: high-rate motor command transmission (10 Hz)
#   - Brain loop:   LLM-driven decisioning + recovery FSM (2 Hz)
#   - Perception loop: vision + telemetry (as fast as VLM allows, capped)
LOOP_HZ = float(os.getenv("LOOP_HZ", "2.0"))                # legacy alias
LOOP_DT = 1.0 / max(0.1, LOOP_HZ)

CONTROL_HZ = float(os.getenv("CONTROL_HZ", "10.0"))
CONTROL_DT = 1.0 / max(0.1, CONTROL_HZ)

BRAIN_HZ = float(os.getenv("BRAIN_HZ", "2.0"))
BRAIN_DT = 1.0 / max(0.1, BRAIN_HZ)

PERCEPTION_HZ_CAP = float(os.getenv("PERCEPTION_HZ_CAP", "4.0"))
PERCEPTION_DT_MIN = 1.0 / max(0.1, PERCEPTION_HZ_CAP)

# Maximum age of a brain decision before control falls back to safe stop
DECISION_MAX_AGE_S = float(os.getenv("DECISION_MAX_AGE_S", "1.5"))

# Data staleness threshold - stop if data older than this (seconds)
STALE_S = float(os.getenv("STALE_S", "1.0"))


# ==============================================================================
# SAFETY LIMITS
# ==============================================================================
# Maximum velocity caps (range: 0.0 to 1.0)
MAX_LINEAR = float(os.getenv("MAX_LINEAR", "0.6"))
MAX_ANGULAR = float(os.getenv("MAX_ANGULAR", "0.8"))

# Battery thresholds
STOP_ON_LOW_BATT = float(os.getenv("STOP_ON_LOW_BATT", "5.0"))    # Hard stop below this %
SLOW_ON_LOW_BATT = float(os.getenv("SLOW_ON_LOW_BATT", "15.0"))   # Reduce speed below this %

# Danger detection keywords (triggers cautious behavior)
DANGER_KEYWORDS = {
    "wall", "obstacle", "blocked", "stairs", "stair", "edge", "cliff", "drop",
    "close", "near", "person", "pedestrian", "barrier",
    "traffic", "tight", "crowded", "fence", "pole",
    "van", "car", "truck", "luggage", "backpack", "suitcase"
}

# Immediate proximity keywords (triggers reflex backup)
PROXIMITY_KEYWORDS = {
    "close up", "too close", "blocked", "blocking", "jammed", "stuck", "crash",
    "covering", "dark", "blank", "blur"
}


# ==============================================================================
# VISION MODEL CONFIGURATION
# ==============================================================================
VLM_MODEL_NAME = os.getenv("VLM_MODEL_NAME", "HuggingFaceTB/SmolVLM-256M-Instruct")
VLM_MAX_NEW_TOKENS = int(os.getenv("VLM_MAX_NEW_TOKENS", "100"))

# Run captions less often than depth — captions are slow-changing.
# 1 = every perception tick, 4 = every 4th tick (default ~1Hz on 4Hz perception)
CAPTION_TICK_STRIDE = int(os.getenv("CAPTION_TICK_STRIDE", "4"))

# Depth-Anything-V2-Metric outputs depth in METERS directly. Indoor / Outdoor
# variants are trained on different distributions; "outdoor" (KITTI) is the
# right fit for sidewalk navigation. Set USE_METRIC_DEPTH=0 to fall back to
# the original relative-depth model and keep the legacy normalization.
USE_METRIC_DEPTH = os.getenv("USE_METRIC_DEPTH", "1") == "1"
DEPTH_MODEL_NAME = os.getenv(
    "DEPTH_MODEL_NAME",
    "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"
    if USE_METRIC_DEPTH
    else "depth-anything/Depth-Anything-V2-Small-hf",
)
# Fallback used if the metric checkpoint fails to load (e.g. older transformers)
DEPTH_MODEL_FALLBACK = os.getenv(
    "DEPTH_MODEL_FALLBACK",
    "depth-anything/Depth-Anything-V2-Small-hf",
)


# ==============================================================================
# CAMERA INTRINSICS / EXTRINSICS
# ==============================================================================
# Earth Rovers carry a wide-angle action camera. Override via env vars after
# you measure your specific bot — these are conservative defaults sized for
# typical 110-130° FOV mounted ~25cm above ground.
CAMERA_HFOV_DEG = float(os.getenv("CAMERA_HFOV_DEG", "120.0"))
CAMERA_VFOV_DEG = float(os.getenv("CAMERA_VFOV_DEG", "75.0"))
CAMERA_MOUNT_HEIGHT_M = float(os.getenv("CAMERA_MOUNT_HEIGHT_M", "0.25"))
# Pitch in degrees: positive = camera tilts down (sees more ground)
CAMERA_PITCH_DEG = float(os.getenv("CAMERA_PITCH_DEG", "0.0"))
# Image dimensions assumed by the depth/detection pipeline after preprocess.
# Region averaging uses ratios so this rarely needs tuning.
IMAGE_WIDTH = int(os.getenv("IMAGE_WIDTH", "256"))
IMAGE_HEIGHT = int(os.getenv("IMAGE_HEIGHT", "256"))

# Telemetry speed → m/s scaling. Earth Rovers' /data 'speed' units are not
# strongly documented; if your rover reports raw motor counts, adjust here.
TELEMETRY_SPEED_TO_MPS = float(os.getenv("TELEMETRY_SPEED_TO_MPS", "1.0"))


# ==============================================================================
# OBJECT DETECTION (LAYER 3)
# ==============================================================================
# Off by default because it adds another GPU model and ~150-300ms per inference.
# When enabled, the brain prompt switches to entity-level reasoning:
#   "person at +18°, 2.3m"  instead of  "sidewalk with trees".
USE_OBJECT_DETECTION = os.getenv("USE_OBJECT_DETECTION", "0") == "1"
OBJECT_DETECTOR_KIND = os.getenv("OBJECT_DETECTOR_KIND", "grounding-dino").lower()
# Model names for each kind
GROUNDING_DINO_MODEL = os.getenv("GROUNDING_DINO_MODEL", "IDEA-Research/grounding-dino-tiny")
YOLO_MODEL = os.getenv("YOLO_MODEL", "ultralytics/yolov8n")
# Open-vocabulary text prompt fed to grounding-dino each frame
DETECTION_TEXT_PROMPT = os.getenv(
    "DETECTION_TEXT_PROMPT",
    "person. car. bicycle. motorcycle. dog. pole. fence. trash can. door. stairs. curb. pothole.",
)
DETECTION_THRESHOLD = float(os.getenv("DETECTION_THRESHOLD", "0.30"))
# Run detection every N perception ticks (people don't move that fast)
DETECTION_TICK_STRIDE = int(os.getenv("DETECTION_TICK_STRIDE", "2"))
# Cap on objects passed to the brain to keep prompts compact
DETECTION_MAX_OBJECTS = int(os.getenv("DETECTION_MAX_OBJECTS", "6"))


# ==============================================================================
# SPATIAL OCCUPANCY GRID (LAYER 2)
# ==============================================================================
# Polar grid centered on the robot. Distance bins are in meters.
# Defaults: 8 angular bins (45° each) × 4 distance bins (0-1m, 1-2m, 2-4m, 4-8m).
GRID_ANGULAR_BINS = int(os.getenv("GRID_ANGULAR_BINS", "8"))
# Comma-separated distance bin EDGES in meters. N bins requires N+1 edges.
GRID_DISTANCE_EDGES = os.getenv("GRID_DISTANCE_EDGES", "0.0,1.0,2.0,4.0,8.0")
# Per-tick occupancy decay (0..1). 0.85 means an unrefreshed cell loses 15% of
# its occupancy every brain tick → forgotten in ~3-5s at BRAIN_HZ=2.
GRID_DECAY = float(os.getenv("GRID_DECAY", "0.85"))
# Threshold above which a cell is considered "blocked"
GRID_OCCUPIED_THRESHOLD = float(os.getenv("GRID_OCCUPIED_THRESHOLD", "0.5"))
# Distance below which an obstacle ahead is considered immediate (meters)
SAFE_FORWARD_CLEARANCE_M = float(os.getenv("SAFE_FORWARD_CLEARANCE_M", "1.2"))


# ==============================================================================
# CLIFF / EDGE DETECTION
# ==============================================================================
# Detect a sudden depth drop in the bottom strip of the frame: ground depth
# at the bottom should be largest (closest); a downstep makes it smaller.
CLIFF_DETECTION_ENABLED = os.getenv("CLIFF_DETECTION_ENABLED", "1") == "1"
# Required fractional drop in depth between adjacent rows of the bottom strip
CLIFF_GRADIENT_DROP = float(os.getenv("CLIFF_GRADIENT_DROP", "0.35"))


# ==============================================================================
# WATCHDOG
# ==============================================================================
# If any worker stops emitting heartbeats for this many seconds, the watchdog
# issues an emergency stop (and warns).
WATCHDOG_TIMEOUT_S = float(os.getenv("WATCHDOG_TIMEOUT_S", "3.0"))
WATCHDOG_CHECK_DT = float(os.getenv("WATCHDOG_CHECK_DT", "0.5"))


# ==============================================================================
# STRUCTURED JSON LOGGING (offline replay / tuning)
# ==============================================================================
# Empty string disables JSONL sink. Set a path like "/tmp/vla.jsonl" to enable.
JSONL_LOG_PATH = os.getenv("JSONL_LOG_PATH", "")

# ==============================================================================
# LLM (DECISION BRAIN) CONFIGURATION
# ==============================================================================
MODEL_PATH = os.getenv("MODEL_PATH", "./models/Phi-3.5-mini-instruct-Q4_K_M.gguf")

# Download URL if model not found locally
MODEL_DOWNLOAD_URL = os.getenv(
    "MODEL_DOWNLOAD_URL",
    "https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF/resolve/main/Phi-3.5-mini-instruct-Q4_K_M.gguf",
)

# LLM inference parameters - tuned for tight JSON output
LLM_CTX = int(os.getenv("LLM_CTX", "1024"))          # Context window size (smaller KV cache = faster)
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "48"))   # Max response tokens (JSON ~25 tokens)
LLM_TEMP = float(os.getenv("LLM_TEMP", "0.0"))       # Temperature (0.0 = fully deterministic)
LLM_GPU_LAYERS = int(os.getenv("LLM_GPU_LAYERS", "-1"))  # -1 = auto (use all GPU layers)
LLM_HISTORY_LEN = int(os.getenv("LLM_HISTORY_LEN", "5"))  # Action history for context

# JSON-schema-guided sampling enforces well-formed output without parse retries
LLM_USE_JSON_SCHEMA = os.getenv("LLM_USE_JSON_SCHEMA", "1") == "1"


# ==============================================================================
# ACTION SMOOTHING & CONTROL FLUIDITY
# ==============================================================================
# Exponential smoothing: smoothed = alpha * new + (1-alpha) * last
ACTION_SMOOTHING_ALPHA = float(os.getenv("ACTION_SMOOTHING_ALPHA", "0.7"))

# Rate limiting: maximum change per second (normal operation)
MAX_DELTA_LINEAR_PER_SEC = float(os.getenv("MAX_DELTA_LINEAR_PER_SEC", "1.0"))
MAX_DELTA_ANGULAR_PER_SEC = float(os.getenv("MAX_DELTA_ANGULAR_PER_SEC", "2.0"))

# Higher rate limits during recovery so backup/rotation can engage promptly
# instead of being throttled by the normal smoother (a 0.5/0.5 cap means
# reversal from +0.5 -> -0.3 takes ~2 ticks; recovery needs to act sooner).
MAX_DELTA_LINEAR_PER_SEC_RECOVERY = float(os.getenv("MAX_DELTA_LINEAR_PER_SEC_RECOVERY", "3.0"))
MAX_DELTA_ANGULAR_PER_SEC_RECOVERY = float(os.getenv("MAX_DELTA_ANGULAR_PER_SEC_RECOVERY", "5.0"))

# Deadband: ignore small changes below these thresholds
DEADBAND_LINEAR = float(os.getenv("DEADBAND_LINEAR", "0.02"))
DEADBAND_ANGULAR = float(os.getenv("DEADBAND_ANGULAR", "0.05"))


# ==============================================================================
# CAPTION STABILITY
# ==============================================================================
# Number of recent captions to keep for stability analysis
CAPTURE_HISTORY = int(os.getenv("CAPTURE_HISTORY", "3"))

# Fraction of captions that must agree for "stable" status
CAPTURE_AGREEMENT = float(os.getenv("CAPTURE_AGREEMENT", "0.66"))

# Similarity threshold for caption grouping (SequenceMatcher ratio)
CAPTURE_SIMILARITY_THRESHOLD = float(os.getenv("CAPTURE_SIMILARITY_THRESHOLD", "0.80"))


# ==============================================================================
# STALE DATA HANDLING
# ==============================================================================
# Number of quick retries before declaring data stale
STALE_RETRY = int(os.getenv("STALE_RETRY", "3"))
STALE_RETRY_DELAY = float(os.getenv("STALE_RETRY_DELAY", "0.25"))

# Maximum backoff sleep when stale data persists
STALE_BACKOFF_MAX_SLEEP = float(os.getenv("STALE_BACKOFF_MAX_SLEEP", "3.0"))


# ==============================================================================
# STUCK / ROTATION TUNING
# ======================================================================
# Target degrees for rotation completion checks (fallback to ticks if not reached)
ROTATE_RIGHT_DEG = float(os.getenv("ROTATE_RIGHT_DEG", "80.0"))
ROTATE_LEFT_DEG = float(os.getenv("ROTATE_LEFT_DEG", "170.0"))

# Angular movement detection thresholds
ANGULAR_STUCK_THRESHOLD = float(os.getenv("ANGULAR_STUCK_THRESHOLD", "0.15"))
ANGULAR_MIN_DEG_PER_TICK = float(os.getenv("ANGULAR_MIN_DEG_PER_TICK", "5.0"))

# Numeric depth threshold: Mid(ahead) value above which the path is blocked
OBSTACLE_AHEAD_THRESHOLD = float(os.getenv("OBSTACLE_AHEAD_THRESHOLD", "0.7"))

# Skip rear vision unless the rover is actually backing up (saves 50% of vision compute).
# When False, rear vision still runs every REAR_VISION_INTERVAL brain ticks.
SKIP_REAR_VISION_WHEN_FORWARD = os.getenv("SKIP_REAR_VISION_WHEN_FORWARD", "1") == "1"
REAR_VISION_INTERVAL = int(os.getenv("REAR_VISION_INTERVAL", "4"))  # tick stride for periodic rear refresh


# ==============================================================================
# INTERVENTION & MISSION SETTINGS
# ==============================================================================
# Enable mission mode (auto-starts mission on agent launch)
MISSION_MODE = os.getenv("MISSION_MODE", "0") == "1"

# Enable auto-intervention on repeated failures/danger
AUTO_INTERVENTION = os.getenv("AUTO_INTERVENTION", "0") == "1"

# Thresholds for triggering auto-intervention
DANGER_STREAK_TO_INTERVENE = int(os.getenv("DANGER_STREAK_TO_INTERVENE", "6"))
FAIL_STREAK_TO_INTERVENE = int(os.getenv("FAIL_STREAK_TO_INTERVENE", "4"))

# Optional mission target waypoint. When MISSION_MODE=1 the brain prompt is
# augmented with bearing_to_target / distance_to_target every tick.
MISSION_TARGET_LAT = os.getenv("MISSION_TARGET_LAT")
MISSION_TARGET_LON = os.getenv("MISSION_TARGET_LON")
MISSION_TARGET_LAT = float(MISSION_TARGET_LAT) if MISSION_TARGET_LAT else None
MISSION_TARGET_LON = float(MISSION_TARGET_LON) if MISSION_TARGET_LON else None
MISSION_REACHED_M = float(os.getenv("MISSION_REACHED_M", "5.0"))


# ==============================================================================
# RECOVERY DIVERSIFICATION
# ==============================================================================
# If we trigger recovery within RECOVERY_GPS_RADIUS_M of the previous trigger,
# escalate: longer backup, lamp on, faster intervention request.
RECOVERY_GPS_RADIUS_M = float(os.getenv("RECOVERY_GPS_RADIUS_M", "3.0"))
RECOVERY_REPEAT_WINDOW_S = float(os.getenv("RECOVERY_REPEAT_WINDOW_S", "30.0"))
# Number of *prior* near-GPS triggers before the next one escalates.
# 1 = second trigger at same spot escalates to level 1; third escalates to level 2.
RECOVERY_ESCALATE_AFTER = int(os.getenv("RECOVERY_ESCALATE_AFTER", "1"))


# ==============================================================================
# LLM SYSTEM PROMPT
# ==============================================================================
# Distances in the prompt are METERS when USE_METRIC_DEPTH=1; otherwise they
# are normalized 0..1 (1.0 = nearest object in frame). Both forms work, but
# metric is much easier for the model to reason about.
SYSTEM_PROMPT = """You are the brain of a small rover robot navigating outdoors.
Your goal: reach the mission target when given, otherwise explore safely.

Output a JSON object only:
- "linear":  float in [-1, 1]   (positive=forward, negative=backward)
- "angular": float in [-1, 1]   (positive=left, negative=right)
- "lamp":    0 or 1

How to read the inputs:
- SPATIAL MAP shows nearest obstacle distance per direction (45° bins around the rover).
  Free directions have distance > 4m or "free". Use these to plan turns.
- DETECTED OBJECTS (when present) lists [class, bearing°, distance_m].
  bearing 0° = directly ahead, +90° = left, -90° = right.
- FRONT/REAR DEPTH gives Mid(ahead) — the closest blob in the camera view.
- CLIFF=true means the bottom of the frame shows a sudden depth drop — DO NOT go forward.
- MISSION (when present) gives bearing & distance to the target. Drive toward bearing 0.

Decision rules:
1. If CLIFF=true OR front clearance < 0.8m -> stop forward and turn.
2. Prefer the freest direction shown in SPATIAL MAP within ±90° of forward.
3. If MISSION target exists, choose the freer of the two directions closest to bearing.
4. Use linear in [0.3, 0.5] when clear, near-zero linear with angular ±0.5 when turning.
5. Avoid pure backup unless front is blocked AND rear is clearly free (>2m).
6. Lamp on (1) only in low-visibility / dark scenes.
7. If history shows you bumped the same obstacle twice, change direction.

Return ONLY the JSON object, no prose.
Example: {"linear": 0.4, "angular": 0.0, "lamp": 0}
"""
