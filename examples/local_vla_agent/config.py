import os
import logging

# ==============================================================================
# LOGGING SETUP
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [VLA] - %(message)s",
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
# Main loop frequency (default 2 Hz = 500ms per cycle)
LOOP_HZ = float(os.getenv("LOOP_HZ", "2.0"))
LOOP_DT = 1.0 / max(0.1, LOOP_HZ)

# Data staleness threshold - stop if data older than this (seconds)
STALE_S = float(os.getenv("STALE_S", "2.0"))


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
# Vision-Language Model for image captioning
# Recommended: "microsoft/git-large-coco" (works reliably with transformers 5.x)
# Note: Florence-2 and Moondream2 have compatibility issues with transformers 5.x
VLM_MODEL_NAME = os.getenv("VLM_MODEL_NAME", "microsoft/git-large-coco")
VLM_MAX_NEW_TOKENS = int(os.getenv("VLM_MAX_NEW_TOKENS", "100"))

# Depth estimation model for obstacle detection
DEPTH_MODEL_NAME = os.getenv("DEPTH_MODEL_NAME", "LiheYoung/depth-anything-small-hf")


# ==============================================================================
# LLM (DECISION BRAIN) CONFIGURATION
# ==============================================================================
# Path to the GGUF quantized LLM model
MODEL_PATH = os.getenv("MODEL_PATH", "./models/stablelm-zephyr-3b.Q4_K_M.gguf")

# Download URL if model not found locally
MODEL_DOWNLOAD_URL = os.getenv(
    "MODEL_DOWNLOAD_URL",
    "https://huggingface.co/TheBloke/stablelm-zephyr-3b-GGUF/resolve/main/stablelm-zephyr-3b.Q4_K_M.gguf",
)

# LLM inference parameters
LLM_CTX = int(os.getenv("LLM_CTX", "2048"))          # Context window size
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "140"))  # Max response tokens
LLM_TEMP = float(os.getenv("LLM_TEMP", "0.2"))       # Temperature (lower = more deterministic)
LLM_GPU_LAYERS = int(os.getenv("LLM_GPU_LAYERS", "-1"))  # -1 = auto (use all GPU layers)
LLM_HISTORY_LEN = int(os.getenv("LLM_HISTORY_LEN", "5"))  # Action history for context


# ==============================================================================
# ACTION SMOOTHING & CONTROL FLUIDITY
# ==============================================================================
# Exponential smoothing: smoothed = alpha * new + (1-alpha) * last
ACTION_SMOOTHING_ALPHA = float(os.getenv("ACTION_SMOOTHING_ALPHA", "0.7"))

# Rate limiting: maximum change per second
MAX_DELTA_LINEAR_PER_SEC = float(os.getenv("MAX_DELTA_LINEAR_PER_SEC", "1.0"))
MAX_DELTA_ANGULAR_PER_SEC = float(os.getenv("MAX_DELTA_ANGULAR_PER_SEC", "2.0"))

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


# ==============================================================================
# INTERVENTION SETTINGS
# ==============================================================================
# Enable mission mode (auto-starts mission on agent launch)
MISSION_MODE = os.getenv("MISSION_MODE", "0") == "1"

# Enable auto-intervention on repeated failures/danger
AUTO_INTERVENTION = os.getenv("AUTO_INTERVENTION", "0") == "1"

# Thresholds for triggering auto-intervention
DANGER_STREAK_TO_INTERVENE = int(os.getenv("DANGER_STREAK_TO_INTERVENE", "6"))
FAIL_STREAK_TO_INTERVENE = int(os.getenv("FAIL_STREAK_TO_INTERVENE", "4"))


# ==============================================================================
# LLM SYSTEM PROMPT
# ==============================================================================
SYSTEM_PROMPT = """You are the brain of a small rover robot.
Your goal is to explore the environment safely.

You must output a JSON object with keys:
- "linear": float in [-1, 1]   (positive=forward, negative=backward)
- "angular": float in [-1, 1]  (positive=left, negative=right)
- "lamp": 0 or 1

Review the 'current status' and 'recent history' carefully.
- If you are STUCK (stuck_level > 0), you MUST backward (-0.2) and turn.
- If you see the same obstacle as before in history, do NOT go forward. Turn away.
- Otherwise, explore cautiously.

Return ONLY the JSON object. Do not explain your reasoning.
Example: {"linear": 0.2, "angular": 0.0, "lamp": 0}
"""
