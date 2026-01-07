import os
import logging

# -------------------- Logging --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [VLA] - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("vla")


# -------------------- Config --------------------
SDK_URL = os.getenv("SDK_URL", "http://localhost:8000").rstrip("/")

CONTROL_ENDPOINT = f"{SDK_URL}/control"
DATA_ENDPOINT = f"{SDK_URL}/data"
V2_SCREENSHOT_ENDPOINT = f"{SDK_URL}/v2/screenshot"

START_MISSION_ENDPOINT = f"{SDK_URL}/start-mission"
INTERVENTION_START_ENDPOINT = f"{SDK_URL}/interventions/start"
INTERVENTION_END_ENDPOINT = f"{SDK_URL}/interventions/end"

# Loop timing
LOOP_HZ = float(os.getenv("LOOP_HZ", "2.0"))
LOOP_DT = 1.0 / max(0.1, LOOP_HZ)

# Networking
HTTP_TIMEOUT_S = float(os.getenv("HTTP_TIMEOUT_S", "2.0"))

# Safety caps
MAX_LINEAR = float(os.getenv("MAX_LINEAR", "0.6"))
MAX_ANGULAR = float(os.getenv("MAX_ANGULAR", "0.8"))

# Stale detection: stop if data or vision timestamps older than this
STALE_S = float(os.getenv("STALE_S", "2.0"))

# Battery behavior
STOP_ON_LOW_BATT = float(os.getenv("STOP_ON_LOW_BATT", "5.0"))
SLOW_ON_LOW_BATT = float(os.getenv("SLOW_ON_LOW_BATT", "15.0"))

# Vision model
BLIP_MODEL_NAME = os.getenv("BLIP_MODEL_NAME", "Salesforce/blip-image-captioning-base")
BLIP_MAX_NEW_TOKENS = int(os.getenv("BLIP_MAX_NEW_TOKENS", "24"))

# LLM model
MODEL_PATH = os.getenv("MODEL_PATH", "./models/stablelm-zephyr-3b.Q4_K_M.gguf")
MODEL_DOWNLOAD_URL = os.getenv(
    "MODEL_DOWNLOAD_URL",
    "https://huggingface.co/TheBloke/stablelm-zephyr-3b-GGUF/resolve/main/stablelm-zephyr-3b.Q4_K_M.gguf",
)

LLM_CTX = int(os.getenv("LLM_CTX", "2048"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "140"))
LLM_TEMP = float(os.getenv("LLM_TEMP", "0.2"))
LLM_GPU_LAYERS = int(os.getenv("LLM_GPU_LAYERS", "-1"))

# Optional features
MISSION_MODE = os.getenv("MISSION_MODE", "0") == "1"
AUTO_INTERVENTION = os.getenv("AUTO_INTERVENTION", "0") == "1"

# Auto intervention thresholds
DANGER_STREAK_TO_INTERVENE = int(os.getenv("DANGER_STREAK_TO_INTERVENE", "6"))
FAIL_STREAK_TO_INTERVENE = int(os.getenv("FAIL_STREAK_TO_INTERVENE", "4"))

# Danger heuristic keywords (cheap safety layer)
DANGER_KEYWORDS = {
    "wall", "obstacle", "blocked", "stairs", "stair", "edge", "cliff", "drop",
    "close", "near", "chair", "table", "person", "pedestrian", "barrier",
    "traffic", "tight", "crowded", "fence", "pole"
}

SYSTEM_PROMPT = """You are the brain of a small rover robot.
Your goal is to explore the environment safely.

You will receive:
1) Visual descriptions (front + rear).
2) Telemetry (battery, speed, gps, vibration, signal, etc.).

You must output a JSON object with keys:
- "linear": float in [-1, 1]   (forward/backward)
- "angular": float in [-1, 1]  (left/right)
- "lamp": 0 or 1

Rules:
- Safety first. If obstacle is near, slow down, turn, or back up.
- Avoid fast forward motion unless the path is clearly open.
- Output STRICTLY valid JSON only. No extra text.
"""
