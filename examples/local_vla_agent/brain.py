"""LLM decision brain.

Inputs (passed to decide()):
- telemetry: raw sensor dict
- caption_front/rear: stable VLM caption (strings)
- depth_summary_front/rear: human-readable depth strings
- spatial_map_summary: forward-cone polar grid summary (string)
- detected_objects: list of (label, bearing, distance) lines
- history: action-outcome tuples
- mission_target: optional dict with bearing_deg, distance_m, target_lat, target_lon
"""

import os
from typing import Dict, Any, List, Optional

from llama_cpp import Llama

try:
    from config import (
        logger,
        MODEL_PATH,
        MODEL_DOWNLOAD_URL,
        LLM_CTX,
        LLM_GPU_LAYERS,
        LLM_MAX_TOKENS,
        LLM_TEMP,
        LLM_USE_JSON_SCHEMA,
        SYSTEM_PROMPT,
        MAX_LINEAR,
        MAX_ANGULAR,
    )
    from utils import download_file
except ImportError:
    from .config import (
        logger,
        MODEL_PATH,
        MODEL_DOWNLOAD_URL,
        LLM_CTX,
        LLM_GPU_LAYERS,
        LLM_MAX_TOKENS,
        LLM_TEMP,
        LLM_USE_JSON_SCHEMA,
        SYSTEM_PROMPT,
        MAX_LINEAR,
        MAX_ANGULAR,
    )
    from .utils import download_file


_ACTION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "linear":  {"type": "number", "minimum": -MAX_LINEAR, "maximum": MAX_LINEAR},
        "angular": {"type": "number", "minimum": -MAX_ANGULAR, "maximum": MAX_ANGULAR},
        "lamp":    {"type": "integer", "enum": [0, 1]},
    },
    "required": ["linear", "angular", "lamp"],
    "additionalProperties": False,
}


class BrainSystem:
    """LLM-based decision system for rover navigation."""

    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            logger.info(
                f"GGUF model not found at {MODEL_PATH}. "
                f"Attempting download from {MODEL_DOWNLOAD_URL}..."
            )
            try:
                download_file(MODEL_DOWNLOAD_URL, MODEL_PATH)
            except Exception as e:
                raise FileNotFoundError(
                    f"Failed to download GGUF model from {MODEL_DOWNLOAD_URL} -> {e}"
                ) from e

        logger.info(f"Loading LLM from {MODEL_PATH} (ctx={LLM_CTX}, gpu_layers={LLM_GPU_LAYERS})...")
        try:
            self.llm = Llama(
                model_path=MODEL_PATH,
                n_ctx=LLM_CTX,
                n_gpu_layers=LLM_GPU_LAYERS,
                verbose=False,
            )
        except Exception as e:
            logger.warning(f"LLM GPU load failed ({e}). Falling back to CPU.")
            self.llm = Llama(
                model_path=MODEL_PATH,
                n_ctx=LLM_CTX,
                n_gpu_layers=0,
                verbose=False,
            )
        logger.info("LLM loaded.")

    def decide(
        self,
        telemetry: Dict[str, Any],
        caption_front: str,
        caption_rear: str,
        history: Optional[List[dict]] = None,
        depth_summary_front: str = "unavailable",
        depth_summary_rear: str = "unavailable",
        mission_target: Optional[Dict[str, Any]] = None,
        spatial_map_summary: Optional[str] = None,
        detected_objects: Optional[List[str]] = None,
        cliff_front: bool = False,
        clearance_front_m: Optional[float] = None,
    ) -> str:
        # ---- History (action -> outcome) ----
        history_str = "None"
        if history:
            lines = []
            for i, h in enumerate(reversed(history), 1):
                act = h.get("action", {})
                outcome = h.get("outcome", "")
                lines.append(
                    f"- T-{i}: lin={act.get('linear', 0):.1f}, ang={act.get('angular', 0):.1f}"
                    f"{(' -> ' + outcome) if outcome else ''}"
                )
            history_str = "\n".join(lines)

        # ---- Mission target ----
        mission_block = ""
        if mission_target:
            bearing = mission_target.get("bearing_deg")
            dist = mission_target.get("distance_m")
            mission_block = (
                f"\nMISSION TARGET:\n"
                f"- bearing: {bearing:+.0f}° (positive=left, negative=right)\n"
                f"- distance: {dist:.1f}m\n"
                f"Steer toward bearing 0 when the path that direction is clear.\n"
            )

        # ---- GPS line ----
        lat = telemetry.get("latitude")
        lon = telemetry.get("longitude")
        heading = telemetry.get("orientation") or telemetry.get("heading")
        gps_line = ""
        if lat is not None and lon is not None:
            gps_line = f"- GPS: ({lat}, {lon}) heading={heading}\n"

        # ---- Spatial map ----
        spatial_block = ""
        if spatial_map_summary:
            spatial_block = f"\nSPATIAL MAP (within ±90° of forward):\n  {spatial_map_summary}\n"

        # ---- Detected objects ----
        objects_block = ""
        if detected_objects:
            obj_lines = "\n".join(f"  - {o}" for o in detected_objects)
            objects_block = f"\nDETECTED OBJECTS:\n{obj_lines}\n"

        # ---- Front clearance / cliff ----
        cliff_line = ""
        if cliff_front:
            cliff_line = "- CLIFF=true (do NOT go forward)\n"
        clearance_line = ""
        if clearance_front_m is not None:
            clearance_line = f"- Front clearance: {clearance_front_m:.1f}m\n"

        user_message = f"""Current Status:
- Battery: {telemetry.get('battery')}%
- Speed: {telemetry.get('speed')}
{gps_line}{cliff_line}{clearance_line}
Current Vision:
- FRONT: {caption_front}
- REAR: {caption_rear}
- FRONT DEPTH: {depth_summary_front}
- REAR DEPTH: {depth_summary_rear}
{spatial_block}{objects_block}{mission_block}
Recent History (most recent first):
{history_str}

Return ONLY a JSON object: {{"linear": <float>, "angular": <float>, "lamp": <0|1>}}.
"""

        kwargs = dict(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            max_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMP,
        )
        if LLM_USE_JSON_SCHEMA:
            kwargs["response_format"] = {
                "type": "json_object",
                "schema": _ACTION_JSON_SCHEMA,
            }

        resp = self.llm.create_chat_completion(**kwargs)
        return (resp["choices"][0]["message"]["content"] or "").strip()


def extract_json_from_text(text: str) -> str:
    """Defensive JSON extractor. Used when LLM_USE_JSON_SCHEMA is disabled."""
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]
    return text
