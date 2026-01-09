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
        SYSTEM_PROMPT,
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
        SYSTEM_PROMPT,
    )
    from .utils import download_file


class BrainSystem:
    """
    LLM-based decision system for rover navigation.
    
    Takes sensor inputs and produces JSON motor commands using a local
    quantized language model. Designed for edge deployment without
    cloud dependencies.
    """
    
    def __init__(self):
        """
        Initialize the brain system by loading the LLM.
        
        Downloads the model from HuggingFace if not found locally.
        Falls back to CPU inference if GPU loading fails.
        """
        # Download model if not present
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

        # Load the LLM
        logger.info(f"Loading LLM from {MODEL_PATH}...")
        try:
            self.llm = Llama(
                model_path=MODEL_PATH,
                n_ctx=LLM_CTX,
                n_gpu_layers=LLM_GPU_LAYERS,
                verbose=False,
            )
        except Exception as e:
            logger.warning(
                f"LLM GPU load failed ({e}). Falling back to CPU (n_gpu_layers=0)."
            )
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
        stuck_status: int = 0,
        depth_analysis_front: str = "unavailable",
        depth_analysis_rear: str = "unavailable"
    ) -> str:
        """
        Generate a navigation decision based on current sensor state.
        
        Args:
            telemetry: Dict with 'battery', 'speed', and other rover stats
            caption_front: Text description of front camera view
            caption_rear: Text description of rear camera view
            history: List of recent actions and their outcomes
            stuck_status: Counter indicating how long rover has been stuck (0=moving)
            depth_analysis_front: Obstacle detection results from front camera
            depth_analysis_rear: Obstacle detection results from rear camera
            
        Returns:
            JSON string with keys: linear, angular, lamp
            Example: '{"linear": 0.2, "angular": 0.0, "lamp": 0}'
        """
        # Format action history for context
        history_str = "None"
        if history:
            history_str = "\n".join([
                f"- T-{i}s: Action(lin={h['action']['linear']:.1f}, "
                f"ang={h['action']['angular']:.1f}) -> Front: '{h['front']}'"
                for i, h in enumerate(reversed(history), 1)
            ])

        # Build the user prompt with all sensor data
        user_message = f"""
Current Status:
- STUCK_LEVEL: {stuck_status} (0=moving, >0=stuck/blocked)
- Battery: {telemetry.get('battery')}%
- Speed: {telemetry.get('speed')}

Current Vision:
- FRONT: {caption_front}
- REAR: {caption_rear}
- FRONT DEPTH: {depth_analysis_front}
- REAR DEPTH: {depth_analysis_rear}

Depth Interpretation (0.0=far, 1.0=very close):
- Close(ground): Floor distance immediately below camera
- Mid(ahead): Object distance in travel path (CRITICAL for collision)
- Far(sky): Background/ceiling distance

OBSTACLE RULES:
1. If FRONT Mid(ahead) > 0.70 AND you want to go FORWARD -> TURN instead, obstacle too close
2. If REAR Mid(ahead) > 0.70 AND you want to BACKUP -> TURN instead, obstacle behind
3. If Mid values are LOW (< 0.40) in travel direction -> Path is CLEAR
4. If STUCK_LEVEL > 0 -> Turn in place (linear near 0, high angular) to find clear path

Navigation Strategy:
- Prefer FORWARD motion when front is clear (Front Mid < 0.50)
- Use BACKUP only when front blocked AND rear is clear (Rear Mid < 0.50)
- Use TURNING (high angular, low linear) to explore and avoid obstacles
- NEVER back up repeatedly - if backing up didn't help in history, TURN instead

Recent History (Actions -> Outcomes):
{history_str}

Return ONLY a JSON object. Do NOT add explanation text before or after because it breaks my parser.
"""

        # Get LLM response
        resp = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            max_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMP,
        )
        
        return (resp["choices"][0]["message"]["content"] or "").strip()


def extract_json_from_text(text: str) -> str:
    """
    Extract JSON object from LLM output that may contain extra text.
    
    Args:
        text: Raw LLM output that may have JSON embedded in prose
        
    Returns:
        Extracted JSON substring, or original text if no braces found
    """
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]
    return text

