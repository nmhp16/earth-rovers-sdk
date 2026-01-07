import os
from typing import Dict, Any
from llama_cpp import Llama
from config import logger, MODEL_PATH, LLM_CTX, LLM_GPU_LAYERS, LLM_MAX_TOKENS, LLM_TEMP, SYSTEM_PROMPT

class BrainSystem:
    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"GGUF model not found at {MODEL_PATH}\n"
                f"Set MODEL_PATH or download the file."
            )
        logger.info(f"Loading LLM from {MODEL_PATH} ...")
        try:
            self.llm = Llama(
                model_path=MODEL_PATH,
                n_ctx=LLM_CTX,
                n_gpu_layers=LLM_GPU_LAYERS,
                verbose=False,
            )
        except Exception as e:
            logger.warning(f"LLM GPU load failed ({e}). Falling back to CPU (n_gpu_layers=0).")
            self.llm = Llama(
                model_path=MODEL_PATH,
                n_ctx=LLM_CTX,
                n_gpu_layers=0,
                verbose=False,
            )
        logger.info("LLM loaded.")

    def decide(self, telemetry: Dict[str, Any], caption_front: str, caption_rear: str) -> str:
        user_message = f"""
Telemetry:
- battery: {telemetry.get('battery')}
- signal_level: {telemetry.get('signal_level')}
- speed: {telemetry.get('speed')}
- gps_signal: {telemetry.get('gps_signal')}
- vibration: {telemetry.get('vibration')}
- latitude: {telemetry.get('latitude')}
- longitude: {telemetry.get('longitude')}
- orientation: {telemetry.get('orientation')}
- lamp: {telemetry.get('lamp')}

Vision:
- front: {caption_front}
- rear: {caption_rear}

Return ONLY a JSON object like:
{{"linear": 0.2, "angular": 0.0, "lamp": 0}}
"""

        resp = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            max_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMP,
        )
        return (resp["choices"][0]["message"]["content"] or "").strip()
