import json
from typing import Any, Dict, Optional

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

def safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default

def extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Extract first {...} JSON object from a text blob (fallback)."""
    if not text:
        return None
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                chunk = text[start : i + 1]
                try:
                    return json.loads(chunk)
                except Exception:
                    return None
    return None

def is_timestamp_stale(ts: Any, now: float, stale_s: float) -> bool:
    tsf = safe_float(ts, 0.0)
    if tsf <= 0.0:
        # If missing timestamp, treat as stale (safer)
        return True
    # If looks like ms epoch, convert
    if tsf > 1e12:
        tsf /= 1000.0
    return (now - tsf) > stale_s
