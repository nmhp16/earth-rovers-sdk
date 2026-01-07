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


def download_file(url: str, dest: str, chunk_size: int = 8 * 1024 * 1024) -> None:
    import os
    import logging
    import requests

    logger = logging.getLogger("vla")

    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    tmp = dest + ".partial"

    logger.info(f"Downloading model from {url} -> {dest}")
    try:
        token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
        headers = {"Authorization": f"Bearer {token}"} if token else None
        with requests.get(url, stream=True, timeout=30, headers=headers) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            downloaded = 0
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if not chunk:
                        continue
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded * 100 / total
                        logger.info(f"Downloading model: {downloaded}/{total} bytes ({pct:.1f}%)")
        # Move into place
        os.replace(tmp, dest)
        logger.info("Model download complete")
    except Exception:
        # Clean up partial file on failure
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass
        raise
