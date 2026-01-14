import json
import os
import logging
from collections import Counter
from difflib import SequenceMatcher
from typing import Any, Dict, Optional, List, Tuple


# ==============================================================================
# VALUE HELPERS
# ==============================================================================

def clamp(x: float, lo: float, hi: float) -> float:
    """Clamp a value to the range [lo, hi]."""
    return max(lo, min(hi, x))


def safe_float(x: Any, default: float = 0.0) -> float:
    """Safely convert a value to float, returning default on failure."""
    try:
        return float(x)
    except Exception:
        return default


def safe_int(x: Any, default: int = 0) -> int:
    """Safely convert a value to int, returning default on failure."""
    try:
        return int(x)
    except Exception:
        return default


# ==============================================================================
# JSON HELPERS
# ==============================================================================

def extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract the first {...} JSON object from a text blob.
    
    Useful for parsing LLM outputs that may contain extra text around JSON.
    
    Args:
        text: Text that may contain a JSON object
        
    Returns:
        Parsed dict, or None if no valid JSON found
    """
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
                chunk = text[start:i + 1]
                try:
                    return json.loads(chunk)
                except Exception:
                    return None
    return None


# ==============================================================================
# TIMESTAMP HELPERS
# ==============================================================================

def is_timestamp_stale(ts: Any, now: float, stale_s: float) -> bool:
    """
    Check if a timestamp is stale (too old).
    
    Handles both seconds and milliseconds epoch formats.
    
    Args:
        ts: Timestamp value (seconds or milliseconds since epoch)
        now: Current time in seconds since epoch
        stale_s: Maximum age in seconds before considered stale
        
    Returns:
        True if timestamp is stale or invalid
    """
    tsf = safe_float(ts, 0.0)
    if tsf <= 0.0:
        # Missing timestamp = treat as stale (safer)
        return True
    
    # If looks like ms epoch, convert to seconds
    if tsf > 1e12:
        tsf /= 1000.0
    
    return (now - tsf) > stale_s


# ==============================================================================
# ORIENTATION HELPERS
# ==============================================================================

def _normalize_angle_deg(angle: float) -> float:
    """Normalize angle to [-180, 180] degrees."""
    a = (angle + 180.0) % 360.0 - 180.0
    return a


def _raw_to_degrees(raw: float, scale: int) -> float:
    """Map a raw orientation value to degrees assuming given max scale.

    scale: e.g., 360 (already degrees) or 256 (raw 0-255 mapping)
    """
    if scale == 360:
        return float(raw)
    # map from 0..(scale-1) to 0..360 deg
    return float(raw) / float(scale - 1) * 360.0


def orientation_delta(a: Any, b: Any) -> float:
    """
    Compute the signed minimal angular delta from a -> b in degrees in range [-180, 180].

    The telemetry 'orientation' may be reported in different scales (e.g. 0..255 or 0..360).
    This helper tries both common interpretations and returns the smallest-magnitude delta in degrees.
    Returns 0.0 if either value is missing or invalid.
    """
    try:
        a_raw = float(a)
        b_raw = float(b)
    except Exception:
        return 0.0

    # Try interpreting as degrees directly (0..360)
    a_deg_360 = _raw_to_degrees(a_raw, 360)
    b_deg_360 = _raw_to_degrees(b_raw, 360)
    delta_360 = _normalize_angle_deg(b_deg_360 - a_deg_360)

    # Try interpreting as 0..255 raw mapping
    a_deg_256 = _raw_to_degrees(a_raw, 256)
    b_deg_256 = _raw_to_degrees(b_raw, 256)
    delta_256 = _normalize_angle_deg(b_deg_256 - a_deg_256)

    # Choose the interpretation with smaller absolute delta
    if abs(delta_360) <= abs(delta_256):
        return delta_360
    return delta_256


# ==============================================================================
# FILE DOWNLOAD
# ==============================================================================

def download_file(url: str, dest: str, chunk_size: int = 8 * 1024 * 1024) -> None:
    """
    Download a file from a URL with progress logging.
    
    Supports HuggingFace authentication via HUGGINGFACE_TOKEN or HF_TOKEN env vars.
    
    Args:
        url: Source URL to download from
        dest: Destination file path
        chunk_size: Download chunk size in bytes (default 8MB)
        
    Raises:
        Exception: If download fails
    """
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
        
        # Move into place atomically
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


# ==============================================================================
# CAPTION STABILITY
# ==============================================================================

def stable_caption_from_history(
    history: List[str],
    agreement_frac: float = 0.66,
    similarity_threshold: float = 0.8
) -> Tuple[str, bool]:
    """
    Determine a stable caption from recent history.
    
    Stability detection:
    1. If a caption appears in >= agreement_frac of history, it's stable
    2. Otherwise, try to find a caption similar to most others
    3. If nothing stable, return most recent caption
    
    Args:
        history: List of recent captions
        agreement_frac: Fraction of history that must agree for exact stability
        similarity_threshold: SequenceMatcher ratio threshold for similarity grouping
        
    Returns:
        Tuple of (caption, is_stable)
    """
    if not history:
        return "", False
    
    n = len(history)
    counts = Counter(history)
    most_common, cnt = counts.most_common(1)[0]
    
    # Check for exact agreement
    if cnt / n >= agreement_frac:
        return most_common, True
    
    # Try similarity grouping
    for candidate in counts.keys():
        sims = [SequenceMatcher(None, candidate, other).ratio() for other in history]
        avg_sim = sum(sims) / n
        if avg_sim >= similarity_threshold:
            return candidate, True
    
    # Not stable - return most recent
    return history[-1], False


# ==============================================================================
# ACTION SMOOTHING
# ==============================================================================

def smooth_action(
    new_linear: float,
    new_angular: float,
    last_linear: float,
    last_angular: float,
    alpha: float,
    max_delta_linear_per_sec: float,
    max_delta_angular_per_sec: float,
    dt: float,
    deadband_linear: float,
    deadband_angular: float,
    max_linear: float,
    max_angular: float
) -> Tuple[float, float]:
    """
    Apply smoothing and rate-limiting to motor commands.
    
    Processing steps:
    1. Exponential smoothing: s = alpha * new + (1-alpha) * last
    2. Deadband: ignore changes smaller than threshold
    3. Rate limiting: clamp change per tick based on max delta/sec
    4. Final clamp to absolute limits
    
    Args:
        new_linear: Target linear velocity
        new_angular: Target angular velocity
        last_linear: Previous linear velocity
        last_angular: Previous angular velocity
        alpha: Smoothing factor (higher = more responsive)
        max_delta_linear_per_sec: Maximum linear change rate
        max_delta_angular_per_sec: Maximum angular change rate
        dt: Time step in seconds
        deadband_linear: Minimum linear change to apply
        deadband_angular: Minimum angular change to apply
        max_linear: Maximum absolute linear velocity
        max_angular: Maximum absolute angular velocity
        
    Returns:
        Tuple of (smoothed_linear, smoothed_angular)
    """
    # 1. Exponential smoothing
    s_lin = alpha * float(new_linear) + (1.0 - alpha) * float(last_linear)
    s_ang = alpha * float(new_angular) + (1.0 - alpha) * float(last_angular)

    # 2. Deadband - ignore tiny changes
    if abs(s_lin - last_linear) < deadband_linear:
        s_lin = last_linear
    if abs(s_ang - last_angular) < deadband_angular:
        s_ang = last_angular

    # 3. Rate limiting - clamp per-tick delta
    max_d_lin = max_delta_linear_per_sec * dt
    max_d_ang = max_delta_angular_per_sec * dt

    d_lin = s_lin - last_linear
    d_ang = s_ang - last_angular

    if d_lin > max_d_lin:
        s_lin = last_linear + max_d_lin
    elif d_lin < -max_d_lin:
        s_lin = last_linear - max_d_lin

    if d_ang > max_d_ang:
        s_ang = last_angular + max_d_ang
    elif d_ang < -max_d_ang:
        s_ang = last_angular - max_d_ang

    # 4. Final clamp to absolute limits
    s_lin = max(-max_linear, min(max_linear, s_lin))
    s_ang = max(-max_angular, min(max_angular, s_ang))

    return s_lin, s_ang

