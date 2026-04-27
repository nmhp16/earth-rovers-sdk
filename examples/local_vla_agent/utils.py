import json
import math
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


def normalize_orientation_to_deg(raw: Any) -> Optional[float]:
    """Normalize a raw orientation telemetry value to degrees in [0, 360).

    Uses the same dual-interpretation logic as orientation_delta — picks
    whichever scale (0..360 or 0..255) yields a value in range. Returns
    None if the value is missing/invalid.
    """
    if raw is None:
        return None
    try:
        v = float(raw)
    except Exception:
        return None
    if 0 <= v <= 360:
        return v % 360.0
    if 0 <= v <= 255:
        return (v / 255.0 * 360.0) % 360.0
    # Unknown scale — best-effort modulo
    return v % 360.0


# ==============================================================================
# GEOMETRY: pixel coords -> bearing / distance, GPS math
# ==============================================================================

def pixel_to_bearing_deg(pixel_x: float, image_width: int, hfov_deg: float) -> float:
    """Map a pixel x-coordinate to a bearing in degrees relative to camera center.

    Returns positive for LEFT of center (matches the rover's angular convention
    where positive angular = turn left). bearing = 0 means dead ahead.
    """
    if image_width <= 0:
        return 0.0
    # Center the pixel: -1 (right edge) .. +1 (left edge)  — note: image x grows
    # rightward, but we want positive=left, so we flip sign.
    centered = (image_width / 2.0 - pixel_x) / (image_width / 2.0)
    return centered * (hfov_deg / 2.0)


def pixel_to_pitch_deg(pixel_y: float, image_height: int, vfov_deg: float,
                       camera_pitch_deg: float = 0.0) -> float:
    """Map a pixel y-coordinate to a pitch in degrees relative to horizon.

    Positive pitch = below horizon (toward the ground); negative = above.
    Includes the camera's static down-tilt (camera_pitch_deg).
    """
    if image_height <= 0:
        return camera_pitch_deg
    # 0 (top) -> -vfov/2, height (bottom) -> +vfov/2
    centered = (pixel_y - image_height / 2.0) / (image_height / 2.0)
    return centered * (vfov_deg / 2.0) + camera_pitch_deg


def ground_distance_from_pixel_y(
    pixel_y: float,
    image_height: int,
    vfov_deg: float,
    camera_height_m: float,
    camera_pitch_deg: float = 0.0,
    max_distance_m: float = 50.0,
) -> Optional[float]:
    """Compute horizontal distance to the ground intersection of a ray
    cast through pixel (_, pixel_y), assuming a flat ground plane.

    Useful as a fallback distance estimator when no metric depth is available.
    Returns None for rays that point at or above the horizon.
    """
    pitch = pixel_to_pitch_deg(pixel_y, image_height, vfov_deg, camera_pitch_deg)
    # Need positive (downward) pitch for a ground intersection
    if pitch <= 0.5:
        return None
    distance = camera_height_m / math.tan(math.radians(pitch))
    if distance <= 0 or distance > max_distance_m:
        return None
    return distance


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in meters between two GPS points."""
    R = 6371000.0  # earth radius in meters
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial bearing from point 1 to point 2 in degrees [0, 360),
    where 0 = North, 90 = East."""
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def relative_bearing_deg(target_bearing_deg: float, heading_deg: float) -> float:
    """Bearing relative to current heading, normalized to (-180, 180].

    Sign convention: positive = need to turn LEFT to face target (matches the
    rover's angular convention).
    """
    rel = (target_bearing_deg - heading_deg + 540.0) % 360.0 - 180.0
    # World bearing is clockwise (0=N, 90=E). Rover convention: +angular = left
    # = counter-clockwise. Flip the sign so the brain prompt matches.
    return -rel


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
    max_angular: float,
    recovery_alpha: Optional[float] = None,
    recovery_max_delta_linear_per_sec: Optional[float] = None,
    recovery_max_delta_angular_per_sec: Optional[float] = None,
    is_recovery: bool = False,
) -> Tuple[float, float]:
    """
    Apply smoothing and rate-limiting to motor commands.

    Processing steps:
    1. Exponential smoothing: s = alpha * new + (1-alpha) * last
    2. Deadband: ignore changes smaller than threshold
    3. Rate limiting: clamp change per tick based on max delta/sec
    4. Final clamp to absolute limits

    Recovery mode raises both alpha (more responsive) and rate caps so the
    BACKING_UP / ROTATE_* states can engage promptly instead of being throttled.
    """
    # Apply recovery overrides
    if is_recovery:
        if recovery_alpha is not None:
            alpha = recovery_alpha
        if recovery_max_delta_linear_per_sec is not None:
            max_delta_linear_per_sec = recovery_max_delta_linear_per_sec
        if recovery_max_delta_angular_per_sec is not None:
            max_delta_angular_per_sec = recovery_max_delta_angular_per_sec

    # 1. Exponential smoothing
    s_lin = alpha * float(new_linear) + (1.0 - alpha) * float(last_linear)
    s_ang = alpha * float(new_angular) + (1.0 - alpha) * float(last_angular)

    # 2. Deadband - ignore tiny changes (skip during recovery so small commands still apply)
    if not is_recovery:
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

