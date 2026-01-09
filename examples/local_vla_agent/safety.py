from typing import Dict, Any, Tuple

try:
    from schema import Action
    from utils import clamp, safe_float, safe_int
    from config import MAX_LINEAR, MAX_ANGULAR, STOP_ON_LOW_BATT, SLOW_ON_LOW_BATT, DANGER_KEYWORDS, PROXIMITY_KEYWORDS
except ImportError:
    from .schema import Action
    from .utils import clamp, safe_float, safe_int
    from .config import MAX_LINEAR, MAX_ANGULAR, STOP_ON_LOW_BATT, SLOW_ON_LOW_BATT, DANGER_KEYWORDS, PROXIMITY_KEYWORDS


def validate_action(obj: Dict[str, Any]) -> Action:
    """
    Validate and sanitize an action dictionary from LLM output.
    
    Args:
        obj: Dictionary with 'linear', 'angular', 'lamp' keys
        
    Returns:
        Validated Action with clamped values
    """
    linear = clamp(safe_float(obj.get("linear", 0.0), 0.0), -1.0, 1.0)
    angular = clamp(safe_float(obj.get("angular", 0.0), 0.0), -1.0, 1.0)
    lamp = 1 if safe_int(obj.get("lamp", 0), 0) == 1 else 0

    # Apply conservative caps
    linear = clamp(linear, -MAX_LINEAR, MAX_LINEAR)
    angular = clamp(angular, -MAX_ANGULAR, MAX_ANGULAR)

    return Action(linear=linear, angular=angular, lamp=lamp)


def safety_override(
    action: Action,
    telemetry: Dict[str, Any],
    caption_front: str,
    caption_rear: str
) -> Tuple[Action, bool]:
    """
    Apply safety overrides to an action based on telemetry and vision.
    
    Safety layers applied:
    1. Hard stop on critical battery
    2. Speed reduction on low battery
    3. Reflex backup on proximity keywords
    4. Speed reduction on danger keywords
    
    Args:
        action: Original action from LLM/recovery
        telemetry: Current rover telemetry (battery, speed, etc.)
        caption_front: Vision caption from front camera
        caption_rear: Vision caption from rear camera
        
    Returns:
        Tuple of (modified action, danger_flag)
    """
    batt = safe_float(telemetry.get("battery", 100.0), 100.0)

    # Hard stop on critical battery
    if batt <= STOP_ON_LOW_BATT:
        return Action(0.0, 0.0, action.lamp), True

    # Slow down on low battery
    if batt <= SLOW_ON_LOW_BATT:
        action.linear *= 0.5
        action.angular *= 0.7

    combined = f"{caption_front} {caption_rear}".lower()
    danger = any(k in combined for k in DANGER_KEYWORDS)

    # Check for immediate proximity (reflex backup)
    # This overrides normal danger logic with a reflexive backup
    if any(k in caption_front.lower() for k in PROXIMITY_KEYWORDS):
        action.linear = -0.25  # Backup
        # Force a turn if one isn't present
        if abs(action.angular) < 0.3:
            action.angular = 0.5
        return Action(linear=action.linear, angular=action.angular, lamp=1), True

    if danger:
        # Reduce forward aggression and prefer some turning
        if action.linear > 0.15:
            action.linear = 0.15
        if abs(action.angular) < 0.2:
            action.angular = 0.35

    # Final clamp to safety limits
    action.linear = clamp(action.linear, -MAX_LINEAR, MAX_LINEAR)
    action.angular = clamp(action.angular, -MAX_ANGULAR, MAX_ANGULAR)
    action.lamp = 1 if action.lamp == 1 else 0
    
    return action, danger
