from typing import Dict, Any, Tuple
from schema import Action
from utils import clamp, safe_float, safe_int
from config import MAX_LINEAR, MAX_ANGULAR, STOP_ON_LOW_BATT, SLOW_ON_LOW_BATT, DANGER_KEYWORDS

def validate_action(obj: Dict[str, Any]) -> Action:
    linear = clamp(safe_float(obj.get("linear", 0.0), 0.0), -1.0, 1.0)
    angular = clamp(safe_float(obj.get("angular", 0.0), 0.0), -1.0, 1.0)
    lamp = 1 if safe_int(obj.get("lamp", 0), 0) == 1 else 0

    # Apply conservative caps
    linear = clamp(linear, -MAX_LINEAR, MAX_LINEAR)
    angular = clamp(angular, -MAX_ANGULAR, MAX_ANGULAR)

    return Action(linear=linear, angular=angular, lamp=lamp)

def safety_override(action: Action, telemetry: Dict[str, Any], caption_front: str, caption_rear: str) -> Tuple[Action, bool]:
    """
    Returns (possibly modified action, danger_flag)
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

    if danger:
        # Reduce forward aggression and prefer some turning
        if action.linear > 0.15:
            action.linear = 0.15
        if abs(action.angular) < 0.2:
            action.angular = 0.35

    # final clamp
    action.linear = clamp(action.linear, -MAX_LINEAR, MAX_LINEAR)
    action.angular = clamp(action.angular, -MAX_ANGULAR, MAX_ANGULAR)
    action.lamp = 1 if action.lamp == 1 else 0
    return action, danger
