from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class Action:
    """
    Motor control action to send to the rover.
    
    Attributes:
        linear: Forward/backward velocity (-1.0 to 1.0, positive = forward)
        angular: Rotation velocity (-1.0 to 1.0, positive = left, negative = right)
        lamp: Lamp state (0 = off, 1 = on)
    """
    linear: float = 0.0
    angular: float = 0.0
    lamp: int = 0

    def to_payload(self) -> Dict[str, Any]:
        """Convert action to API payload format."""
        return {
            "command": {
                "linear": self.linear,
                "angular": self.angular,
                "lamp": self.lamp
            }
        }
