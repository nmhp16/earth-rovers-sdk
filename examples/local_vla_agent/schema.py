from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class Action:
    linear: float = 0.0
    angular: float = 0.0
    lamp: int = 0

    def to_payload(self) -> Dict[str, Any]:
        return {"command": {"linear": self.linear, "angular": self.angular, "lamp": self.lamp}}
