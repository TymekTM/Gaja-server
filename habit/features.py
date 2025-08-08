from __future__ import annotations
from datetime import datetime, timezone
import math
from collections import deque
from typing import Any

class FeatureService:
    def __init__(self, max_events: int = 500):
        self.recent_actions = deque(maxlen=max_events)

    def featurize(self, event: dict[str, Any]) -> dict[str, float | int]:
        raw_ts = event["ts"]
        if isinstance(raw_ts, str):
            ts = datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
        else:
            ts = raw_ts
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        hour = ts.hour + ts.minute / 60.0
        hour_sin = math.sin(2 * math.pi * hour / 24)
        hour_cos = math.cos(2 * math.pi * hour / 24)
        dow = ts.weekday()
        features: dict[str, float | int] = {"hour_sin": hour_sin, "hour_cos": hour_cos}
        for i in range(7):
            features[f"dow_{i}"] = 1 if i == dow else 0
        cutoff = ts.timestamp() - 15 * 60
        for (t, verb, obj) in list(self.recent_actions):
            if t >= cutoff:
                features[f"prev_{verb}_{obj}_15m"] = 1
        self.recent_actions.append((ts.timestamp(), event.get("verb"), event.get("object")))
        props = event.get("props", {}) or {}
        loc = props.get("location")
        if loc:
            features[f"location_{loc}"] = 1
        dev = props.get("device")
        if dev:
            features[f"device_{dev}"] = 1
        # generic boolean flags
        for k, v in props.items():
            if isinstance(v, bool):
                features[f"prop_{k}_{int(v)}"] = 1
        return features
