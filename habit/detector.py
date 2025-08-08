from __future__ import annotations
from collections import defaultdict
from typing import Dict, Any, Tuple

class HabitDetector:
    """Heurystyczne wykrywanie kandydatów na nawyki (support/confidence/lift)."""

    def __init__(self, min_support=0.05, min_conf=0.5, min_lift=1.0, min_obs=3):  # lowered thresholds for initial detection
        self.min_support = min_support
        self.min_conf = min_conf
        self.min_lift = min_lift
        self.min_obs = min_obs
        # Counters
        self.count_all = 0
        self.count_action: Dict[Tuple[str, str], int] = defaultdict(int)
        self.count_context: Dict[str, int] = defaultdict(int)
        # key: (bucket, verb, object)
        self.count_pair: Dict[Tuple[str, str, str], int] = defaultdict(int)

    def _context_bucket(self, features: dict[str, Any]) -> str:
        hour_cos = features.get('hour_cos', 0)
        hour_bucket = int(hour_cos * 1000)
        loc_keys = [k for k in features if k.startswith('location_') and features[k] == 1]
        dow = [k for k in features if k.startswith('dow_') and features[k] == 1]
        return f"{hour_bucket}|{','.join(sorted(loc_keys))}|{','.join(sorted(dow))}"

    def observe(self, features: dict[str, Any], action: dict[str, Any]):
        verb = action.get('verb') or ''
        obj = action.get('object') or ''
        if not verb or not obj:
            return
        self.count_all += 1
        act_key = (verb, obj)
        self.count_action[act_key] += 1
        bucket = self._context_bucket(features)
        self.count_context[bucket] += 1
        self.count_pair[(bucket, verb, obj)] += 1

    def candidates(self):
        results = []
        total = self.count_all or 1
        for (bucket, verb, obj), c_pair in self.count_pair.items():
            c_context = self.count_context[bucket]
            act_key = (verb, obj)
            c_action = self.count_action[act_key]
            support = c_pair / total
            confidence = c_pair / c_context if c_context else 0.0
            p_action = c_action / total
            # lift simplified: confidence / p_action
            lift = confidence / (p_action if p_action else 1e-9)
            if support >= self.min_support and confidence >= self.min_conf and lift >= self.min_lift and c_pair >= self.min_obs:
                results.append({
                    'bucket': bucket,
                    'verb': verb,
                    'object': obj,
                    'support': round(support,4),
                    'confidence': round(confidence,4),
                    'lift': round(lift,4),
                    'observations': c_pair
                })
        return results
