from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class Event:
    id: str
    ts: datetime
    actor: str
    verb: str
    object: str
    props: Dict[str, Any] = field(default_factory=dict)
    source: str = "ui"
    confidence: float = 1.0


@dataclass
class Action:
    verb: str
    object: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Habit:
    habit_id: str
    action: Action
    context_proto: Dict[str, Any]
    mode: str  # candidate|suggest|auto
    stats: Dict[str, Any]
    bandit_state: Dict[str, Any] | None = None
    explanation_template: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass
class Decision:
    id: str
    ts: datetime
    context: Dict[str, Any]
    action: Action
    mode: str
    reason: str
    predicted_reward: float
    executed: bool = False


@dataclass
class Feedback:
    id: str
    decision_id: str
    ts: datetime
    outcome: str  # accept|reject|snooze|undo
    latency_ms: int | None = None
