from __future__ import annotations
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
from loguru import logger

from .models import Habit, Action, Decision, Feedback


class HabitStorage:
    """Lightweight storage layer for habit learning tables (SQLite)."""

    def __init__(self, db_path: str = "server_data.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._ensure_schema()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    @contextmanager
    def _cursor(self):
        conn = self._get_conn()
        cur = conn.cursor()
        try:
            yield cur
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cur.close()

    def _ensure_schema(self):
        with self._get_conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id TEXT PRIMARY KEY,
                    ts TEXT NOT NULL,
                    actor TEXT, verb TEXT, object TEXT,
                    props TEXT, source TEXT, confidence REAL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS decisions (
                    id TEXT PRIMARY KEY,
                    ts TEXT NOT NULL,
                    context TEXT NOT NULL,
                    action TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    reason TEXT,
                    predicted_reward REAL,
                    executed INTEGER DEFAULT 0,
                    decision_vector TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback (
                    id TEXT PRIMARY KEY,
                    decision_id TEXT,
                    ts TEXT NOT NULL,
                    outcome TEXT,
                    latency_ms INTEGER,
                    FOREIGN KEY(decision_id) REFERENCES decisions(id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS habits (
                    id TEXT PRIMARY KEY,
                    action TEXT NOT NULL,
                    context_proto TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    stats TEXT NOT NULL,
                    bandit_state TEXT,
                    explanation_template TEXT,
                    created_at TEXT,
                    updated_at TEXT
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_decisions_ts ON decisions(ts)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_ts ON feedback(ts)")
            try:
                conn.execute("ALTER TABLE decisions ADD COLUMN decision_vector TEXT")
            except Exception:
                pass

    # --- Habit CRUD ---
    def upsert_habit(self, habit: Habit):
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO habits (id, action, context_proto, mode, stats, bandit_state, explanation_template, created_at, updated_at)
                VALUES (?,?,?,?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET action=excluded.action, context_proto=excluded.context_proto, mode=excluded.mode, stats=excluded.stats, bandit_state=excluded.bandit_state, explanation_template=excluded.explanation_template, updated_at=excluded.updated_at
                """,
                (
                    habit.habit_id,
                    json.dumps(habit.action.__dict__),
                    json.dumps(habit.context_proto),
                    habit.mode,
                    json.dumps(habit.stats),
                    json.dumps(habit.bandit_state) if habit.bandit_state else None,
                    habit.explanation_template,
                    (habit.created_at or datetime.now(timezone.utc)).isoformat(),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def get_habits(self, mode: str | None = None) -> List[Habit]:
        q = "SELECT * FROM habits"
        params: tuple[Any, ...] = ()
        if mode:
            q += " WHERE mode=?"
            params = (mode,)
        with self._cursor() as cur:
            cur.execute(q, params)
            rows = cur.fetchall()
        habits: List[Habit] = []
        for r in rows:
            habits.append(
                Habit(
                    habit_id=r["id"],
                    action=Action(**json.loads(r["action"])),
                    context_proto=json.loads(r["context_proto"]),
                    mode=r["mode"],
                    stats=json.loads(r["stats"]),
                    bandit_state=json.loads(r["bandit_state"]) if r["bandit_state"] else None,
                    explanation_template=r["explanation_template"],
                    created_at=datetime.fromisoformat(r["created_at"]) if r["created_at"] else None,
                    updated_at=datetime.fromisoformat(r["updated_at"]) if r["updated_at"] else None,
                )
            )
        return habits

    def get_habit(self, habit_id: str) -> Habit | None:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM habits WHERE id=?", (habit_id,))
            r = cur.fetchone()
        if not r:
            return None
        return Habit(
            habit_id=r["id"],
            action=Action(**json.loads(r["action"])),
            context_proto=json.loads(r["context_proto"]),
            mode=r["mode"],
            stats=json.loads(r["stats"]),
            bandit_state=json.loads(r["bandit_state"]) if r["bandit_state"] else None,
            explanation_template=r["explanation_template"],
            created_at=datetime.fromisoformat(r["created_at"]) if r["created_at"] else None,
            updated_at=datetime.fromisoformat(r["updated_at"]) if r["updated_at"] else None,
        )

    def update_habit_mode(self, habit_id: str, mode: str):
        with self._cursor() as cur:
            cur.execute(
                "UPDATE habits SET mode=?, updated_at=? WHERE id=?",
                (mode, datetime.now(timezone.utc).isoformat(), habit_id),
            )

    def delete_habit(self, habit_id: str):
        with self._cursor() as cur:
            cur.execute("DELETE FROM habits WHERE id=?", (habit_id,))

    # --- Events / Decisions / Feedback ---
    def insert_event(self, event: dict[str, Any]):
        with self._cursor() as cur:
            cur.execute(
                "INSERT OR REPLACE INTO events (id, ts, actor, verb, object, props, source, confidence) VALUES (?,?,?,?,?,?,?,?)",
                (
                    event["id"],
                    event["ts"],
                    event.get("actor"),
                    event.get("verb"),
                    event.get("object"),
                    json.dumps(event.get("props")),
                    event.get("source"),
                    event.get("confidence", 1.0),
                ),
            )

    def recent_events(self, limit: int = 1000) -> list[dict[str, Any]]:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM events ORDER BY ts DESC LIMIT ?", (limit,))
            rows = cur.fetchall()
        out = []
        for r in rows:
            out.append(
                {
                    "id": r["id"],
                    "ts": r["ts"],
                    "actor": r["actor"],
                    "verb": r["verb"],
                    "object": r["object"],
                    "props": json.loads(r["props"]) if r["props"] else {},
                    "source": r["source"],
                    "confidence": r["confidence"],
                }
            )
        return out

    def insert_decision(self, decision: Decision, decision_vector: dict[str, Any] | None = None):
        with self._cursor() as cur:
            cur.execute(
                "INSERT OR REPLACE INTO decisions (id, ts, context, action, mode, reason, predicted_reward, executed, decision_vector) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    decision.id,
                    decision.ts.isoformat(),
                    json.dumps(decision.context),
                    json.dumps(decision.action.__dict__),
                    decision.mode,
                    decision.reason,
                    decision.predicted_reward,
                    1 if decision.executed else 0,
                    json.dumps(decision_vector) if decision_vector else None,
                ),
            )

    def get_decision(self, decision_id: str) -> dict | None:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM decisions WHERE id=?", (decision_id,))
            r = cur.fetchone()
        if not r:
            return None
        return {
            'id': r['id'],
            'ts': r['ts'],
            'context': json.loads(r['context']),
            'action': json.loads(r['action']),
            'mode': r['mode'],
            'reason': r['reason'],
            'predicted_reward': r['predicted_reward'],
            'executed': bool(r['executed']),
            'decision_vector': json.loads(r['decision_vector']) if r['decision_vector'] else None
        }

    def insert_feedback(self, fb: Feedback):
        with self._cursor() as cur:
            cur.execute(
                "INSERT OR REPLACE INTO feedback (id, decision_id, ts, outcome, latency_ms) VALUES (?,?,?,?,?)",
                (
                    fb.id,
                    fb.decision_id,
                    fb.ts.isoformat(),
                    fb.outcome,
                    fb.latency_ms,
                ),
            )

    def decisions_last_days(self, days: int = 7) -> list[sqlite3.Row]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM decisions WHERE ts >= datetime('now', ?)",
                (f"-{days} days",),
            )
            return cur.fetchall()

    def feedback_for_habit(self, habit_id: str, days: int = 7) -> list[dict[str, Any]]:
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT f.* FROM feedback f
                JOIN decisions d ON f.decision_id = d.id
                JOIN habits h ON json_extract(d.action,'$.verb') = json_extract(h.action,'$.verb')
                  AND json_extract(d.action,'$.object') = json_extract(h.action,'$.object')
                WHERE h.id=? AND f.ts >= datetime('now', ?)
                """,
                (habit_id, f"-{days} days"),
            )
            rows = cur.fetchall()
        out = []
        for r in rows:
            out.append(
                {
                    "id": r["id"],
                    "decision_id": r["decision_id"],
                    "ts": r["ts"],
                    "outcome": r["outcome"],
                    "latency_ms": r["latency_ms"],
                }
            )
        return out

    def update_decision_executed(self, decision_id: str):
        with self._cursor() as cur:
            cur.execute("UPDATE decisions SET executed=1 WHERE id=?", (decision_id,))

    def list_decisions(self, limit: int = 50):
        with self._cursor() as cur:
            cur.execute("SELECT * FROM decisions ORDER BY ts DESC LIMIT ?", (limit,))
            rows = cur.fetchall()
        out = []
        for r in rows:
            out.append({
                'id': r['id'],
                'ts': r['ts'],
                'action': json.loads(r['action']),
                'mode': r['mode'],
                'predicted_reward': r['predicted_reward'],
                'executed': bool(r['executed'])
            })
        return out

    def list_feedback(self, limit: int = 50):
        with self._cursor() as cur:
            cur.execute("SELECT * FROM feedback ORDER BY ts DESC LIMIT ?", (limit,))
            rows = cur.fetchall()
        out = []
        for r in rows:
            out.append({
                'id': r['id'],
                'decision_id': r['decision_id'],
                'ts': r['ts'],
                'outcome': r['outcome'],
                'latency_ms': r['latency_ms']
            })
        return out
