from __future__ import annotations
import uuid
from datetime import datetime, timezone
from typing import Any
from loguru import logger
from threading import Lock
import numpy as np
import threading

from .models import Action, Decision, Feedback, Habit
from .storage import HabitStorage
from .features import FeatureService
from .detector import HabitDetector
from .policy import PolicyManager

class HabitEngine:
    def __init__(self, db_path: str = "server_data.db", decision_cooldown: int = 0, suggestion_limit_per_hour: int = 1000, auto_exec_limit_per_hour: int = 20, scheduler_interval: int = 30):
        self.storage = HabitStorage(db_path)
        self.features = FeatureService()
        self.detector = HabitDetector()
        self.policy = PolicyManager()
        self._decision_vectors: dict[str, tuple[str, Any]] = {}
        self._lock = Lock()
        self._load_bandits()
        self.decision_cooldown = decision_cooldown
        self.suggestion_limit_per_hour = suggestion_limit_per_hour
        self.auto_exec_limit_per_hour = auto_exec_limit_per_hour
        self.scheduler_interval = scheduler_interval
        self._last_decision_ts: dict[str, float] = {}
        from collections import deque
        self._recent_suggestions = deque(maxlen=500)
        self._recent_auto_exec = deque(maxlen=200)
        self._scheduler_thread: threading.Thread | None = None
        self._scheduler_stop = threading.Event()
        self._last_features: dict[str, Any] | None = None
        self._active_decision_ids: set[str] = set()

    def _load_bandits(self):
        try:
            for h in self.storage.get_habits():
                if h.bandit_state and 'model' in h.bandit_state:
                    state = h.bandit_state['model']
                    feats = h.bandit_state.get('feature_index', [])
                    self.policy.load_action_state(f"{h.action.verb}|{h.action.object}", state, feats)
        except Exception:
            logger.exception('Failed loading bandit states')

    def _persist_bandit(self, habit: Habit):
        try:
            action_key = f"{habit.action.verb}|{habit.action.object}"
            model_state = self.policy.get_action_state(action_key)
            habit.bandit_state = {
                'model': model_state,
                'feature_index': list(self.policy.export_feature_index().keys())
            }
            self.storage.upsert_habit(habit)
        except Exception:
            logger.exception('Persist bandit failed')

    def log_event(self, event: dict[str, Any]):
        # persist
        if 'id' not in event:
            event['id'] = str(uuid.uuid4())
        if 'ts' not in event:
            event['ts'] = datetime.now(timezone.utc).isoformat()
        self.storage.insert_event(event)
        # featurize & detector observe
        feats = self.features.featurize(event)
        action = { 'verb': event.get('verb'), 'object': event.get('object') }
        self.detector.observe(feats, action)
        self._last_features = feats
        return event['id']

    def scan_new_habits(self):
        cands = self.detector.candidates()
        created = []
        for c in cands:
            habit_id = f"hab_{c['verb']}_{c['object']}_{abs(hash(c['bucket']))%10000}"
            existing = self.storage.get_habit(habit_id)
            if existing:
                # refresh stats for already detected habit
                existing.stats['support'] = c['support']
                existing.stats['confidence'] = c['confidence']
                existing.stats['lift'] = c['lift']
                self.storage.upsert_habit(existing)
                continue
            habit = Habit(
                habit_id=habit_id,
                action=Action(verb=c['verb'], object=c['object']),
                context_proto={'bucket': c['bucket']},
                mode='suggest',
                stats={'support': c['support'], 'confidence': c['confidence'], 'lift': c['lift'], 'successes':0,'failures':0,'accept_rate_7d':0.0},
                bandit_state=None,
                explanation_template=f"Zwykle w tym kontekście uruchamiasz {c['verb']} {c['object']}"
            )
            self.storage.upsert_habit(habit)
            created.append(habit_id)
        return created

    def decide(self, context: dict[str, Any]):
        # conflict guard: limit równoległych aktywnych decyzji bez feedbacku
        if len(self._active_decision_ids) >= 5:
            return None
        habits = self.storage.get_habits()
        if not habits:
            return None
        now_ts = datetime.now(timezone.utc).timestamp()
        # prune recent suggestions >1h
        while self._recent_suggestions and now_ts - self._recent_suggestions[0] > 3600:
            self._recent_suggestions.popleft()
        if len(self._recent_suggestions) >= self.suggestion_limit_per_hour:
            return None
        best = None
        best_val = -1e9
        for h in habits:
            bucket = h.context_proto.get('bucket')
            feats = context.get('features') or context
            variants = self._bucket_variants(feats)
            if bucket and bucket not in variants:
                # fallback: allow if same verb/object even if bucket drift minor (ignore location part)
                pass
            upper, mean, x_vec = self.policy.predict(f"{h.action.verb}|{h.action.object}", feats)
            if upper > best_val:
                best_val = upper
                best = (h, upper, mean, x_vec, feats)
        if not best:
            return None
        h, upper, mean, x_vec, feats = best
        action_key = f"{h.action.verb}|{h.action.object}"
        last_ts = self._last_decision_ts.get(action_key)
        if self.decision_cooldown and last_ts and now_ts - last_ts < self.decision_cooldown:
            return None
        explanation = h.explanation_template or 'Sugestia nawyku'
        try:
            theta_info = self.policy.get_theta(f"{h.action.verb}|{h.action.object}")
            if theta_info:
                feat_names, theta_vals = theta_info
                contrib = sorted(zip(feat_names, theta_vals), key=lambda x: abs(x[1]), reverse=True)[:3]
                parts = [f"{name} ({val:+.2f})" for name, val in contrib if name in feats and feats[name] != 0]
                if parts:
                    explanation += ' (cechy: ' + ', '.join(parts) + ')'
        except Exception:
            logger.exception('Explain failed')
        decision = Decision(
            id=str(uuid.uuid4()),
            ts=datetime.now(timezone.utc),
            context=context,
            action=h.action,
            mode=h.mode,
            reason=explanation,
            predicted_reward=mean,
            executed=(h.mode == 'auto')
        )
        self.storage.insert_decision(decision, decision_vector={'action_key': f"{h.action.verb}|{h.action.object}", 'x_vec': x_vec.tolist()})
        with self._lock:
            self._decision_vectors[decision.id] = (f"{h.action.verb}|{h.action.object}", x_vec)
        self._last_decision_ts[action_key] = now_ts
        self._recent_suggestions.append(now_ts)
        self._active_decision_ids.add(decision.id)
        return decision

    def feedback(self, decision_id: str, outcome: str, latency_ms: int | None = None):
        reward = 0.0
        undo_adjust = None
        if outcome == 'accept':
            reward = 1.0
        elif outcome in ('reject', 'undo'):
            reward = -1.0 if outcome != 'undo' else 0.0
        fb = Feedback(id=str(uuid.uuid4()), decision_id=decision_id, ts=datetime.now(timezone.utc), outcome=outcome, latency_ms=latency_ms)
        self.storage.insert_feedback(fb)
        with self._lock:
            data = self._decision_vectors.pop(decision_id, None)
        if not data:
            # attempt restore from DB
            dec = self.storage.get_decision(decision_id)
            if dec and dec.get('decision_vector'):
                dv = dec['decision_vector']
                x_vec = np.array(dv['x_vec'], dtype=float)
                data = (dv['action_key'], x_vec)
        if data:
            action_key, x_vec = data
            if isinstance(x_vec, list):
                x_vec = np.array(x_vec, dtype=float)
            self.policy.update(action_key, x_vec, reward)
            # update habit stats only if we have action_key
            try:
                verb, obj = action_key.split('|',1)
                for h in self.storage.get_habits():
                    if h.action.verb == verb and h.action.object == obj:
                        if outcome == 'accept':
                            h.stats['successes'] = h.stats.get('successes',0)+1
                        elif outcome == 'reject':
                            h.stats['failures'] = h.stats.get('failures',0)+1
                        elif outcome == 'undo':
                            # revert last success if any
                            if h.stats.get('successes',0) > 0:
                                h.stats['successes'] -= 1
                        total = max(1, h.stats.get('successes',0)+h.stats.get('failures',0))
                        h.stats['accept_rate_7d'] = h.stats.get('successes',0)/total
                        if h.mode == 'suggest' and total >= 5 and h.stats['accept_rate_7d'] >= 0.8:
                            h.mode = 'auto'
                        if h.mode == 'auto' and total >= 5 and h.stats['accept_rate_7d'] < 0.5:
                            h.mode = 'suggest'
                        self._persist_bandit(h)
                        break
            except Exception:
                logger.exception('Failed updating habit stats')
        self._active_decision_ids.discard(decision_id)
        return fb.id

    def _bucket_variants(self, features: dict[str, Any]):
        hour_cos = features.get('hour_cos',0)
        hour_bucket = int(hour_cos * 1000)
        loc_keys = [k for k in features if k.startswith('location_') and features[k]==1]
        dow = [k for k in features if k.startswith('dow_') and features[k]==1]
        return {f"{hour_bucket}|{','.join(sorted(loc_keys))}|{','.join(sorted(dow))}"}

    def execute_action(self, decision_id: str):
        dec = self.storage.get_decision(decision_id)
        if not dec:
            return False
        # Placeholder execution logic
        self.storage.update_decision_executed(decision_id)
        return True

    def start_scheduler(self):
        if self._scheduler_thread and self._scheduler_thread.is_alive():
            return
        self._scheduler_stop.clear()
        self._scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self._scheduler_thread.start()

    def stop_scheduler(self):
        if self._scheduler_thread and self._scheduler_thread.is_alive():
            self._scheduler_stop.set()
            self._scheduler_thread.join(timeout=2)

    def _scheduler_loop(self):
        while not self._scheduler_stop.is_set():
            try:
                self._auto_tick()
            except Exception:
                logger.exception('Scheduler tick failed')
            self._scheduler_stop.wait(self.scheduler_interval)

    def _auto_tick(self):
        # prune auto exec >1h
        now_ts = datetime.now(timezone.utc).timestamp()
        while self._recent_auto_exec and now_ts - self._recent_auto_exec[0] > 3600:
            self._recent_auto_exec.popleft()
        if len(self._recent_auto_exec) >= self.auto_exec_limit_per_hour:
            return
        # build context
        if not self._last_features:
            return
        # evaluate only auto habits
        habits = [h for h in self.storage.get_habits() if h.mode == 'auto']
        if not habits:
            return
        context = {'features': self._last_features}
        decision = self.decide(context)
        if decision and decision.executed:
            # treat as implicitly accepted for training
            self.feedback(decision.id, 'accept', latency_ms=0)
            self._recent_auto_exec.append(now_ts)

    # --- State export/import ---
    def export_state(self) -> dict[str, Any]:
        habits_dump = []
        for h in self.storage.get_habits():
            habits_dump.append({
                'habit_id': h.habit_id,
                'action': {'verb': h.action.verb, 'object': h.action.object},
                'context_proto': h.context_proto,
                'mode': h.mode,
                'stats': h.stats,
                'bandit_state': h.bandit_state,
                'explanation_template': h.explanation_template
            })
        return {
            'version': 1,
            'exported_at': datetime.now(timezone.utc).isoformat(),
            'habits': habits_dump
        }

    def import_state(self, payload: dict[str, Any], merge: bool = True):
        habits_data = payload.get('habits', [])
        existing = {h.habit_id: h for h in self.storage.get_habits()}
        for h in habits_data:
            hid = h['habit_id']
            if not merge and hid in existing:
                continue
            habit = Habit(
                habit_id=hid,
                action=Action(verb=h['action']['verb'], object=h['action']['object']),
                context_proto=h['context_proto'],
                mode=h['mode'],
                stats=h.get('stats', {}),
                bandit_state=h.get('bandit_state'),
                explanation_template=h.get('explanation_template')
            )
            self.storage.upsert_habit(habit)
        # reload bandits after import
        self.policy = PolicyManager()
        self._load_bandits()
