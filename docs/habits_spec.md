# Gaja Habit Learning System (Uczenie Nawy\u00f3w)

Status: Draft v1 (implement immediately)
Owner: Habit/Personalization Team
Last Updated: 2025-08-08

## 1. Cel
Lekki, transparentny system uczenia nawyk\u00f3w: obserwuj \u2192 sugeruj \u2192 automatyzuj z opcj\u0105 cofni\u0119cia. Minimalny koszt zasob\u00f3w, szybka warto\u015b\u0107 dla u\u017cytkownika.

## 2. Modu\u0142y
- EventBus (ingest)
- FeatureService (featuryzacja + cache)
- HabitDetector (heurystyki: support/conf/lift, sekwencje w rozszerzeniu)
- Policy/Bandit (LinUCB / Thompson)
- Scheduler (okna czasu + planowanie sugestii / automatyzacji)
- ActionExecutor (wykonanie + log)
- FeedbackService (przechwyt odpowiedzi i implicit)
- HabitStore (persist: habits, stats, bandit_state)
- Explainer (why strings)
- Guardrails (rate limits, konflikty, polityki)
- Telemetry & Eval (metryki, widoki SQL, raporty)

## 3. Schematy danych (SQL - minimal)
```sql
CREATE TABLE events(
 id TEXT PRIMARY KEY,
 ts TIMESTAMPTZ NOT NULL,
 actor TEXT,
 verb TEXT,
 object TEXT,
 props JSONB,
 source TEXT,
 confidence REAL
);
CREATE INDEX ON events(ts);

CREATE TABLE decisions(
 id TEXT PRIMARY KEY,
 ts TIMESTAMPTZ NOT NULL,
 context JSONB NOT NULL,
 action JSONB NOT NULL,
 mode TEXT NOT NULL,
 reason TEXT,
 predicted_reward REAL,
 executed BOOLEAN DEFAULT FALSE
);
CREATE INDEX ON decisions(ts);

CREATE TABLE feedback(
 id TEXT PRIMARY KEY,
 decision_id TEXT REFERENCES decisions(id),
 ts TIMESTAMPTZ NOT NULL,
 outcome TEXT, -- accept|reject|snooze|undo
 latency_ms INT
);
CREATE INDEX ON feedback(ts);

CREATE TABLE habits(
 id TEXT PRIMARY KEY,
 action JSONB NOT NULL,
 context_proto JSONB NOT NULL,
 mode TEXT NOT NULL, -- candidate|suggest|auto
 stats JSONB NOT NULL,
 bandit_state JSONB,
 explanation_template TEXT,
 created_at TIMESTAMPTZ DEFAULT NOW(),
 updated_at TIMESTAMPTZ DEFAULT NOW()
);
```

## 4. Modele Pydantic (Python)
```python
class Event(BaseModel):
    id: str; ts: datetime; actor: str; verb: str; object: str
    props: dict = {}; source: str = "ui"; confidence: float = 1.0

class Context(BaseModel):
    ts: datetime; features: dict

class Action(BaseModel):
    verb: str; object: str; params: dict = {}

class Decision(BaseModel):
    id: str; ts: datetime; ctx: Context; action: Action
    mode: str; reason: str; predicted_reward: float

class Habit(BaseModel):
    habit_id: str; action: Action; context_proto: dict
    mode: str; stats: dict; bandit_state: dict; explanation_template: str
```

## 5. Featuryzacja (FeatureService)
- Czas: hour_sin, hour_cos, dow_onehot[7], is_workday
- Ostatnie akcje: prev_<verb>_<object>_15m (0/1), n_actions_last_15m
- Lokalizacja / stan: location_<v>, presence_<v>
- Kalendarium: next_event_within_60m (0/1)
- Hashing dla nazw / playlist: hash32(name) -> bucket (opcjonalnie)
Cache: in-memory TTL 15m. Przy starcie odtwarzaj minimalny stan z ostatnich N zdarzeń.

## 6. Heurystyczny HabitDetector
Dla ka\u017cdego kandydata (context bucket -> action):
- support = count(A,X)/N
- confidence = count(A,X)/count(X)
- lift = P(A,X)/(P(A)*P(X))
Progi promocji candidate->suggest:
- support >= 0.1
- confidence >= 0.6
- lift > 1.2
- obserwacje >= 7

## 7. Bandit Policy (LinUCB)
- Ka\u017cda akcja posiada (A_matrix, b_vector)
- Predict: upper_conf = mu + alpha * sqrt(x^T A^-1 x)
- alpha start 0.25 (feature-flag)
- Decay dzienny ~0.98 (aplikowany przy update lub okresowo co 24h)
- Reward: +1 accept, 0 snooze, -1 reject/undo
- Exploration limits: max 1 nowa eksploracyjna akcja / h, max 3 / d
Persist bandit_state w habits.bandit_state = {"A": [...], "b": [...], "updated_at": ts}

## 8. Scheduler
- Dopasowuje kontekstowe okno czasu: z context_proto.time_window (HH:MM-HH:MM)
- Przy wysokim skupieniu czasowym (κ wysoki): zaw\u0119\u017c okno do ±7.5 min; inaczej bucket 15 min
- Tworzy wpis w decisions (mode suggest lub auto) i wysy\u0142a webhook/UI trigger

## 9. Promocja / Degradacja
- suggest -> auto gdy: accept_rate_7d >= 0.8 AND min_accepts >=5 AND hard_rejects_7d=0
- auto -> suggest gdy: undo >=2 lub rejects >=3 (7d)
- Rate limits (global): ≤1 promocja nawyku / 6h, ≤3 / d

## 10. Guardrails
- Quiet hours (night) => brak auto (tylko suggest) unless user_night_opt_in
- Kategorie z blacklist: tylko suggest
- Konflikt: je\u015bli dwie akcje koliduj\u0105 (definicja: z tej samej kategorii ekskluzywnej) wybierz wy\u017cszy predicted_reward, pozosta\u0142e jako opcje UI

## 11. Explainer v1
- Zbiera wk\u0142ad cech: contribution_i = x_i * theta_i
- Wybiera top 3 dodatnie
- Render: "Bo zwykle {feature_phrases} o tej porze uruchamiasz {action}".
Fallback: explanation_template z habit

## 12. API Endpoints (REST)
```
POST /events              body=Event
POST /decide              body={context} -> {action?, reason, confidence, mode}
GET  /habits?mode=...     -> list
POST /habits/{id}/mode    body={mode}
POST /habits/{id}/snooze  body={until_ts}
POST /habits/{id}/optout
POST /suggestions         body={habit_id, context, scheduled_ts}
POST /feedback            body={suggestion_id, outcome, latency_ms}
```
Autoryzacja: internal token / session (TODO).

## 13. Payload Examples
```json
{
  "id":"evt_123","ts":"2025-08-08T20:30:05Z","actor":"user","verb":"open","object":"spotify","props":{"location":"home","device":"pc"},"source":"ui","confidence":1.0
}
```
Decision:
```json
{
  "id":"dec_001","ts":"2025-08-08T20:29:55Z","context":{"ts":"2025-08-08T20:29:55Z","features":{"hour_sin":0.91,"hour_cos":-0.41,"dow_4":1,"location_home":1}},"action":{"verb":"open","object":"spotify","params":{"playlist":"focus"}},"mode":"suggest","reason":"Wieczorem w domu zwykle w\u0142\u0105czasz Spotify.","predicted_reward":0.72
}
```
Feedback:
```json
{"id":"fb_777","decision_id":"dec_001","ts":"2025-08-08T20:30:07Z","outcome":"accept","latency_ms":12000}
```
Habit:
```json
{
  "habit_id":"hab_spotify_evening_focus","action":{"verb":"open","object":"spotify","params":{"playlist":"focus"}},"context_proto":{"location":"home","time_window":"20:15-22:45"},"mode":"suggest","stats":{"support":0.18,"confidence":0.63,"lift":1.41,"successes":5,"failures":2,"last_seen":"2025-08-08T20:30:05Z","accept_rate_7d":0.83},"bandit_state":{"A_diag":[...],"b":[...]},"explanation_template":"Zwykle wieczorem w domu w\u0142\u0105czasz playlist\u0119 fokus."}
```

## 14. Kontekst Matching
```
matches(context, context_proto):
  - location equality
  - time_window: now in [start-end] inclusive (z tolerancj\u0105 ±delta gdy κ wysoki)
  - opcjonalne: after_event occurs within Δt (rozszerzenie)
```

## 15. Metryki & SQL
Acceptance 7d:
```sql
SELECT COALESCE(
  SUM(CASE WHEN f.outcome='accept' THEN 1 END)::float / NULLIF(COUNT(*),0),0) AS accept_rate_7d
FROM feedback f WHERE f.ts > NOW() - INTERVAL '7 days';
```
Spam score (reject share 7d):
```sql
SELECT COALESCE(SUM(CASE WHEN f.outcome='reject' THEN 1 END)::float / NULLIF(COUNT(*),0),0) AS spam_score
FROM feedback f WHERE ts > NOW() - INTERVAL '7 days';
```
Regret proxy (negatives):
```sql
SELECT AVG(CASE WHEN f.outcome IN ('reject','undo') THEN 1 ELSE 0 END) AS regret_proxy_7d
FROM feedback f WHERE ts > NOW() - INTERVAL '7 days';
```
Per-habit stats refresh job (co 5 min) agreguje successes/failures.

## 16. Kolejno\u015b\u0107 Implementacji (Checklist)
[ ] 1. EventBus + storage (endpoint POST /events, walidacja, insert)
[ ] 2. FeatureService (czas/dow/location/prev actions). In-memory cache + interface.
[ ] 3. HabitDetector (support/conf/lift). Batch co 5 min + on-demand update nowego zdarzenia.
[ ] 4. Scheduler + Suggest API (/decide, /suggestions, /feedback). Rate limits.
[ ] 5. Explainer v1 (regu\u0142y + wk\u0142ad cech gdy dost\u0119pny theta).
[ ] 6. LinUCB integracja (persist state). Online update po feedback.
[ ] 7. Promocja/degradacja + enforcement rate limits.
[ ] 8. Auto-mode + Undo (okno 5 min). Guardrails (quiet hours).
[ ] 9. Telemetry (widoki / endpoints read-only). Dashboard stub.
[ ] 10. Rozszerzenia: sekwencje, κ-czas, klasyfikator per-akcja.

## 17. Test Plan
Jednostkowe:
- matches(context_proto)
- compute_support_conf_lift()
- linucb.predict/update poprawno\u015b\u0107 wymiar\u00f3w i efekt decay
- promotion/demotion logic
- explainer top-k cech
Symulacyjne:
- Synthetic generator (poranek/ wiecz\u00f3r). Mierzymy konwergencj\u0119 accept_rate.
E2E:
- event -> feature -> detection -> suggestion -> feedback -> bandit update -> promotion

## 18. Interfejs (Fasada)
```python
class HabitEngine:
    def log_event(self, event: Event): ...
    def decide(self, context: Context) -> Decision | None: ...
    def schedule(self, habit_id: str, ts: datetime): ...
    def feedback(self, decision_id: str, outcome: str, latency_ms: int): ...
    def set_mode(self, habit_id: str, mode: str): ...
    def opt_out(self, habit_id: str): ...
```

## 19. Feature Flags (konfiguracja)
Przechowuj w tabeli/JSON: alpha_linucb, promotion_thresholds, quiet_hours, exploration_limits, enable_sequences, enable_kappa_time.

## 20. Bezpiecze\u0144stwo i Prywatno\u015b\u0107
- Domy\u015blnie dane lokalnie (SQLite). Eksport agregat\u00f3w tylko po anonimizacji.
- Data retention: events raw 60 dni (konfig), agregaty (habits.stats) d\u0142u\u017cej.
- One-tap opt-out global i per habit.

## 21. Przyk\u0142adowa Implementacja LinUCB
```python
class LinUCB:
    def __init__(self, d, alpha=0.25, decay=0.98):
        self.A = np.eye(d); self.b = np.zeros((d,1))
        self.alpha = alpha; self.decay = decay
    def predict(self, x):
        A_inv = np.linalg.inv(self.A)
        theta = A_inv @ self.b
        mu = float(x.T @ theta)
        conf = self.alpha * float(np.sqrt(x.T @ A_inv @ x))
        return mu + conf, mu
    def update(self, x, reward):
        self.A = self.decay*self.A + x@x.T
        self.b = self.decay*self.b + reward*x
```

## 22. Przysz\u0142e Rozszerzenia
- Per-action classifier (FTRL) jako prior -> bandit.
- Kappa time concentration (von Mises) do w\u0105skiego scheduling.
- Sekwencyjne n-gramy i mini-SPADE.
- Personalizowane cost funkcje (priorytety u\u017cytkownika).

## 23. Uwagi Implementacyjne
- U\u017cyj timezone Europe/Warsaw w kontek\u015bcie.
- Normalizuj cechy ci\u0105g\u0142e (z-score / [0,1]) przed LinUCB.
- Persist theta snapshot co X feedback\u00f3w dla debug.
- Log latency_ms jako proxy wa\u017cno\u015bci.

## 24. ASCII Flow
```
Event -> FeatureService -> HabitDetector (update stats) -> Scheduler/Policy decide -> Decision row -> UI Suggest -> Feedback -> Bandit update -> Habit stats -> (Promotion) -> Auto -> Undo?
```

## 25. Gotowe Do Startu
Po scaleniu pliku mo\u017cna zaczyna\u0107 od punktu 1 checklisty.

---
END OF SPEC
