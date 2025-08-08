import sys, pathlib, uuid
from datetime import datetime, timezone

# ensure root path for imports
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from habit.engine import HabitEngine  # noqa: E402


def test_engine_basic_flow(tmp_path):
    db = tmp_path / "test.db"
    eng = HabitEngine(str(db))

    now = datetime.now(timezone.utc)
    for _ in range(10):
        evt = {
            'id': str(uuid.uuid4()),
            'ts': now.isoformat(),
            'actor': 'user',
            'verb': 'open',
            'object': 'spotify',
            'props': {'location':'home','device':'pc'}
        }
        eng.log_event(evt)

    new_habits = eng.scan_new_habits()
    assert new_habits

    sample_event = {
        'id': 'ctx', 'ts': now.isoformat(), 'actor': 'user', 'verb':'open','object':'spotify', 'props': {'location':'home','device':'pc'}
    }
    feats = eng.features.featurize(sample_event)
    decision = eng.decide({'features':feats})
    assert decision is not None

    fb_id = eng.feedback(decision, 'accept', latency_ms=1000)
    assert fb_id


def test_bandit_persistence(tmp_path):
    db = tmp_path / 'persist.db'
    eng = HabitEngine(str(db))
    # generate events to form habit (ensure min_obs>=3 and some support)
    for _ in range(8):
        eng.log_event({'verb':'open','object':'spotify_playlist','props':{'hour_cos':0.1,'hour_sin':0.2,'location':'office','dow':0}})
    eng.scan_new_habits()
    habits = eng.storage.get_habits()
    assert habits, 'No habits detected; detector thresholds may be too high'
    # build context using feature service for consistency
    evt_ctx = {'id':'ctx','ts': datetime.now(timezone.utc).isoformat(),'verb':'open','object':'spotify_playlist','props':{'hour_cos':0.1,'hour_sin':0.2,'location':'office','dow':0}}
    feats = {'hour_cos':0.1,'hour_sin':0.2,'location_office':1,'dow_0':1}
    d = eng.decide({'features':feats})
    assert d, 'Decision not generated'
    eng.feedback(d.id, 'accept')
    h_mid = eng.storage.get_habits()[0]
    assert h_mid.bandit_state is not None
    # restart engine
    eng2 = HabitEngine(str(db))
    h_after = eng2.storage.get_habits()[0]
    assert h_after.bandit_state is not None
    d2 = eng2.decide({'features':{'hour_cos':0.1,'hour_sin':0.2,'location_office':1,'dow_0':1}})
    assert d2


def test_rate_limiting(tmp_path):
    db = tmp_path / 'limit.db'
    eng = HabitEngine(str(db), decision_cooldown=5, suggestion_limit_per_hour=3)
    for _ in range(5):
        eng.log_event({'verb':'open','object':'spotify_playlist','props':{'hour_cos':0.2,'hour_sin':0.3,'location':'office','dow':0}})
    eng.scan_new_habits()
    feats = {'hour_cos':0.2,'hour_sin':0.3,'location_office':1,'dow_0':1}
    d1 = eng.decide({'features':feats})
    assert d1
    d2 = eng.decide({'features':feats})  # cooldown blocks
    assert d2 is None
    # simulate time passage for cooldown
    eng._last_decision_ts[f"{d1.action.verb}|{d1.action.object}"] -= 10
    d3 = eng.decide({'features':feats})
    assert d3
    d4 = eng.decide({'features':feats})  # limit per hour reached (3 suggestions)
    assert d4 is None


def test_auto_scheduler(tmp_path):
    db = tmp_path / 'auto.db'
    eng = HabitEngine(str(db), decision_cooldown=0, suggestion_limit_per_hour=100, auto_exec_limit_per_hour=10, scheduler_interval=1)
    # create habit
    for _ in range(8):
        eng.log_event({'verb':'open','object':'spotify_playlist','props':{'hour_cos':0.3,'hour_sin':0.4,'location':'office','dow':0}})
    eng.scan_new_habits()
    feats = {'hour_cos':0.3,'hour_sin':0.4,'location_office':1,'dow_0':1}
    # generate accept feedback to promote
    for _ in range(5):
        d = eng.decide({'features':feats})
        if d:
            eng.feedback(d.id, 'accept')
    # ensure at least one habit auto
    auto_habits = [h for h in eng.storage.get_habits() if h.mode=='auto']
    assert auto_habits, 'Habit not promoted to auto'
    eng.start_scheduler()
    import time
    time.sleep(2.5)
    eng.stop_scheduler()
    # should have recorded auto executions (success increments)
    h = eng.storage.get_habits()[0]
    assert h.stats.get('successes',0) >= 5
