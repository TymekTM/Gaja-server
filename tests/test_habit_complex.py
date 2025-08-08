import sys, pathlib, random
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from habit.engine import HabitEngine  # noqa: E402


def test_multiple_habits_with_noise(tmp_path):
    db = tmp_path / 'complex.db'
    eng = HabitEngine(str(db))

    now = datetime.now(timezone.utc)

    # Config: three target habits A,B (same bucket office) and C (home), plus noise (varied objects/locations) below thresholds
    # A: open spotify_playlist (office)
    # B: launch voicemeeter (office)
    # C: open editor (home)
    # All events share same timestamp hour to keep bucket stable

    def add_event(verb, obj, loc):
        evt = {'verb': verb, 'object': obj, 'props': {'location': loc}, 'ts': now.isoformat()}
        eng.log_event(evt)

    # Generate interleaved pattern occurrences
    pattern_A = [('open', 'spotify_playlist', 'office')]*12
    pattern_B = [('launch', 'voicemeeter', 'office')]*12
    pattern_C = [('open', 'editor', 'home')]*8

    # Noise: 30 events w innej lokalizacji (cafe) aby nie rozmywać confidence habitu C
    noise_candidates = []
    for i in range(30):
        noise_candidates.append((f'act{i%5}', f'obj{i}', 'cafe'))

    all_events = pattern_A + pattern_B + pattern_C + noise_candidates
    random.shuffle(all_events)

    for idx,(v,o,l) in enumerate(all_events):
        add_event(v,o,l)
        if idx % 10 == 0:
            eng.scan_new_habits()
    eng.scan_new_habits()

    habits = eng.storage.get_habits()
    # Expect at least the 3 target habits
    objects = {h.action.object: h for h in habits}
    assert 'spotify_playlist' in objects, 'Habit A missing'
    assert 'voicemeeter' in objects, 'Habit B missing'
    assert 'editor' in objects, 'Habit C missing'

    # Validate metrics
    a = objects['spotify_playlist']
    b = objects['voicemeeter']
    c = objects['editor']

    # Confidence thresholds
    assert a.stats['confidence'] >= 0.5
    assert b.stats['confidence'] >= 0.5
    assert c.stats['confidence'] >= 0.5

    # Każdy habit powinien mieć sensowny support
    for obj, h in [('A', a), ('B', b), ('C', c)]:
        assert h.stats['support'] >= 0.05, f"Zbyt niski support dla {obj}"

    # A i B (ten sam kontekst) powinny mieć zbliżony support (różnica <= 0.15)
    assert abs(a.stats['support'] - b.stats['support']) <= 0.15

    # Decision in office context yields one of A or B
    # Build features from an artificial event in office
    office_evt = {'verb': 'noop', 'object': 'context', 'ts': now.isoformat(), 'props': {'location': 'office'}}
    feats = eng.features.featurize(office_evt)
    d = eng.decide({'features': feats})
    assert d is not None
    assert d.action.object in ('spotify_playlist', 'voicemeeter')

    # Provide feedback loop to adapt bandit (accept first 3 decisions, reject next 2 for exploration)
    accepted = 0
    rejected = 0
    accepted_actions = []
    for i in range(5):
        d_loop = eng.decide({'features': feats})
        if not d_loop:
            break
        if i < 3:
            eng.feedback(d_loop.id, 'accept')
            accepted += 1
            accepted_actions.append(d_loop.action.object)
        else:
            eng.feedback(d_loop.id, 'reject')
            rejected += 1
    assert accepted >= 1
    refreshed = eng.storage.get_habits()
    updated_objects = {h.action.object: h for h in refreshed}
    # Sprawdź że dla każdej zaakceptowanej akcji wzrosły successes >= 1 łącznie
    for obj in set(accepted_actions):
        assert updated_objects[obj].stats.get('successes',0) >= 1, f"Brak sukcesów zarejestrowanych dla {obj}"
