import sys, pathlib
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from habit.engine import HabitEngine  # noqa: E402


def test_synthetic_habit_emergence(tmp_path):
    db = tmp_path / 'synthetic.db'
    eng = HabitEngine(str(db))

    now = datetime.now(timezone.utc).isoformat()

    # Noise (ale powtarzalny - będzie wykryty jako nawyk, to akceptujemy)
    for _ in range(4):
        eng.log_event({'verb':'open','object':'editor','props':{'location':'home'}, 'ts': now})
    eng.scan_new_habits()
    # editor może stać się pierwszym nawykiem
    initial_habits = eng.storage.get_habits()

    # Drugi wzorzec docelowy, który chcemy aby także został wykryty i porównamy support
    pattern_event = {'verb':'open','object':'spotify_playlist','props':{'location':'office'}, 'ts': now}
    for _ in range(3):
        eng.log_event(pattern_event.copy())
    eng.scan_new_habits()

    habits = eng.storage.get_habits()
    verbs = {h.action.object: h for h in habits}
    assert 'spotify_playlist' in verbs, 'Docelowy habit nie został wykryty'
    # Sprawdź że habit ma sensowne metryki
    sp = verbs['spotify_playlist']
    assert sp.stats['support'] >= 0.05
    assert sp.stats['confidence'] >= 0.5
