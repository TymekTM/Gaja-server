import sys, pathlib, time, json
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402
from server_main import app  # noqa: E402


def _build_features_from_bucket(bucket: str):
    hour_part, loc_part, dow_part = bucket.split('|')
    hour_cos = int(hour_part)/1000.0 if hour_part else 0.0
    feats = {'hour_cos': hour_cos, 'hour_sin': 0.0}
    if loc_part:
        for l in loc_part.split(','):
            if l:
                feats[l] = 1
    if dow_part:
        for d in dow_part.split(','):
            if d:
                feats[d] = 1
    return feats


def test_habit_api_e2e():
    client = TestClient(app)

    # 1. generate events to create habit
    for _ in range(6):
        resp = client.post('/api/v1/habit/events', json={'verb':'open','object':'spotify_playlist','props':{'location':'office'}})
        assert resp.status_code == 200
        time.sleep(0.01)

    # 2. poll for habit detection
    habit_id = None
    bucket = None
    for _ in range(10):
        r = client.get('/api/v1/habit/habits')
        assert r.status_code == 200
        data = r.json()['habits']
        if data:
            habit_id = data[0]['id']
            bucket = data[0]['context_proto']['bucket']
            break
        time.sleep(0.05)
    assert habit_id, 'Habit not detected'
    assert bucket

    # 3. decide suggestion
    feats = _build_features_from_bucket(bucket)
    decision_resp = client.post('/api/v1/habit/decide', json={'features':feats})
    assert decision_resp.status_code == 200
    decision = decision_resp.json()['decision']
    assert decision and decision['mode'] == 'suggest'

    # 4. feedback accept multiple times to promote
    for _ in range(5):
        d = client.post('/api/v1/habit/decide', json={'features':feats}).json().get('decision')
        if not d:
            continue
        fb = client.post('/api/v1/habit/feedback', params={'decision_id': d['id'], 'outcome':'accept'})
        assert fb.status_code == 200

    # refresh habit
    h = client.get('/api/v1/habit/habits').json()['habits'][0]
    assert h['stats']['successes'] >= 1

    # 5. telemetry
    tele = client.get('/api/v1/habit/telemetry')
    assert tele.status_code == 200
    assert tele.json()['habits']

    # 6. export/import
    export = client.get('/api/v1/habit/export').json()
    assert export['habits']
    # modify mode in export to suggest
    export['habits'][0]['mode'] = 'suggest'
    imp = client.post('/api/v1/habit/import', json={'data': export, 'merge': True})
    assert imp.status_code == 200

    # 7. execute decision & undo
    d2 = client.post('/api/v1/habit/decide', json={'features':feats}).json().get('decision')
    if d2:
        ex = client.post('/api/v1/habit/execute', json={'decision_id': d2['id']})
        assert ex.status_code == 200
        fb2 = client.post('/api/v1/habit/feedback', params={'decision_id': d2['id'], 'outcome':'undo'})
        assert fb2.status_code == 200

    # 8. delete habit
    delr = client.delete(f'/api/v1/habit/{habit_id}')
    assert delr.status_code == 200
    habits_after = client.get('/api/v1/habit/habits').json()['habits']
    assert not habits_after
