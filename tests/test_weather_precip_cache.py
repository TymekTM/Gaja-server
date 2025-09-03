import importlib.util
from pathlib import Path
import pytest
import types
from datetime import datetime, timedelta

BASE_DIR = Path(__file__).resolve().parent.parent
MODULES_DIR = BASE_DIR / "modules"
USER_ID = 2

def _load_module(name: str):
    path = MODULES_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"modules.{name}", path)
    if not spec or not spec.loader:
        raise RuntimeError(f"Spec load failed for {name}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

@pytest.mark.asyncio
async def test_mock_precipitation_chance_range():
    weather_mod_module = _load_module("weather_module")
    result = await weather_mod_module.execute_function(
        "get_weather", {"location": "Warszawa", "test_mode": True}, USER_ID
    )
    assert result.get("success"), result
    curr = result["data"]["current"]
    pc = curr.get("precipitation_chance")
    assert isinstance(pc, (int, float))
    assert 0 <= pc <= 100

@pytest.mark.asyncio
async def test_cache_prevents_second_call(monkeypatch):
    weather_mod_module = _load_module("weather_module")
    weather_mod = weather_mod_module.weather_module  # instance

    calls = {"count": 0}

    async def fake_provider(user_id, location, api_key):
        calls["count"] += 1
        return {
            "location": f"{location}, Polska",
            "temperature": 20,
            "feels_like": 20,
            "humidity": 50,
            "pressure": 1000,
            "description": "Pochmurnie",
            "wind_speed": 2,
            "wind_direction": "N",
            "visibility": 10,
            "uv_index": 3,
            "precipitation_chance": None,
            "cloud_cover": 80,
            "timestamp": datetime.now().isoformat(),
        }

    # Podmieniamy provider i klucz API + cache_duration skracamy do 10s
    weather_mod.weather_providers["weatherapi"] = fake_provider
    weather_mod.cache_duration = timedelta(seconds=10)

    # Monkeypatch _get_user_api_key aby nie zależeć od DB
    async def fake_key(user, provider):
        return "dummy"
    weather_mod._get_user_api_key = fake_key  # type: ignore

    # pierwsze wywołanie -> 1 zapytanie
    r1 = await weather_mod.execute_function("get_weather", {"location": "Lodz"}, USER_ID)
    assert r1.get("success"), r1
    assert calls["count"] == 1

    # drugie wywołanie (ten sam location) w czasie < cache -> brak nowego zapytania
    r2 = await weather_mod.execute_function("get_weather", {"location": "Lodz"}, USER_ID)
    assert r2.get("success")
    assert calls["count"] == 1, "Cache nie zadziałał - provider wywołany ponownie"

    # Invalidate czasowo: przesuwamy timestamp w cache wstecz aby minęła ważność
    for key, entry in list(weather_mod.weather_cache.items()):
        data, ts, meta = entry
        weather_mod.weather_cache[key] = (data, ts - timedelta(seconds=3600), meta)

    r3 = await weather_mod.execute_function("get_weather", {"location": "Lodz"}, USER_ID)
    assert r3.get("success")
    assert calls["count"] == 2, "Po wygaśnięciu cache powinno być nowe wywołanie"

@pytest.mark.asyncio
async def test_precipitation_fallback_when_none(monkeypatch):
    weather_mod_module = _load_module("weather_module")
    weather_mod = weather_mod_module.weather_module

    async def fake_provider(user_id, location, api_key):
        return {
            "location": f"{location}, Polska",
            "temperature": 21,
            "feels_like": 21,
            "humidity": 40,
            "pressure": 1005,
            "description": "Zachmurzenie duze",  # brak słowa deszcz ale wysokie chmury
            "wind_speed": 3,
            "wind_direction": "N",
            "visibility": 10,
            "uv_index": 3,
            "precipitation_chance": None,
            "cloud_cover": 95,
            "timestamp": datetime.now().isoformat(),
        }

    weather_mod.weather_providers["weatherapi"] = fake_provider
    weather_mod.cache_duration = timedelta(seconds=0)  # zawsze odświeżaj

    async def fake_key(user, provider):
        return "dummy"
    weather_mod._get_user_api_key = fake_key  # type: ignore

    res = await weather_mod.execute_function("get_weather", {"location": "Gdansk"}, USER_ID)
    assert res.get("success")
    pc = res["data"]["current"].get("precipitation_chance")
    assert pc is not None and 0 <= pc <= 100
    # Przy 95% chmur heurystyka powinna dać co najmniej 55
    assert pc >= 55
