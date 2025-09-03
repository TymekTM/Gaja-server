import importlib.util
from pathlib import Path
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
MODULES_DIR = BASE_DIR / "modules"
USER_ID = 2

def _load_module(path: Path):
    name = path.stem
    spec = importlib.util.spec_from_file_location(f"modules.{name}", path)
    if not spec or not spec.loader:
        raise RuntimeError(f"Spec load failed for {name}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.asyncio
async def test_get_weather_unified_structure():
    path = MODULES_DIR / "weather_module.py"
    mod = _load_module(path)
    result = await mod.execute_function(
        "get_weather", {"location": "Warszawa", "test_mode": True}, USER_ID
    )
    assert result.get("success") is True, result
    data = result.get("data")
    assert isinstance(data, dict)
    assert "current" in data and isinstance(data["current"], dict)
    assert "temperature" in data["current"], "current.temperature missing"
    assert "forecast" in data and isinstance(data["forecast"], list)
    assert data["forecast"], "forecast list empty"
    first = data["forecast"][0]
    assert "min_temp" in first and "max_temp" in first, "min/max temp missing in forecast[0]"


@pytest.mark.asyncio
async def test_get_forecast_unified_structure():
    path = MODULES_DIR / "weather_module.py"
    mod = _load_module(path)
    result = await mod.execute_function(
        "get_forecast", {"location": "Krakow", "days": 2, "test_mode": True}, USER_ID
    )
    assert result.get("success") is True, result
    data = result.get("data")
    assert isinstance(data, dict)
    assert "forecast" in data and isinstance(data["forecast"], list)
    assert data["forecast"], "forecast list empty"
    first = data["forecast"][0]
    assert "min_temp" in first and "max_temp" in first, "Unified forecast day missing min/max temp"
    # current may be None by design for pure forecast
    assert "current" in data, "Unified forecast should contain 'current' key (can be None)"
