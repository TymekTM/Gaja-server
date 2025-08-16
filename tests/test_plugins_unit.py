"""Jednostkowe testy wszystkich modułów pluginów wywoływanych przez AI.

Cel:
 1. Sprawdzenie że każdy plugin posiada `get_functions()` zwracające listę opisów.
 2. Próba wykonania każdej funkcji poprzez `execute_function` z minimalnym zestawem parametrów.
 3. Wymuszenie trybu testowego (test_mode=True) gdzie to możliwe, aby uniknąć działań ubocznych
    (otwieranie przeglądarki, sterowanie multimediami, realne API / sieć itp.).

UWAGA:
 - Niektóre funkcje (np. API external) nie mają parametru test_mode – używamy bezpiecznego URL.
 - Jeśli funkcja wymaga dodatkowych danych których nie potrafimy odtworzyć, oznaczamy ją jako pominiętą.
 - Test ma charakter smoke: gwarantuje że ścieżki kodu się nie wywalają, nie ocenia jakości treści.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
from datetime import datetime, timedelta

import pytest

MODULES_DIR = Path(__file__).resolve().parent.parent / "modules"
USER_ID = 2

# Nazwy modułów do pominięcia (np. sam AI moduł / monitorujące / wymagające środowiska specjalnego)
SKIP_MODULES = {
    "ai_module",  # orchestration not a tool provider for tests
    "plugin_monitor_module",  # obserwacja, brak sensu w bezpośrednim execute
    "server_performance_monitor",  # monitoring systemowy
    # Moduły, które nie expose'ują interfejsu get_functions/execute_function (helpery / procesy w tle)
    "active_window_module",
    "day_narrative_module",
    "day_summary_module",
    "memory_module",
    "proactive_assistant_module",
    "routines_learner_module",
    "user_behavior_module",
    # Tymczasowo pomijamy api_module – wymaga poprawnego user_id w DB (FOREIGN KEY). Dodamy osobny test integracyjny.
    "api_module",
}

# Domyślne wartości dla wymaganych pól parametrów
DEFAULT_PARAM_VALUES = {
    "duration": "5s",
    "label": "test",
    "title": "Test Event",
    "date": datetime.now().strftime("%Y-%m-%d"),
    "time": "12:00",
    "text": "Tekst testowy",
    "question": "Co doprecyzować?",
    "task": "Zadanie testowe",
    "task_id": 0,
    "list_name": "lista",
    "item": "element",
    "location": "Warszawa",
    "provider": "openweather",
    "days": 2,
    "query": "Test sztucznej inteligencji",
    "engine": "duckduckgo",
    "max_results": 3,
    "language": "pl",
    "action": "play",
    "platform": "auto",
    "method": "GET",
    "url": "https://example.com",
    "headers": {},
    "params": {},
    "json_data": {},
}


def _load_module(module_path: Path):
    name = module_path.stem
    spec = importlib.util.spec_from_file_location(f"modules.{name}", module_path)
    if not spec or not spec.loader:
        raise RuntimeError(f"Nie udało się utworzyć spec dla {name}")
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "modules"
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def _build_params(param_schema: dict) -> dict:
    if not isinstance(param_schema, dict):
        return {}
    props = param_schema.get("properties") or {}
    required = param_schema.get("required") or []
    out = {}
    for key in props.keys():
        # Uzupełnij wymagane lub jeśli mamy wartość domyślną w mapie
        if key in required or key in DEFAULT_PARAM_VALUES:
            if key in DEFAULT_PARAM_VALUES:
                out[key] = DEFAULT_PARAM_VALUES[key]
    # Wymuś test_mode jeśli dostępne
    if "test_mode" in props:
        out["test_mode"] = True
    # Specjalny przypadek: set_reminder w core_module wymaga ISO datetime w polu 'time'
    # Heurystyka: jeśli 'time' jest wymagane oraz występuje 'text' (ale brak 'title'), to traktujemy to jako reminder/time ISO
    if (
        "time" in required
        and "text" in required
        and "title" not in required
        and "time" in out
        and len(out["time"]) <= 5  # np. '12:00'
    ):
        # ustaw ISO teraz + 5 minut
        iso_time = (
            datetime.now() + timedelta(minutes=5)
        ).replace(second=0, microsecond=0).isoformat()
        out["time"] = iso_time
    return out


def discover_plugin_modules():
    for path in MODULES_DIR.glob("*_module.py"):
        name = path.stem
        if name in SKIP_MODULES:
            continue
        yield path


@pytest.mark.parametrize("module_path", list(discover_plugin_modules()))
def test_get_functions_schema(module_path: Path):
    mod = _load_module(module_path)
    assert hasattr(mod, "get_functions"), f"Brak get_functions w {module_path.name}"
    funcs = mod.get_functions()
    assert isinstance(funcs, list) and funcs, f"get_functions pusty w {module_path.name}"
    for f in funcs:
        assert "name" in f, f"Brak nazwy funkcji w {module_path.name}"
        assert "parameters" in f, f"Brak schematu parametrów w {module_path.name}"


@pytest.mark.asyncio
@pytest.mark.parametrize("module_path", list(discover_plugin_modules()))
async def test_execute_functions_smoke(module_path: Path):
    mod = _load_module(module_path)
    if not hasattr(mod, "get_functions") or not hasattr(mod, "execute_function"):
        pytest.skip(f"Moduł {module_path.stem} nie ma standardowego interfejsu")
    # Specjalna inicjalizacja dla core_module (naprawa potencjalnie uszkodzonego pliku storage)
    if module_path.stem == "core_module":  # pragma: no cover - setup specyficzny
        try:
            storage_path = getattr(mod, "STORAGE_FILE", None)
            if storage_path:
                # Nadpisz plik poprawną strukturą jeśli niepoprawny
                needs_init = True
                if Path(storage_path).exists():
                    try:
                        with open(storage_path, "r", encoding="utf-8") as f:
                            json.load(f)
                            needs_init = False
                    except Exception:
                        needs_init = True
                if needs_init:
                    with open(storage_path, "w", encoding="utf-8") as f:
                        json.dump(
                            {
                                "timers": [],
                                "events": [],
                                "reminders": [],
                                "shopping_list": [],
                                "tasks": [],
                                "lists": {},
                            },
                            f,
                        )
        except Exception:
            pass
    funcs = mod.get_functions()
    failures: list[str] = []
    for func in funcs:
        fname = func.get("name")
        if not fname:
            continue
        param_schema = func.get("parameters", {})
        params = _build_params(param_schema)
        try:
            result = await mod.execute_function(fname, params, USER_ID)  # type: ignore[attr-defined]
            if not isinstance(result, dict) or not result.get("success", False):
                failures.append(f"{fname}: result={result}")
        except Exception as e:  # pragma: no cover - diagnostyka
            failures.append(f"{fname}: exception={e}")
    assert not failures, f"Niepowodzenia funkcji w {module_path.stem}: {failures}"
