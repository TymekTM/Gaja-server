from datetime import date, datetime
from unittest.mock import patch

from zoneinfo import ZoneInfo

import httpx

from integrations.telegram.service import TelegramBotService, TelegramConfig


class _DummyServer:
    pass


def _make_service(group: str = "1") -> TelegramBotService:
    config = TelegramConfig(enabled=True, timetable_group=group)
    return TelegramBotService(_DummyServer(), config)


def test_parse_timetable_filters_group_and_extracts_fields():
    service = _make_service()
    ics_text = """BEGIN:VCALENDAR\nBEGIN:VEVENT\nDTSTART;TZID=Europe/Warsaw:20241003T080000\nDTEND;TZID=Europe/Warsaw:20241003T093000\nSUMMARY=Analiza Matematyczna [grupa 1]\nDESCRIPTION=Prowadzący: dr Jan Kowalski\\nGrupa: 1\nLOCATION=Aula 101\nEND:VEVENT\nBEGIN:VEVENT\nDTSTART;TZID=Europe/Warsaw:20241003T100000\nDTEND;TZID=Europe/Warsaw:20241003T113000\nSUMMARY=Fizyka (grupa 2)\nDESCRIPTION=Prowadząca: dr Anna Nowak\\nGrupa: 2\nLOCATION=Aula 102\nEND:VEVENT\nEND:VCALENDAR\n"""

    result = service._parse_timetable_by_date(ics_text, "1")
    target_date = date(2024, 10, 3)
    assert target_date in result

    entries = result[target_date]
    assert len(entries) == 1

    entry = entries[0]
    assert entry.subject == "Analiza Matematyczna [grupa 1]"
    assert entry.location == "Aula 101"
    assert entry.teacher == "dr Jan Kowalski"
    assert entry.group_label == "gr. 1"


def test_parse_timetable_keeps_entries_without_group_hint():
    service = _make_service()
    ics_text = """BEGIN:VCALENDAR\nBEGIN:VEVENT\nDTSTART;TZID=Europe/Warsaw:20241004T120000\nDTEND;TZID=Europe/Warsaw:20241004T133000\nSUMMARY=Laboratorium Automatyki\nDESCRIPTION=Prowadzący: mgr Piotr Zieliński\\n\nLOCATION=Lab 201\nEND:VEVENT\nEND:VCALENDAR\n"""

    result = service._parse_timetable_by_date(ics_text, "1")
    target_date = date(2024, 10, 4)
    assert target_date in result
    entries = result[target_date]
    assert len(entries) == 1


def test_detect_day_selector_variants():
    service = _make_service()

    assert service._detect_day_selector("Plan na jutro") == "tomorrow"
    assert service._detect_day_selector("Jaki plan lekcji dzisiaj?") == "today"
    assert service._detect_day_selector("Hej, jaki mam plan lekcji?") == "tomorrow"
    assert service._detect_day_selector("Czy zaplanujesz jutro zadania?") is None


def test_detect_day_selector_polish_weekday_command_arg():
    service = _make_service()

    assert service._detect_day_selector("poniedziałek", allow_without_keyword=True) == "weekday:0"
    assert service._detect_day_selector("piątek", allow_without_keyword=True) == "weekday:4"
    assert service._detect_day_selector("Plan na wtorek") == "weekday:1"


def test_apply_timetable_fallback_sequence():
    service = _make_service()

    assert service._timetable_use_legacy_tls is False
    assert service._timetable_use_http_fallback is False

    first = service._apply_timetable_connect_fallback(httpx.ConnectError("fail"))
    assert first is True
    assert service._timetable_use_legacy_tls is True
    assert service._timetable_use_http_fallback is False

    second = service._apply_timetable_connect_fallback(httpx.ConnectError("fail again"))
    assert second is True
    assert service._timetable_use_http_fallback is True

    third = service._apply_timetable_connect_fallback(httpx.ConnectError("still failing"))
    assert third is False


def test_resolve_timetable_request_url_switches_to_http():
    service = _make_service()

    assert service._resolve_timetable_request_url().startswith("https://")

    service._timetable_use_http_fallback = True
    resolved = service._resolve_timetable_request_url()
    assert resolved.startswith("http://")
    assert "plan.polsl.pl" in resolved


def test_resolve_target_date_for_polish_weekday():
    service = _make_service()
    service._timetable_zone = ZoneInfo("Europe/Warsaw")

    reference = datetime(2025, 10, 3, 12, 0, tzinfo=ZoneInfo("Europe/Warsaw"))  # friday
    with patch.object(service, "_current_time", return_value=reference):
        monday_date, monday_label = service._resolve_target_date("weekday:0")
        friday_date, friday_label = service._resolve_target_date("weekday:4")

    assert monday_date == date(2025, 10, 6)
    assert monday_label == "poniedziałek"
    assert friday_date == date(2025, 10, 3)
    assert friday_label == "piątek"
