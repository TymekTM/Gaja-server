"""Plugin sterujący lokalnym API Daikin."""

from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, NamedTuple, Optional

import httpx

from core.base_module import (
    BaseModule,
    TestModeSupport,
    create_standard_function_schema,
)


PLUGIN_NAME = "daikin_module"
PLUGIN_DESCRIPTION = "Steruje klimatyzacją Daikin przez lokalne API proxy."
PLUGIN_VERSION = "0.1.0"
PLUGIN_AUTHOR = "GAJA Team"
PLUGIN_DEPENDENCIES: list[str] = ["httpx"]


class _RequestResult(NamedTuple):
    ok: bool
    data: Any
    status: int


class DaikinModule(BaseModule, TestModeSupport):
    """Lekki wrapper na lokalny serwer Daikin (LAN WebUI)."""

    def __init__(self) -> None:
        super().__init__("daikin")
        self.base_url = os.getenv("DAIKIN_API_BASE_URL", "http://192.168.0.107:6969").rstrip("/")
        self._client: Optional[httpx.AsyncClient] = None
        self._client_lock = asyncio.Lock()

    async def _initialize_impl(self) -> None:
        if self._client is None:
            auth = None
            user = os.getenv("DAIKIN_API_USER")
            password = os.getenv("DAIKIN_API_PASS")
            if user and password:
                auth = httpx.BasicAuth(user, password)
            self._client = httpx.AsyncClient(timeout=10.0, auth=auth)

    async def _ensure_client(self) -> httpx.AsyncClient:
        async with self._client_lock:
            if self._client is None:
                await self.initialize()
        assert self._client is not None
        return self._client

    def get_function_schemas(self):  # type: ignore[override]
        return [
            create_standard_function_schema(
                name="discover_devices",
                description="Skanuj sieć w poszukiwaniu jednostek.",
                properties={},
                required=[],
            ),
            create_standard_function_schema(
                name="get_device_state",
                description="Pobierz stan urządzenia.",
                properties={
                    "ip": {"type": "string", "description": "Adres IP"},
                },
                required=["ip"],
            ),
            create_standard_function_schema(
                name="set_power_state",
                description="Włącz lub wyłącz urządzenie.",
                properties={
                    "ip": {"type": "string", "description": "Adres IP"},
                    "power": {
                        "type": "integer",
                        "description": "1 = ON, 0 = OFF",
                        "enum": [0, 1],
                    },
                },
                required=["ip", "power"],
            ),
            create_standard_function_schema(
                name="set_temperature",
                description="Ustaw temperaturę docelową.",
                properties={
                    "ip": {"type": "string", "description": "Adres IP"},
                    "temperature": {
                        "type": "number",
                        "description": "Zakres 18-32°C",
                        "minimum": 18.0,
                        "maximum": 32.0,
                    },
                },
                required=["ip", "temperature"],
            ),
            create_standard_function_schema(
                name="set_mode",
                description="Zmień tryb pracy.",
                properties={
                    "ip": {"type": "string", "description": "Adres IP"},
                    "mode": {
                        "type": "string",
                        "description": "AUTO/DRY/COOL/HEAT/FAN lub kod liczbowy.",
                    },
                    "temperature": {
                        "type": "number",
                        "description": "Temp. wymagana przy przejściu z FAN.",
                        "minimum": 18.0,
                        "maximum": 32.0,
                    },
                },
                required=["ip", "mode"],
            ),
            create_standard_function_schema(
                name="set_fan_rate",
                description="Ustaw prędkość wentylatora.",
                properties={
                    "ip": {"type": "string", "description": "Adres IP"},
                    "fan_rate": {
                        "type": "string",
                        "description": "A (auto), B (cichy) lub 3-7.",
                    },
                },
                required=["ip", "fan_rate"],
            ),
            create_standard_function_schema(
                name="set_fan_direction",
                description="Ustaw kierunek nadmuchu.",
                properties={
                    "ip": {"type": "string", "description": "Adres IP"},
                    "fan_dir": {
                        "type": "string",
                        "description": "0 stop, 1 pion, 2 poziom, 3 oba.",
                    },
                },
                required=["ip", "fan_dir"],
            ),
            create_standard_function_schema(
                name="set_humidity",
                description="Ustaw docelową wilgotność.",
                properties={
                    "ip": {"type": "string", "description": "Adres IP"},
                    "humidity": {
                        "type": "integer",
                        "description": "Poziom wilgotności (np. 50).",
                    },
                },
                required=["ip", "humidity"],
            ),
        ]

    async def execute_function(self, function_name: str, parameters: Dict[str, Any], user_id: str):  # type: ignore[override]
        handler = getattr(self, f"_{function_name}", None)
        if not handler:
            return self._create_error_response(f"Unknown function {function_name}")
        return await handler(parameters)

    def get_mock_data(self, function_name: str, parameters: Dict[str, Any]):  # type: ignore[override]
        stub_ip = parameters.get("ip", "192.168.1.100")
        if function_name == "discover_devices":
            return self._create_success_response(
                {"found": 1, "devices": [stub_ip]},
                message="Mock discovery",
                test_mode=True,
            )
        if function_name == "get_device_state":
            return self._create_success_response(
                {
                    "available": True,
                    "ip": stub_ip,
                    "mode": "COOL",
                    "pow": 1,
                    "stemp": "23.0",
                },
                message="Mock state",
                test_mode=True,
            )
        return self._create_success_response({"ip": stub_ip}, message="Mock", test_mode=True)

    async def _discover_devices(self, _: Dict[str, Any]):
        ok, data, status = await self._request("GET", "/api/devices/discover")
        if ok:
            return self._create_success_response(data, message="Discovery ok")
        return self._error_from_response(data, status)

    async def _get_device_state(self, params: Dict[str, Any]):
        ip = self._require_ip(params)
        if not ip:
            return self._create_error_response("Missing ip")
        ok, data, status = await self._request("GET", f"/api/device/{ip}/state")
        if ok:
            return self._create_success_response(data, message=f"State {ip}")
        return self._error_from_response(data, status)

    async def _set_power_state(self, params: Dict[str, Any]):
        ip = self._require_ip(params)
        if ip is None:
            return self._create_error_response("Missing ip")
        power = params.get("power")
        if power not in (0, 1, "0", "1", False, True):
            return self._create_error_response("power must be 0/1")
        payload = {"pow": 1 if str(power) in {"1", "True", "true"} or power is True else 0}
        ok, data, status = await self._request("POST", f"/api/device/{ip}/power", json=payload)
        if ok:
            return self._create_success_response(data, message=f"Power {payload['pow']}")
        return self._error_from_response(data, status)

    async def _set_temperature(self, params: Dict[str, Any]):
        ip = self._require_ip(params)
        if ip is None:
            return self._create_error_response("Missing ip")
        temperature = params.get("temperature")
        if temperature is None:
            return self._create_error_response("Missing temperature")
        try:
            temp_value = float(temperature)
        except (TypeError, ValueError):
            return self._create_error_response("temperature must be number")
        if not 18.0 <= temp_value <= 32.0:
            return self._create_error_response("temperature out of range")
        payload = {"stemp": f"{temp_value:.1f}"}
        ok, data, status = await self._request("POST", f"/api/device/{ip}/temp", json=payload)
        if ok:
            return self._create_success_response(data, message=f"Temp {payload['stemp']}")
        return self._error_from_response(data, status)

    async def _set_mode(self, params: Dict[str, Any]):
        ip = self._require_ip(params)
        if ip is None:
            return self._create_error_response("Missing ip")
        mode = params.get("mode")
        if not mode:
            return self._create_error_response("Missing mode")
        mode_value = str(mode).upper()
        payload: Dict[str, Any] = {"mode": mode_value}
        if "temperature" in params and params["temperature"] is not None:
            try:
                temp_value = float(params["temperature"])
            except (TypeError, ValueError):
                return self._create_error_response("temperature must be number")
            payload["stemp"] = f"{temp_value:.1f}"
        ok, data, status = await self._request("POST", f"/api/device/{ip}/mode", json=payload)
        if ok:
            return self._create_success_response(data, message=f"Mode {mode_value}")
        return self._error_from_response(data, status)

    async def _set_fan_rate(self, params: Dict[str, Any]):
        ip = self._require_ip(params)
        if ip is None:
            return self._create_error_response("Missing ip")
        rate = params.get("fan_rate")
        if not rate:
            return self._create_error_response("Missing fan_rate")
        payload = {"f_rate": str(rate).upper()}
        ok, data, status = await self._request("POST", f"/api/device/{ip}/fan_rate", json=payload)
        if ok:
            return self._create_success_response(data, message=f"Fan {payload['f_rate']}")
        return self._error_from_response(data, status)

    async def _set_fan_direction(self, params: Dict[str, Any]):
        ip = self._require_ip(params)
        if ip is None:
            return self._create_error_response("Missing ip")
        direction = params.get("fan_dir")
        if direction is None:
            return self._create_error_response("Missing fan_dir")
        payload = {"f_dir": str(direction)}
        ok, data, status = await self._request("POST", f"/api/device/{ip}/fan_dir", json=payload)
        if ok:
            return self._create_success_response(data, message=f"Dir {payload['f_dir']}")
        return self._error_from_response(data, status)

    async def _set_humidity(self, params: Dict[str, Any]):
        ip = self._require_ip(params)
        if ip is None:
            return self._create_error_response("Missing ip")
        humidity = params.get("humidity")
        if humidity is None:
            return self._create_error_response("Missing humidity")
        try:
            humidity_value = int(humidity)
        except (TypeError, ValueError):
            return self._create_error_response("humidity must be int")
        payload = {"shum": str(humidity_value)}
        ok, data, status = await self._request("POST", f"/api/device/{ip}/humidity", json=payload)
        if ok:
            return self._create_success_response(data, message=f"Humidity {payload['shum']}")
        return self._error_from_response(data, status)

    async def _request(self, method: str, path: str, *, json: Optional[Dict[str, Any]] = None) -> _RequestResult:
        client = await self._ensure_client()
        url = f"{self.base_url}{path}"
        try:
            response = await client.request(method, url, json=json)
        except httpx.HTTPError as exc:
            return _RequestResult(False, {"detail": str(exc)}, 0)
        status = response.status_code
        try:
            payload = response.json()
        except ValueError:
            payload = {"detail": response.text}
        if status >= 400:
            return _RequestResult(False, payload, status)
        return _RequestResult(True, payload, status)

    @staticmethod
    def _require_ip(params: Dict[str, Any]) -> Optional[str]:
        ip = params.get("ip")
        if not ip:
            return None
        return str(ip)

    def _error_from_response(self, data: Any, status: int):
        if isinstance(data, dict):
            detail = data.get("detail") or data.get("message") or str(data)
        else:
            detail = str(data)
        return self._create_error_response(detail, code=str(status or "request_error"))


_daikin_module_instance: Optional[DaikinModule] = None


def _get_module() -> DaikinModule:
    global _daikin_module_instance
    if _daikin_module_instance is None:
        _daikin_module_instance = DaikinModule()
    return _daikin_module_instance


def get_functions():
    return _get_module().get_functions()


async def execute_function(function_name: str, parameters: Dict[str, Any], user_id: str):
    module = _get_module()
    if not module._initialized:
        await module.initialize()
    return await module.safe_execute(function_name, parameters, user_id)
