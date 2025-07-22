import logging
from datetime import datetime, timedelta
from typing import Any

from config_manager import get_database_manager

from .api_module import get_api_module

logger = logging.getLogger(__name__)


# Required plugin functions
def get_functions() -> list[dict[str, Any]]:
    """Zwraca listę dostępnych funkcji w pluginie."""
    return [
        {
            "name": "get_weather",
            "description": "Pobiera dane pogodowe dla określonej lokalizacji",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "Lokalizacja (miasto, kraj)",
                    },
                    "provider": {
                        "type": "string",
                        "description": "Provider pogodowy (openweather, weatherapi)",
                        "enum": ["openweather", "weatherapi"],
                        "default": "openweather",
                    },
                    "test_mode": {
                        "type": "boolean",
                        "description": "Tryb testowy (używa mock danych)",
                        "default": False,
                    },
                },
                "required": ["location"],
            },
        },
        {
            "name": "get_forecast",
            "description": "Pobiera prognozę pogody na kilka dni",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "Lokalizacja (miasto, kraj)",
                    },
                    "days": {
                        "type": "integer",
                        "description": "Liczba dni prognozy",
                        "default": 3,
                        "minimum": 1,
                        "maximum": 7,
                    },
                    "test_mode": {
                        "type": "boolean",
                        "description": "Tryb testowy (używa mock danych)",
                        "default": False,
                    },
                },
                "required": ["location"],
            },
        },
    ]


async def execute_function(
    function_name: str, parameters: dict[str, Any], user_id: int
) -> dict[str, Any]:
    """Wykonuje funkcję pluginu."""
    weather_module = WeatherModule()
    await weather_module.initialize()

    try:
        if function_name == "get_weather":
            location = parameters.get("location")
            provider = parameters.get("provider", "openweather")
            test_mode = parameters.get("test_mode", False)

            # Sprawdź czy jest tryb testowy lub brak klucza API
            api_key = await weather_module._get_user_api_key(user_id, provider)
            if not api_key or test_mode:
                # Zwróć mock dane
                mock_data = weather_module._get_mock_weather_data(location)
                return {
                    "success": True,
                    "data": mock_data,
                    "message": f"Pobrano dane pogodowe dla {location} (tryb testowy)",
                    "test_mode": True,
                }

            result = await weather_module.get_weather(
                user_id, location, api_key, provider
            )
            return {
                "success": True,
                "data": result,
                "message": f"Pobrano dane pogodowe dla {location}",
            }

        elif function_name == "get_forecast":
            location = parameters.get("location")
            days = parameters.get("days", 3)
            test_mode = parameters.get("test_mode", False)

            # Sprawdź czy jest tryb testowy lub brak klucza API
            api_key = await weather_module._get_user_api_key(user_id, "openweather")
            if not api_key or test_mode:
                # Zwróć mock dane
                mock_data = weather_module._get_mock_forecast_data(location, days)
                return {
                    "success": True,
                    "data": mock_data,
                    "message": f"Pobrano prognozę pogody dla {location} na {days} dni (tryb testowy)",
                    "test_mode": True,
                }

            result = await weather_module.get_forecast(user_id, location, api_key, days)
            return {
                "success": True,
                "data": result,
                "message": f"Pobrano prognozę na {days} dni dla {location}",
            }

        else:
            return {"success": False, "error": f"Unknown function: {function_name}"}

    except Exception as e:
        logger.error(f"Error executing {function_name}: {e}")
        return {"success": False, "error": str(e)}


class WeatherModule:
    """Moduł pogodowy dla asystenta."""

    def __init__(self):
        self.api_module = None
        self.db_manager = get_database_manager()
        self.weather_providers = {
            "openweather": self._get_openweather_data,
            "weatherapi": self._get_weatherapi_data,
        }

        # Cache pogody (żeby nie wykonywać zbyt wielu zapytań)
        self.weather_cache = {}
        self.cache_duration = timedelta(minutes=30)

    async def initialize(self):
        """Inicjalizuje moduł pogodowy."""
        self.api_module = await get_api_module()
        logger.info("WeatherModule initialized")

    async def get_weather(
        self, user_id: int, location: str, api_key: str, provider: str = "openweather"
    ) -> dict[str, Any]:
        """Pobiera dane pogodowe dla lokalizacji.

        Args:
            user_id: ID użytkownika
            location: Lokalizacja (miasto, kraj)
            api_key: Klucz API
            provider: Provider pogodowy

        Returns:
            Dane pogodowe
        """
        if not self.api_module:
            await self.initialize()

        # Sprawdź cache
        cache_key = f"{provider}_{location.lower()}"
        if cache_key in self.weather_cache:
            cached_data, timestamp = self.weather_cache[cache_key]
            if datetime.now() - timestamp < self.cache_duration:
                logger.info(f"Returning cached weather data for {location}")
                return cached_data

        if provider not in self.weather_providers:
            return {
                "error": f"Nieobsługiwany provider pogodowy: {provider}",
                "available_providers": list(self.weather_providers.keys()),
            }

        try:
            weather_func = self.weather_providers[provider]
            weather_data = await weather_func(user_id, location, api_key)

            # Cache wyników
            if "error" not in weather_data:
                self.weather_cache[cache_key] = (weather_data, datetime.now())

            return weather_data

        except Exception as e:
            logger.error(f"Weather error with {provider}: {e}")
            return {
                "error": f"Błąd pobierania pogody: {str(e)}",
                "location": location,
                "provider": provider,
            }

    async def _get_openweather_data(
        self, user_id: int, location: str, api_key: str
    ) -> dict[str, Any]:
        """Pobiera dane z OpenWeatherMap API."""

        # Aktualna pogoda
        current_url = "https://api.openweathermap.org/data/2.5/weather"
        current_params = {
            "q": location,
            "appid": api_key,
            "units": "metric",
            "lang": "pl",
        }

        current_response = await self.api_module.get(
            user_id, current_url, params=current_params
        )

        if current_response.get("status") != 200:
            return {
                "error": "Błąd pobierania aktualnej pogody",
                "details": current_response,
                "provider": "openweather",
            }

        current_data = current_response.get("data", {})

        # Prognoza 5-dniowa
        forecast_url = "https://api.openweathermap.org/data/2.5/forecast"
        forecast_params = {
            "q": location,
            "appid": api_key,
            "units": "metric",
            "lang": "pl",
        }

        forecast_response = await self.api_module.get(
            user_id, forecast_url, params=forecast_params
        )
        forecast_data = (
            forecast_response.get("data", {})
            if forecast_response.get("status") == 200
            else {}
        )

        # Przetwórz dane
        weather_info = self._process_openweather_data(
            current_data, forecast_data, location
        )
        weather_info["provider"] = "openweather"
        weather_info["timestamp"] = datetime.now().isoformat()

        return weather_info

    def _process_openweather_data(
        self, current_data: dict, forecast_data: dict, location: str
    ) -> dict[str, Any]:
        """Przetwarza dane z OpenWeatherMap."""

        # Podstawowe informacje
        main = current_data.get("main", {})
        weather = current_data.get("weather", [{}])[0]
        wind = current_data.get("wind", {})
        clouds = current_data.get("clouds", {})
        sys = current_data.get("sys", {})
        coord = current_data.get("coord", {})

        # Aktualna pogoda
        current_weather = {
            "temperature": round(float(main.get("temp"), 0)),
            "feels_like": round(float(main.get("feels_like"), 0)),
            "humidity": main.get("humidity", 0),
            "pressure": main.get("pressure", 0),
            "description": weather.get("description", "").capitalize(),
            "icon": weather.get("icon", ""),
            "wind_speed": wind.get("speed", 0),
            "wind_direction": wind.get("deg", 0),
            "cloudiness": clouds.get("all", 0),
            "visibility": (
                current_data.get("visibility", 0) // 1000
                if current_data.get("visibility")
                else None
            ),
            "sunrise": (
                datetime.fromtimestamp(sys.get("sunrise", 0)).strftime("%H:%M")
                if sys.get("sunrise")
                else None
            ),
            "sunset": (
                datetime.fromtimestamp(sys.get("sunset", 0)).strftime("%H:%M")
                if sys.get("sunset")
                else None
            ),
        }

        # Lokalizacja
        location_info = {
            "name": current_data.get("name", location),
            "country": sys.get("country", ""),
            "latitude": coord.get("lat"),
            "longitude": coord.get("lon"),
        }

        # Prognoza
        forecast = []
        if forecast_data.get("list"):
            # Grupuj prognozy po dniach
            daily_forecasts = {}

            for item in forecast_data["list"]:
                dt = datetime.fromtimestamp(item["dt"])
                date_str = dt.strftime("%Y-%m-%d")

                if date_str not in daily_forecasts:
                    daily_forecasts[date_str] = {
                        "date": date_str,
                        "day_name": dt.strftime("%A"),
                        "temperatures": [],
                        "descriptions": [],
                        "humidity": [],
                        "wind_speed": [],
                        "hourly": [],
                    }

                # Dodaj dane godzinowe
                hourly_data = {
                    "time": dt.strftime("%H:%M"),
                    "temperature": round(item["main"]["temp"]),
                    "description": item["weather"][0]["description"],
                    "icon": item["weather"][0]["icon"],
                    "humidity": item["main"]["humidity"],
                    "wind_speed": item["wind"]["speed"],
                }

                daily_forecasts[date_str]["hourly"].append(hourly_data)
                daily_forecasts[date_str]["temperatures"].append(item["main"]["temp"])
                daily_forecasts[date_str]["descriptions"].append(
                    item["weather"][0]["description"]
                )
                daily_forecasts[date_str]["humidity"].append(item["main"]["humidity"])
                daily_forecasts[date_str]["wind_speed"].append(item["wind"]["speed"])

            # Przetwórz dzienne podsumowania
            for date_str, day_data in daily_forecasts.items():
                temps = day_data["temperatures"]
                forecast.append(
                    {
                        "date": date_str,
                        "day_name": day_data["day_name"],
                        "min_temp": round(min(temps)) if temps else 0,
                        "max_temp": round(max(temps)) if temps else 0,
                        "avg_temp": round(sum(1 for x in temps if x) / len(temps))
                        if temps
                        else 0,
                        "description": (
                            max(
                                set(day_data["descriptions"]),
                                key=day_data["descriptions"].count,
                            )
                            if day_data["descriptions"]
                            else ""
                        ),
                        "avg_humidity": (
                            round(
                                sum(1 for x in day_data["humidity"] if x)
                                / len(day_data["humidity"])
                            )
                            if day_data["humidity"]
                            else 0
                        ),
                        "avg_wind_speed": (
                            round(
                                sum(1 for x in day_data["wind_speed"] if x)
                                / len(day_data["wind_speed"]),
                                1,
                            )
                            if day_data["wind_speed"]
                            else 0
                        ),
                        "hourly": day_data["hourly"][
                            :8
                        ],  # Tylko pierwsze 8 godzin na dzień
                    }
                )

        return {
            "location": location_info,
            "current": current_weather,
            "forecast": forecast[:5],  # 5 dni
            "units": {
                "temperature": "°C",
                "wind_speed": "m/s",
                "pressure": "hPa",
                "visibility": "km",
            },
        }

    async def _get_weatherapi_data(
        self, user_id: int, location: str, api_key: str
    ) -> dict[str, Any]:
        """Pobiera dane z WeatherAPI.com."""

        # Aktualna pogoda + prognoza 3-dniowa
        url = "http://api.weatherapi.com/v1/forecast.json"
        params = {
            "key": api_key,
            "q": location,
            "days": 5,
            "aqi": "yes",  # Jakość powietrza
            "alerts": "yes",  # Alerty pogodowe
            "lang": "pl",
        }

        response = await self.api_module.get(user_id, url, params=params)

        if response.get("status") != 200:
            return {
                "error": "Błąd pobierania danych pogodowych",
                "details": response,
                "provider": "weatherapi",
            }

        data = response.get("data", {})
        weather_info = self._process_weatherapi_data(data, location)
        weather_info["provider"] = "weatherapi"
        weather_info["timestamp"] = datetime.now().isoformat()

        return weather_info

    def _process_weatherapi_data(self, data: dict, location: str) -> dict[str, Any]:
        """Przetwarza dane z WeatherAPI."""

        current = data.get("current", {})
        location_data = data.get("location", {})
        forecast_data = data.get("forecast", {})

        # Aktualna pogoda
        current_weather = {
            "temperature": round(float(current.get("temp_c"), 0)),
            "feels_like": round(float(current.get("feelslike_c"), 0)),
            "humidity": current.get("humidity", 0),
            "pressure": current.get("pressure_mb", 0),
            "description": current.get("condition", {}).get("text", ""),
            "icon": current.get("condition", {}).get("icon", ""),
            "wind_speed": current.get("wind_kph", 0) / 3.6,  # Konwersja na m/s
            "wind_direction": current.get("wind_dir", ""),
            "cloudiness": current.get("cloud", 0),
            "visibility": current.get("vis_km", 0),
            "uv_index": current.get("uv", 0),
            "air_quality": self._process_air_quality(
                data.get("current", {}).get("air_quality", {})
            ),
        }

        # Lokalizacja
        location_info = {
            "name": location_data.get("name", location),
            "country": location_data.get("country", ""),
            "region": location_data.get("region", ""),
            "latitude": location_data.get("lat"),
            "longitude": location_data.get("lon"),
            "timezone": location_data.get("tz_id", ""),
            "local_time": location_data.get("localtime", ""),
        }

        # Prognoza
        forecast = []
        for day_data in forecast_data.get("forecastday", []):
            day = day_data.get("day", {})
            date_str = day_data.get("date", "")

            # Godzinowa prognoza
            hourly = []
            for hour_data in day_data.get("hour", []):
                hour_dt = datetime.fromisoformat(hour_data.get("time", ""))
                hourly.append(
                    {
                        "time": hour_dt.strftime("%H:%M"),
                        "temperature": round(float(hour_data.get("temp_c"), 0)),
                        "description": hour_data.get("condition", {}).get("text", ""),
                        "icon": hour_data.get("condition", {}).get("icon", ""),
                        "humidity": hour_data.get("humidity", 0),
                        "wind_speed": hour_data.get("wind_kph", 0) / 3.6,
                        "chance_of_rain": hour_data.get("chance_of_rain", 0),
                    }
                )

            forecast.append(
                {
                    "date": date_str,
                    "day_name": datetime.strptime(date_str, "%Y-%m-%d").strftime("%A"),
                    "min_temp": round(float(day.get("mintemp_c"), 0)),
                    "max_temp": round(float(day.get("maxtemp_c"), 0)),
                    "avg_temp": round(float(day.get("avgtemp_c"), 0)),
                    "description": day.get("condition", {}).get("text", ""),
                    "icon": day.get("condition", {}).get("icon", ""),
                    "max_wind_speed": day.get("maxwind_kph", 0) / 3.6,
                    "total_precipitation": day.get("totalprecip_mm", 0),
                    "avg_humidity": day.get("avghumidity", 0),
                    "chance_of_rain": day.get("daily_chance_of_rain", 0),
                    "chance_of_snow": day.get("daily_chance_of_snow", 0),
                    "uv_index": day.get("uv", 0),
                    "hourly": hourly,
                }
            )

        # Alerty pogodowe
        alerts = []
        for alert in data.get("alerts", {}).get("alert", []):
            alerts.append(
                {
                    "headline": alert.get("headline", ""),
                    "description": alert.get("desc", ""),
                    "severity": alert.get("severity", ""),
                    "urgency": alert.get("urgency", ""),
                    "areas": alert.get("areas", ""),
                    "category": alert.get("category", ""),
                    "certainty": alert.get("certainty", ""),
                    "event": alert.get("event", ""),
                    "effective": alert.get("effective", ""),
                    "expires": alert.get("expires", ""),
                }
            )

        return {
            "location": location_info,
            "current": current_weather,
            "forecast": forecast,
            "alerts": alerts,
            "units": {
                "temperature": "°C",
                "wind_speed": "m/s",
                "pressure": "mb",
                "visibility": "km",
                "precipitation": "mm",
            },
        }

    def _process_air_quality(self, aqi_data: dict) -> dict[str, Any]:
        """Przetwarza dane o jakości powietrza."""
        if not aqi_data:
            return {}

        return {
            "co": aqi_data.get("co", 0),
            "no2": aqi_data.get("no2", 0),
            "o3": aqi_data.get("o3", 0),
            "so2": aqi_data.get("so2", 0),
            "pm2_5": aqi_data.get("pm2_5", 0),
            "pm10": aqi_data.get("pm10", 0),
            "us_epa_index": aqi_data.get("us-epa-index", 0),
            "gb_defra_index": aqi_data.get("gb-defra-index", 0),
        }

    def format_weather_response(self, weather_data: dict[str, Any]) -> str:
        """Formatuje dane pogodowe do czytelnej odpowiedzi.

        Args:
            weather_data: Dane pogodowe

        Returns:
            Sformatowana odpowiedź tekstowa
        """
        if "error" in weather_data:
            return f"Przepraszam, nie mogłem pobrać danych pogodowych: {weather_data['error']}"

        current = weather_data.get("current", {})
        location = weather_data.get("location", {})
        forecast = weather_data.get("forecast", [])

        # Podstawowe informacje
        response_parts = [
            f"🌍 Pogoda dla: {location.get('name', 'Nieznana lokalizacja')}",
            f"🌡️ Temperatura: {current.get('temperature', 0)}°C (odczuwalna: {current.get('feels_like', 0)}°C)",
            f"☁️ Opis: {current.get('description', 'Brak opisu')}",
            f"💧 Wilgotność: {current.get('humidity', 0)}%",
            f"💨 Wiatr: {current.get('wind_speed', 0)} m/s",
        ]

        # Dodaj ciśnienie jeśli dostępne
        if current.get("pressure"):
            response_parts.append(f"📊 Ciśnienie: {current.get('pressure')} hPa")

        # Dodaj wschód/zachód słońca jeśli dostępne
        if current.get("sunrise") and current.get("sunset"):
            response_parts.extend(
                [
                    f"🌅 Wschód słońca: {current.get('sunrise')}",
                    f"🌅 Zachód słońca: {current.get('sunset')}",
                ]
            )

        # Prognoza na najbliższe dni
        if forecast:
            response_parts.append("\n📅 Prognoza na najbliższe dni:")
            for day in forecast[:3]:  # Tylko 3 dni
                response_parts.append(
                    f"• {day.get('day_name', 'Nieznany dzień')}: {day.get('min_temp', 0)}°C - {day.get('max_temp', 0)}°C, {day.get('description', 'Brak opisu')}"
                )

        # Alerty pogodowe
        alerts = weather_data.get("alerts", [])
        if alerts:
            response_parts.append("\n⚠️ Alerty pogodowe:")
            for alert in alerts[:2]:  # Maksymalnie 2 alerty
                response_parts.append(f"• {alert.get('headline', 'Alert pogodowy')}")

        return "\n".join(response_parts)

    async def get_weather_advice(
        self, user_id: int, weather_data: dict[str, Any]
    ) -> str:
        """Generuje porady na podstawie danych pogodowych.

        Args:
            user_id: ID użytkownika
            weather_data: Dane pogodowe

        Returns:
            Porady dotyczące pogody
        """
        if "error" in weather_data:
            return "Nie mogę udzielić porad bez danych pogodowych."

        current = weather_data.get("current", {})
        temp = current.get("temperature", 0)
        humidity = current.get("humidity", 0)
        wind_speed = current.get("wind_speed", 0)
        description = current.get("description", "").lower()

        advice_parts = []

        # Porady dotyczące temperatury
        if temp < 0:
            advice_parts.append(
                "🧥 Na zewnątrz jest bardzo zimno - załóż ciepłą kurtkę, czapkę i rękawiczki!"
            )
        elif temp < 10:
            advice_parts.append("🧥 Jest chłodno - warto założyć kurtkę lub sweter.")
        elif temp > 25:
            advice_parts.append("👕 Jest ciepło - lekkie ubrania będą odpowiednie.")
        elif temp > 30:
            advice_parts.append(
                "🌡️ Jest bardzo gorąco - noś lekkie ubrania i pamiętaj o nawodnieniu!"
            )

        # Porady dotyczące opadów
        if "deszcz" in description or "rain" in description:
            advice_parts.append("☔ Zabierz parasol lub kurtkę przeciwdeszczową!")
        elif "śnieg" in description or "snow" in description:
            advice_parts.append("❄️ Uwaga na śliskie drogi i chodniki!")

        # Porady dotyczące wiatru
        if wind_speed > 10:
            advice_parts.append(
                "💨 Silny wiatr - uważaj na spadające gałęzie i trzymaj mocno parasol!"
            )

        # Porady dotyczące wilgotności
        if humidity > 80:
            advice_parts.append("💧 Wysoka wilgotność - może być duszno.")
        elif humidity < 30:
            advice_parts.append("🏜️ Niska wilgotność - pamiętaj o nawilżaniu skóry.")

        # UV Index jeśli dostępny
        uv_index = current.get("uv_index", 0)
        if uv_index > 6:
            advice_parts.append(
                "☀️ Wysokie promieniowanie UV - używaj kremu z filtrem!"
            )

        return (
            "\n".join(advice_parts)
            if advice_parts
            else "Pogoda wygląda na stabilną - ubierz się zgodnie z temperaturą!"
        )

    async def _get_user_api_key(self, user_id: int, provider: str) -> str | None:
        """Pobiera klucz API użytkownika dla danego providera.

        Args:
            user_id: ID użytkownika
            provider: Provider (openweather, weatherapi)

        Returns:
            Klucz API lub None
        """
        try:
            return self.db_manager.get_user_api_key(user_id, provider)
        except Exception as e:
            logger.error(
                f"Error getting API key for user {user_id}, provider {provider}: {e}"
            )
            return None

    def _get_mock_weather_data(self, location: str) -> dict[str, Any]:
        """Zwraca przykładowe dane pogodowe (mock) dla testów."""
        return {
            "location": {
                "name": location,
                "country": "PL",
                "latitude": 52.2297,
                "longitude": 21.0122,
            },
            "current": {
                "temperature": 10,
                "feels_like": 8,
                "humidity": 80,
                "pressure": 1013,
                "description": "Częściowe zachmurzenie",
                "icon": "04d",
                "wind_speed": 3,
                "wind_direction": 180,
                "cloudiness": 75,
                "visibility": 10,
                "sunrise": "06:00",
                "sunset": "18:00",
            },
            "forecast": [
                {
                    "date": "2023-10-01",
                    "day_name": "Poniedziałek",
                    "min_temp": 8,
                    "max_temp": 12,
                    "avg_temp": 10,
                    "description": "Zachmurzenie",
                    "avg_humidity": 75,
                    "avg_wind_speed": 3,
                    "hourly": [
                        {
                            "time": "06:00",
                            "temperature": 8,
                            "description": "Zachmurzenie",
                            "icon": "04n",
                            "humidity": 85,
                            "wind_speed": 2,
                        },
                        {
                            "time": "12:00",
                            "temperature": 10,
                            "description": "Częściowe zachmurzenie",
                            "icon": "03d",
                            "humidity": 70,
                            "wind_speed": 3,
                        },
                        {
                            "time": "18:00",
                            "temperature": 9,
                            "description": "Zachmurzenie",
                            "icon": "04d",
                            "humidity": 80,
                            "wind_speed": 4,
                        },
                    ],
                },
                {
                    "date": "2023-10-02",
                    "day_name": "Wtorek",
                    "min_temp": 9,
                    "max_temp": 13,
                    "avg_temp": 11,
                    "description": "Słonecznie",
                    "avg_humidity": 70,
                    "avg_wind_speed": 2,
                    "hourly": [
                        {
                            "time": "06:00",
                            "temperature": 9,
                            "description": "Słonecznie",
                            "icon": "01n",
                            "humidity": 80,
                            "wind_speed": 2,
                        },
                        {
                            "time": "12:00",
                            "temperature": 12,
                            "description": "Czyste niebo",
                            "icon": "01d",
                            "humidity": 60,
                            "wind_speed": 3,
                        },
                        {
                            "time": "18:00",
                            "temperature": 10,
                            "description": "Słonecznie",
                            "icon": "01d",
                            "humidity": 75,
                            "wind_speed": 2,
                        },
                    ],
                },
            ],
            "units": {
                "temperature": "°C",
                "wind_speed": "m/s",
                "pressure": "hPa",
                "visibility": "km",
            },
        }

    def _get_mock_forecast_data(self, location: str, days: int) -> dict[str, Any]:
        """Zwraca przykładowe dane prognozy (mock) dla testów."""
        forecast = []
        for i in range(days):
            date = datetime.now() + timedelta(days=i)
            forecast.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "day_name": date.strftime("%A"),
                    "min_temp": 5 + i,
                    "max_temp": 10 + i,
                    "avg_temp": 7 + i,
                    "description": "Przykładowy opis pogody",
                    "avg_humidity": 70 + i * 5,
                    "avg_wind_speed": 3 + i * 0.5,
                    "hourly": [
                        {
                            "time": date.replace(hour=6, minute=0).strftime("%H:%M"),
                            "temperature": 6 + i,
                            "description": "Przykładowy opis",
                            "icon": "01n",
                            "humidity": 80,
                            "wind_speed": 2,
                        },
                        {
                            "time": date.replace(hour=12, minute=0).strftime("%H:%M"),
                            "temperature": 9 + i,
                            "description": "Przykładowy opis",
                            "icon": "01d",
                            "humidity": 60,
                            "wind_speed": 3,
                        },
                        {
                            "time": date.replace(hour=18, minute=0).strftime("%H:%M"),
                            "temperature": 8 + i,
                            "description": "Przykładowy opis",
                            "icon": "01d",
                            "humidity": 75,
                            "wind_speed": 2,
                        },
                    ],
                }
            )

        return {
            "location": {
                "name": location,
                "country": "PL",
                "latitude": 52.2297,
                "longitude": 21.0122,
            },
            "forecast": forecast,
            "units": {"temperature": "°C", "wind_speed": "m/s", "precipitation": "mm"},
        }


# Globalna instancja
_weather_module = None


async def get_weather_module() -> WeatherModule:
    """Pobiera globalną instancję modułu pogodowego."""
    global _weather_module
    if _weather_module is None:
        _weather_module = WeatherModule()
        await _weather_module.initialize()
    return _weather_module

    async def get_forecast(
        self, user_id: int, location: str, api_key: str, days: int = 3
    ) -> dict[str, Any]:
        """Pobiera prognozę pogody na kilka dni.

        Args:
            user_id: ID użytkownika
            location: Lokalizacja
            api_key: Klucz API
            days: Liczba dni prognozy

        Returns:
            Prognoza pogody
        """
        if not self.api_module:
            await self.initialize()

        try:
            # OpenWeatherMap 5 day forecast
            url = "https://api.openweathermap.org/data/2.5/forecast"
            params = {
                "q": location,
                "appid": api_key,
                "units": "metric",
                "lang": "pl",
                "cnt": days * 8,  # 8 prognoz na dzień (co 3 godziny)
            }

            response = await self.api_module.get(user_id, url, params=params)

            if response.get("status") != 200:
                return {
                    "error": "Błąd pobierania prognozy pogody",
                    "details": response,
                    "provider": "openweather",
                }

            data = response.get("data", {})
            forecast_info = self._process_forecast_data(data, location, days)
            forecast_info["provider"] = "openweather"
            forecast_info["timestamp"] = datetime.now().isoformat()

            return forecast_info

        except Exception as e:
            logger.error(f"Forecast error: {e}")
            return {
                "error": f"Błąd pobierania prognozy: {str(e)}",
                "location": location,
            }

    def _process_forecast_data(
        self, data: dict, location: str, days: int
    ) -> dict[str, Any]:
        """Przetwarza dane prognozy z OpenWeatherMap.

        Args:
            data: Dane z API
            location: Lokalizacja
            days: Liczba dni

        Returns:
            Przetworzona prognoza
        """
        forecast_list = data.get("list", [])
        city_info = data.get("city", {})

        # Grupuj prognozy po dniach
        daily_forecasts = {}

        for item in forecast_list:
            dt = datetime.fromtimestamp(item["dt"])
            date_str = dt.strftime("%Y-%m-%d")

            if date_str not in daily_forecasts:
                daily_forecasts[date_str] = {
                    "date": date_str,
                    "day_name": dt.strftime("%A"),
                    "temperatures": [],
                    "descriptions": [],
                    "humidity": [],
                    "wind_speed": [],
                    "precipitation": [],
                    "hourly": [],
                }

            # Dodaj dane godzinowe
            hourly_data = {
                "time": dt.strftime("%H:%M"),
                "temperature": round(item["main"]["temp"]),
                "feels_like": round(item["main"]["feels_like"]),
                "description": item["weather"][0]["description"],
                "icon": item["weather"][0]["icon"],
                "humidity": item["main"]["humidity"],
                "wind_speed": item["wind"]["speed"],
                "precipitation": item.get("rain", {}).get("3h", 0)
                + item.get("snow", {}).get("3h", 0),
            }

            daily_forecasts[date_str]["hourly"].append(hourly_data)
            daily_forecasts[date_str]["temperatures"].append(item["main"]["temp"])
            daily_forecasts[date_str]["descriptions"].append(
                item["weather"][0]["description"]
            )
            daily_forecasts[date_str]["humidity"].append(item["main"]["humidity"])
            daily_forecasts[date_str]["wind_speed"].append(item["wind"]["speed"])
            daily_forecasts[date_str]["precipitation"].append(
                hourly_data["precipitation"]
            )

        # Przetwórz dzienne podsumowania
        forecast = []
        for date_str, day_data in list(daily_forecasts.items())[:days]:
            temps = day_data["temperatures"]
            precipitations = day_data["precipitation"]

            forecast.append(
                {
                    "date": date_str,
                    "day_name": day_data["day_name"],
                    "min_temp": round(min(temps)) if temps else 0,
                    "max_temp": round(max(temps)) if temps else 0,
                    "avg_temp": round(sum(1 for x in temps if x) / len(temps))
                    if temps
                    else 0,
                    "description": (
                        max(
                            set(day_data["descriptions"]),
                            key=day_data["descriptions"].count,
                        )
                        if day_data["descriptions"]
                        else ""
                    ),
                    "avg_humidity": (
                        round(
                            sum(1 for x in day_data["humidity"] if x)
                            / len(day_data["humidity"])
                        )
                        if day_data["humidity"]
                        else 0
                    ),
                    "avg_wind_speed": (
                        round(
                            sum(1 for x in day_data["wind_speed"] if x)
                            / len(day_data["wind_speed"]),
                            1,
                        )
                        if day_data["wind_speed"]
                        else 0
                    ),
                    "total_precipitation": (
                        round(sum(1 for x in precipitations if x), 1)
                        if precipitations
                        else 0
                    ),
                    "hourly": day_data["hourly"][:8],  # Maksymalnie 8 godzin na dzień
                }
            )

        return {
            "location": {
                "name": city_info.get("name", location),
                "country": city_info.get("country", ""),
                "latitude": city_info.get("coord", {}).get("lat"),
                "longitude": city_info.get("coord", {}).get("lon"),
            },
            "forecast": forecast,
            "units": {"temperature": "°C", "wind_speed": "m/s", "precipitation": "mm"},
        }
