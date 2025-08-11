import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List

from config.config_manager import get_database_manager
from core.base_module import BaseModule, FunctionSchema, TestModeSupport, create_standard_function_schema

from .api_module import get_api_module

logger = logging.getLogger(__name__)


class WeatherModule(BaseModule, TestModeSupport):
    """Moduł pogodowy dla asystenta."""

    def __init__(self):
        super().__init__("weather")
        self.api_module = None
        self.db_manager = get_database_manager()
        self.weather_providers = {
            "openweather": self._get_openweather_data,
            "weatherapi": self._get_weatherapi_data,
        }

        # Cache pogody (żeby nie wykonywać zbyt wielu zapytań)
        self.weather_cache = {}
        self.cache_duration = timedelta(minutes=30)

    async def _initialize_impl(self):
        """Inicjalizuje moduł pogodowy."""
        self.api_module = await get_api_module()
        self.logger.info("WeatherModule initialized")

    def get_function_schemas(self) -> List[FunctionSchema]:
        """Return list of function schemas provided by this module."""
        return [
            create_standard_function_schema(
                name="get_weather",
                description="Pobiera dane pogodowe dla określonej lokalizacji",
                properties={
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
                },
                required=["location"]
            ),
            create_standard_function_schema(
                name="get_forecast",
                description="Pobiera prognozę pogody na kilka dni",
                properties={
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
                },
                required=["location"]
            ),
        ]

    def get_mock_data(self, function_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Provide mock data for test mode."""
        location = parameters.get("location", "Warszawa")
        
        if function_name == "get_weather":
            return self._create_success_response(
                data=self._get_mock_weather_data(location),
                message=f"Pobrano dane pogodowe dla {location} (tryb testowy)",
                test_mode=True
            )
        elif function_name == "get_forecast":
            days = parameters.get("days", 3)
            return self._create_success_response(
                data=self._get_mock_forecast_data(location, days),
                message=f"Pobrano prognozę pogody dla {location} na {days} dni (tryb testowy)",
                test_mode=True
            )
        
        return super().get_mock_data(function_name, parameters)

    async def execute_function(self, function_name: str, parameters: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """Execute a function provided by this module."""
        if function_name == "get_weather":
            return await self._get_weather(parameters, user_id)
        elif function_name == "get_forecast":
            return await self._get_forecast(parameters, user_id)
        else:
            return self._create_error_response(f"Unknown function: {function_name}")

    async def _get_weather(self, parameters: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """Handle get_weather function."""
        location = parameters.get("location")
        if not location:
            return self._create_error_response("Location parameter is required")
        
        provider = parameters.get("provider", "openweather")

        # Check API key availability
        api_key = await self._get_user_api_key(user_id, provider)
        if not api_key:
            return self._create_error_response(
                f"Brak klucza API dla providera {provider}",
                code="missing_api_key"
            )

        try:
            result = await self.get_weather(user_id, location, api_key, provider)
            return self._create_success_response(
                data=result,
                message=f"Pobrano dane pogodowe dla {location}"
            )
        except Exception as e:
            return self._create_error_response(f"Błąd pobierania pogody: {str(e)}")

    async def _get_forecast(self, parameters: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """Handle get_forecast function."""
        location = parameters.get("location")
        if not location:
            return self._create_error_response("Location parameter is required")
        
        days = parameters.get("days", 3)

        # Check API key availability
        api_key = await self._get_user_api_key(user_id, "openweather")
        if not api_key:
            return self._create_error_response(
                "Brak klucza API dla OpenWeather",
                code="missing_api_key"
            )

        try:
            result = await self.get_forecast(user_id, location, api_key, days)
            return self._create_success_response(
                data=result,
                message=f"Pobrano prognozę na {days} dni dla {location}"
            )
        except Exception as e:
            return self._create_error_response(f"Błąd pobierania prognozy: {str(e)}")

    async def get_weather(
        self, user_id: str, location: str, api_key: str, provider: str = "openweather"
    ) -> Dict[str, Any]:
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
                self.logger.info(f"Returning cached weather data for {location}")
                return cached_data

        if provider not in self.weather_providers:
            raise ValueError(f"Nieobsługiwany provider pogodowy: {provider}")

        try:
            weather_func = self.weather_providers[provider]
            weather_data = await weather_func(user_id, location, api_key)

            # Cache wyników
            if "error" not in weather_data:
                self.weather_cache[cache_key] = (weather_data, datetime.now())

            return weather_data

        except Exception as e:
            self.logger.error(f"Weather error with {provider}: {e}")
            raise

    async def get_forecast(
        self, user_id: str, location: str, api_key: str, days: int = 3
    ) -> Dict[str, Any]:
        """Pobiera prognozę pogody na kilka dni.

        Args:
            user_id: ID użytkownika
            location: Lokalizacja
            api_key: Klucz API OpenWeather
            days: Liczba dni prognozy

        Returns:
            Prognoza pogody
        """
        if not self.api_module:
            await self.initialize()

        # Sprawdź cache
        cache_key = f"forecast_{location.lower()}_{days}"
        if cache_key in self.weather_cache:
            cached_data, timestamp = self.weather_cache[cache_key]
            if datetime.now() - timestamp < self.cache_duration:
                self.logger.info(f"Returning cached forecast data for {location}")
                return cached_data

        try:
            forecast_data = await self._get_openweather_forecast(
                user_id, location, api_key, days
            )

            # Cache wyników
            if "error" not in forecast_data:
                self.weather_cache[cache_key] = (forecast_data, datetime.now())

            return forecast_data

        except Exception as e:
            self.logger.error(f"Forecast error: {e}")
            raise

    def _get_mock_weather_data(self, location: str) -> Dict[str, Any]:
        """Mock dane pogodowe do testów."""
        return {
            "location": location,
            "temperature": 22.5,
            "feels_like": 25.0,
            "humidity": 65,
            "pressure": 1013,
            "description": "Pochmurnie",
            "wind_speed": 3.2,
            "wind_direction": "SW",
            "visibility": 10,
            "uv_index": 5,
            "timestamp": datetime.now().isoformat(),
        }

    def _get_mock_forecast_data(self, location: str, days: int) -> Dict[str, Any]:
        """Mock dane prognozy do testów."""
        forecast_days = []
        for i in range(days):
            day_data = {
                "date": (datetime.now() + timedelta(days=i)).date().isoformat(),
                "temperature_max": 25 + i,
                "temperature_min": 15 + i,
                "description": "Słonecznie" if i % 2 == 0 else "Pochmurnie",
                "humidity": 60 + i * 5,
                "wind_speed": 2.5 + i * 0.5,
            }
            forecast_days.append(day_data)

        return {
            "location": location,
            "days": days,
            "forecast": forecast_days,
            "timestamp": datetime.now().isoformat(),
        }

    async def _get_user_api_key(self, user_id: str, provider: str) -> str:
        """Pobiera klucz API użytkownika dla providera."""
        try:
            if self.db_manager:
                # Convert user_id to int as database expects it
                user_id_int = int(user_id) if isinstance(user_id, str) else user_id
                api_key = self.db_manager.get_user_api_key(user_id_int, provider)
                if api_key:
                    return api_key
            return ""
        except Exception as e:
            self.logger.warning(f"Failed to get API key for {provider}: {e}")
            return ""

    async def _get_openweather_data(
        self, user_id: str, location: str, api_key: str
    ) -> Dict[str, Any]:
        """Pobiera dane z OpenWeatherMap API."""
        if not self.api_module:
            await self.initialize()
            
        url = f"http://api.openweathermap.org/data/2.5/weather"
        params = {
            "q": location,
            "appid": api_key,
            "units": "metric",
            "lang": "pl"
        }

        # Convert user_id to int for API module
        user_id_int = int(user_id) if isinstance(user_id, str) else user_id
        response = await self.api_module.make_request(
            user_id=user_id_int,
            method="GET", 
            url=url, 
            params=params
        )
        
        if response.get("status") == 200 and "data" in response:
            data = response["data"]
            return {
                "location": f"{data['name']}, {data['sys']['country']}",
                "temperature": data["main"]["temp"],
                "feels_like": data["main"]["feels_like"],
                "humidity": data["main"]["humidity"],
                "pressure": data["main"]["pressure"],
                "description": data["weather"][0]["description"],
                "wind_speed": data.get("wind", {}).get("speed", 0),
                "wind_direction": self._wind_direction_from_degrees(
                    data.get("wind", {}).get("deg", 0)
                ),
                "visibility": data.get("visibility", 0) / 1000,  # Convert to km
                "timestamp": datetime.now().isoformat(),
            }
        else:
            raise Exception(f"OpenWeather API error: {response.get('error', 'Unknown error')}")

    async def _get_weatherapi_data(
        self, user_id: str, location: str, api_key: str
    ) -> Dict[str, Any]:
        """Pobiera dane z WeatherAPI."""
        if not self.api_module:
            await self.initialize()
            
        url = f"http://api.weatherapi.com/v1/current.json"
        params = {
            "key": api_key,
            "q": location,
            "lang": "pl"
        }

        # Convert user_id to int for API module
        user_id_int = int(user_id) if isinstance(user_id, str) else user_id
        response = await self.api_module.make_request(
            user_id=user_id_int,
            method="GET",
            url=url,
            params=params
        )
        
        if response.get("status") == 200 and "data" in response:
            data = response["data"]
            current = data["current"]
            location_data = data["location"]
            
            return {
                "location": f"{location_data['name']}, {location_data['country']}",
                "temperature": current["temp_c"],
                "feels_like": current["feelslike_c"],
                "humidity": current["humidity"],
                "pressure": current["pressure_mb"],
                "description": current["condition"]["text"],
                "wind_speed": current["wind_kph"] / 3.6,  # Convert to m/s
                "wind_direction": current["wind_dir"],
                "visibility": current["vis_km"],
                "uv_index": current["uv"],
                "timestamp": datetime.now().isoformat(),
            }
        else:
            raise Exception(f"WeatherAPI error: {response.get('error', 'Unknown error')}")

    async def _get_openweather_forecast(
        self, user_id: str, location: str, api_key: str, days: int
    ) -> Dict[str, Any]:
        """Pobiera prognozę z OpenWeatherMap API."""
        if not self.api_module:
            await self.initialize()
            
        url = f"http://api.openweathermap.org/data/2.5/forecast"
        params = {
            "q": location,
            "appid": api_key,
            "units": "metric",
            "lang": "pl"
        }

        # Convert user_id to int for API module
        user_id_int = int(user_id) if isinstance(user_id, str) else user_id
        response = await self.api_module.make_request(
            user_id=user_id_int,
            method="GET",
            url=url,
            params=params
        )
        
        if response.get("status") == 200 and "data" in response:
            data = response["data"]
            forecast_list = data["list"]
            
            # Group by day
            daily_forecasts = {}
            for forecast in forecast_list[:days * 8]:  # 8 forecasts per day (3-hour intervals)
                date = datetime.fromtimestamp(forecast["dt"]).date()
                if date not in daily_forecasts:
                    daily_forecasts[date] = {
                        "temperatures": [],
                        "descriptions": [],
                        "humidity": [],
                        "wind_speed": []
                    }
                
                daily_forecasts[date]["temperatures"].append(forecast["main"]["temp"])
                daily_forecasts[date]["descriptions"].append(forecast["weather"][0]["description"])
                daily_forecasts[date]["humidity"].append(forecast["main"]["humidity"])
                daily_forecasts[date]["wind_speed"].append(forecast.get("wind", {}).get("speed", 0))

            # Convert to final format
            forecast_days = []
            for date, day_data in list(daily_forecasts.items())[:days]:
                forecast_days.append({
                    "date": date.isoformat(),
                    "temperature_max": max(day_data["temperatures"]),
                    "temperature_min": min(day_data["temperatures"]),
                    "description": max(set(day_data["descriptions"]), key=day_data["descriptions"].count),
                    "humidity": sum(day_data["humidity"]) // len(day_data["humidity"]),
                    "wind_speed": sum(day_data["wind_speed"]) / len(day_data["wind_speed"]),
                })

            return {
                "location": f"{data['city']['name']}, {data['city']['country']}",
                "days": days,
                "forecast": forecast_days,
                "timestamp": datetime.now().isoformat(),
            }
        else:
            raise Exception(f"OpenWeather forecast API error: {response.get('error', 'Unknown error')}")

    def _wind_direction_from_degrees(self, degrees: float) -> str:
        """Konwertuje stopnie na kierunek wiatru."""
        directions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                     "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
        index = round(degrees / (360 / len(directions))) % len(directions)
        return directions[index]


# Create module instance
weather_module = WeatherModule()


# Required plugin functions (for compatibility)
def get_functions() -> List[Dict[str, Any]]:
    """Zwraca listę dostępnych funkcji w pluginie."""
    return weather_module.get_functions()


async def execute_function(
    function_name: str, parameters: Dict[str, Any], user_id: str
) -> Dict[str, Any]:
    """Wykonuje funkcję pluginu."""
    if not weather_module._initialized:
        await weather_module.initialize()
    
    return await weather_module.safe_execute(function_name, parameters, user_id)
