"""Daily Briefing Module for Asystent Provides proactive morning briefings with weather,
calendar, holidays, and personalized information."""

import asyncio
import json
import logging
import random
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import aiohttp

# AI and memory integration imports
logger = logging.getLogger(__name__)

try:
    from ai_module import generate_response

    AI_MODULE_AVAILABLE = True
except ImportError:
    AI_MODULE_AVAILABLE = False
    logger.warning("AI module not available - falling back to template-based briefings")

try:
    from modules.memory_module import search_memories

    MEMORY_MODULE_AVAILABLE = True
except ImportError:
    MEMORY_MODULE_AVAILABLE = False
    logger.warning("Memory module not available - skipping memory integration")


class DailyBriefingModule:
    """Handles proactive daily briefings with weather, calendar, holidays, and
    memories."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.user_name = config.get("USER_NAME") or config.get(
            "daily_briefing", {}
        ).get("user_name", "Tymek")
        self.location = config.get("daily_briefing", {}).get("location", "Warsaw,PL")
        self.enabled = config.get("daily_briefing", {}).get("enabled", True)
        self.briefing_time = config.get("daily_briefing", {}).get(
            "briefing_time", "08:00"
        )
        self.startup_briefing = config.get("daily_briefing", {}).get(
            "startup_briefing", True
        )
        self.scheduled_briefing = config.get("daily_briefing", {}).get(
            "scheduled_briefing", False
        )
        self.include_weather = config.get("daily_briefing", {}).get(
            "include_weather", True
        )
        self.include_calendar = config.get("daily_briefing", {}).get(
            "include_calendar", True
        )
        self.include_holidays = config.get("daily_briefing", {}).get(
            "include_holidays", True
        )
        self.include_memories = config.get("daily_briefing", {}).get(
            "include_memories", False
        )
        self.include_quote = config.get("daily_briefing", {}).get("include_quote", True)
        self.weekend_briefing = config.get("daily_briefing", {}).get(
            "weekend_briefing", True
        )
        self.min_interval_hours = config.get("daily_briefing", {}).get(
            "min_interval_hours", 12
        )
        self.briefing_style = config.get("daily_briefing", {}).get(
            "briefing_style", "normal"
        )  # normal, funny, serious
        self.use_ai_generation = (
            config.get("daily_briefing", {}).get("use_ai_generation", True)
            and AI_MODULE_AVAILABLE
        )
        self.use_dynamic_location = config.get("daily_briefing", {}).get(
            "use_dynamic_location", True
        )

        # Session tracking
        self.session_file = Path("user_data") / "daily_briefing_session.json"
        self.session_file.parent.mkdir(exist_ok=True)

        # Scheduling
        self._scheduler_task = None
        self._scheduler_running = False
        self._stop_scheduler = False

        # Polish holidays API
        self.holidays_api_url = "https://date.nager.at/api/v3/PublicHolidays"
        self.quote_api_url = "https://api.quotable.io/random"

        # Day names in Polish
        self.polish_days = {
            0: "poniedziałek",
            1: "wtorek",
            2: "środa",
            3: "czwartek",
            4: "piątek",
            5: "sobota",
            6: "niedziela",
        }

        # Month names in Polish
        self.polish_months = {
            1: "stycznia",
            2: "lutego",
            3: "marca",
            4: "kwietnia",
            5: "maja",
            6: "czerwca",
            7: "lipca",
            8: "sierpnia",
            9: "września",
            10: "października",
            11: "listopada",
            12: "grudnia",
        }

    def should_deliver_briefing(self, force_delivery: bool = False) -> bool:
        """Check if briefing should be delivered today."""
        if not self.enabled:
            return False

        if force_delivery:
            return True

        today = datetime.now().date()

        # Check if it's weekend and weekend briefing is disabled
        if today.weekday() >= 5 and not self.weekend_briefing:
            return False

        # Check session file for last delivery
        if self.session_file.exists():
            try:
                with open(self.session_file, encoding="utf-8") as f:
                    session_data = json.load(f)

                last_delivery = session_data.get("last_briefing_date")
                if last_delivery == today.isoformat():
                    # Already delivered today
                    return False

                # Check minimum interval
                if last_delivery:
                    last_date = datetime.fromisoformat(last_delivery + "T00:00:00")
                    hours_since_last = (
                        datetime.now() - last_date
                    ).total_seconds() / 3600
                    if hours_since_last < self.min_interval_hours:
                        return False

            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.warning(f"Error reading session file: {e}")

        return True

    def _mark_briefing_delivered(self):
        """Mark briefing as delivered today."""
        today = datetime.now().date()
        session_data = {
            "last_briefing_date": today.isoformat(),
            "last_delivery_time": datetime.now().isoformat(),
        }

        try:
            with open(self.session_file, "w", encoding="utf-8") as f:
                json.dump(session_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error writing session file: {e}")

    def start_scheduler(self, delivery_callback):
        """Start the scheduler for automatic briefings."""
        if not self.scheduled_briefing or self._scheduler_running:
            return

        self._stop_scheduler = False
        self._scheduler_task = threading.Thread(
            target=self._scheduler_loop, args=(delivery_callback,), daemon=True
        )
        self._scheduler_task.start()
        self._scheduler_running = True
        logger.info(f"Daily briefing scheduler started for {self.briefing_time}")

    def stop_scheduler(self):
        """Stop the scheduler."""
        self._stop_scheduler = True
        self._scheduler_running = False
        if self._scheduler_task:
            self._scheduler_task.join(timeout=1)
        logger.info("Daily briefing scheduler stopped")

    def _scheduler_loop(self, delivery_callback):
        """Main scheduler loop that runs in background thread."""
        while not self._stop_scheduler:
            try:
                now = datetime.now()
                target_time = datetime.strptime(self.briefing_time, "%H:%M").time()
                target_datetime = datetime.combine(now.date(), target_time)

                # If target time has passed today, schedule for tomorrow
                if target_datetime <= now:
                    target_datetime += timedelta(days=1)

                wait_seconds = (target_datetime - now).total_seconds()
                # Wait with periodic checks for stop signal
                while wait_seconds > 0 and not self._stop_scheduler:
                    sleep_time = min(60, wait_seconds)  # Check every minute
                    threading.Event().wait(sleep_time)
                    wait_seconds -= sleep_time

                if not self._stop_scheduler:
                    # Check if we should deliver briefing
                    if self.should_deliver_briefing():
                        briefing_text = asyncio.run(self.deliver_briefing())
                        if briefing_text and delivery_callback:
                            delivery_callback(briefing_text)

                    # Wait a bit to avoid immediate re-trigger
                    threading.Event().wait(60)

            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                threading.Event().wait(300)  # Wait 5 minutes before retrying

    async def get_weather_data(self) -> dict[str, Any] | None:
        """Get current weather data."""
        if not self.include_weather:
            return None

        try:
            from modules.weather_module import WeatherModule

            weather_module = WeatherModule()
            await weather_module.initialize()

            current_location = await self.get_current_location()
            api_key = self.config.get("api_keys", {}).get("openweather", "")
            result = await weather_module.get_weather(0, current_location, api_key)
            if result and "current" in result:
                text = result["current"]["description"]
                return {"current_text": text, "location": current_location}
        except Exception as e:
            logger.error(f"Could not get weather data: {e}")

        return None

    async def get_polish_holidays(self) -> list[dict[str, Any]]:
        """Get Polish holidays for today."""
        if not self.include_holidays:
            return []

        try:
            year = datetime.now().year
            url = f"{self.holidays_api_url}/{year}/PL"
            headers = {
                "User-Agent": "GAJA Assistant/1.0 (https://github.com/TymekTM/Asystent)"
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=5), headers=headers
                ) as response:
                    response.raise_for_status()
                    holidays = await response.json()

            today = datetime.now().date()
            today_holidays = []

            for holiday in holidays:
                holiday_date = datetime.fromisoformat(holiday["date"]).date()
                if holiday_date == today:
                    today_holidays.append(
                        {"name": holiday["localName"], "date": holiday["date"]}
                    )

            return today_holidays

        except Exception as e:
            logger.error(f"Could not get Polish holidays: {e}")
            return []

    async def get_calendar_events(self) -> list[dict[str, Any]]:
        """Get today's calendar events and reminders."""
        if not self.include_calendar:
            return []

        try:
            # Import core module functions
            from modules.core_module import get_reminders_for_today

            reminders = await get_reminders_for_today({})
            return reminders or []
        except Exception as e:
            logger.error(f"Could not get calendar events: {e}")
            return []

    async def get_memories_data(self) -> list[dict[str, Any]]:
        """Get relevant memories or historical data."""
        if not self.include_memories or not MEMORY_MODULE_AVAILABLE:
            return []

        try:
            memories = []

            # Get recent memories (last 30 days)
            recent = await search_memories("0", "")
            if recent.get("success"):
                for item in recent.get("results", [])[:10]:
                    memories.append(
                        {"content": item.get("content", ""), "type": "recent_memory"}
                    )

            # Get memories related to daily briefings
            briefing = await search_memories("0", "briefing dzień dobry pogoda")
            if briefing.get("success"):
                for item in briefing.get("results", [])[:5]:
                    memories.append(
                        {"content": item.get("content", ""), "type": "briefing_memory"}
                    )

            return memories[:5]  # Limit to 5 most relevant memories

        except Exception as e:
            logger.error(f"Could not get memories data: {e}")
            return []

    async def get_quote(self) -> dict[str, str] | None:
        """Fetch a motivational quote."""
        if not self.include_quote:
            return None

        try:
            headers = {
                "User-Agent": "GAJA Assistant/1.0 (https://github.com/TymekTM/Asystent)"
            }
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.quote_api_url,
                    timeout=aiohttp.ClientTimeout(total=5),
                    headers=headers,
                    ssl=False,
                ) as r:
                    if r.status == 200:
                        data = await r.json()
                        if data.get("content"):
                            return {
                                "text": data.get("content"),
                                "author": data.get("author", ""),
                            }
        except Exception as e:
            logger.warning(f"Quote API failed: {e}")

        quotes = [
            {
                "text": "Uśmiech to najkrótsza droga do drugiego człowieka.",
                "author": "H. Sienkiewicz",
            },
            {"text": "Nie ma nic wielkiego bez pasji.", "author": "Voltaire"},
            {
                "text": "Każdy dzień to nowa szansa, aby zmienić swoje życie.",
                "author": "Anonim",
            },
        ]
        return random.choice(quotes)

    def _format_date_polish(self, date: datetime) -> str:
        """Format date in Polish."""
        day_name = self.polish_days[date.weekday()]
        day = date.day
        month_name = self.polish_months[date.month]
        year = date.year

        return f"{day_name}, {day} {month_name} {year}"

    def _format_weather_info(self, weather_data: dict[str, Any] | None) -> str:
        """Format weather information for briefing."""
        if not weather_data or not weather_data.get("current_text"):
            return ""

        current_text = weather_data["current_text"]
        if current_text and "Wystąpił błąd" not in current_text:
            return f"Jeśli chodzi o pogodę - {current_text.lower()}"

        return ""

    def _format_holidays_info(self, holidays: list[dict[str, Any]]) -> str:
        """Format holidays information for briefing."""
        if not holidays:
            return ""

        if len(holidays) == 1:
            return f"Dziś obchodzimy {holidays[0]['name']}."
        else:
            holiday_names = ", ".join([h["name"] for h in holidays])
            return f"Dziś obchodzimy {holiday_names}."

    def _format_events_info(self, events: list[dict[str, Any]]) -> str:
        """Format events information for briefing."""
        if not events:
            return ""

        if len(events) == 1:
            return f"Masz dziś zaplanowane: {events[0]}."
        else:
            return f"Masz dziś zaplanowane {len(events)} wydarzeń."

    def _format_quote_info(self, quote: dict[str, str] | None) -> str:
        if not quote or not quote.get("text"):
            return ""
        text = quote["text"].strip()
        author = quote.get("author")
        if author:
            return f'"{text}" — {author}'
        return f'"{text}"'

    async def generate_briefing_content(self) -> dict[str, Any]:
        """Generate the complete briefing content."""
        logger.info("Generating daily briefing content...")

        # Collect all data in parallel
        now = datetime.now()

        tasks = [
            self.get_weather_data(),
            self.get_polish_holidays(),
            self.get_calendar_events(),
            self.get_memories_data(),
            self.get_quote(),
        ]

        weather_data, holidays, events, memories, quote = await asyncio.gather(
            *tasks, return_exceptions=True
        )

        # Handle exceptions
        if isinstance(weather_data, Exception):
            logger.error(f"Weather data error: {weather_data}")
            weather_data = None

        if isinstance(holidays, Exception):
            logger.error(f"Holidays data error: {holidays}")
            holidays = []

        if isinstance(events, Exception):
            logger.error(f"Events data error: {events}")
            events = []

        if isinstance(memories, Exception):
            logger.error(f"Memories data error: {memories}")
            memories = []

        if isinstance(quote, Exception):
            logger.error(f"Quote data error: {quote}")
            quote = None

        # Build briefing content
        content = {
            "greeting": f"Hej {self.user_name}",
            "date": self._format_date_polish(now),
            "weather": self._format_weather_info(
                weather_data if isinstance(weather_data, dict) else None
            ),
            "holidays": self._format_holidays_info(
                holidays if isinstance(holidays, list) else []
            ),
            "events": self._format_events_info(
                events if isinstance(events, list) else []
            ),
            "memories": memories if isinstance(memories, list) else [],
            "quote": self._format_quote_info(
                quote if isinstance(quote, dict) else None
            ),
            "timestamp": now.isoformat(),
        }

        return content

    async def generate_ai_briefing(self, content: dict[str, Any]) -> str | None:
        """Generate personalized briefing using AI with language detection."""
        if not self.use_ai_generation or not AI_MODULE_AVAILABLE:
            return None

        try:
            # Detect user language from previous interactions or use Polish as default
            user_language = "Polish"  # Default for this assistant
            # Build context for AI generation
            briefing_context = {
                "user_name": self.user_name,
                "location": await self.get_current_location(),
                "date": content["date"],
                "weather": content["weather"],
                "holidays": content["holidays"],
                "events": content["events"],
                "memories": content["memories"],
                "quote": content["quote"],
                "style": self.briefing_style,
                "language": user_language,
            }
            # Create AI prompt for briefing generation
            ai_prompt = self._build_ai_prompt(briefing_context)

            # Generate response using AI
            logger.info("Generating AI-powered daily briefing...")
            import json
            from collections import deque

            response_json = await generate_response(
                conversation_history=deque([{"role": "user", "content": ai_prompt}]),
                detected_language="pl",
                user_name=self.user_name,
            )

            # Parse the JSON response
            if response_json:
                try:
                    response_data = json.loads(response_json)
                    response = response_data.get("text", "")
                except json.JSONDecodeError:
                    response = response_json  # Fallback to raw response
            else:
                response = None

            if response and isinstance(response, str):
                # Clean up the response
                cleaned_response = response.strip()
                if cleaned_response:
                    logger.info("AI briefing generated successfully")
                    return cleaned_response

            logger.warning("AI generation failed, falling back to template")
            return None

        except Exception as e:
            logger.error(f"Error generating AI briefing: {e}")
            return None

    def _build_ai_prompt(self, context: dict[str, Any]) -> str:
        """Build AI prompt for briefing generation."""
        style_instructions = {
            "normal": "Napisz naturalny, przyjazny dzienny briefing w języku polskim.",
            "funny": "Napisz zabawny, energiczny dzienny briefing z humorem i emoji w języku polskim.",
            "serious": "Napisz profesjonalny, merytoryczny dzienny briefing w formalnym tonie w języku polskim.",
        }

        style_instruction = style_instructions.get(
            context["style"], style_instructions["normal"]
        )

        prompt = f"""
{style_instruction}

Informacje do uwzględnienia:
- Imię użytkownika: {context['user_name']}
- Data: {context['date']}
- Lokalizacja: {context['location']}

"""

        if context["weather"]:
            prompt += f"- Pogoda: {context['weather']}\n"

        if context["holidays"]:
            prompt += f"- Święta: {context['holidays']}\n"

        if context["events"]:
            prompt += f"- Wydarzenia: {context['events']}\n"

        if context["memories"] and len(context["memories"]) > 0:
            prompt += "- Wspomnienia/historie:\n"
            for memory in context["memories"][:3]:  # Limit to top 3
                prompt += f"  • {memory.get('content', '')}\n"

        if context.get("quote"):
            prompt += f"- Cytat dnia: {context['quote']}\n"

        prompt += f"""
Styl odpowiedzi: {context['style']}
Język: {context['language']}

Wygeneruj kompletny dzienny briefing (2-4 zdania) zawierający powitanie, informacje o dacie, pogodzie, świętach, wydarzeniach i zakończenie odpowiednie do stylu. Odpowiedz TYLKO briefingiem, bez dodatkowych komentarzy.
"""

        return prompt.strip()

    def build_briefing_text(self, content: dict[str, Any]) -> str:
        """Build the final briefing text from content."""
        # Style-specific greetings
        if self.briefing_style == "funny":
            greeting = f"Siema {content['greeting'].split()[1]}! 😄"
        elif self.briefing_style == "serious":
            greeting = f"Dzień dobry, {content['greeting'].split()[1]}."
        else:
            greeting = content["greeting"]

        parts = [f"{greeting}, dziś jest {content['date']}."]

        if content["weather"]:
            if self.briefing_style == "funny":
                parts.append(
                    f"Co się dzieje z pogodą? {content['weather'].replace('Jeśli chodzi o pogodę - ', '').capitalize()} 🌤️"
                )
            else:
                parts.append(content["weather"])

        if content["holidays"]:
            if self.briefing_style == "funny":
                parts.append(f"O! {content['holidays']} 🎉")
            else:
                parts.append(content["holidays"])

        if content["events"]:
            if self.briefing_style == "funny":
                parts.append(f"Uwaga, uwaga! {content['events']} 📅")
            else:
                parts.append(content["events"])

        if content.get("quote"):
            if self.briefing_style == "funny":
                parts.append(f"Na koniec mądrość dnia: {content['quote']} 🧐")
            else:
                parts.append(f"Cytat dnia: {content['quote']}")

        # Style-specific endings
        weekday = datetime.now().weekday()
        if self.briefing_style == "funny":
            if weekday == 0:  # Monday
                parts.append("No to lecimy z tym poniedziałkiem! 💪")
            elif weekday == 4:  # Friday
                parts.append("Weekend już blisko - trzymaj się! 🎉")
            elif weekday >= 5:  # Weekend
                parts.append("Weekend mode: ON! Odpocznij sobie! 😎")
            else:
                parts.append("Dawaj z tym dniem! 🚀")
        elif self.briefing_style == "serious":
            if weekday == 0:  # Monday
                parts.append(
                    "Życzę efektywnego i skoncentrowanego rozpoczęcia tygodnia pracy."
                )
            elif weekday == 4:  # Friday
                parts.append(
                    "Życzę satysfakcjonującego zakończenia tygodnia roboczego."
                )
            elif weekday >= 5:  # Weekend
                parts.append("Życzę regenerującego i przyjemnego weekendu.")
            else:
                parts.append("Życzę konstruktywnego i wydajnego dnia.")
        else:
            # Normal style (existing logic)
            if weekday == 0:  # Monday
                parts.append("Życzę produktywnego początku tygodnia!")
            elif weekday == 4:  # Friday
                parts.append("Życzę udanego zakończenia tygodnia!")
            elif weekday >= 5:  # Weekend
                parts.append("Życzę miłego weekendu!")
            else:
                parts.append("Życzę miłego dnia!")

        return " ".join(parts)

    async def deliver_briefing(self, force_delivery: bool = False) -> str | None:
        """Generate and deliver the daily briefing."""
        try:
            if not self.should_deliver_briefing(force_delivery=force_delivery):
                return None

            logger.info("Generating daily briefing...")

            # Generate briefing content
            content = await self.generate_briefing_content()

            # Try AI generation first if enabled
            briefing_text = None
            if self.use_ai_generation:
                ai_briefing = await self.generate_ai_briefing(content)
                if ai_briefing:
                    briefing_text = ai_briefing
                    logger.info("Using AI-generated briefing")

            # Fallback to template-based briefing if AI generation failed
            if not briefing_text:
                briefing_text = self.build_briefing_text(content)
                logger.info("Using template-based briefing")

            # Mark as delivered (only if not forced, to avoid blocking manual requests)
            if not force_delivery:
                self._mark_briefing_delivered()

            logger.info(f"Daily briefing generated: {briefing_text}")
            return briefing_text

        except Exception as e:
            logger.error(f"Error delivering daily briefing: {e}", exc_info=True)
            return "Przepraszam, wystąpił błąd podczas przygotowywania dzisiejszego briefingu."

    async def get_location_from_ip(self):
        """Get user location based on IP address."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "http://ip-api.com/json/", timeout=5
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("status") == "success":
                            city = data.get("city", "Warsaw")
                            country_code = data.get("countryCode", "PL")
                            return f"{city},{country_code}"
        except Exception as e:
            logger.warning(f"Failed to get location from IP: {e}")
        return "Warsaw,PL"  # Default fallback

    async def get_current_location(self) -> str:
        """Get current location - either from config or dynamically from IP."""
        if self.use_dynamic_location:
            return await self.get_location_from_ip()
        return self.location

    async def generate_daily_briefing(self, user_id: str = "default") -> str:
        """Generate daily briefing for a specific user.

        Args:
            user_id: User identifier for personalized briefing

        Returns:
            Generated briefing text
        """
        try:
            logger.info(f"Generating daily briefing for user: {user_id}")

            # Generate briefing content
            content = await self.generate_briefing_content()

            # Try AI generation first if enabled
            briefing_text = None
            if self.use_ai_generation:
                ai_briefing = await self.generate_ai_briefing(content)
                if ai_briefing:
                    briefing_text = ai_briefing
                    logger.info("Using AI-generated briefing")

            # Fallback to template-based briefing if AI generation failed
            if not briefing_text:
                briefing_text = self.build_briefing_text(content)
                logger.info("Using template-based briefing")

            logger.info(
                f"Daily briefing generated for {user_id}: {len(briefing_text)} characters"
            )
            return briefing_text

        except Exception as e:
            logger.error(
                f"Error generating daily briefing for {user_id}: {e}", exc_info=True
            )
            return "Dzień dobry! Przepraszam, wystąpił błąd podczas przygotowywania briefingu. Mam nadzieję, że dzień będzie udany!"


# Handler function for integration with assistant
async def handle_daily_briefing(query: str, params: dict[str, Any]) -> dict[str, Any]:
    """Handle daily briefing requests."""
    try:
        # Get config from params or default config
        config = params.get("config", {})

        # Create briefing module instance
        briefing_module = DailyBriefingModule(config)

        # Force delivery for manual requests
        briefing_text = await briefing_module.deliver_briefing(force_delivery=True)

        if briefing_text:
            return {"success": True, "message": briefing_text, "should_speak": True}
        else:
            return {
                "success": False,
                "message": "Briefing nie jest dostępny w tej chwili.",
                "should_speak": True,
            }
    except Exception as e:
        logger.error(f"Error in daily briefing handler: {e}")
        return {
            "success": False,
            "message": f"Wystąpił błąd podczas generowania briefingu: {str(e)}",
            "should_speak": True,
        }


# Test functions
async def test_ai_briefing():
    """Test AI briefing generation."""
    config = {
        "daily_briefing": {
            "enabled": True,
            "user_name": "Tymek",
            "location": "Warsaw,PL",
            "briefing_style": "funny",
            "use_ai_generation": True,
            "include_weather": True,
            "include_holidays": True,
            "include_calendar": True,
            "include_memories": True,
        }
    }

    briefing = DailyBriefingModule(config)
    result = await briefing.deliver_briefing(force_delivery=True)
    logger.info(f"AI Briefing: {result}")


async def test_template_briefing():
    """Test template-based briefing generation."""
    config = {
        "daily_briefing": {
            "enabled": True,
            "user_name": "Tymek",
            "location": "Warsaw,PL",
            "briefing_style": "normal",
            "use_ai_generation": False,
            "include_weather": True,
            "include_holidays": True,
            "include_calendar": True,
            "include_memories": False,
        }
    }

    briefing = DailyBriefingModule(config)
    result = await briefing.deliver_briefing(force_delivery=True)
    logger.info(f"Template Briefing: {result}")


def register():
    """Register the daily briefing plugin."""
    return {
        "command": "briefing",
        "aliases": ["brief", "podsumowanie", "dzien"],
        "description": "Generuje codzienne podsumowanie dnia",
        "handler": lambda params=None, **kwargs: asyncio.run(
            handle_daily_briefing("", {"config": kwargs.get("config", {})})
        ),
    }


def get_functions() -> list[dict[str, Any]]:
    return [
        {
            "name": "daily_briefing",
            "description": "Return a daily briefing text",
            "parameters": {"type": "object", "properties": {}},
        }
    ]


async def execute_function(
    function_name: str, parameters: dict[str, Any], user_id: int
) -> dict[str, Any]:
    if function_name == "daily_briefing":
        result = await handle_daily_briefing("", {"config": {}})
        return {"success": True, "message": result.get("message", "")}
    return {"success": False, "error": "Unknown function"}


if __name__ == "__main__":
    asyncio.run(test_ai_briefing())
    asyncio.run(test_template_briefing())
