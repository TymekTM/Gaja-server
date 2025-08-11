"""Day Summary Module for Asystent Zbiera dane o aktywności użytkownika przez cały dzień
i tworzy podsumowania."""

import asyncio
import json
import logging
import threading
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    from ai_module import generate_response

    AI_MODULE_AVAILABLE = True
except ImportError:
    AI_MODULE_AVAILABLE = False
    logger.warning("AI module not available for day summaries")


class DaySummaryModule:
    """Moduł zbierający dane o aktywności użytkownika i tworzący podsumowania dnia."""

    def __init__(self, config: dict[str, Any], db_manager=None):
        self.config = config
        self.db_manager = db_manager
        self.enabled = config.get("day_summary", {}).get("enabled", True)
        self.user_name = config.get("USER_NAME", "User")

        # Tracking configuration
        self.track_applications = config.get("day_summary", {}).get(
            "track_applications", True
        )
        self.track_interactions = config.get("day_summary", {}).get(
            "track_interactions", True
        )
        self.track_productivity = config.get("day_summary", {}).get(
            "track_productivity", True
        )
        self.summary_time = config.get("day_summary", {}).get("summary_time", "20:00")
        self.auto_summary = config.get("day_summary", {}).get("auto_summary", True)

        # Data storage
        self.data_dir = Path("user_data") / "day_summaries"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Current day tracking
        self.current_day_data = {
            "date": datetime.now().date().isoformat(),
            "start_time": datetime.now().isoformat(),
            "applications": defaultdict(float),  # app_name -> total_seconds
            "interactions": [],  # list of user interactions
            "active_periods": [],  # periods of activity
            "productivity_score": 0.0,
            "breaks": [],  # detected break periods
            "last_activity": datetime.now().isoformat(),
        }

        # Activity tracking
        self.last_activity_time = datetime.now()
        self.current_application = None
        self.application_start_time = None
        self.inactive_threshold_minutes = 5  # Consider user inactive after 5 minutes

        # Productivity categories
        self.productivity_categories = {
            "high_productivity": [
                "vscode",
                "visual studio code",
                "pycharm",
                "intellij",
                "sublime",
                "atom",
                "notepad++",
                "vim",
                "emacs",
            ],
            "medium_productivity": [
                "browser",
                "chrome",
                "firefox",
                "edge",
                "office",
                "word",
                "excel",
                "powerpoint",
                "outlook",
            ],
            "low_productivity": [
                "youtube",
                "facebook",
                "twitter",
                "instagram",
                "tiktok",
                "netflix",
                "games",
            ],
            "communication": [
                "discord",
                "slack",
                "teams",
                "zoom",
                "skype",
                "telegram",
                "whatsapp",
            ],
            "system": ["explorer", "finder", "terminal", "cmd", "powershell"],
        }

        # Background thread for continuous tracking
        self._tracking_thread = None
        self._stop_tracking = False

    async def initialize(self):
        """Inicjalizacja modułu."""
        if not self.enabled:
            logger.info("Day Summary Module disabled")
            return

        # Load today's data if exists
        await self._load_current_day_data()

        # Start background tracking
        self._start_background_tracking()

        logger.info("Day Summary Module initialized")

    def _start_background_tracking(self):
        """Uruchom śledzenie w tle."""
        if self._tracking_thread and self._tracking_thread.is_alive():
            return

        self._stop_tracking = False
        self._tracking_thread = threading.Thread(
            target=self._tracking_loop, daemon=True
        )
        self._tracking_thread.start()
        logger.info("Background activity tracking started")

    def _tracking_loop(self):
        """Główna pętla śledzenia aktywności."""
        while not self._stop_tracking:
            try:
                # Track current application if enabled
                if self.track_applications:
                    self._track_current_application()

                # Check for inactivity
                self._check_inactivity()

                # Save data periodically
                asyncio.run(self._save_current_day_data())

                # Sleep for tracking interval
                threading.Event().wait(30)  # Track every 30 seconds

            except Exception as e:
                logger.error(f"Error in tracking loop: {e}")
                threading.Event().wait(60)  # Wait longer on error

    def _track_current_application(self):
        """Śledź aktualnie aktywną aplikację."""
        try:
            # Import here to avoid circular dependencies
            from active_window_module import get_active_window_title

            current_app = get_active_window_title()
            now = datetime.now()

            if current_app != self.current_application:
                # Application changed
                if self.current_application and self.application_start_time:
                    # Record time spent in previous application
                    duration = (now - self.application_start_time).total_seconds()
                    self.current_day_data["applications"][
                        self.current_application
                    ] += duration

                # Start tracking new application
                self.current_application = current_app
                self.application_start_time = now

                logger.debug(f"Application switched to: {current_app}")

            # Update last activity time
            self.last_activity_time = now
            self.current_day_data["last_activity"] = now.isoformat()

        except Exception as e:
            logger.error(f"Error tracking application: {e}")

    def _check_inactivity(self):
        """Sprawdź okresy nieaktywności."""
        now = datetime.now()
        time_since_activity = (
            now - self.last_activity_time
        ).total_seconds() / 60  # minutes

        if time_since_activity > self.inactive_threshold_minutes:
            # User has been inactive - record as break
            if self.current_application:
                # Stop tracking current application
                if self.application_start_time:
                    duration = (
                        self.last_activity_time - self.application_start_time
                    ).total_seconds()
                    self.current_day_data["applications"][
                        self.current_application
                    ] += duration

                self.current_application = None
                self.application_start_time = None

            # Record break if not already recorded
            if (
                not self.current_day_data["breaks"]
                or (
                    datetime.fromisoformat(self.current_day_data["breaks"][-1]["end"])
                    + timedelta(minutes=self.inactive_threshold_minutes)
                )
                < self.last_activity_time
            ):
                break_record = {
                    "start": self.last_activity_time.isoformat(),
                    "end": now.isoformat(),
                    "duration_minutes": time_since_activity,
                }
                self.current_day_data["breaks"].append(break_record)
                logger.debug(f"Break detected: {time_since_activity:.1f} minutes")

    async def record_interaction(
        self, interaction_type: str, content: str = "", metadata: dict = None
    ):
        """Zapisz interakcję użytkownika z asystentem."""
        if not self.track_interactions:
            return

        interaction = {
            "timestamp": datetime.now().isoformat(),
            "type": interaction_type,  # 'voice_command', 'text_query', 'function_call', etc.
            "content": content[:200],  # Limit content length
            "metadata": metadata or {},
        }

        self.current_day_data["interactions"].append(interaction)
        logger.debug(f"Recorded interaction: {interaction_type}")

    def _calculate_productivity_score(self) -> float:
        """Oblicz wynik produktywności na podstawie używanych aplikacji."""
        if not self.current_day_data["applications"]:
            return 0.0

        total_time = sum(self.current_day_data["applications"].values())
        if total_time == 0:
            return 0.0

        productivity_score = 0.0

        for app_name, duration in self.current_day_data["applications"].items():
            app_lower = app_name.lower()
            weight = 0.0

            # Determine productivity weight
            for category, apps in self.productivity_categories.items():
                if any(productive_app in app_lower for productive_app in apps):
                    if category == "high_productivity":
                        weight = 1.0
                    elif category == "medium_productivity":
                        weight = 0.7
                    elif category == "communication":
                        weight = 0.5
                    elif category == "system":
                        weight = 0.3
                    elif category == "low_productivity":
                        weight = 0.1
                    break

            # Calculate weighted score
            productivity_score += (duration / total_time) * weight

        return min(productivity_score, 1.0)  # Cap at 1.0

    async def _load_current_day_data(self):
        """Załaduj dane dla bieżącego dnia."""
        today = datetime.now().date()
        data_file = self.data_dir / f"day_data_{today.isoformat()}.json"

        if data_file.exists():
            try:
                with open(data_file, encoding="utf-8") as f:
                    loaded_data = json.load(f)

                # Convert defaultdict for applications
                applications = defaultdict(float)
                applications.update(loaded_data.get("applications", {}))
                loaded_data["applications"] = applications

                self.current_day_data = loaded_data
                logger.info(f"Loaded existing day data for {today}")

            except Exception as e:
                logger.error(f"Error loading day data: {e}")

    async def _save_current_day_data(self):
        """Zapisz dane bieżącego dnia."""
        try:
            today = datetime.now().date()
            data_file = self.data_dir / f"day_data_{today.isoformat()}.json"

            # Update productivity score
            self.current_day_data[
                "productivity_score"
            ] = self._calculate_productivity_score()
            self.current_day_data["last_updated"] = datetime.now().isoformat()

            # Convert defaultdict to regular dict for JSON serialization
            save_data = dict(self.current_day_data)
            save_data["applications"] = dict(save_data["applications"])

            with open(data_file, "w", encoding="utf-8") as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error(f"Error saving day data: {e}")

    async def generate_day_summary(self, date: str | None = None) -> dict[str, Any]:
        """Wygeneruj podsumowanie dnia."""
        if date is None:
            date = datetime.now().date().isoformat()

        # Load data for the specified date
        data_file = self.data_dir / f"day_data_{date}.json"

        if not data_file.exists():
            return {"success": False, "error": f"No data available for {date}"}

        try:
            with open(data_file, encoding="utf-8") as f:
                day_data = json.load(f)

            # Calculate summary statistics
            total_active_time = (
                sum(day_data.get("applications", {}).values()) / 3600
            )  # hours
            total_interactions = len(day_data.get("interactions", []))
            total_breaks = len(day_data.get("breaks", []))
            break_time = (
                sum(
                    break_data.get("duration_minutes", 0)
                    for break_data in day_data.get("breaks", [])
                )
                / 60
            )  # hours
            productivity_score = day_data.get("productivity_score", 0.0)

            # Top applications
            applications = day_data.get("applications", {})
            top_apps = sorted(applications.items(), key=lambda x: x[1], reverse=True)[
                :5
            ]
            top_apps_formatted = [
                (app, duration / 3600) for app, duration in top_apps
            ]  # Convert to hours

            # Activity patterns
            interactions_by_hour = defaultdict(int)
            for interaction in day_data.get("interactions", []):
                hour = datetime.fromisoformat(interaction["timestamp"]).hour
                interactions_by_hour[hour] += 1

            peak_hour = (
                max(interactions_by_hour.items(), key=lambda x: x[1])[0]
                if interactions_by_hour
                else None
            )

            summary = {
                "success": True,
                "date": date,
                "statistics": {
                    "total_active_time_hours": round(total_active_time, 2),
                    "total_interactions": total_interactions,
                    "total_breaks": total_breaks,
                    "break_time_hours": round(break_time, 2),
                    "productivity_score": round(productivity_score, 2),
                    "peak_activity_hour": peak_hour,
                },
                "top_applications": top_apps_formatted,
                "activity_patterns": dict(interactions_by_hour),
                "breaks": day_data.get("breaks", []),
            }

            # Generate AI narrative if available
            if AI_MODULE_AVAILABLE:
                try:
                    narrative = await self._generate_ai_summary(summary)
                    summary["ai_narrative"] = narrative
                except Exception as e:
                    logger.error(f"Error generating AI narrative: {e}")
                    summary["ai_narrative"] = None

            return summary

        except Exception as e:
            logger.error(f"Error generating day summary: {e}")
            return {"success": False, "error": str(e)}

    async def _generate_ai_summary(self, summary_data: dict) -> str | None:
        """Wygeneruj narracyjne podsumowanie używając AI."""
        try:
            from collections import deque

            # Prepare context for AI
            stats = summary_data["statistics"]
            top_apps = summary_data["top_applications"]

            context = f"""
Przygotuj krótkie, naturalne podsumowanie dnia użytkownika {self.user_name} na podstawie danych:

Statystyki:
- Czas aktywności: {stats['total_active_time_hours']} godzin
- Liczba interakcji: {stats['total_interactions']}
- Liczba przerw: {stats['total_breaks']}
- Czas przerw: {stats['break_time_hours']} godzin
- Wynik produktywności: {stats['productivity_score']} (0-1)
- Najbardziej aktywna godzina: {stats.get('peak_activity_hour', 'brak danych')}

Najczęściej używane aplikacje:
{', '.join([f"{app} ({hours:.1f}h)" for app, hours in top_apps[:3]])}

Napisz krótkie, przyjazne podsumowanie w 2-3 zdaniach po polsku.
"""

            # Generate AI response
            response_json = generate_response(
                conversation_history=deque([{"role": "user", "content": context}]),
                detected_language="pl",
                user_name=self.user_name,
            )

            if response_json:
                try:
                    response_data = json.loads(response_json)
                    return response_data.get("text", "")
                except json.JSONDecodeError:
                    return response_json  # Fallback to raw response

            return None

        except Exception as e:
            logger.error(f"Error generating AI summary: {e}")
            return None

    async def get_week_summary(self) -> dict[str, Any]:
        """Wygeneruj podsumowanie tygodnia."""
        try:
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=6)  # Last 7 days

            week_data = []
            total_stats = {
                "total_active_time": 0.0,
                "total_interactions": 0,
                "total_breaks": 0,
                "average_productivity": 0.0,
            }

            # Collect data for each day
            for i in range(7):
                date = start_date + timedelta(days=i)
                day_summary = await self.generate_day_summary(date.isoformat())

                if day_summary.get("success"):
                    week_data.append(day_summary)
                    stats = day_summary["statistics"]
                    total_stats["total_active_time"] += stats["total_active_time_hours"]
                    total_stats["total_interactions"] += stats["total_interactions"]
                    total_stats["total_breaks"] += stats["total_breaks"]
                    total_stats["average_productivity"] += stats["productivity_score"]

            # Calculate averages
            days_with_data = len(week_data)
            if days_with_data > 0:
                total_stats["average_productivity"] /= days_with_data
                total_stats["average_active_time_per_day"] = (
                    total_stats["total_active_time"] / days_with_data
                )
                total_stats["average_interactions_per_day"] = (
                    total_stats["total_interactions"] / days_with_data
                )

            return {
                "success": True,
                "period": f"{start_date} to {end_date}",
                "days_with_data": days_with_data,
                "total_statistics": total_stats,
                "daily_summaries": week_data,
            }

        except Exception as e:
            logger.error(f"Error generating week summary: {e}")
            return {"success": False, "error": str(e)}

    def stop_tracking(self):
        """Zatrzymaj śledzenie."""
        self._stop_tracking = True
        if self._tracking_thread:
            self._tracking_thread.join(timeout=5)
        logger.info("Day summary tracking stopped")


# Plugin functions for function calling system
async def get_day_summary(user_id: str, date: str = None) -> dict[str, Any]:
    """Pobierz podsumowanie dnia."""
    try:
        from server_main import server_app

        if hasattr(server_app, "day_summary"):
            return await server_app.day_summary.generate_day_summary(date)
        else:
            return {"success": False, "error": "Day summary module not available"}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def get_week_summary(user_id: str) -> dict[str, Any]:
    """Pobierz podsumowanie tygodnia."""
    try:
        from server_main import server_app

        if hasattr(server_app, "day_summary"):
            return await server_app.day_summary.get_week_summary()
        else:
            return {"success": False, "error": "Day summary module not available"}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def record_user_interaction(
    user_id: str, interaction_type: str, content: str = "", metadata: dict = None
) -> dict[str, Any]:
    """Zapisz interakcję użytkownika."""
    try:
        from server_main import server_app

        if hasattr(server_app, "day_summary"):
            await server_app.day_summary.record_interaction(
                interaction_type, content, metadata
            )
            return {"success": True, "message": "Interaction recorded"}
        else:
            return {"success": False, "error": "Day summary module not available"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# Plugin metadata
PLUGIN_FUNCTIONS = {
    "get_day_summary": {
        "function": get_day_summary,
        "description": "Get summary of user's day activities",
        "parameters": {
            "type": "object",
            "properties": {
                "date": {
                    "type": "string",
                    "description": "Date in YYYY-MM-DD format (optional, defaults to today)",
                }
            },
        },
    },
    "get_week_summary": {
        "function": get_week_summary,
        "description": "Get summary of user's week activities",
        "parameters": {"type": "object", "properties": {}},
    },
    "record_user_interaction": {
        "function": record_user_interaction,
        "description": "Record a user interaction with the assistant",
        "parameters": {
            "type": "object",
            "properties": {
                "interaction_type": {
                    "type": "string",
                    "description": "Type of interaction (voice_command, text_query, etc.)",
                },
                "content": {
                    "type": "string",
                    "description": "Content of the interaction",
                },
                "metadata": {
                    "type": "object",
                    "description": "Additional metadata about the interaction",
                },
            },
            "required": ["interaction_type"],
        },
    },
}
