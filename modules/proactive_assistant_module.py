"""Proactive Assistant Module Automatically suggests actions, predicts user needs and
sends notifications."""

import asyncio
import json
import logging
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger


@dataclass
class ProactiveNotification:
    """Struktura dla proaktywnych powiadomień."""

    type: str  # 'wellness', 'productivity', 'routine', 'prediction', 'optimization'
    priority: str  # 'low', 'medium', 'high', 'critical'
    title: str
    message: str
    suggested_action: str | None = None
    context: dict | None = None
    timestamp: datetime | None = None
    dismissed: bool = False


@dataclass
class UserContext:
    """Kontekst użytkownika dla predykcji."""

    current_activity: str | None = None
    work_intensity: float = 0.0  # 0-1
    break_needed: bool = False
    focus_level: float = 0.0  # 0-1
    energy_level: float = 0.0  # 0-1
    productivity_trend: str = "stable"  # 'increasing', 'decreasing', 'stable'
    last_break_time: datetime | None = None
    session_duration: int = 0  # minutes
    daily_goals_progress: float = 0.0  # 0-1


class ProactiveAssistantModule:
    """Moduł proaktywnego asystenta."""

    def __init__(self, config: dict, database_path: str = "gaja_memory.db"):
        self.config = config.get("proactive_assistant", {})
        self.database_path = database_path
        self.logger = logging.getLogger(__name__)
        self.scheduler = BackgroundScheduler()
        self.notification_queue = []
        self.user_context = UserContext()
        self.is_active = False
        self.lock = asyncio.Lock()

        # Konfiguracja domyślna
        self.default_config = {
            "enabled": True,
            "wellness_check_interval": 20,  # minuty
            "break_reminder_interval": 60,  # minuty
            "productivity_check_interval": 30,  # minuty
            "prediction_interval": 45,  # minuty
            "notification_cooldown": 5,  # minuty między powiadomieniami tego samego typu
            "work_session_max": 90,  # maksymalny czas pracy bez przerwy (minuty)
            "daily_goals_check_time": "17:00",  # sprawdzanie postępu celów
            "end_of_day_summary_time": "18:00",  # podsumowanie dnia
            "wellness_thresholds": {
                "break_needed_after": 75,  # minuty
                "low_energy_threshold": 0.3,
                "low_focus_threshold": 0.4,
            },
            "proactive_features": {
                "wellness_monitoring": True,
                "break_reminders": True,
                "productivity_optimization": True,
                "routine_suggestions": True,
                "goal_tracking": True,
                "context_awareness": True,
            },
        }

        # Merge z konfiguracją użytkownika
        for key, value in self.default_config.items():
            if key not in self.config:
                self.config[key] = value

        self.init_database()
        self.logger.info("ProactiveAssistantModule zainicjalizowany")

    def init_database(self):
        """Inicjalizacja tabel bazy danych."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()

            # Tabela dla proaktywnych powiadomień
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS proactive_notifications (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    type TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    title TEXT NOT NULL,
                    message TEXT NOT NULL,
                    suggested_action TEXT,
                    context TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    dismissed BOOLEAN DEFAULT FALSE,
                    user_response TEXT
                )
            """
            )

            # Tabela dla kontekstu użytkownika
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS user_context_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    activity TEXT,
                    work_intensity REAL,
                    focus_level REAL,
                    energy_level REAL,
                    session_duration INTEGER,
                    break_taken BOOLEAN DEFAULT FALSE
                )
            """
            )

            # Tabela dla predykcji i sugestii
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS proactive_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    prediction_type TEXT NOT NULL,
                    prediction_data TEXT,
                    confidence REAL,
                    outcome TEXT,
                    accuracy REAL
                )
            """
            )

            conn.commit()
            conn.close()
            self.logger.info("Proactive assistant database initialized")

        except Exception as e:
            self.logger.error(f"Error initializing database: {e}")

    def start(self):
        """Uruchomienie proaktywnego asystenta."""
        if not self.config.get("enabled", True):
            self.logger.info("Proactive assistant disabled in configuration")
            return

        try:
            self.is_active = True

            # Konfiguracja schedulera
            if self.config["proactive_features"]["wellness_monitoring"]:
                self.scheduler.add_job(
                    self.wellness_check,
                    IntervalTrigger(minutes=self.config["wellness_check_interval"]),
                    id="wellness_check",
                    max_instances=1,
                )

            if self.config["proactive_features"]["break_reminders"]:
                self.scheduler.add_job(
                    self.break_reminder_check,
                    IntervalTrigger(minutes=self.config["break_reminder_interval"]),
                    id="break_reminder",
                    max_instances=1,
                )

            if self.config["proactive_features"]["productivity_optimization"]:
                self.scheduler.add_job(
                    self.productivity_check,
                    IntervalTrigger(minutes=self.config["productivity_check_interval"]),
                    id="productivity_check",
                    max_instances=1,
                )

            if self.config["proactive_features"]["routine_suggestions"]:
                self.scheduler.add_job(
                    self.routine_optimization_check,
                    IntervalTrigger(minutes=self.config["prediction_interval"]),
                    id="routine_optimization",
                    max_instances=1,
                )

            if self.config["proactive_features"]["goal_tracking"]:
                # Parse time format "17:00" to hour and minute
                time_parts = self.config["daily_goals_check_time"].split(":")
                hour = int(time_parts[0])
                minute = int(time_parts[1])
                self.scheduler.add_job(
                    self.daily_goals_check,
                    CronTrigger(hour=hour, minute=minute),
                    id="daily_goals_check",
                    max_instances=1,
                )

            # Podsumowanie dnia
            time_parts = self.config["end_of_day_summary_time"].split(":")
            hour = int(time_parts[0])
            minute = int(time_parts[1])
            self.scheduler.add_job(
                self.end_of_day_summary,
                CronTrigger(hour=hour, minute=minute),
                id="end_of_day_summary",
                max_instances=1,
            )

            self.scheduler.start()
            self.logger.info("Proactive assistant started")

        except Exception as e:
            self.logger.error(f"Error starting proactive assistant: {e}")

    def stop(self):
        """Zatrzymanie proaktywnego asystenta."""
        try:
            self.is_active = False
            if self.scheduler.running:
                self.scheduler.shutdown()
            self.logger.info("Proactive assistant stopped")
        except Exception as e:
            self.logger.error(f"Error stopping proactive assistant: {e}")

    def update_user_context(self, context_data: dict):
        """Aktualizacja kontekstu użytkownika."""
        try:
            with self.lock:
                if "activity" in context_data:
                    self.user_context.current_activity = context_data["activity"]
                if "work_intensity" in context_data:
                    self.user_context.work_intensity = context_data["work_intensity"]
                if "focus_level" in context_data:
                    self.user_context.focus_level = context_data["focus_level"]
                if "energy_level" in context_data:
                    self.user_context.energy_level = context_data["energy_level"]
                if "session_duration" in context_data:
                    self.user_context.session_duration = context_data[
                        "session_duration"
                    ]
                if "daily_goals_progress" in context_data:
                    self.user_context.daily_goals_progress = context_data[
                        "daily_goals_progress"
                    ]

                # Zapisz do bazy danych
                self._save_context_to_db()

        except Exception as e:
            self.logger.error(f"Error updating user context: {e}")

    def wellness_check(self):
        """Sprawdzenie stanu zdrowia i samopoczucia użytkownika."""
        try:
            if not self.is_active:
                return

            # Sprawdź czy potrzebna przerwa
            if (
                self.user_context.session_duration
                > self.config["wellness_thresholds"]["break_needed_after"]
            ):
                self._create_notification(
                    "wellness",
                    "medium",
                    "Czas na przerwę!",
                    f"Pracujesz już {self.user_context.session_duration} minut. Zalecam 5-10 minutową przerwę.",
                    "Weź krótką przerwę - wstań, rozciągnij się lub napij się wody",
                )

            # Sprawdź poziom energii
            if (
                self.user_context.energy_level
                < self.config["wellness_thresholds"]["low_energy_threshold"]
            ):
                self._create_notification(
                    "wellness",
                    "medium",
                    "Niski poziom energii",
                    "Wygląda na to, że Twoja energia spadła. Może czas na przerwę lub zmianę aktywności?",
                    "Rozważ krótką przerwę, przekąskę lub zmianę typu zadań",
                )

            # Sprawdź poziom koncentracji
            if (
                self.user_context.focus_level
                < self.config["wellness_thresholds"]["low_focus_threshold"]
            ):
                self._create_notification(
                    "wellness",
                    "low",
                    "Spadek koncentracji",
                    "Zauważyłem, że Twoja koncentracja może być niższa. Chcesz dostać sugestie?",
                    "Spróbuj techniki Pomodoro lub zmień środowisko pracy",
                )

        except Exception as e:
            self.logger.error(f"Error checking wellness: {e}")

    def break_reminder_check(self):
        """Sprawdzenie czy czas na przypomnienie o przerwie."""
        try:
            if not self.is_active:
                return

            if self.user_context.session_duration >= self.config["work_session_max"]:
                self._create_notification(
                    "wellness",
                    "high",
                    "Długa sesja pracy",
                    f"Pracujesz bez przerwy już {self.user_context.session_duration} minut. To może wpłynąć na Twoją produktywność.",
                    "Zalecam 15-20 minutową przerwę",
                )

        except Exception as e:
            self.logger.error(f"Error checking breaks: {e}")

    def productivity_check(self):
        """Analiza i optymalizacja produktywności."""
        try:
            if not self.is_active:
                return

            # Analiza trendu produktywności
            productivity_data = self._analyze_productivity_trend()

            if productivity_data["trend"] == "decreasing":
                self._create_notification(
                    "productivity",
                    "medium",
                    "Spadek produktywności",
                    f"Zauważyłem {productivity_data['decline_percentage']:.1f}% spadek produktywności w ostatniej godzinie.",
                    "Może warto zmienić typ zadań lub wziąć krótką przerwę?",
                )
            elif productivity_data["trend"] == "increasing":
                self._create_notification(
                    "productivity",
                    "low",
                    "Świetna produktywność!",
                    f"Twoja produktywność wzrosła o {productivity_data['improvement_percentage']:.1f}%. Dobra robota!",
                    "Kontynuuj w tym tempie, ale pamiętaj o regularnych przerwach",
                )

        except Exception as e:
            self.logger.error(f"Error checking productivity: {e}")

    def routine_optimization_check(self):
        """Sprawdzenie i optymalizacja rutyn."""
        try:
            if not self.is_active:
                return

            # Pobierz dane o rutynach z routines_learner_module
            routine_suggestions = self._get_routine_suggestions()

            if routine_suggestions:
                self._create_notification(
                    "routine",
                    "low",
                    "Sugestia optymalizacji rutyny",
                    routine_suggestions["message"],
                    routine_suggestions["action"],
                )

        except Exception as e:
            self.logger.error(f"Error optimizing routines: {e}")

    def daily_goals_check(self):
        """Sprawdzenie postępu celów dziennych."""
        try:
            if not self.is_active:
                return

            progress = self.user_context.daily_goals_progress
            current_hour = datetime.now().hour

            # Sprawdź postęp o 17:00
            if current_hour == 17:
                if progress < 0.7:  # Mniej niż 70% celów
                    self._create_notification(
                        "productivity",
                        "medium",
                        "Postęp celów dziennych",
                        f"Do końca dnia pozostało niewiele czasu, a realizacja celów to {progress*100:.0f}%.",
                        "Może warto skupić się na najważniejszych zadaniach?",
                    )
                else:
                    self._create_notification(
                        "productivity",
                        "low",
                        "Świetny postęp!",
                        f"Realizacja dziennych celów: {progress*100:.0f}%. Jesteś na dobrej drodze!",
                        "Kontynuuj dobrą pracę",
                    )

        except Exception as e:
            self.logger.error(f"Error checking goals: {e}")

    def end_of_day_summary(self):
        """Podsumowanie końca dnia."""
        try:
            if not self.is_active:
                return

            # Generuj proaktywne podsumowanie dnia
            summary_data = self._generate_proactive_day_summary()

            self._create_notification(
                "productivity",
                "medium",
                "Podsumowanie dnia",
                summary_data["message"],
                summary_data["tomorrow_suggestion"],
            )

        except Exception as e:
            self.logger.error(f"Error generating day summary: {e}")

    def predict_user_needs(self) -> list[dict]:
        """Przewidywanie potrzeb użytkownika."""
        try:
            predictions = []
            current_time = datetime.now()

            # Predykcja na podstawie wzorców czasowych
            if current_time.hour >= 14 and current_time.hour <= 16:
                if self.user_context.energy_level < 0.6:
                    predictions.append(
                        {
                            "type": "energy_dip",
                            "confidence": 0.8,
                            "message": "Popołudniowy spadek energii jest normalny",
                            "suggestion": "Rozważ lekką przekąskę lub krótką przerwę",
                        }
                    )

            # Predykcja końca sesji pracy
            if self.user_context.session_duration > 60:
                end_probability = min(self.user_context.session_duration / 120, 0.9)
                predictions.append(
                    {
                        "type": "session_end",
                        "confidence": end_probability,
                        "message": f"Prawdopodobieństwo zakończenia sesji: {end_probability*100:.0f}%",
                        "suggestion": "Przygotuj się do podsumowania i planowania następnych kroków",
                    }
                )

            return predictions

        except Exception as e:
            self.logger.error(f"Error predicting needs: {e}")
            return []

    def get_pending_notifications(self) -> list[ProactiveNotification]:
        """Pobierz oczekujące powiadomienia."""
        try:
            with self.lock:
                # Usuń stare powiadomienia (starsze niż 1 godzina)
                cutoff_time = datetime.now() - timedelta(hours=1)
                self.notification_queue = [
                    notif
                    for notif in self.notification_queue
                    if notif.timestamp and notif.timestamp > cutoff_time
                ]
                return self.notification_queue.copy()
        except Exception as e:
            self.logger.error(f"Error fetching notifications: {e}")
            return []

    def _dismiss_notification_sync(
        self, notification_id: int, user_response: str = None
    ):
        """Odrzuć powiadomienie."""
        try:
            # Usuń z kolejki
            with self.lock:
                self.notification_queue = [
                    notif
                    for notif in self.notification_queue
                    if id(notif) != notification_id
                ]

            # Zaktualizuj w bazie danych
            if user_response:
                self._update_notification_response(notification_id, user_response)

        except Exception as e:
            self.logger.error(f"Error dismissing notification: {e}")

    def _create_notification(
        self,
        type_: str,
        priority: str,
        title: str,
        message: str,
        suggested_action: str = None,
        context: dict = None,
    ):
        """Stwórz nowe powiadomienie."""
        try:
            # Sprawdź cooldown dla tego typu powiadomienia
            if self._is_notification_on_cooldown(type_):
                return

            notification = ProactiveNotification(
                type=type_,
                priority=priority,
                title=title,
                message=message,
                suggested_action=suggested_action,
                context=context,
                timestamp=datetime.now(),
            )

            with self.lock:
                self.notification_queue.append(notification)

            # Zapisz do bazy danych
            self._save_notification_to_db(notification)

            self.logger.info(f"Created proactive notification: {title}")

        except Exception as e:
            self.logger.error(f"Error creating notification: {e}")

    def _is_notification_on_cooldown(self, notification_type: str) -> bool:
        """Sprawdź czy typ powiadomienia jest w cooldown."""
        try:
            cooldown_minutes = self.config.get("notification_cooldown", 5)
            cutoff_time = datetime.now() - timedelta(minutes=cooldown_minutes)

            with self.lock:
                recent_notifications = [
                    notif
                    for notif in self.notification_queue
                    if notif.type == notification_type and notif.timestamp > cutoff_time
                ]

            return len(recent_notifications) > 0

        except Exception as e:
            self.logger.error(f"Error checking cooldown: {e}")
            return False

    def _analyze_productivity_trend(self) -> dict:
        """Analiza trendu produktywności."""
        try:
            # Domyślne dane - w rzeczywistej implementacji pobierz z user_behavior_module
            return {
                "trend": "stable",
                "decline_percentage": 0.0,
                "improvement_percentage": 0.0,
                "current_level": self.user_context.work_intensity,
            }
        except Exception as e:
            self.logger.error(f"Error analyzing productivity: {e}")
            return {
                "trend": "stable",
                "decline_percentage": 0.0,
                "improvement_percentage": 0.0,
            }

    def _get_routine_suggestions(self) -> dict | None:
        """Pobierz sugestie optymalizacji rutyn."""
        try:
            # Przykładowe sugestie - w rzeczywistej implementacji integruj z routines_learner_module
            suggestions = [
                {
                    "message": "Zauważyłem, że najproduktywniej pracujesz rano. Może warto zaplanować trudne zadania na godziny 9-11?",
                    "action": "Przenieś wymagające zadania na godziny porannej",
                },
                {
                    "message": "Twoje sesje pracy są najdłuższe we wtorek i środę. Czy chcesz zaplanować więcej przerw w te dni?",
                    "action": "Dodaj dodatkowe przypomnienia o przerwach",
                },
            ]

            # Zwróć losową sugestię jeśli dostępne
            import random

            if suggestions and random.random() < 0.3:  # 30% szans na sugestię
                return random.choice(suggestions)

            return None

        except Exception as e:
            self.logger.error(f"Error fetching routine suggestions: {e}")
            return None

    def _generate_proactive_day_summary(self) -> dict:
        """Generuj proaktywne podsumowanie dnia."""
        try:
            return {
                "message": f"Dzień dobiegł końca! Realizacja celów: {self.user_context.daily_goals_progress*100:.0f}%, sesje pracy: {self.user_context.session_duration//60}h.",
                "tomorrow_suggestion": "Na jutro sugeruję zaplanowanie najważniejszych zadań na godziny poranne.",
            }
        except Exception as e:
            self.logger.error(f"Error generating summary: {e}")
            return {
                "message": "Koniec dnia!",
                "tomorrow_suggestion": "Miłego wieczoru!",
            }

    def _save_context_to_db(self):
        """Zapisz kontekst do bazy danych."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO user_context_history
                (activity, work_intensity, focus_level, energy_level, session_duration)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    self.user_context.current_activity,
                    self.user_context.work_intensity,
                    self.user_context.focus_level,
                    self.user_context.energy_level,
                    self.user_context.session_duration,
                ),
            )

            conn.commit()
            conn.close()

        except Exception as e:
            self.logger.error(f"Error saving context: {e}")

    def _save_notification_to_db(self, notification: ProactiveNotification):
        """Zapisz powiadomienie do bazy danych."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO proactive_notifications
                (type, priority, title, message, suggested_action, context, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    notification.type,
                    notification.priority,
                    notification.title,
                    notification.message,
                    notification.suggested_action,
                    json.dumps(notification.context) if notification.context else None,
                    notification.timestamp,
                ),
            )

            conn.commit()
            conn.close()

        except Exception as e:
            self.logger.error(f"Error saving notification: {e}")

    def _update_notification_response(self, notification_id: int, user_response: str):
        """Zaktualizuj odpowiedź użytkownika na powiadomienie."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                UPDATE proactive_notifications
                SET dismissed = TRUE, user_response = ?
                WHERE id = ?
            """,
                (user_response, notification_id),
            )

            conn.commit()
            conn.close()

        except Exception as e:
            self.logger.error(f"Error updating response: {e}")

    def get_status(self) -> dict:
        """Pobierz status modułu."""
        return {
            "active": self.is_active,
            "scheduler_running": (
                self.scheduler.running if hasattr(self.scheduler, "running") else False
            ),
            "pending_notifications": len(self.notification_queue),
            "user_context": asdict(self.user_context),
            "config": self.config,
        }

    # Async interface methods for integration with server
    async def get_notifications(self, user_id: str) -> list[dict]:
        """Pobierz powiadomienia dla użytkownika (async interface)"""
        try:
            notifications = self.get_pending_notifications()
            # Convert to dict format for JSON serialization
            return [
                {
                    "id": notif.id,
                    "type": notif.type,
                    "priority": notif.priority,
                    "title": notif.title,
                    "message": notif.message,
                    "suggested_action": notif.suggested_action,
                    "timestamp": (
                        notif.timestamp.isoformat() if notif.timestamp else None
                    ),
                    "context": notif.context,
                }
                for notif in notifications
            ]
        except Exception as e:
            self.logger.error(f"Error fetching notifications: {e}")
            return []

    async def add_notification(
        self,
        user_id: str,
        notification_type: str,
        message: str,
        priority: str = "medium",
        title: str = None,
        suggested_action: str = None,
        context: dict = None,
    ):
        """Dodaj powiadomienie (async interface)"""
        try:
            if not title:
                title = f"{notification_type.title()} Notification"

            notification = self._create_notification(
                notification_type, priority, title, message, suggested_action, context
            )
            if notification:
                self._save_notification_to_db(notification)
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error adding notification: {e}")
            return False

    async def dismiss_notification(self, user_id: str, notification_id: str) -> bool:
        """Odrzuć powiadomienie (async interface)"""
        try:
            # Convert string ID to int if needed
            if isinstance(notification_id, str):
                notification_id = int(notification_id)

            self._dismiss_notification_sync(notification_id)
            return True
        except Exception as e:
            self.logger.error(f"Error dismissing notification: {e}")
            return False

    async def update_user_context_async(self, user_id: str, context_data: dict):
        """Aktualizuj kontekst użytkownika (async interface)"""
        try:
            # Update the user context object
            if "activity" in context_data:
                self.user_context.current_activity = context_data["activity"]
            if "work_intensity" in context_data:
                self.user_context.work_intensity = context_data["work_intensity"]
            if "focus_level" in context_data:
                self.user_context.focus_level = context_data["focus_level"]
            if "energy_level" in context_data:
                self.user_context.energy_level = context_data["energy_level"]

            # Save to database
            await self._save_context_to_db(context_data)

        except Exception as e:
            self.logger.error(f"Error updating context: {e}")

    async def get_predictions(self, user_id: str) -> list[dict]:
        """Pobierz predykcje dla użytkownika (async interface)"""
        try:
            # For now, return some sample predictions
            # In a full implementation, this would analyze user patterns and generate real predictions
            predictions = [
                {
                    "type": "productivity",
                    "message": "Based on your pattern, your productivity typically peaks around 10 AM.",
                    "confidence": 0.8,
                    "suggested_action": "Schedule important tasks for 10-11 AM",
                },
                {
                    "type": "wellness",
                    "message": "You usually need a break after 45 minutes of focused work.",
                    "confidence": 0.75,
                    "suggested_action": "Set a 45-minute work timer",
                },
            ]
            return predictions
        except Exception as e:
            self.logger.error(f"Error generating predictions: {e}")
            return []

    async def _save_context_to_db(self, context_data: dict):
        """Zapisz kontekst do bazy danych."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO user_context_history
                (activity, work_intensity, focus_level, energy_level, session_duration)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    context_data.get("activity"),
                    context_data.get("work_intensity"),
                    context_data.get("focus_level"),
                    context_data.get("energy_level"),
                    context_data.get("session_duration"),
                ),
            )

            conn.commit()
            conn.close()

        except Exception as e:
            self.logger.error(f"Error saving context: {e}")
