"""User Behavior Learning Module for Asystent System uczący się nawyków i wzorców
zachowań użytkownika."""

import json
import logging
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class UserBehaviorModule:
    """Moduł uczący się zachowań i nawyków użytkownika."""

    def __init__(self, config: dict[str, Any], db_manager=None):
        self.config = config
        self.db_manager = db_manager
        self.enabled = config.get("user_behavior", {}).get("enabled", True)
        self.user_name = config.get("USER_NAME", "User")

        # Learning configuration
        self.learning_period_days = config.get("user_behavior", {}).get(
            "learning_period_days", 30
        )
        self.min_data_points = config.get("user_behavior", {}).get("min_data_points", 5)
        self.confidence_threshold = config.get("user_behavior", {}).get(
            "confidence_threshold", 0.7
        )

        # Data storage
        self.data_dir = Path("user_data") / "behavior_patterns"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Behavior patterns file
        self.patterns_file = self.data_dir / "learned_patterns.json"
        self.raw_data_file = self.data_dir / "behavior_data.json"

        # Current patterns
        self.learned_patterns = {
            "daily_routines": {},
            "work_patterns": {},
            "break_patterns": {},
            "interaction_patterns": {},
            "application_preferences": {},
            "time_preferences": {},
            "productivity_patterns": {},
        }

        # Raw behavior data
        self.behavior_data = {
            "daily_sessions": [],  # When user starts/stops working
            "interaction_times": [],  # When user interacts with assistant
            "application_usage": [],  # What apps are used when
            "break_patterns": [],  # When and how long breaks are taken
            "productivity_scores": [],  # Daily productivity over time
            "wake_times": [],  # When user becomes active
            "work_end_times": [],  # When user stops working
        }

        # Real-time tracking
        self.current_session = {
            "start_time": None,
            "interactions": [],
            "applications_used": [],
            "breaks_taken": [],
            "productivity_events": [],
        }

        # Pattern confidence scores
        self.pattern_confidence = {}

    async def initialize(self):
        """Inicjalizacja modułu uczenia się."""
        if not self.enabled:
            logger.info("User Behavior Learning Module disabled")
            return

        # Load existing patterns and data
        await self._load_patterns()
        await self._load_behavior_data()

        # Start current session
        self._start_session()

        logger.info("User Behavior Learning Module initialized")

    def _start_session(self):
        """Rozpocznij nową sesję użytkownika."""
        self.current_session = {
            "start_time": datetime.now().isoformat(),
            "interactions": [],
            "applications_used": [],
            "breaks_taken": [],
            "productivity_events": [],
        }
        logger.debug("New behavior learning session started")

    async def _load_patterns(self):
        """Załaduj nauczone wzorce."""
        if self.patterns_file.exists():
            try:
                with open(self.patterns_file, encoding="utf-8") as f:
                    data = json.load(f)
                    self.learned_patterns = data.get("patterns", self.learned_patterns)
                    self.pattern_confidence = data.get("confidence", {})
                logger.info("Loaded existing behavior patterns")
            except Exception as e:
                logger.error(f"Error loading behavior patterns: {e}")

    async def _load_behavior_data(self):
        """Załaduj surowe dane o zachowaniach."""
        if self.raw_data_file.exists():
            try:
                with open(self.raw_data_file, encoding="utf-8") as f:
                    self.behavior_data = json.load(f)
                logger.info("Loaded existing behavior data")
            except Exception as e:
                logger.error(f"Error loading behavior data: {e}")

    async def _save_patterns(self):
        """Zapisz nauczone wzorce."""
        try:
            data = {
                "patterns": self.learned_patterns,
                "confidence": self.pattern_confidence,
                "last_updated": datetime.now().isoformat(),
            }
            with open(self.patterns_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error saving behavior patterns: {e}")

    async def _save_behavior_data(self):
        """Zapisz surowe dane o zachowaniach."""
        try:
            # Limit data size - keep only recent data
            cutoff_date = datetime.now() - timedelta(days=self.learning_period_days * 2)

            # Filter old data
            for key in self.behavior_data:
                if isinstance(self.behavior_data[key], list):
                    self.behavior_data[key] = [
                        item
                        for item in self.behavior_data[key]
                        if datetime.fromisoformat(item.get("timestamp", "1970-01-01"))
                        > cutoff_date
                    ]

            with open(self.raw_data_file, "w", encoding="utf-8") as f:
                json.dump(self.behavior_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error saving behavior data: {e}")

    async def record_interaction(
        self, interaction_type: str, content: str = "", metadata: dict = None
    ):
        """Zapisz interakcję użytkownika."""
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "type": interaction_type,
            "content": content[:100],  # Limit length
            "metadata": metadata or {},
            "hour": datetime.now().hour,
            "day_of_week": datetime.now().weekday(),
            "session_id": id(self.current_session),
        }

        # Add to current session
        self.current_session["interactions"].append(interaction)

        # Add to historical data
        self.behavior_data["interaction_times"].append(interaction)

        logger.debug(
            f"Recorded interaction: {interaction_type} at {interaction['hour']}:xx"
        )

    async def record_application_usage(self, app_name: str, duration_seconds: float):
        """Zapisz użycie aplikacji."""
        usage = {
            "timestamp": datetime.now().isoformat(),
            "application": app_name,
            "duration_seconds": duration_seconds,
            "hour": datetime.now().hour,
            "day_of_week": datetime.now().weekday(),
            "session_id": id(self.current_session),
        }

        # Add to current session
        self.current_session["applications_used"].append(usage)

        # Add to historical data
        self.behavior_data["application_usage"].append(usage)

        logger.debug(f"Recorded app usage: {app_name} for {duration_seconds}s")

    async def record_break(self, duration_minutes: float):
        """Zapisz przerwę w pracy."""
        break_data = {
            "timestamp": datetime.now().isoformat(),
            "duration_minutes": duration_minutes,
            "hour": datetime.now().hour,
            "day_of_week": datetime.now().weekday(),
            "session_id": id(self.current_session),
        }

        # Add to current session
        self.current_session["breaks_taken"].append(break_data)

        # Add to historical data
        self.behavior_data["break_patterns"].append(break_data)

        logger.debug(
            f"Recorded break: {duration_minutes} minutes at {break_data['hour']}:xx"
        )

    async def record_productivity_score(self, score: float, context: dict = None):
        """Zapisz wynik produktywności."""
        productivity = {
            "timestamp": datetime.now().isoformat(),
            "score": score,
            "context": context or {},
            "hour": datetime.now().hour,
            "day_of_week": datetime.now().weekday(),
            "date": datetime.now().date().isoformat(),
        }

        # Add to current session
        self.current_session["productivity_events"].append(productivity)

        # Add to historical data
        self.behavior_data["productivity_scores"].append(productivity)

        logger.debug(f"Recorded productivity score: {score}")

    async def record_session_start(self):
        """Zapisz rozpoczęcie sesji pracy."""
        session_start = {
            "timestamp": datetime.now().isoformat(),
            "hour": datetime.now().hour,
            "day_of_week": datetime.now().weekday(),
            "date": datetime.now().date().isoformat(),
        }

        self.behavior_data["daily_sessions"].append(session_start)
        self.behavior_data["wake_times"].append(session_start)

        logger.debug(f"Recorded session start at {session_start['hour']}:xx")

    async def record_session_end(self):
        """Zapisz zakończenie sesji pracy."""
        session_end = {
            "timestamp": datetime.now().isoformat(),
            "hour": datetime.now().hour,
            "day_of_week": datetime.now().weekday(),
            "date": datetime.now().date().isoformat(),
            "session_data": self.current_session,
        }

        self.behavior_data["work_end_times"].append(session_end)

        # Save data after session ends
        await self._save_behavior_data()

        logger.debug(f"Recorded session end at {session_end['hour']}:xx")

    async def learn_patterns(self):
        """Analiza danych i uczenie się wzorców."""
        logger.info("Starting pattern learning analysis...")

        try:
            # Learn different types of patterns
            await self._learn_daily_routines()
            await self._learn_work_patterns()
            await self._learn_break_patterns()
            await self._learn_interaction_patterns()
            await self._learn_application_preferences()
            await self._learn_time_preferences()
            await self._learn_productivity_patterns()

            # Calculate confidence scores
            self._calculate_pattern_confidence()

            # Save learned patterns
            await self._save_patterns()

            logger.info("Pattern learning completed")

        except Exception as e:
            logger.error(f"Error during pattern learning: {e}")

    async def _learn_daily_routines(self):
        """Ucz się codziennych rutyn."""
        wake_times = self.behavior_data.get("wake_times", [])
        work_end_times = self.behavior_data.get("work_end_times", [])

        if len(wake_times) >= self.min_data_points:
            # Analyze wake times
            wake_hours = [
                datetime.fromisoformat(item["timestamp"]).hour
                for item in wake_times[-30:]
            ]  # Last 30 days
            avg_wake_time = statistics.mean(wake_hours)
            wake_time_std = statistics.stdev(wake_hours) if len(wake_hours) > 1 else 0

            self.learned_patterns["daily_routines"]["average_wake_time"] = {
                "hour": round(avg_wake_time, 1),
                "consistency": max(
                    0, 1 - (wake_time_std / 4)
                ),  # Normalize by 4-hour range
                "data_points": len(wake_hours),
            }

        if len(work_end_times) >= self.min_data_points:
            # Analyze work end times
            end_hours = [
                datetime.fromisoformat(item["timestamp"]).hour
                for item in work_end_times[-30:]
            ]
            avg_end_time = statistics.mean(end_hours)
            end_time_std = statistics.stdev(end_hours) if len(end_hours) > 1 else 0

            self.learned_patterns["daily_routines"]["average_work_end_time"] = {
                "hour": round(avg_end_time, 1),
                "consistency": max(0, 1 - (end_time_std / 4)),
                "data_points": len(end_hours),
            }

    async def _learn_work_patterns(self):
        """Ucz się wzorców pracy."""
        sessions = self.behavior_data.get("daily_sessions", [])

        if len(sessions) >= self.min_data_points:
            # Work days patterns
            work_days = defaultdict(int)
            work_hours_by_day = defaultdict(list)

            for session in sessions[-50:]:  # Last 50 sessions
                day_of_week = session["day_of_week"]
                hour = session["hour"]
                work_days[day_of_week] += 1
                work_hours_by_day[day_of_week].append(hour)

            # Most common work days
            total_sessions = sum(work_days.values())
            work_day_preferences = {
                day: count / total_sessions for day, count in work_days.items()
            }

            # Average start time by day
            avg_start_times = {
                day: statistics.mean(hours) if hours else 0
                for day, hours in work_hours_by_day.items()
            }

            self.learned_patterns["work_patterns"] = {
                "work_day_preferences": work_day_preferences,
                "average_start_times_by_day": avg_start_times,
                "most_active_days": sorted(
                    work_days.items(), key=lambda x: x[1], reverse=True
                )[:3],
            }

    async def _learn_break_patterns(self):
        """Ucz się wzorców przerw."""
        breaks = self.behavior_data.get("break_patterns", [])

        if len(breaks) >= self.min_data_points:
            # Break timing patterns
            break_hours = [break_data["hour"] for break_data in breaks[-100:]]
            break_durations = [
                break_data["duration_minutes"] for break_data in breaks[-100:]
            ]

            # Most common break times
            break_hour_counter = Counter(break_hours)
            common_break_hours = break_hour_counter.most_common(5)

            # Average break duration
            avg_break_duration = statistics.mean(break_durations)

            # Break frequency by day of week
            break_by_day = defaultdict(list)
            for break_data in breaks[-100:]:
                break_by_day[break_data["day_of_week"]].append(break_data)

            avg_breaks_per_day = {
                day: len(day_breaks)
                / max(1, len({item["timestamp"][:10] for item in day_breaks}))
                for day, day_breaks in break_by_day.items()
            }

            self.learned_patterns["break_patterns"] = {
                "common_break_hours": common_break_hours,
                "average_break_duration": avg_break_duration,
                "average_breaks_per_day": avg_breaks_per_day,
                "break_frequency": len(breaks)
                / max(1, len({item["timestamp"][:10] for item in breaks})),
            }

    async def _learn_interaction_patterns(self):
        """Ucz się wzorców interakcji."""
        interactions = self.behavior_data.get("interaction_times", [])

        if len(interactions) >= self.min_data_points:
            # Interaction frequency by hour
            interaction_hours = [
                interaction["hour"] for interaction in interactions[-200:]
            ]
            hour_counter = Counter(interaction_hours)

            # Peak interaction hours
            peak_hours = hour_counter.most_common(5)

            # Interaction types frequency
            interaction_types = [
                interaction["type"] for interaction in interactions[-200:]
            ]
            type_counter = Counter(interaction_types)

            # Daily interaction frequency
            interactions_by_date = defaultdict(int)
            for interaction in interactions[-200:]:
                date = interaction["timestamp"][:10]
                interactions_by_date[date] += 1

            avg_daily_interactions = (
                statistics.mean(interactions_by_date.values())
                if interactions_by_date
                else 0
            )

            self.learned_patterns["interaction_patterns"] = {
                "peak_interaction_hours": peak_hours,
                "common_interaction_types": type_counter.most_common(10),
                "average_daily_interactions": avg_daily_interactions,
                "interaction_distribution": dict(hour_counter),
            }

    async def _learn_application_preferences(self):
        """Ucz się preferencji aplikacji."""
        app_usage = self.behavior_data.get("application_usage", [])

        if len(app_usage) >= self.min_data_points:
            # Total time per application
            app_time = defaultdict(float)
            app_frequency = defaultdict(int)

            for usage in app_usage[-500:]:  # Last 500 app usage records
                app_time[usage["application"]] += usage["duration_seconds"]
                app_frequency[usage["application"]] += 1

            # Most used applications by time
            top_apps_by_time = sorted(
                app_time.items(), key=lambda x: x[1], reverse=True
            )[:10]

            # Most frequently used applications
            top_apps_by_frequency = sorted(
                app_frequency.items(), key=lambda x: x[1], reverse=True
            )[:10]

            # Application usage by time of day
            app_by_hour = defaultdict(lambda: defaultdict(float))
            for usage in app_usage[-500:]:
                hour = usage["hour"]
                app = usage["application"]
                app_by_hour[hour][app] += usage["duration_seconds"]

            self.learned_patterns["application_preferences"] = {
                "top_applications_by_time": top_apps_by_time,
                "top_applications_by_frequency": top_apps_by_frequency,
                "usage_by_hour": {
                    hour: sorted(apps.items(), key=lambda x: x[1], reverse=True)[:3]
                    for hour, apps in app_by_hour.items()
                },
            }

    async def _learn_time_preferences(self):
        """Ucz się preferencji czasowych."""
        all_activities = []

        # Collect all timestamped activities
        for activity_type, activities in self.behavior_data.items():
            if isinstance(activities, list):
                for activity in activities:
                    if "timestamp" in activity:
                        all_activities.append(
                            {
                                "timestamp": activity["timestamp"],
                                "hour": activity.get(
                                    "hour",
                                    datetime.fromisoformat(activity["timestamp"]).hour,
                                ),
                                "type": activity_type,
                            }
                        )

        if len(all_activities) >= self.min_data_points:
            # Activity distribution by hour
            activity_by_hour = defaultdict(int)
            for activity in all_activities[-1000:]:  # Last 1000 activities
                activity_by_hour[activity["hour"]] += 1

            # Find peak activity periods
            if activity_by_hour:
                max_activity = max(activity_by_hour.values())
                peak_hours = [
                    hour
                    for hour, count in activity_by_hour.items()
                    if count > max_activity * 0.7
                ]

                # Morning/afternoon/evening preferences
                morning_activity = sum(
                    count for hour, count in activity_by_hour.items() if 6 <= hour <= 12
                )
                afternoon_activity = sum(
                    count
                    for hour, count in activity_by_hour.items()
                    if 12 <= hour <= 18
                )
                evening_activity = sum(
                    count
                    for hour, count in activity_by_hour.items()
                    if 18 <= hour <= 24
                )

                total_activity = (
                    morning_activity + afternoon_activity + evening_activity
                )

                if total_activity > 0:
                    time_preferences = {
                        "morning": morning_activity / total_activity,
                        "afternoon": afternoon_activity / total_activity,
                        "evening": evening_activity / total_activity,
                    }

                    preferred_time = max(time_preferences.items(), key=lambda x: x[1])[
                        0
                    ]

                    self.learned_patterns["time_preferences"] = {
                        "peak_activity_hours": peak_hours,
                        "time_distribution": time_preferences,
                        "preferred_time_period": preferred_time,
                        "activity_by_hour": dict(activity_by_hour),
                    }

    async def _learn_productivity_patterns(self):
        """Ucz się wzorców produktywności."""
        productivity_data = self.behavior_data.get("productivity_scores", [])

        if len(productivity_data) >= self.min_data_points:
            # Productivity by hour
            productivity_by_hour = defaultdict(list)
            productivity_by_day = defaultdict(list)

            for score_data in productivity_data[-100:]:
                hour = score_data["hour"]
                day = score_data["day_of_week"]
                score = score_data["score"]

                productivity_by_hour[hour].append(score)
                productivity_by_day[day].append(score)

            # Average productivity by hour
            avg_productivity_by_hour = {
                hour: statistics.mean(scores)
                for hour, scores in productivity_by_hour.items()
                if scores
            }

            # Average productivity by day
            avg_productivity_by_day = {
                day: statistics.mean(scores)
                for day, scores in productivity_by_day.items()
                if scores
            }

            # Find most productive times
            if avg_productivity_by_hour:
                most_productive_hours = sorted(
                    avg_productivity_by_hour.items(), key=lambda x: x[1], reverse=True
                )[:5]

                most_productive_days = sorted(
                    avg_productivity_by_day.items(), key=lambda x: x[1], reverse=True
                )[:3]

                self.learned_patterns["productivity_patterns"] = {
                    "most_productive_hours": most_productive_hours,
                    "most_productive_days": most_productive_days,
                    "productivity_by_hour": avg_productivity_by_hour,
                    "productivity_by_day": avg_productivity_by_day,
                    "overall_average": statistics.mean(
                        [score["score"] for score in productivity_data[-50:]]
                    ),
                }

    def _calculate_pattern_confidence(self):
        """Oblicz poziom pewności dla każdego wzorca."""
        for pattern_category, patterns in self.learned_patterns.items():
            if not patterns:
                self.pattern_confidence[pattern_category] = 0.0
                continue

            # Base confidence on data points and consistency
            confidence = 0.0

            for pattern_name, pattern_data in patterns.items():
                if isinstance(pattern_data, dict):
                    # Check for data points
                    data_points = pattern_data.get("data_points", 0)
                    consistency = pattern_data.get("consistency", 0)

                    if data_points >= self.min_data_points:
                        confidence += (
                            min(1.0, data_points / (self.min_data_points * 2)) * 0.5
                        )

                    if consistency > 0:
                        confidence += consistency * 0.5

            # Average confidence for the category
            num_patterns = len(patterns)
            if num_patterns > 0:
                confidence = confidence / num_patterns

            self.pattern_confidence[pattern_category] = min(1.0, confidence)

    async def get_predictions(self) -> dict[str, Any]:
        """Wygeneruj przewidywania na podstawie nauczonych wzorców."""
        predictions = {}
        current_hour = datetime.now().hour
        current_day = datetime.now().weekday()

        try:
            # Predict next break time
            break_patterns = self.learned_patterns.get("break_patterns", {})
            if (
                break_patterns
                and self.pattern_confidence.get("break_patterns", 0)
                > self.confidence_threshold
            ):
                common_break_hours = break_patterns.get("common_break_hours", [])
                if common_break_hours:
                    next_break_hours = [
                        hour for hour, _ in common_break_hours if hour > current_hour
                    ]
                    if next_break_hours:
                        predictions["next_likely_break"] = {
                            "hour": min(next_break_hours),
                            "confidence": self.pattern_confidence["break_patterns"],
                        }

            # Predict productivity level
            productivity_patterns = self.learned_patterns.get(
                "productivity_patterns", {}
            )
            if (
                productivity_patterns
                and self.pattern_confidence.get("productivity_patterns", 0)
                > self.confidence_threshold
            ):
                productivity_by_hour = productivity_patterns.get(
                    "productivity_by_hour", {}
                )
                if str(current_hour) in productivity_by_hour:
                    predictions["current_hour_productivity"] = {
                        "expected_score": productivity_by_hour[str(current_hour)],
                        "confidence": self.pattern_confidence["productivity_patterns"],
                    }

            # Predict work end time
            daily_routines = self.learned_patterns.get("daily_routines", {})
            if (
                daily_routines
                and self.pattern_confidence.get("daily_routines", 0)
                > self.confidence_threshold
            ):
                work_end = daily_routines.get("average_work_end_time", {})
                if work_end:
                    predictions["expected_work_end"] = {
                        "hour": work_end["hour"],
                        "confidence": work_end["consistency"],
                    }

            return {
                "success": True,
                "predictions": predictions,
                "confidence_scores": self.pattern_confidence,
            }

        except Exception as e:
            logger.error(f"Error generating predictions: {e}")
            return {"success": False, "error": str(e)}

    async def get_behavior_insights(self) -> dict[str, Any]:
        """Zwróć szczegółowe wglądy w zachowania użytkownika."""
        try:
            insights = {
                "patterns": self.learned_patterns,
                "confidence_scores": self.pattern_confidence,
                "data_summary": {
                    "total_sessions": len(self.behavior_data.get("daily_sessions", [])),
                    "total_interactions": len(
                        self.behavior_data.get("interaction_times", [])
                    ),
                    "total_app_usage_records": len(
                        self.behavior_data.get("application_usage", [])
                    ),
                    "total_breaks": len(self.behavior_data.get("break_patterns", [])),
                    "learning_period_days": self.learning_period_days,
                },
                "recommendations": await self._generate_recommendations(),
            }

            return {"success": True, "insights": insights}

        except Exception as e:
            logger.error(f"Error getting behavior insights: {e}")
            return {"success": False, "error": str(e)}

    async def _generate_recommendations(self) -> list[str]:
        """Wygeneruj rekomendacje na podstawie nauczonych wzorców."""
        recommendations = []

        try:
            # Productivity recommendations
            productivity_patterns = self.learned_patterns.get(
                "productivity_patterns", {}
            )
            if productivity_patterns:
                most_productive_hours = productivity_patterns.get(
                    "most_productive_hours", []
                )
                if most_productive_hours:
                    best_hour = most_productive_hours[0][0]
                    recommendations.append(
                        f"Twoja najbardziej produktywna godzina to {best_hour}:00. "
                        f"Planuj najważniejsze zadania na ten czas."
                    )

            # Break recommendations
            break_patterns = self.learned_patterns.get("break_patterns", {})
            if break_patterns:
                avg_break_duration = break_patterns.get("average_break_duration", 0)
                if avg_break_duration > 30:
                    recommendations.append(
                        f"Twoje przerwy są dość długie (średnio {avg_break_duration:.1f} min). "
                        "Rozważ krótsze, ale częstsze przerwy dla lepszej produktywności."
                    )
                elif avg_break_duration < 5:
                    recommendations.append(
                        "Robisz bardzo krótkie przerwy. Rozważ dłuższe przerwy dla lepszej regeneracji."
                    )

            # Work pattern recommendations
            work_patterns = self.learned_patterns.get("work_patterns", {})
            if work_patterns:
                work_day_prefs = work_patterns.get("work_day_preferences", {})
                if work_day_prefs:
                    # Find least active day
                    least_active_day = min(work_day_prefs.items(), key=lambda x: x[1])
                    if least_active_day[1] < 0.8:  # Less than 80% of average activity
                        day_names = [
                            "poniedziałek",
                            "wtorek",
                            "środa",
                            "czwartek",
                            "piątek",
                            "sobota",
                            "niedziela",
                        ]
                        recommendations.append(
                            f"W {day_names[least_active_day[0]]} jesteś mniej aktywny. "
                            f"Zaplanuj lżejsze zadania na ten dzień."
                        )

            return recommendations

        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return ["Błąd podczas generowania rekomendacji."]


# Plugin functions for function calling system
async def get_behavior_insights(user_id: str) -> dict[str, Any]:
    """Pobierz wglądy w zachowania użytkownika."""
    try:
        from server_main import server_app

        if hasattr(server_app, "user_behavior"):
            return await server_app.user_behavior.get_behavior_insights()
        else:
            return {"success": False, "error": "User behavior module not available"}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def get_behavior_predictions(user_id: str) -> dict[str, Any]:
    """Pobierz przewidywania zachowań."""
    try:
        from server_main import server_app

        if hasattr(server_app, "user_behavior"):
            return await server_app.user_behavior.get_predictions()
        else:
            return {"success": False, "error": "User behavior module not available"}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def learn_user_patterns(user_id: str) -> dict[str, Any]:
    """Uruchom proces uczenia się wzorców."""
    try:
        from server_main import server_app

        if hasattr(server_app, "user_behavior"):
            await server_app.user_behavior.learn_patterns()
            return {"success": True, "message": "Pattern learning completed"}
        else:
            return {"success": False, "error": "User behavior module not available"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# Plugin metadata
PLUGIN_FUNCTIONS = {
    "get_behavior_insights": {
        "function": get_behavior_insights,
        "description": "Get detailed insights into user behavior patterns",
        "parameters": {"type": "object", "properties": {}},
    },
    "get_behavior_predictions": {
        "function": get_behavior_predictions,
        "description": "Get predictions based on learned user behavior patterns",
        "parameters": {"type": "object", "properties": {}},
    },
    "learn_user_patterns": {
        "function": learn_user_patterns,
        "description": "Trigger learning process for user behavior patterns",
        "parameters": {"type": "object", "properties": {}},
    },
}
