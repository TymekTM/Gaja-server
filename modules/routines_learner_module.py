"""Routines Learner Module for Asystent AI system wykrywający i analizujący powtarzalne
wzorce dnia użytkownika."""

import asyncio
import json
import logging
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    from ai_module import generate_response

    AI_MODULE_AVAILABLE = True
except ImportError:
    AI_MODULE_AVAILABLE = False
    logger.warning("AI module not available for routines learning")

try:
    import pickle

    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    KMeans = None
    DBSCAN = None
    StandardScaler = None
    pickle = None
    logger.warning("scikit-learn not available for advanced pattern recognition")

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logger.warning("NumPy not available for advanced calculations")


class RoutinesLearnerModule:
    """Moduł AI do wykrywania i uczenia się rutyn użytkownika."""

    def __init__(self, config: dict[str, Any], db_manager=None):
        self.config = config
        self.db_manager = db_manager
        self.enabled = config.get("routines_learner", {}).get("enabled", True)
        self.user_name = config.get("USER_NAME", "User")

        # AI Learning configuration
        self.min_pattern_frequency = config.get("routines_learner", {}).get(
            "min_pattern_frequency", 3
        )
        self.pattern_similarity_threshold = config.get("routines_learner", {}).get(
            "similarity_threshold", 0.8
        )
        self.learning_window_days = config.get("routines_learner", {}).get(
            "learning_window_days", 14
        )
        self.prediction_confidence_threshold = config.get("routines_learner", {}).get(
            "prediction_threshold", 0.7
        )

        # Data storage
        self.data_dir = Path("user_data") / "routines_ai"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Model files
        self.routines_file = self.data_dir / "detected_routines.json"
        self.models_file = self.data_dir / "ml_models.pkl"
        self.sequences_file = self.data_dir / "activity_sequences.json"

        # Detected routines
        self.detected_routines = {
            "morning_routines": [],
            "work_routines": [],
            "break_routines": [],
            "evening_routines": [],
            "weekly_patterns": [],
            "project_patterns": [],
        }

        # Activity sequences for pattern recognition
        self.activity_sequences = []

        # ML Models
        self.ml_models = {
            "activity_clustering": None,
            "sequence_predictor": None,
            "routine_classifier": None,
        }

        # Real-time tracking
        self.current_sequence = []
        self.sequence_start_time = None
        self.last_activity_time = None

        # Pattern templates
        self.routine_templates = {
            "morning": {
                "time_range": (6, 12),
                "typical_activities": [
                    "start_session",
                    "first_interaction",
                    "check_briefing",
                ],
                "duration_range": (30, 180),  # minutes
            },
            "work_block": {
                "time_range": (9, 18),
                "typical_activities": [
                    "focused_work",
                    "application_usage",
                    "productivity",
                ],
                "duration_range": (60, 480),  # minutes
            },
            "break": {
                "time_range": (0, 24),
                "typical_activities": ["break_start", "short_pause", "return_to_work"],
                "duration_range": (5, 60),  # minutes
            },
            "evening": {
                "time_range": (17, 24),
                "typical_activities": ["work_end", "summary_request", "final_tasks"],
                "duration_range": (30, 180),  # minutes
            },
        }

    async def initialize(self):
        """Inicjalizacja modułu uczenia się rutyn."""
        if not self.enabled:
            logger.info("Routines Learner Module disabled")
            return

        # Load existing routines and sequences
        await self._load_detected_routines()
        await self._load_activity_sequences()
        await self._load_ml_models()

        # Start new sequence tracking
        self._start_new_sequence()

        logger.info("Routines Learner Module initialized")

    async def _load_detected_routines(self):
        """Załaduj wykryte rutyny."""
        if self.routines_file.exists():
            try:
                with open(self.routines_file, encoding="utf-8") as f:
                    self.detected_routines = json.load(f)
                logger.info("Loaded detected routines")
            except Exception as e:
                logger.error(f"Error loading routines: {e}")

    async def _load_activity_sequences(self):
        """Załaduj sekwencje aktywności."""
        if self.sequences_file.exists():
            try:
                with open(self.sequences_file, encoding="utf-8") as f:
                    self.activity_sequences = json.load(f)
                logger.info(f"Loaded {len(self.activity_sequences)} activity sequences")
            except Exception as e:
                logger.error(f"Error loading sequences: {e}")

    async def _load_ml_models(self):
        """Załaduj modele machine learning."""
        if self.models_file.exists() and SKLEARN_AVAILABLE and pickle:
            try:
                with open(self.models_file, "rb") as f:
                    self.ml_models = pickle.load(f)
                logger.info("Loaded ML models")
            except Exception as e:
                logger.error(f"Error loading ML models: {e}")

    async def _save_detected_routines(self):
        """Zapisz wykryte rutyny."""
        try:
            with open(self.routines_file, "w", encoding="utf-8") as f:
                json.dump(self.detected_routines, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error saving routines: {e}")

    async def _save_activity_sequences(self):
        """Zapisz sekwencje aktywności."""
        try:
            # Keep only recent sequences to manage file size
            cutoff_date = datetime.now() - timedelta(days=self.learning_window_days * 2)

            filtered_sequences = [
                seq
                for seq in self.activity_sequences
                if datetime.fromisoformat(seq.get("start_time", "1970-01-01"))
                > cutoff_date
            ]

            self.activity_sequences = filtered_sequences

            with open(self.sequences_file, "w", encoding="utf-8") as f:
                json.dump(self.activity_sequences, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error saving sequences: {e}")

    async def _save_ml_models(self):
        """Zapisz modele ML."""
        if SKLEARN_AVAILABLE and pickle:
            try:
                with open(self.models_file, "wb") as f:
                    pickle.dump(self.ml_models, f)
            except Exception as e:
                logger.error(f"Error saving ML models: {e}")

    def _start_new_sequence(self):
        """Rozpocznij nową sekwencję aktywności."""
        if self.current_sequence:
            # Save previous sequence if it has activities
            asyncio.create_task(self._finalize_current_sequence())

        self.current_sequence = []
        self.sequence_start_time = datetime.now()
        self.last_activity_time = datetime.now()

        logger.debug("Started new activity sequence")

    async def _finalize_current_sequence(self):
        """Finalizuj bieżącą sekwencję."""
        if not self.current_sequence or not self.sequence_start_time:
            return

        sequence_data = {
            "start_time": self.sequence_start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "duration_minutes": (
                datetime.now() - self.sequence_start_time
            ).total_seconds()
            / 60,
            "activities": self.current_sequence.copy(),
            "day_of_week": self.sequence_start_time.weekday(),
            "start_hour": self.sequence_start_time.hour,
            "activity_count": len(self.current_sequence),
        }

        # Add sequence classification
        sequence_data["routine_type"] = self._classify_sequence(sequence_data)

        self.activity_sequences.append(sequence_data)

        # Trigger routine analysis if we have enough sequences
        if len(self.activity_sequences) % 10 == 0:  # Analyze every 10 sequences
            await self._analyze_routines()

        logger.debug(
            f"Finalized sequence: {len(self.current_sequence)} activities, "
            f"{sequence_data['duration_minutes']:.1f} minutes"
        )

    def _classify_sequence(self, sequence_data: dict) -> str:
        """Klasyfikuj sekwencję do typu rutyny."""
        start_hour = sequence_data["start_hour"]
        duration = sequence_data["duration_minutes"]
        activity_count = sequence_data["activity_count"]

        # Rule-based classification
        if 6 <= start_hour <= 10 and duration <= 120:
            return "morning_routine"
        elif 9 <= start_hour <= 17 and duration >= 60:
            return "work_block"
        elif duration <= 30 and activity_count <= 3:
            return "break_routine"
        elif 17 <= start_hour <= 23 and duration <= 180:
            return "evening_routine"
        else:
            return "general_activity"

    async def record_activity(self, activity_type: str, details: dict = None):
        """Zapisz aktywność w bieżącej sekwencji."""
        activity = {
            "timestamp": datetime.now().isoformat(),
            "type": activity_type,
            "details": details or {},
            "hour": datetime.now().hour,
            "sequence_position": len(self.current_sequence),
        }

        self.current_sequence.append(activity)
        self.last_activity_time = datetime.now()

        # Check if we should start a new sequence (gap in activity)
        if len(self.current_sequence) > 1:
            last_activity = self.current_sequence[-2]
            time_gap = (
                datetime.now() - datetime.fromisoformat(last_activity["timestamp"])
            ).total_seconds() / 60

            if time_gap > 30:  # More than 30 minutes gap
                await self._finalize_current_sequence()
                self._start_new_sequence()
                # Add the current activity to new sequence
                self.current_sequence.append(activity)

        logger.debug(f"Recorded activity: {activity_type}")

    async def _analyze_routines(self):
        """Przeprowadź analizę rutyn używając AI i ML."""
        logger.info("Starting routine analysis...")

        try:
            # Get recent sequences for analysis
            recent_sequences = self._get_recent_sequences()

            if len(recent_sequences) < self.min_pattern_frequency:
                logger.info("Not enough sequences for routine analysis")
                return

            # Perform different types of analysis
            await self._detect_time_based_patterns(recent_sequences)
            await self._detect_activity_patterns(recent_sequences)
            await self._detect_weekly_patterns(recent_sequences)

            if SKLEARN_AVAILABLE:
                await self._ml_pattern_detection(recent_sequences)

            if AI_MODULE_AVAILABLE:
                await self._ai_routine_analysis(recent_sequences)

            # Save results
            await self._save_detected_routines()
            await self._save_activity_sequences()

            logger.info("Routine analysis completed")

        except Exception as e:
            logger.error(f"Error during routine analysis: {e}")

    def _get_recent_sequences(self, days: int = None) -> list[dict]:
        """Pobierz niedawne sekwencje do analizy."""
        if days is None:
            days = self.learning_window_days

        cutoff_date = datetime.now() - timedelta(days=days)

        recent = [
            seq
            for seq in self.activity_sequences
            if datetime.fromisoformat(seq["start_time"]) > cutoff_date
        ]

        return recent

    async def _detect_time_based_patterns(self, sequences: list[dict]):
        """Wykryj wzorce oparte na czasie."""
        # Group sequences by hour
        sequences_by_hour = defaultdict(list)
        for seq in sequences:
            hour = seq["start_hour"]
            sequences_by_hour[hour].append(seq)

        # Find recurring time patterns
        time_patterns = []
        for hour, hour_sequences in sequences_by_hour.items():
            if len(hour_sequences) >= self.min_pattern_frequency:
                # Analyze activities in this hour
                activity_types = [
                    act["type"] for seq in hour_sequences for act in seq["activities"]
                ]
                common_activities = Counter(activity_types).most_common(5)

                # Calculate consistency
                total_days = len(
                    {
                        datetime.fromisoformat(seq["start_time"]).date()
                        for seq in hour_sequences
                    }
                )
                frequency = len(hour_sequences) / max(1, total_days)

                if frequency >= 0.5:  # Occurs at least 50% of the time
                    pattern = {
                        "type": "time_based",
                        "hour": hour,
                        "frequency": frequency,
                        "common_activities": common_activities,
                        "average_duration": statistics.mean(
                            [seq["duration_minutes"] for seq in hour_sequences]
                        ),
                        "confidence": min(1.0, frequency * (len(hour_sequences) / 10)),
                    }
                    time_patterns.append(pattern)

        # Update detected routines
        self.detected_routines["time_based_patterns"] = time_patterns

    async def _detect_activity_patterns(self, sequences: list[dict]):
        """Wykryj wzorce sekwencji aktywności."""
        # Extract activity sequences
        activity_sequences = []
        for seq in sequences:
            activity_types = [act["type"] for act in seq["activities"]]
            if len(activity_types) >= 2:  # At least 2 activities
                activity_sequences.append(activity_types)

        # Find common subsequences
        sequence_patterns = []
        min_seq_length = 2
        max_seq_length = 5

        for seq_length in range(min_seq_length, max_seq_length + 1):
            subsequences = []
            for seq in activity_sequences:
                if len(seq) >= seq_length:
                    for i in range(len(seq) - seq_length + 1):
                        subseq = tuple(seq[i : i + seq_length])
                        subsequences.append(subseq)

            # Count occurrences
            subseq_counts = Counter(subsequences)

            # Find patterns that occur frequently
            for subseq, count in subseq_counts.items():
                if count >= self.min_pattern_frequency:
                    frequency = count / len(subsequences) if subsequences else 0

                    if frequency >= 0.1:  # At least 10% frequency
                        pattern = {
                            "type": "activity_sequence",
                            "sequence": list(subseq),
                            "frequency": frequency,
                            "occurrences": count,
                            "confidence": min(1.0, frequency * 2),
                        }
                        sequence_patterns.append(pattern)

        # Sort by confidence
        sequence_patterns.sort(key=lambda x: x["confidence"], reverse=True)
        self.detected_routines["activity_patterns"] = sequence_patterns[
            :20
        ]  # Keep top 20

    async def _detect_weekly_patterns(self, sequences: list[dict]):
        """Wykryj wzorce tygodniowe."""
        # Group by day of week
        sequences_by_day = defaultdict(list)
        for seq in sequences:
            day = seq["day_of_week"]
            sequences_by_day[day].append(seq)

        weekly_patterns = []
        day_names = [
            "poniedziałek",
            "wtorek",
            "środa",
            "czwartek",
            "piątek",
            "sobota",
            "niedziela",
        ]

        for day, day_sequences in sequences_by_day.items():
            if len(day_sequences) >= 3:  # At least 3 occurrences
                # Analyze this day's patterns
                hours = [seq["start_hour"] for seq in day_sequences]
                durations = [seq["duration_minutes"] for seq in day_sequences]
                activity_counts = [seq["activity_count"] for seq in day_sequences]

                pattern = {
                    "type": "weekly_pattern",
                    "day_of_week": day,
                    "day_name": day_names[day],
                    "typical_start_hour": statistics.mean(hours),
                    "average_duration": statistics.mean(durations),
                    "average_activities": statistics.mean(activity_counts),
                    "frequency": len(day_sequences)
                    / max(1, len(sequences) / 7),  # Normalized frequency
                    "consistency": 1
                    - (
                        statistics.stdev(hours) / 12 if len(hours) > 1 else 0
                    ),  # Hour consistency
                }

                if pattern["frequency"] >= 0.3:  # Occurs at least 30% of expected times
                    weekly_patterns.append(pattern)

        self.detected_routines["weekly_patterns"] = weekly_patterns

    async def _ml_pattern_detection(self, sequences: list[dict]):
        """Użyj machine learning do wykrywania wzorców."""
        if not SKLEARN_AVAILABLE or not NUMPY_AVAILABLE or len(sequences) < 10:
            return

        try:
            # Prepare features for clustering
            features = []
            for seq in sequences:
                feature_vector = [
                    seq["start_hour"],
                    seq["duration_minutes"],
                    seq["activity_count"],
                    seq["day_of_week"],
                    len(
                        {act["type"] for act in seq["activities"]}
                    ),  # Unique activity types
                ]
                features.append(feature_vector)

            features = np.array(features)

            # Normalize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)

            # Perform clustering
            n_clusters = min(5, len(sequences) // 3)  # Adaptive number of clusters
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(features_scaled)

            # Analyze clusters
            ml_patterns = []
            for cluster_id in range(n_clusters):
                cluster_sequences = [
                    seq
                    for i, seq in enumerate(sequences)
                    if cluster_labels[i] == cluster_id
                ]

                if len(cluster_sequences) >= 2:
                    # Calculate cluster characteristics
                    avg_start_hour = statistics.mean(
                        [seq["start_hour"] for seq in cluster_sequences]
                    )
                    avg_duration = statistics.mean(
                        [seq["duration_minutes"] for seq in cluster_sequences]
                    )
                    common_activities = Counter(
                        [
                            act["type"]
                            for seq in cluster_sequences
                            for act in seq["activities"]
                        ]
                    ).most_common(5)

                    pattern = {
                        "type": "ml_cluster",
                        "cluster_id": int(cluster_id),
                        "size": len(cluster_sequences),
                        "average_start_hour": avg_start_hour,
                        "average_duration": avg_duration,
                        "common_activities": common_activities,
                        "confidence": len(cluster_sequences) / len(sequences),
                    }
                    ml_patterns.append(pattern)

            self.detected_routines["ml_patterns"] = ml_patterns

            # Save the model
            self.ml_models["activity_clustering"] = {
                "model": kmeans,
                "scaler": scaler,
                "last_trained": datetime.now().isoformat(),
            }

            await self._save_ml_models()

        except Exception as e:
            logger.error(f"Error in ML pattern detection: {e}")

    async def _ai_routine_analysis(self, sequences: list[dict]):
        """Użyj AI do analizy rutyn i generowania insights."""
        try:
            from collections import deque

            # Prepare data summary for AI analysis
            data_summary = {
                "total_sequences": len(sequences),
                "time_range": f"{min(seq['start_hour'] for seq in sequences)} - {max(seq['start_hour'] for seq in sequences)}",
                "average_duration": statistics.mean(
                    [seq["duration_minutes"] for seq in sequences]
                ),
                "most_common_activities": Counter(
                    [act["type"] for seq in sequences for act in seq["activities"]]
                ).most_common(10),
                "days_covered": len(
                    {
                        datetime.fromisoformat(seq["start_time"]).date()
                        for seq in sequences
                    }
                ),
            }

            # Create AI prompt for routine analysis
            prompt = f"""
Przeanalizuj wzorce aktywności użytkownika {self.user_name} i zidentyfikuj rutyny:

Dane:
- Łącznie sekwencji: {data_summary['total_sequences']}
- Zakres godzinowy: {data_summary['time_range']}
- Średni czas trwania: {data_summary['average_duration']:.1f} minut
- Dni objęte analizą: {data_summary['days_covered']}

Najczęstsze aktywności:
{', '.join([f"{act} ({count}x)" for act, count in data_summary['most_common_activities'][:5]])}

Przeanalizuj te dane i wskaż:
1. Główne rutyny użytkownika
2. Wzorce czasowe
3. Rekomendacje optymalizacji
4. Przewidywalne zachowania

Odpowiedz w formacie JSON z kluczami: routines, time_patterns, recommendations, predictions
"""

            # Generate AI analysis
            response_json = generate_response(
                conversation_history=deque([{"role": "user", "content": prompt}]),
                detected_language="pl",
                user_name=self.user_name,
            )

            if response_json:
                try:
                    response_data = json.loads(response_json)
                    ai_analysis = response_data.get("text", "")

                    # Try to parse AI response as JSON
                    try:
                        ai_insights = json.loads(ai_analysis)
                        self.detected_routines["ai_insights"] = ai_insights
                    except json.JSONDecodeError:
                        # Store as text if not valid JSON
                        self.detected_routines["ai_insights"] = {
                            "analysis": ai_analysis,
                            "generated_at": datetime.now().isoformat(),
                        }

                except json.JSONDecodeError:
                    # Fallback for non-JSON response
                    self.detected_routines["ai_insights"] = {
                        "analysis": response_json,
                        "generated_at": datetime.now().isoformat(),
                    }

        except Exception as e:
            logger.error(f"Error in AI routine analysis: {e}")

    async def get_predictions(self) -> dict[str, Any]:
        """Wygeneruj przewidywania następnych aktywności."""
        try:
            predictions = {}
            current_hour = datetime.now().hour
            current_day = datetime.now().weekday()

            # Time-based predictions
            time_patterns = self.detected_routines.get("time_based_patterns", [])
            for pattern in time_patterns:
                if (
                    pattern["hour"] > current_hour
                    and pattern["confidence"] > self.prediction_confidence_threshold
                ):
                    predictions[f"activity_at_{pattern['hour']}"] = {
                        "type": "time_based",
                        "hour": pattern["hour"],
                        "expected_activities": [
                            act[0] for act in pattern["common_activities"][:3]
                        ],
                        "confidence": pattern["confidence"],
                        "estimated_duration": pattern["average_duration"],
                    }

            # Weekly pattern predictions
            weekly_patterns = self.detected_routines.get("weekly_patterns", [])
            for pattern in weekly_patterns:
                if pattern["day_of_week"] == current_day and pattern["frequency"] > 0.5:
                    predictions["today_pattern"] = {
                        "type": "weekly_pattern",
                        "expected_start_hour": pattern["typical_start_hour"],
                        "expected_duration": pattern["average_duration"],
                        "confidence": pattern["frequency"],
                    }

            # Sequence-based predictions
            if self.current_sequence:
                current_activities = [act["type"] for act in self.current_sequence]
                activity_patterns = self.detected_routines.get("activity_patterns", [])

                for pattern in activity_patterns:
                    sequence = pattern["sequence"]
                    if len(current_activities) < len(sequence):
                        # Check if current sequence matches the beginning of this pattern
                        if current_activities == sequence[: len(current_activities)]:
                            next_activity = sequence[len(current_activities)]
                            predictions["next_activity"] = {
                                "type": "sequence_based",
                                "activity": next_activity,
                                "confidence": pattern["confidence"],
                                "pattern_sequence": sequence,
                            }
                            break

            return {
                "success": True,
                "predictions": predictions,
                "current_sequence_length": len(self.current_sequence),
                "analysis_timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error generating predictions: {e}")
            return {"success": False, "error": str(e)}

    async def get_routine_insights(self) -> dict[str, Any]:
        """Zwróć szczegółowe informacje o wykrytych rutynach."""
        try:
            insights = {
                "detected_routines": self.detected_routines,
                "statistics": {
                    "total_sequences_analyzed": len(self.activity_sequences),
                    "recent_sequences": len(self._get_recent_sequences()),
                    "routine_categories": len(self.detected_routines),
                    "ml_models_available": SKLEARN_AVAILABLE,
                    "ai_analysis_available": AI_MODULE_AVAILABLE,
                },
                "recommendations": await self._generate_routine_recommendations(),
                "next_analysis_suggestions": self._get_analysis_suggestions(),
            }

            return {"success": True, "insights": insights}

        except Exception as e:
            logger.error(f"Error getting routine insights: {e}")
            return {"success": False, "error": str(e)}

    async def _generate_routine_recommendations(self) -> list[str]:
        """Wygeneruj rekomendacje na podstawie wykrytych rutyn."""
        recommendations = []

        try:
            # Time-based recommendations
            time_patterns = self.detected_routines.get("time_based_patterns", [])
            if time_patterns:
                most_consistent = max(time_patterns, key=lambda x: x["confidence"])
                recommendations.append(
                    f"Twoja najbardziej konsystentna rutyna zaczyna się o {most_consistent['hour']}:00. "
                    f"Planuj ważne zadania na ten czas."
                )

            # Weekly recommendations
            weekly_patterns = self.detected_routines.get("weekly_patterns", [])
            if weekly_patterns:
                most_productive_day = max(
                    weekly_patterns, key=lambda x: x["average_activities"]
                )
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
                    f"Najbardziej aktywny jesteś w {day_names[most_productive_day['day_of_week']]}. "
                    f"Zaplanuj na ten dzień najważniejsze projekty."
                )

            # AI insights recommendations
            ai_insights = self.detected_routines.get("ai_insights", {})
            if ai_insights and "recommendations" in ai_insights:
                if isinstance(ai_insights["recommendations"], list):
                    recommendations.extend(ai_insights["recommendations"][:3])

            # Activity pattern recommendations
            activity_patterns = self.detected_routines.get("activity_patterns", [])
            if activity_patterns:
                strongest_pattern = max(
                    activity_patterns, key=lambda x: x["confidence"]
                )
                recommendations.append(
                    f"Wykryto silny wzorzec aktywności: {' → '.join(strongest_pattern['sequence'])}. "
                    f"Ta sekwencja może być zoptymalizowana."
                )

            return recommendations

        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return ["Błąd podczas generowania rekomendacji."]

    def _get_analysis_suggestions(self) -> list[str]:
        """Wygeneruj sugestie dla poprawy analizy."""
        suggestions = []

        recent_sequences = self._get_recent_sequences()

        if len(recent_sequences) < 20:
            suggestions.append(
                "Zbierz więcej danych (aktualnie: "
                + str(len(recent_sequences))
                + " sekwencji) dla lepszej analizy wzorców."
            )

        if not SKLEARN_AVAILABLE:
            suggestions.append(
                "Zainstaluj scikit-learn dla zaawansowanej analizy wzorców ML."
            )

        if not AI_MODULE_AVAILABLE:
            suggestions.append("Włącz moduł AI dla głębszej analizy rutyn.")

        # Check data diversity
        unique_activities = set()
        for seq in recent_sequences:
            for act in seq["activities"]:
                unique_activities.add(act["type"])

        if len(unique_activities) < 5:
            suggestions.append(
                "Zróżnicuj aktywności dla bogatszego modelowania wzorców."
            )

        return suggestions


# Plugin functions for function calling system
async def get_routine_insights(user_id: str) -> dict[str, Any]:
    """Pobierz informacje o wykrytych rutynach."""
    try:
        from server_main import server_app

        if hasattr(server_app, "routines_learner"):
            return await server_app.routines_learner.get_routine_insights()
        else:
            return {"success": False, "error": "Routines learner module not available"}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def get_routine_predictions(user_id: str) -> dict[str, Any]:
    """Pobierz przewidywania rutyn."""
    try:
        from server_main import server_app

        if hasattr(server_app, "routines_learner"):
            return await server_app.routines_learner.get_predictions()
        else:
            return {"success": False, "error": "Routines learner module not available"}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def analyze_routines(user_id: str) -> dict[str, Any]:
    """Uruchom analizę rutyn."""
    try:
        from server_main import server_app

        if hasattr(server_app, "routines_learner"):
            await server_app.routines_learner._analyze_routines()
            return {"success": True, "message": "Routine analysis completed"}
        else:
            return {"success": False, "error": "Routines learner module not available"}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def record_routine_activity(
    user_id: str, activity_type: str, details: dict = None
) -> dict[str, Any]:
    """Zapisz aktywność rutyny."""
    try:
        from server_main import server_app

        if hasattr(server_app, "routines_learner"):
            await server_app.routines_learner.record_activity(activity_type, details)
            return {"success": True, "message": "Activity recorded"}
        else:
            return {"success": False, "error": "Routines learner module not available"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# Plugin metadata
PLUGIN_FUNCTIONS = {
    "get_routine_insights": {
        "function": get_routine_insights,
        "description": "Get detailed insights about detected user routines",
        "parameters": {"type": "object", "properties": {}},
    },
    "get_routine_predictions": {
        "function": get_routine_predictions,
        "description": "Get predictions about upcoming activities based on learned routines",
        "parameters": {"type": "object", "properties": {}},
    },
    "analyze_routines": {
        "function": analyze_routines,
        "description": "Trigger analysis of user routines using AI and ML",
        "parameters": {"type": "object", "properties": {}},
    },
    "record_routine_activity": {
        "function": record_routine_activity,
        "description": "Record an activity for routine learning",
        "parameters": {
            "type": "object",
            "properties": {
                "activity_type": {
                    "type": "string",
                    "description": "Type of activity being recorded",
                },
                "details": {
                    "type": "object",
                    "description": "Additional details about the activity",
                },
            },
            "required": ["activity_type"],
        },
    },
}
