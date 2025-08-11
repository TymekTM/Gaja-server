"""Day Narrative Module for Asystent System generujący narracyjne podsumowania dnia
używając AI."""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    from ai_module import generate_response

    AI_MODULE_AVAILABLE = True
except ImportError:
    AI_MODULE_AVAILABLE = False
    logger.warning("AI module not available for day narratives")


class DayNarrativeModule:
    """Moduł generujący narracyjne, spójne opowieści o dniu użytkownika."""

    def __init__(self, config: dict[str, Any], db_manager=None):
        self.config = config
        self.db_manager = db_manager
        self.enabled = config.get("day_narrative", {}).get("enabled", True)
        self.user_name = config.get("USER_NAME", "User")

        # Narrative configuration
        self.narrative_style = config.get("day_narrative", {}).get(
            "style", "friendly"
        )  # friendly, professional, casual, poetic
        self.narrative_length = config.get("day_narrative", {}).get(
            "length", "medium"
        )  # short, medium, long
        self.include_emotions = config.get("day_narrative", {}).get(
            "include_emotions", True
        )
        self.include_insights = config.get("day_narrative", {}).get(
            "include_insights", True
        )
        self.include_comparisons = config.get("day_narrative", {}).get(
            "include_comparisons", True
        )
        self.auto_generate = config.get("day_narrative", {}).get("auto_generate", True)
        self.generation_time = config.get("day_narrative", {}).get(
            "generation_time", "21:00"
        )

        # Data storage
        self.data_dir = Path("user_data") / "day_narratives"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Narrative templates
        self.narrative_templates = {
            "friendly": {
                "opening": [
                    "Hej {user_name}! Jak przebiegł Twój dzień?",
                    "Cześć {user_name}! Opowiem Ci o Twoim dniu.",
                    "Witaj {user_name}! Czas na podsumowanie dnia.",
                ],
                "transitions": [
                    "Następnie",
                    "Potem",
                    "W międzyczasie",
                    "Dalej",
                    "Później",
                ],
                "closing": [
                    "To było naprawdę produktywne {period}!",
                    "Dobra robota dzisiaj!",
                    "Mam nadzieję, że jutro będzie równie udane!",
                ],
            },
            "professional": {
                "opening": [
                    "Przygotowałem raport z Twojej dzisiejszej aktywności.",
                    "Analiza dnia {user_name} - {date}",
                    "Podsumowanie produktywności za dzień {date}",
                ],
                "transitions": [
                    "Następnie zaobserwowano",
                    "W kolejnej fazie",
                    "Kontynuując",
                    "Dalej",
                    "W trakcie",
                ],
                "closing": [
                    "Ogólnie dzień można ocenić jako produktywny.",
                    "Końcowe wnioski wskazują na efektywne wykorzystanie czasu.",
                    "Rekomendacje na jutro będą dostępne w briefingu porannym.",
                ],
            },
            "casual": {
                "opening": [
                    "No to jak tam dzisiaj? 😊",
                    "Dzisiaj było naprawdę ciekawie!",
                    "Sprawdźmy co działo się dzisiaj!",
                ],
                "transitions": ["A potem", "I wtedy", "No i", "Później", "A następnie"],
                "closing": [
                    "W sumie niezły dzień! 👍",
                    "Można było lepiej, ale było okej!",
                    "Jutro będzie jeszcze lepiej! 🚀",
                ],
            },
            "poetic": {
                "opening": [
                    "Jak promień słońca przecina chmury, tak Twój dzień miał swoje światła i cienie...",
                    "Każdy dzień to opowieść, a Twoja dzisiejsza była wyjątkowa.",
                    "W rytmie godzin i minut układała się Twoja dzisiejsza symfonia.",
                ],
                "transitions": [
                    "Niczym fale na morzu",
                    "W harmonii z czasem",
                    "Jak nić w tkaninie",
                    "W naturalnym rytmie",
                ],
                "closing": [
                    "I tak zakończył się kolejny rozdział Twojej opowieści.",
                    "Dzień przeminął, pozostawiając ślad w czasie.",
                    "Jutrzejszy świt przyniesie nowe możliwości.",
                ],
            },
        }

        # Emotion keywords for detection
        self.emotion_keywords = {
            "productive": [
                "completed",
                "finished",
                "accomplished",
                "productive",
                "efficient",
            ],
            "focused": ["focused", "concentrated", "deep_work", "flow_state"],
            "stressed": ["rushed", "pressure", "deadline", "urgent", "overwhelmed"],
            "relaxed": ["break", "rest", "leisure", "calm", "peaceful"],
            "social": ["meeting", "call", "collaboration", "team", "discussion"],
            "creative": ["design", "create", "brainstorm", "innovative", "artistic"],
        }

        # Cached data for performance
        self._cached_summaries = {}
        self._cache_date = None

    async def initialize(self):
        """Inicjalizacja modułu narracji."""
        if not self.enabled:
            logger.info("Day Narrative Module disabled")
            return

        if not AI_MODULE_AVAILABLE:
            logger.warning(
                "AI module not available - narratives will use templates only"
            )

        logger.info("Day Narrative Module initialized")

    async def generate_day_narrative(
        self, date: str | None = None, style: str | None = None
    ) -> dict[str, Any]:
        """Wygeneruj narracyjne podsumowanie dnia."""
        try:
            if date is None:
                date = datetime.now().date().isoformat()

            if style is None:
                style = self.narrative_style

            logger.info(f"Generating day narrative for {date} in {style} style")

            # Collect data from all modules
            day_data = await self._collect_day_data(date)

            if not day_data["has_data"]:
                return {"success": False, "error": f"No data available for {date}"}

            # Generate narrative using AI if available
            if AI_MODULE_AVAILABLE:
                narrative = await self._generate_ai_narrative(day_data, style)
            else:
                narrative = await self._generate_template_narrative(day_data, style)

            # Save narrative
            await self._save_narrative(date, narrative, style)

            # Add metadata
            result = {
                "success": True,
                "date": date,
                "narrative": narrative,
                "style": style,
                "word_count": len(narrative.split()) if narrative else 0,
                "generated_at": datetime.now().isoformat(),
                "data_summary": day_data["summary"],
            }

            return result

        except Exception as e:
            logger.error(f"Error generating day narrative: {e}")
            return {"success": False, "error": str(e)}

    async def _collect_day_data(self, date: str) -> dict[str, Any]:
        """Zbierz dane o dniu ze wszystkich modułów."""
        day_data = {
            "date": date,
            "has_data": False,
            "summary": {},
            "day_summary": None,
            "behavior_data": None,
            "routines_data": None,
            "interactions": [],
            "emotions": [],
            "achievements": [],
            "challenges": [],
        }

        try:
            # Get day summary data
            from server_main import server_app

            if hasattr(server_app, "day_summary"):
                day_summary = await server_app.day_summary.generate_day_summary(date)
                if day_summary.get("success"):
                    day_data["day_summary"] = day_summary
                    day_data["has_data"] = True
                    day_data["summary"]["total_active_time"] = day_summary[
                        "statistics"
                    ]["total_active_time_hours"]
                    day_data["summary"]["interactions"] = day_summary["statistics"][
                        "total_interactions"
                    ]
                    day_data["summary"]["productivity"] = day_summary["statistics"][
                        "productivity_score"
                    ]

            # Get behavior insights
            if hasattr(server_app, "user_behavior"):
                behavior_insights = (
                    await server_app.user_behavior.get_behavior_insights()
                )
                if behavior_insights.get("success"):
                    day_data["behavior_data"] = behavior_insights
                    day_data["has_data"] = True

            # Get routine insights
            if hasattr(server_app, "routines_learner"):
                routine_insights = (
                    await server_app.routines_learner.get_routine_insights()
                )
                if routine_insights.get("success"):
                    day_data["routines_data"] = routine_insights
                    day_data["has_data"] = True

            # Get interactions from database
            if hasattr(server_app, "db_manager"):
                interactions = await server_app.db_manager.get_user_history(
                    "1"
                )  # Main user
                # Filter interactions for the specific date
                date_interactions = [
                    interaction
                    for interaction in interactions
                    if interaction.get("timestamp", "").startswith(date)
                ]
                day_data["interactions"] = date_interactions[
                    -20:
                ]  # Last 20 interactions
                if date_interactions:
                    day_data["has_data"] = True

            # Analyze emotions and themes
            day_data["emotions"] = self._analyze_emotions(day_data)
            day_data["achievements"] = self._identify_achievements(day_data)
            day_data["challenges"] = self._identify_challenges(day_data)

            return day_data

        except Exception as e:
            logger.error(f"Error collecting day data: {e}")
            return day_data

    def _analyze_emotions(self, day_data: dict) -> list[str]:
        """Analizuj emocje i nastrój dnia."""
        emotions = []

        try:
            # Analyze from interactions
            interactions = day_data.get("interactions", [])
            all_text = " ".join(
                [interaction.get("content", "").lower() for interaction in interactions]
            )

            # Check emotion keywords
            for emotion, keywords in self.emotion_keywords.items():
                if any(keyword in all_text for keyword in keywords):
                    emotions.append(emotion)

            # Analyze from productivity score
            if day_data.get("day_summary"):
                productivity = day_data["day_summary"]["statistics"].get(
                    "productivity_score", 0
                )
                if productivity > 0.8:
                    emotions.append("accomplished")
                elif productivity > 0.6:
                    emotions.append("productive")
                elif productivity < 0.3:
                    emotions.append("unfocused")

            # Analyze from break patterns
            if day_data.get("day_summary"):
                breaks = day_data["day_summary"]["statistics"].get("total_breaks", 0)
                break_time = day_data["day_summary"]["statistics"].get(
                    "break_time_hours", 0
                )

                if breaks == 0:
                    emotions.append("intense")
                elif break_time > 2:
                    emotions.append("relaxed")
                elif breaks > 8:
                    emotions.append("fragmented")

            return list(set(emotions))  # Remove duplicates

        except Exception as e:
            logger.error(f"Error analyzing emotions: {e}")
            return []

    def _identify_achievements(self, day_data: dict) -> list[str]:
        """Zidentyfikuj osiągnięcia dnia."""
        achievements = []

        try:
            # High productivity achievement
            if day_data.get("day_summary"):
                productivity = day_data["day_summary"]["statistics"].get(
                    "productivity_score", 0
                )
                if productivity > 0.8:
                    achievements.append("Bardzo wysoka produktywność")
                elif productivity > 0.6:
                    achievements.append("Dobra produktywność")

            # Long focus sessions
            if day_data.get("day_summary"):
                active_time = day_data["day_summary"]["statistics"].get(
                    "total_active_time_hours", 0
                )
                if active_time > 8:
                    achievements.append("Długa sesja pracy")
                elif active_time > 6:
                    achievements.append("Solidna sesja pracy")

            # Many interactions (engagement)
            interactions_count = len(day_data.get("interactions", []))
            if interactions_count > 20:
                achievements.append("Wysoka aktywność w komunikacji")
            elif interactions_count > 10:
                achievements.append("Dobra aktywność w komunikacji")

            # Consistent routine following
            if day_data.get("routines_data"):
                routine_insights = day_data["routines_data"]["insights"]
                if (
                    routine_insights.get("statistics", {}).get("recent_sequences", 0)
                    > 5
                ):
                    achievements.append("Konsystentne przestrzeganie rutyn")

            return achievements

        except Exception as e:
            logger.error(f"Error identifying achievements: {e}")
            return []

    def _identify_challenges(self, day_data: dict) -> list[str]:
        """Zidentyfikuj wyzwania dnia."""
        challenges = []

        try:
            # Low productivity
            if day_data.get("day_summary"):
                productivity = day_data["day_summary"]["statistics"].get(
                    "productivity_score", 0
                )
                if productivity < 0.3:
                    challenges.append("Niska produktywność")
                elif productivity < 0.5:
                    challenges.append("Produktywność poniżej średniej")

            # Too many or too few breaks
            if day_data.get("day_summary"):
                breaks = day_data["day_summary"]["statistics"].get("total_breaks", 0)
                break_time = day_data["day_summary"]["statistics"].get(
                    "break_time_hours", 0
                )

                if breaks == 0:
                    challenges.append("Brak przerw - ryzyko przepracowania")
                elif break_time > 3:
                    challenges.append("Zbyt dużo czasu na przerwach")
                elif breaks > 10:
                    challenges.append("Fragmentacja pracy - zbyt częste przerwy")

            # Low engagement
            interactions_count = len(day_data.get("interactions", []))
            if interactions_count < 3:
                challenges.append("Niska aktywność - mało interakcji")

            # Short work day
            if day_data.get("day_summary"):
                active_time = day_data["day_summary"]["statistics"].get(
                    "total_active_time_hours", 0
                )
                if active_time < 3:
                    challenges.append("Krótki dzień pracy")

            return challenges

        except Exception as e:
            logger.error(f"Error identifying challenges: {e}")
            return []

    async def _generate_ai_narrative(self, day_data: dict, style: str) -> str:
        """Wygeneruj narrację używając AI."""
        try:
            from collections import deque

            # Prepare comprehensive context for AI
            context = self._build_narrative_context(day_data, style)

            # Create AI prompt
            prompt = f"""
Napisz narracyjne podsumowanie dnia użytkownika {self.user_name} w stylu {style}.

Dane dnia ({day_data['date']}):
{context}

Wymagania:
1. Styl: {style}
2. Długość: {self.narrative_length}
3. Język: polski
4. Ton: {'osobisty i empatyczny' if self.include_emotions else 'rzeczowy'}
5. Uwzględnij {'wglądy i analizy' if self.include_insights else 'tylko fakty'}

Stwórz spójną opowieść o dniu, która:
- Ma jasny początek, środek i koniec
- Łączy wydarzenia w logiczną narrację
- Używa przejść między tematami
- Kończy się pozytywną nutą lub konstruktywną refleksją
- Ma długość około {200 if self.narrative_length == 'short' else 400 if self.narrative_length == 'medium' else 600} słów

Odpowiedz tylko narracją, bez dodatkowych komentarzy.
"""

            # Generate AI response
            response_json = generate_response(
                conversation_history=deque([{"role": "user", "content": prompt}]),
                detected_language="pl",
                user_name=self.user_name,
            )

            if response_json:
                try:
                    response_data = json.loads(response_json)
                    narrative = response_data.get("text", "")

                    if narrative and len(narrative) > 50:  # Reasonable length check
                        return narrative.strip()

                except json.JSONDecodeError:
                    # Try to use raw response
                    if isinstance(response_json, str) and len(response_json) > 50:
                        return response_json.strip()

            # Fallback to template if AI fails
            logger.warning("AI narrative generation failed, falling back to template")
            return await self._generate_template_narrative(day_data, style)

        except Exception as e:
            logger.error(f"Error generating AI narrative: {e}")
            return await self._generate_template_narrative(day_data, style)

    def _build_narrative_context(self, day_data: dict, style: str) -> str:
        """Zbuduj kontekst dla AI do generowania narracji."""
        context_parts = []

        # Summary statistics
        if day_data.get("day_summary"):
            stats = day_data["day_summary"]["statistics"]
            context_parts.append(
                f"Czas aktywności: {stats.get('total_active_time_hours', 0):.1f}h"
            )
            context_parts.append(f"Interakcje: {stats.get('total_interactions', 0)}")
            context_parts.append(
                f"Produktywność: {stats.get('productivity_score', 0):.2f}"
            )
            context_parts.append(f"Przerwy: {stats.get('total_breaks', 0)}")

        # Top applications
        if day_data.get("day_summary", {}).get("top_applications"):
            apps = day_data["day_summary"]["top_applications"][:3]
            app_list = ", ".join([f"{app[0]} ({app[1]:.1f}h)" for app in apps])
            context_parts.append(f"Główne aplikacje: {app_list}")

        # Emotions and themes
        if day_data.get("emotions"):
            emotions_text = ", ".join(day_data["emotions"])
            context_parts.append(f"Nastrój/tematy: {emotions_text}")

        # Achievements
        if day_data.get("achievements"):
            achievements_text = ", ".join(day_data["achievements"])
            context_parts.append(f"Osiągnięcia: {achievements_text}")

        # Challenges
        if day_data.get("challenges"):
            challenges_text = ", ".join(day_data["challenges"])
            context_parts.append(f"Wyzwania: {challenges_text}")

        # Recent interactions themes
        interactions = day_data.get("interactions", [])
        if interactions:
            interaction_types = [
                interaction.get("role", "unknown") for interaction in interactions[-5:]
            ]
            context_parts.append(
                f"Ostatnie interakcje: {len(interactions)} ({', '.join(set(interaction_types))})"
            )

        return "\n".join(context_parts)

    async def _generate_template_narrative(self, day_data: dict, style: str) -> str:
        """Wygeneruj narrację używając szablonów."""
        try:
            template = self.narrative_templates.get(
                style, self.narrative_templates["friendly"]
            )

            # Choose random templates
            import random

            opening = random.choice(template["opening"]).format(
                user_name=self.user_name, date=day_data["date"], period="dzień"
            )

            # Build narrative parts
            narrative_parts = [opening]

            # Add activity summary
            if day_data.get("day_summary"):
                stats = day_data["day_summary"]["statistics"]
                active_time = stats.get("total_active_time_hours", 0)
                productivity = stats.get("productivity_score", 0)

                if active_time > 0:
                    narrative_parts.append(
                        f"Pracowałeś przez {active_time:.1f} godzin z produktywnością {productivity:.0%}."
                    )

                # Add top applications
                if day_data["day_summary"].get("top_applications"):
                    top_app = day_data["day_summary"]["top_applications"][0]
                    narrative_parts.append(
                        f"Najwięcej czasu spędziłeś w aplikacji {top_app[0]} ({top_app[1]:.1f}h)."
                    )

            # Add transition
            transition = random.choice(template["transitions"])

            # Add achievements or challenges
            if day_data.get("achievements"):
                narrative_parts.append(
                    f"{transition} udało Ci się osiągnąć: {', '.join(day_data['achievements'][:2])}."
                )
            elif day_data.get("challenges"):
                narrative_parts.append(
                    f"{transition} napotkałeś wyzwania: {', '.join(day_data['challenges'][:2])}."
                )

            # Add interactions if available
            interactions_count = len(day_data.get("interactions", []))
            if interactions_count > 0:
                narrative_parts.append(
                    f"Miałeś {interactions_count} interakcji z asystentem, co świadczy o aktywnym dniu."
                )

            # Add closing
            closing = random.choice(template["closing"]).format(
                period=(
                    "dzień"
                    if datetime.fromisoformat(day_data["date"]).weekday() < 5
                    else "dzień weekendowy"
                )
            )
            narrative_parts.append(closing)

            return " ".join(narrative_parts)

        except Exception as e:
            logger.error(f"Error generating template narrative: {e}")
            return f"Dzień {day_data['date']} minął, ale wystąpił błąd podczas generowania podsumowania."

    async def _save_narrative(self, date: str, narrative: str, style: str):
        """Zapisz wygenerowaną narrację."""
        try:
            narrative_data = {
                "date": date,
                "narrative": narrative,
                "style": style,
                "generated_at": datetime.now().isoformat(),
                "word_count": len(narrative.split()),
                "character_count": len(narrative),
            }

            filename = f"narrative_{date}_{style}.json"
            filepath = self.data_dir / filename

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(narrative_data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error(f"Error saving narrative: {e}")

    async def get_narrative_history(self, days: int = 7) -> dict[str, Any]:
        """Pobierz historię narracji."""
        try:
            narratives = []

            for i in range(days):
                date = (datetime.now().date() - timedelta(days=i)).isoformat()

                # Look for narrative files
                pattern = f"narrative_{date}_*.json"
                narrative_files = list(self.data_dir.glob(pattern))

                for file_path in narrative_files:
                    try:
                        with open(file_path, encoding="utf-8") as f:
                            narrative_data = json.load(f)
                            narratives.append(narrative_data)
                    except Exception as e:
                        logger.error(f"Error reading narrative file {file_path}: {e}")

            # Sort by date (newest first)
            narratives.sort(key=lambda x: x["date"], reverse=True)

            return {
                "success": True,
                "narratives": narratives,
                "total_count": len(narratives),
            }

        except Exception as e:
            logger.error(f"Error getting narrative history: {e}")
            return {"success": False, "error": str(e)}

    async def compare_days(self, date1: str, date2: str) -> dict[str, Any]:
        """Porównaj dwa dni i wygeneruj narrację porównawczą."""
        try:
            # Get data for both days
            day1_data = await self._collect_day_data(date1)
            day2_data = await self._collect_day_data(date2)

            if not day1_data["has_data"] or not day2_data["has_data"]:
                return {"success": False, "error": "Insufficient data for comparison"}

            # Generate comparison narrative
            if AI_MODULE_AVAILABLE:
                comparison = await self._generate_ai_comparison(day1_data, day2_data)
            else:
                comparison = await self._generate_template_comparison(
                    day1_data, day2_data
                )

            return {
                "success": True,
                "date1": date1,
                "date2": date2,
                "comparison_narrative": comparison,
                "generated_at": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error comparing days: {e}")
            return {"success": False, "error": str(e)}

    async def _generate_ai_comparison(self, day1_data: dict, day2_data: dict) -> str:
        """Wygeneruj porównanie dni używając AI."""
        try:
            from collections import deque

            prompt = f"""
Porównaj dwa dni użytkownika {self.user_name} i napisz narracyjne porównanie:

DZIEŃ 1 ({day1_data['date']}):
{self._build_narrative_context(day1_data, 'professional')}

DZIEŃ 2 ({day2_data['date']}):
{self._build_narrative_context(day2_data, 'professional')}

Napisz porównanie w stylu naturalnej rozmowy, które:
1. Porówna produktywność, aktywność i nastrój
2. Wskaże, który dzień był lepszy i dlaczego
3. Zasugeruje wnioski na przyszłość
4. Będzie napisane po polsku w przyjaznym tonie

Odpowiedz tylko porównaniem, bez dodatkowych komentarzy.
"""

            response_json = generate_response(
                conversation_history=deque([{"role": "user", "content": prompt}]),
                detected_language="pl",
                user_name=self.user_name,
            )

            if response_json:
                try:
                    response_data = json.loads(response_json)
                    return response_data.get("text", "").strip()
                except json.JSONDecodeError:
                    return (
                        response_json.strip() if isinstance(response_json, str) else ""
                    )

            return await self._generate_template_comparison(day1_data, day2_data)

        except Exception as e:
            logger.error(f"Error generating AI comparison: {e}")
            return await self._generate_template_comparison(day1_data, day2_data)

    async def _generate_template_comparison(
        self, day1_data: dict, day2_data: dict
    ) -> str:
        """Wygeneruj porównanie używając szablonów."""
        try:
            comparison_parts = []

            # Compare productivity
            if day1_data.get("day_summary") and day2_data.get("day_summary"):
                prod1 = day1_data["day_summary"]["statistics"].get(
                    "productivity_score", 0
                )
                prod2 = day2_data["day_summary"]["statistics"].get(
                    "productivity_score", 0
                )

                if prod1 > prod2:
                    comparison_parts.append(
                        f"Dzień {day1_data['date']} był bardziej produktywny "
                        f"({prod1:.0%}) niż {day2_data['date']} ({prod2:.0%})."
                    )
                elif prod2 > prod1:
                    comparison_parts.append(
                        f"Dzień {day2_data['date']} był bardziej produktywny "
                        f"({prod2:.0%}) niż {day1_data['date']} ({prod1:.0%})."
                    )
                else:
                    comparison_parts.append("Oba dni miały podobną produktywność.")

            # Compare active time
            if day1_data.get("day_summary") and day2_data.get("day_summary"):
                time1 = day1_data["day_summary"]["statistics"].get(
                    "total_active_time_hours", 0
                )
                time2 = day2_data["day_summary"]["statistics"].get(
                    "total_active_time_hours", 0
                )

                time_diff = abs(time1 - time2)
                if time_diff > 1:  # Significant difference
                    longer_day = (
                        day1_data["date"] if time1 > time2 else day2_data["date"]
                    )
                    longer_time = max(time1, time2)
                    comparison_parts.append(
                        f"W dniu {longer_day} pracowałeś dłużej ({longer_time:.1f}h)."
                    )

            # Compare achievements
            achievements1 = len(day1_data.get("achievements", []))
            achievements2 = len(day2_data.get("achievements", []))

            if achievements1 > achievements2:
                comparison_parts.append(f"Więcej osiągnięć miałeś {day1_data['date']}.")
            elif achievements2 > achievements1:
                comparison_parts.append(f"Więcej osiągnięć miałeś {day2_data['date']}.")

            if not comparison_parts:
                comparison_parts.append(
                    "Oba dni były bardzo podobne pod względem aktywności."
                )

            return " ".join(comparison_parts)

        except Exception as e:
            logger.error(f"Error generating template comparison: {e}")
            return "Nie udało się porównać dni z powodu błędu."


# Plugin functions for function calling system
async def generate_day_narrative(
    user_id: str, date: str = None, style: str = None
) -> dict[str, Any]:
    """Wygeneruj narracyjne podsumowanie dnia."""
    try:
        from server_main import server_app

        if hasattr(server_app, "day_narrative"):
            return await server_app.day_narrative.generate_day_narrative(date, style)
        else:
            return {"success": False, "error": "Day narrative module not available"}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def get_narrative_history(user_id: str, days: int = 7) -> dict[str, Any]:
    """Pobierz historię narracji."""
    try:
        from server_main import server_app

        if hasattr(server_app, "day_narrative"):
            return await server_app.day_narrative.get_narrative_history(days)
        else:
            return {"success": False, "error": "Day narrative module not available"}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def compare_days(user_id: str, date1: str, date2: str) -> dict[str, Any]:
    """Porównaj dwa dni."""
    try:
        from server_main import server_app

        if hasattr(server_app, "day_narrative"):
            return await server_app.day_narrative.compare_days(date1, date2)
        else:
            return {"success": False, "error": "Day narrative module not available"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# Plugin metadata
PLUGIN_FUNCTIONS = {
    "generate_day_narrative": {
        "function": generate_day_narrative,
        "description": "Generate a narrative summary of the user's day",
        "parameters": {
            "type": "object",
            "properties": {
                "date": {
                    "type": "string",
                    "description": "Date in YYYY-MM-DD format (optional, defaults to today)",
                },
                "style": {
                    "type": "string",
                    "description": "Narrative style: friendly, professional, casual, poetic",
                },
            },
        },
    },
    "get_narrative_history": {
        "function": get_narrative_history,
        "description": "Get history of generated day narratives",
        "parameters": {
            "type": "object",
            "properties": {
                "days": {
                    "type": "integer",
                    "description": "Number of days to look back (default: 7)",
                }
            },
        },
    },
    "compare_days": {
        "function": compare_days,
        "description": "Compare two days and generate a comparative narrative",
        "parameters": {
            "type": "object",
            "properties": {
                "date1": {
                    "type": "string",
                    "description": "First date in YYYY-MM-DD format",
                },
                "date2": {
                    "type": "string",
                    "description": "Second date in YYYY-MM-DD format",
                },
            },
            "required": ["date1", "date2"],
        },
    },
}
