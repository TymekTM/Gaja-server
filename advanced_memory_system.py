"""
advanced_memory_system.py - Zaawansowany system zarządzania pamięcią AI

System implementuje trzystopniową strukturę pamięci:
- ShortTermMemory: kontekst tymczasowy (15-20 minut)
- MidTermMemory: kontekst dzienny (do końca dnia)
- LongTermMemory: zapis trwały, historyczny

Zawiera mechanizmy wygasania, przenoszenia danych i analizy ważności.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from enum import Enum
from typing import Any

# Import database models

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Enums and Constants
# -----------------------------------------------------------------------------


class MemoryType(Enum):
    SHORT_TERM = "short_term"
    MID_TERM = "mid_term"
    LONG_TERM = "long_term"


class AnalyticsAction(Enum):
    ADD = "add"
    SEARCH = "search"
    PROMOTE = "promote"
    DELETE = "delete"
    CREATE = "create"
    ACCESS = "access"
    EXPIRE = "expire"
    ANALYZE = "analyze"


class ContextType(Enum):
    CONVERSATION = "conversation"
    TASK = "task"
    PERSONAL = "personal"
    SYSTEM = "system"
    LEARNING = "learning"
    EMOTIONAL = "emotional"
    USER_PREFERENCE = "user_preference"
    SYSTEM_EVENT = "system_event"


# Thresholds for importance scoring
IMPORTANCE_THRESHOLDS = {
    "low": 0.3,
    "medium": 0.6,
    "high": 0.8,
    "critical": 0.95,
    "short_to_mid": 0.4,
    "mid_to_long": 0.7,
    "direct_long_term": 0.9,
}

# Time constants
SHORT_TERM_DEFAULT_MINUTES = 20
MID_TERM_EXPIRES_END_OF_DAY = True

# Keywords for emotional content detection
EMOTIONAL_KEYWORDS = {
    "positive": [
        "szczęśliwy",
        "radosny",
        "zadowolony",
        "excited",
        "happy",
        "joy",
        "love",
        "great",
        "amazing",
        "wonderful",
    ],
    "negative": [
        "smutny",
        "zły",
        "frustrated",
        "angry",
        "sad",
        "upset",
        "worried",
        "anxious",
        "stressed",
        "depressed",
    ],
    "important": [
        "ważne",
        "important",
        "critical",
        "urgent",
        "remember",
        "pamiętaj",
        "nie zapomnij",
        "don't forget",
    ],
}

# Personal pronouns and relationship indicators
PERSONAL_INDICATORS = [
    "moja",
    "mój",
    "moje",
    "my",
    "mine",
    "I",
    "ja",
    "me",
    "mnie",
    "rodzina",
    "family",
    "żona",
    "wife",
    "mąż",
    "husband",
    "dziecko",
    "child",
    "mama",
    "tata",
    "mother",
    "father",
    "brat",
    "brother",
    "siostra",
    "sister",
]

# -----------------------------------------------------------------------------
# Core Data Classes
# -----------------------------------------------------------------------------


@dataclass
class MemoryEntry:
    """Universal memory entry that can represent any memory type."""

    content: str
    user: str
    memory_type: MemoryType
    importance_score: float = 0.0
    created_at: datetime = None
    expires_at: datetime = None
    context_type: ContextType | None = None
    context_tags: str | None = None
    access_count: int = 0
    is_important: bool = False
    id: int | None = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

    @classmethod
    def from_short_term(cls, st_memory) -> MemoryEntry:
        """Convert ShortTermMemory to MemoryEntry."""
        return cls(
            id=st_memory.id,
            content=st_memory.content,
            user=st_memory.user,
            memory_type=MemoryType.SHORT_TERM,
            importance_score=st_memory.importance_score,
            created_at=st_memory.created_at,
            expires_at=st_memory.expires_at,
            context_tags=st_memory.context_tags,
        )

    @classmethod
    def from_mid_term(cls, mt_memory) -> MemoryEntry:
        """Convert MidTermMemory to MemoryEntry."""
        return cls(
            id=mt_memory.id,
            content=mt_memory.content,
            user=mt_memory.user,
            memory_type=MemoryType.MID_TERM,
            importance_score=mt_memory.importance_score,
            created_at=mt_memory.created_at,
            expires_at=mt_memory.expires_at,
            context_type=(
                ContextType.TASK
                if mt_memory.context_type == "task"
                else (
                    ContextType.CONVERSATION
                    if mt_memory.context_type == "conversation"
                    else None
                )
            ),
            access_count=mt_memory.access_count,
            context_tags=mt_memory.context_tags,
        )

    @classmethod
    def from_long_term(cls, lt_memory) -> MemoryEntry:
        """Convert LongTermMemory to MemoryEntry."""
        return cls(
            id=lt_memory.id,
            content=lt_memory.content,
            user=lt_memory.user,
            memory_type=MemoryType.LONG_TERM,
            importance_score=lt_memory.importance_score,
            created_at=lt_memory.created_at,
            is_important=lt_memory.is_important,
            access_count=lt_memory.access_count,
            context_tags=lt_memory.context_tags,
        )


@dataclass
class MemorySearchResult:
    """Result from memory search operations."""

    memories: list[MemoryEntry]
    total_count: int
    search_time_ms: float
    relevance_scores: dict[int, float]


@dataclass
class MemoryAnalysisResult:
    """Result from memory importance analysis."""

    original_score: float
    new_score: float
    factors: dict[str, float]
    should_promote: bool
    recommended_type: MemoryType


# -----------------------------------------------------------------------------
# Memory Analyzer - Heuristics for Importance Detection
# -----------------------------------------------------------------------------


class MemoryAnalyzer:
    """Analyzes memories to determine importance and promotion eligibility."""

    def __init__(self):
        self.user_context = defaultdict(list)  # Track user-specific patterns
        self.global_patterns = defaultdict(int)  # Track global usage patterns

    def analyze_importance(
        self, content: str, user: str, context: dict[str, Any] = None
    ) -> MemoryAnalysisResult:
        """Analyze memory content to determine importance score."""
        original_score = 0.0
        factors = {}

        # Base content analysis
        factors["length"] = self._analyze_length(content)
        factors["emotional"] = self._analyze_emotional_content(content)
        factors["personal"] = self._analyze_personal_content(content)
        factors["keywords"] = self._analyze_important_keywords(content)
        factors["structure"] = self._analyze_content_structure(content)

        # Context analysis if provided
        if context:
            factors["repetition"] = self._analyze_repetition_context(
                content, user, context
            )
            factors["user_focus"] = self._analyze_user_focus(content, context)
            factors["temporal"] = self._analyze_temporal_importance(content, context)

        # Calculate weighted importance score
        weights = {
            "length": 0.1,
            "emotional": 0.25,
            "personal": 0.2,
            "keywords": 0.2,
            "structure": 0.1,
            "repetition": 0.1,
            "user_focus": 0.05,
            "temporal": 0.1,
        }

        importance_score = sum(
            factors.get(factor, 0) * weight for factor, weight in weights.items()
        )
        importance_score = max(0.0, min(1.0, importance_score))  # Clamp to [0,1]

        # Determine if should promote and recommended type
        should_promote = importance_score > IMPORTANCE_THRESHOLDS["medium"]
        recommended_type = self._recommend_memory_type(importance_score, factors)

        return MemoryAnalysisResult(
            original_score=original_score,
            new_score=importance_score,
            factors=factors,
            should_promote=should_promote,
            recommended_type=recommended_type,
        )

    def _analyze_length(self, content: str) -> float:
        """Analyze content length - longer content might be more important."""
        words = len(content.split())
        if words == 0:
            return 0.0  # Empty content has no length value
        elif words < 3:
            return 0.1
        elif words < 10:
            return 0.3
        elif words < 30:
            return 0.6
        else:
            return 0.8

    def _analyze_emotional_content(self, content: str) -> float:
        """Detect emotional indicators."""
        content_lower = content.lower()
        score = 0.0

        for emotion_type, keywords in EMOTIONAL_KEYWORDS.items():
            matches = sum(1 for keyword in keywords if keyword in content_lower)
            if emotion_type == "important":
                score += matches * 0.3  # Important keywords get higher weight
            else:
                score += matches * 0.2

        return min(1.0, score)

    def _analyze_personal_content(self, content: str) -> float:
        """Detect personal information indicators."""
        content_lower = content.lower()
        matches = sum(
            1 for indicator in PERSONAL_INDICATORS if indicator in content_lower
        )
        return min(1.0, matches * 0.25)

    def _analyze_important_keywords(self, content: str) -> float:
        """Analyze presence of important keywords."""
        important_patterns = [
            r"\b(spotkanie|meeting|termin|deadline|appointment)\b",
            r"\b(urodziny|birthday|anniversary|rocznica)\b",
            r"\b(praca|work|projekt|project|zadanie|task)\b",
            r"\b(telefon|phone|call|email|mail)\b",
            r"\b(lekarz|doctor|hospital|szpital|health|zdrowie)\b",
            r"\b(pieniądze|money|płatność|payment|bill|rachunek)\b",
        ]

        score = 0.0
        for pattern in important_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                score += 0.2

        return min(1.0, score)

    def _analyze_content_structure(self, content: str) -> float:
        """Analyze content structure (dates, times, names, etc.)."""
        score = 0.0

        # Date patterns
        date_patterns = [
            r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b",  # 01/02/2024
            r"\b\d{1,2}\s+(stycznia|lutego|marca|kwietnia|maja|czerwca|lipca|sierpnia|września|października|listopada|grudnia)\b",
            r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|poniedziałek|wtorek|środa|czwartek|piątek|sobota|niedziela)\b",
        ]

        # Time patterns
        time_patterns = [
            r"\b\d{1,2}:\d{2}\b",  # 14:30
            r"\b\d{1,2}:\d{2}:\d{2}\b",  # 14:30:00
        ]

        # Check for structured content
        for pattern in date_patterns + time_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                score += 0.15

        # Check for proper names (capitalized words)
        proper_names = re.findall(r"\b[A-Z][a-z]+\b", content)
        if len(proper_names) > 0:
            score += min(0.3, len(proper_names) * 0.1)

        return min(1.0, score)

    def _analyze_repetition_context(
        self, content: str, user: str, context: dict[str, Any]
    ) -> float:
        """Analyze if user has mentioned this topic multiple times."""
        # This would analyze conversation history for repeated mentions
        # For now, return baseline score
        return 0.0

    def _analyze_user_focus(self, content: str, context: dict[str, Any]) -> float:
        """Analyze if user is focusing on this topic in conversation."""
        # This would analyze conversation flow and topic coherence
        # For now, return baseline score
        return 0.0

    def _analyze_temporal_importance(
        self, content: str, context: dict[str, Any]
    ) -> float:
        """Analyze temporal importance (upcoming events, deadlines, etc.)."""
        # Look for temporal indicators
        temporal_keywords = [
            "dziś",
            "today",
            "jutro",
            "tomorrow",
            "wczoraj",
            "yesterday",
            "wkrótce",
            "soon",
            "później",
            "later",
            "nigdy",
            "never",
        ]

        urgent_keywords = [
            "pilne",
            "urgent",
            "natychmiast",
            "immediately",
            "szybko",
            "quickly",
        ]

        content_lower = content.lower()
        score = 0.0

        for keyword in temporal_keywords:
            if keyword in content_lower:
                score += 0.1

        for keyword in urgent_keywords:
            if keyword in content_lower:
                score += 0.3

        return min(1.0, score)

    def _recommend_memory_type(
        self, importance_score: float, factors: dict[str, float]
    ) -> MemoryType:
        """Recommend memory type based on importance score and factors."""
        if importance_score >= IMPORTANCE_THRESHOLDS["high"]:
            return MemoryType.LONG
        elif importance_score >= IMPORTANCE_THRESHOLDS["medium"]:
            # Check if it's more of a daily context
            if factors.get("temporal", 0) > 0.5 or factors.get("structure", 0) > 0.5:
                return MemoryType.MID
            else:
                return MemoryType.LONG
        else:
            return MemoryType.SHORT_TERM


# -----------------------------------------------------------------------------
# Memory Search Engine
# -----------------------------------------------------------------------------


class MemorySearchEngine:
    """Handles searching and retrieving memories with relevance scoring."""

    def __init__(self):
        self.analyzer = MemoryAnalyzer()

    def search_all_memories(
        self,
        query: str,
        limit: int = 50,
        user: str = None,
        memory_types: list[MemoryType] = None,
    ) -> MemorySearchResult:
        """Search across all memory types."""
        start_time = datetime.now()

        if memory_types is None:
            memory_types = [
                MemoryType.SHORT_TERM,
                MemoryType.MID_TERM,
                MemoryType.LONG_TERM,
            ]

        all_memories = []
        relevance_scores = {}

        # Search each memory type
        for memory_type in memory_types:
            memories = self._search_memory_type(query, memory_type, limit, user)
            all_memories.extend(memories)

        # Calculate relevance scores
        for memory in all_memories:
            relevance_scores[memory.id] = self._calculate_relevance(
                query, memory.content
            )

        # Sort by relevance and importance
        all_memories.sort(
            key=lambda m: (
                relevance_scores.get(m.id, 0),
                m.importance_score,
                m.created_at,
            ),
            reverse=True,
        )

        # Limit results
        all_memories = all_memories[:limit]

        search_time = (datetime.now() - start_time).total_seconds() * 1000

        return MemorySearchResult(
            memories=all_memories,
            total_count=len(all_memories),
            search_time_ms=search_time,
            relevance_scores=relevance_scores,
        )

    def _search_memory_type(
        self, query: str, memory_type: MemoryType, limit: int, user: str = None
    ) -> list[MemoryEntry]:
        """Search specific memory type."""
        memories = []

        try:
            if memory_type == MemoryType.SHORT_TERM:
                short_memories = get_short_term_memories(
                    limit=limit, user=user, exclude_expired=True
                )
                memories = [
                    self._convert_to_memory_entry(m, MemoryType.SHORT_TERM)
                    for m in short_memories
                ]

            elif memory_type == MemoryType.MID_TERM:
                mid_memories = get_mid_term_memories(
                    limit=limit, user=user, exclude_expired=True
                )
                memories = [
                    self._convert_to_memory_entry(m, MemoryType.MID_TERM)
                    for m in mid_memories
                ]

            elif memory_type == MemoryType.LONG_TERM:
                long_memories = get_long_term_memories_enhanced(
                    limit=limit, query=query, user=user
                )
                memories = [
                    self._convert_to_memory_entry(m, MemoryType.LONG_TERM)
                    for m in long_memories
                ]

        except Exception as e:
            logger.error(f"Error searching {memory_type.value} memories: {e}")
        # Filter by query if not already filtered (for short/mid term)
        if query and memory_type in [MemoryType.SHORT_TERM, MemoryType.MID_TERM]:
            query_lower = query.lower()
            memories = [m for m in memories if query_lower in m.content.lower()]

        return memories

    def _convert_to_memory_entry(
        self,
        memory_obj: ShortTermMemory | MidTermMemory | LongTermMemory,
        memory_type: MemoryType,
    ) -> MemoryEntry:
        """Convert database memory object to universal MemoryEntry."""
        context_type = None
        expires_at = None

        if memory_type == MemoryType.SHORT_TERM:
            expires_at = memory_obj.expires_at
        elif memory_type == MemoryType.MID_TERM:
            expires_at = memory_obj.expires_at
            context_type = getattr(memory_obj, "context_type", None)
            if context_type:
                try:
                    context_type = ContextType(context_type)
                except ValueError:
                    context_type = ContextType.CONVERSATION

        return MemoryEntry(
            id=memory_obj.id,
            content=memory_obj.content,
            user=memory_obj.user,
            memory_type=memory_type,
            importance_score=memory_obj.importance_score,
            created_at=(
                memory_obj.created_at
                if hasattr(memory_obj, "created_at")
                else datetime.now()
            ),
            expires_at=expires_at,
            context_type=context_type,
            context_tags=getattr(memory_obj, "context_tags", None),
            access_count=getattr(memory_obj, "access_count", 0),
            is_important=getattr(memory_obj, "is_important", False),
        )

    def _calculate_relevance(self, query: str, content: str) -> float:
        """Calculate relevance score between query and content."""
        if not query:
            return 0.5

        query_lower = query.lower()
        content_lower = content.lower()

        # Exact match gets highest score
        if query_lower in content_lower:
            return 1.0

        # Use sequence matcher for fuzzy matching
        return SequenceMatcher(None, query_lower, content_lower).ratio()

    def search_memories(
        self,
        memories: list[MemoryEntry],
        query: str,
        limit: int = 50,
        user: str = None,
        memory_type: MemoryType = None,
    ) -> list[MemoryEntry]:
        """Search within a provided list of memories."""
        if not memories:
            return []

        filtered_memories = memories

        # Filter by user if specified
        if user:
            filtered_memories = [m for m in filtered_memories if m.user == user]

        # Filter by memory type if specified
        if memory_type:
            filtered_memories = [
                m for m in filtered_memories if m.memory_type == memory_type
            ]

        # Calculate relevance scores
        scored_memories = []
        for memory in filtered_memories:
            relevance = self._calculate_relevance(query, memory.content)

            # Also check context tags for relevance
            if memory.context_tags:
                context_tags_str = (
                    memory.context_tags
                    if isinstance(memory.context_tags, str)
                    else ",".join(memory.context_tags)
                )
                tag_relevance = self._calculate_relevance(query, context_tags_str)
                relevance = max(relevance, tag_relevance)

            # Only include memories with significant relevance
            if relevance > 0.3:  # Increase threshold for better matching
                scored_memories.append((memory, relevance))

        # Sort by relevance score and importance
        scored_memories.sort(key=lambda x: (x[1], x[0].importance_score), reverse=True)

        # Return top results
        return [memory for memory, score in scored_memories[:limit]]


# -----------------------------------------------------------------------------
# Main Memory Manager
# -----------------------------------------------------------------------------


class MemoryManager:
    """Main manager for the advanced memory system."""

    def __init__(self):
        self.analyzer = MemoryAnalyzer()
        self.search_engine = MemorySearchEngine()
        self.cleanup_running = False

    def add_memory(
        self,
        content: str,
        user: str = "assistant",
        memory_type: MemoryType = None,
        context: dict[str, Any] = None,
    ) -> MemoryEntry:
        """Add new memory with automatic type detection and importance analysis."""

        # Analyze importance if not explicitly specified
        analysis = self.analyzer.analyze_importance(content, user, context)

        # Determine memory type
        if memory_type is None:
            memory_type = analysis.recommended_type

        # Create memory entry
        entry = MemoryEntry(
            content=content,
            user=user,
            memory_type=memory_type,
            importance_score=analysis.new_score,
            context_type=self._determine_context_type(content, context),
            context_tags=self._generate_context_tags(content, context),
            is_important=analysis.new_score >= IMPORTANCE_THRESHOLDS["high"],
        )
        # Store in appropriate table
        memory_id = self._store_memory(entry)
        entry.id = memory_id

        # Add analytics entry
        # Serialize context properly, handling enums
        context_serializable = {}
        if context:
            for key, value in context.items():
                if hasattr(value, "value"):  # Handle enums
                    context_serializable[key] = value.value
                else:
                    context_serializable[key] = value

        add_memory_analytics_entry(
            memory_id=memory_id,
            memory_type=memory_type.value,
            action=AnalyticsAction.CREATE.value,
            importance_change=analysis.new_score,
            context_data=json.dumps(
                {"factors": analysis.factors, "context": context_serializable}
            ),
        )

        logger.info(
            f"Added {memory_type.value} memory {memory_id} with importance {analysis.new_score:.2f}"
        )
        return entry

    def get_memories(
        self,
        query: str = "",
        memory_types: list[MemoryType] = None,
        user: str = None,
        limit: int = 50,
    ) -> MemorySearchResult:
        """Get memories with search and filtering."""
        return self.search_engine.search_all_memories(
            query=query, limit=limit, user=user, memory_types=memory_types
        )

    def search_memories(
        self,
        query: str = "",
        memory_types: list[MemoryType] = None,
        user: str = None,
        limit: int = 50,
    ) -> MemorySearchResult:
        """Search memories (alias for get_memories for compatibility)."""
        return self.get_memories(
            query=query, memory_types=memory_types, user=user, limit=limit
        )

    def promote_memories(self) -> dict[str, int]:
        """Promote memories based on importance and access patterns."""
        results = {
            "short_to_mid": 0,
            "mid_to_long": 0,
            "expired_short": 0,
            "expired_mid": 0,
        }

        # Get candidates for promotion
        short_memories = get_short_term_memories(limit=1000, exclude_expired=False)
        mid_memories = get_mid_term_memories(limit=1000, exclude_expired=False)

        # Promote short to mid term
        for memory in short_memories:
            if self._should_promote_short_to_mid(memory):
                self._promote_short_to_mid(memory)
                results["short_to_mid"] += 1

        # Promote mid to long term
        for memory in mid_memories:
            if self._should_promote_mid_to_long(memory):
                self._promote_mid_to_long(memory)
                results["mid_to_long"] += 1

        # Clean up expired memories
        results["expired_short"] = delete_expired_short_term_memories()
        results["expired_mid"] = delete_expired_mid_term_memories()

        logger.info(f"Memory promotion completed: {results}")
        return results

    def cleanup_expired_memories(self) -> int:
        """Clean up expired memories and return total cleaned."""
        if self.cleanup_running:
            return 0

        self.cleanup_running = True
        try:
            short_deleted = delete_expired_short_term_memories()
            mid_deleted = delete_expired_mid_term_memories()
            total = short_deleted + mid_deleted

            if total > 0:
                logger.info(
                    f"Cleaned up {total} expired memories ({short_deleted} short, {mid_deleted} mid)"
                )

            return total
        finally:
            self.cleanup_running = False

    def analyze_memory_patterns(
        self, user: str = None, days: int = 7
    ) -> dict[str, Any]:
        """Analyze memory usage patterns."""
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)

        analytics = get_memory_analytics()

        # Filter by time range and user
        filtered_analytics = [
            a
            for a in analytics
            if a.timestamp >= start_time
            and (user is None or user in str(a.context_data))
        ]

        # Analyze patterns
        patterns = {
            "total_entries": len(filtered_analytics),
            "by_action": defaultdict(int),
            "by_memory_type": defaultdict(int),
            "importance_trends": [],
            "peak_usage_hours": defaultdict(int),
        }

        for entry in filtered_analytics:
            patterns["by_action"][entry.action] += 1
            patterns["by_memory_type"][entry.memory_type] += 1
            patterns["peak_usage_hours"][entry.timestamp.hour] += 1

            if entry.importance_change is not None:
                patterns["importance_trends"].append(
                    {
                        "timestamp": entry.timestamp.isoformat(),
                        "change": entry.importance_change,
                    }
                )

        return dict(patterns)

    def _store_memory(self, entry: MemoryEntry) -> int:
        """Store memory in appropriate table based on type."""
        if entry.memory_type == MemoryType.SHORT_TERM:
            return add_short_term_memory(
                content=entry.content,
                user=entry.user,
                importance_score=entry.importance_score,
                context_tags=entry.context_tags,
            )
        elif entry.memory_type == MemoryType.MID_TERM:
            return add_mid_term_memory(
                content=entry.content,
                user=entry.user,
                context_type=(
                    entry.context_type.value
                    if entry.context_type
                    else ContextType.CONVERSATION.value
                ),
                importance_score=entry.importance_score,
                context_tags=entry.context_tags,
            )
        else:  # LONG_TERM
            return add_long_term_memory_enhanced(
                content=entry.content,
                user=entry.user,
                importance_score=entry.importance_score,
                is_important=entry.is_important,
                memory_type=(
                    entry.context_type.value if entry.context_type else "general"
                ),
                context_tags=entry.context_tags,
            )

    def _determine_context_type(
        self, content: str, context: dict[str, Any] = None
    ) -> ContextType:
        """Determine context type from content and context."""
        content_lower = content.lower()

        # Check for task-related content
        task_keywords = [
            "zadanie",
            "task",
            "praca",
            "work",
            "projekt",
            "project",
            "deadline",
        ]
        if any(keyword in content_lower for keyword in task_keywords):
            return ContextType.TASK

        # Check for personal content
        if any(indicator in content_lower for indicator in PERSONAL_INDICATORS):
            return ContextType.PERSONAL

        # Check for emotional content
        emotional_found = any(
            any(keyword in content_lower for keyword in keywords)
            for keywords in EMOTIONAL_KEYWORDS.values()
        )
        if emotional_found:
            return ContextType.EMOTIONAL

        # Check for learning content
        learning_keywords = [
            "nauczyć",
            "learn",
            "understand",
            "rozumieć",
            "explain",
            "wyjaśnić",
        ]
        if any(keyword in content_lower for keyword in learning_keywords):
            return ContextType.LEARNING

        # Default to conversation
        return ContextType.CONVERSATION

    def _generate_context_tags(
        self, content: str, context: dict[str, Any] = None
    ) -> str:
        """Generate context tags from content and context."""
        tags = set()
        content_lower = content.lower()

        # Add topic tags
        topic_mapping = {
            "work": ["praca", "work", "job", "office", "biuro"],
            "family": [
                "rodzina",
                "family",
                "żona",
                "wife",
                "mąż",
                "husband",
                "dziecko",
                "child",
            ],
            "health": ["zdrowie", "health", "lekarz", "doctor", "hospital", "szpital"],
            "money": ["pieniądze", "money", "płatność", "payment", "bill", "rachunek"],
            "meeting": ["spotkanie", "meeting", "termin", "appointment"],
            "food": [
                "jedzenie",
                "food",
                "restauracja",
                "restaurant",
                "kolacja",
                "dinner",
            ],
        }

        for tag, keywords in topic_mapping.items():
            if any(keyword in content_lower for keyword in keywords):
                tags.add(tag)

        # Add temporal tags
        temporal_patterns = {
            "urgent": ["pilne", "urgent", "natychmiast", "immediately"],
            "future": ["jutro", "tomorrow", "następny", "next", "wkrótce", "soon"],
            "past": [
                "wczoraj",
                "yesterday",
                "wcześniej",
                "earlier",
                "poprzedni",
                "previous",
            ],
        }

        for tag, keywords in temporal_patterns.items():
            if any(keyword in content_lower for keyword in keywords):
                tags.add(tag)

        return ",".join(sorted(tags)) if tags else None

    def _should_promote_short_to_mid(self, memory: ShortTermMemory) -> bool:
        """Determine if short-term memory should be promoted to mid-term."""
        # Don't promote if already expired
        if memory.expires_at <= datetime.now():
            return False

        # Promote if importance is high enough
        if memory.importance_score >= IMPORTANCE_THRESHOLDS["medium"]:
            return True

        # Promote if accessed multiple times (would need tracking)
        # For now, use creation time as proxy
        age_hours = (datetime.now() - memory.created_at).total_seconds() / 3600
        if age_hours > 1 and memory.importance_score > IMPORTANCE_THRESHOLDS["low"]:
            return True

        return False

    def _should_promote_mid_to_long(self, memory: MidTermMemory) -> bool:
        """Determine if mid-term memory should be promoted to long-term."""
        # Don't promote if already expired
        if memory.expires_at <= datetime.now():
            return False

        # Promote if importance is high
        if memory.importance_score >= IMPORTANCE_THRESHOLDS["high"]:
            return True

        # Promote if accessed frequently
        if memory.access_count >= 3:
            return True

        # Promote based on context type
        important_contexts = ["personal", "task", "emotional"]
        if (
            memory.context_type in important_contexts
            and memory.importance_score > IMPORTANCE_THRESHOLDS["medium"]
        ):
            return True

        return False

    def _promote_short_to_mid(self, memory: ShortTermMemory) -> bool:
        """Promote short-term memory to mid-term."""
        try:
            # Add to mid-term
            mid_id = add_mid_term_memory(
                content=memory.content,
                user=memory.user,
                context_type=ContextType.CONVERSATION.value,  # Default context
                importance_score=memory.importance_score,
                context_tags=memory.context_tags,
            )

            # Remove from short-term
            from database_models import get_connection

            with get_connection() as conn:
                conn.execute("DELETE FROM short_term_memory WHERE id = ?", (memory.id,))
            # Add analytics
            add_memory_analytics_entry(
                memory_id=mid_id,
                memory_type=MemoryType.MID_TERM.value,
                action=AnalyticsAction.PROMOTE.value,
                context_data=json.dumps(
                    {"from_memory_id": memory.id, "from_type": "short"}
                ),
            )

            logger.info(f"Promoted short-term memory {memory.id} to mid-term {mid_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to promote short-term memory {memory.id}: {e}")
            return False

    def _promote_mid_to_long(self, memory: MidTermMemory) -> bool:
        """Promote mid-term memory to long-term."""
        try:
            # Add to long-term
            long_id = add_long_term_memory_enhanced(
                content=memory.content,
                user=memory.user,
                importance_score=memory.importance_score,
                is_important=memory.importance_score >= IMPORTANCE_THRESHOLDS["high"],
                memory_type=memory.context_type or "general",
                context_tags=memory.context_tags,
            )

            # Remove from mid-term
            from database_models import get_connection

            with get_connection() as conn:
                conn.execute(
                    "DELETE FROM mid_term_memory WHERE id = ?", (memory.id,)
                )  # Add analytics
            add_memory_analytics_entry(
                memory_id=long_id,
                memory_type=MemoryType.LONG_TERM.value,
                action=AnalyticsAction.PROMOTE.value,
                context_data=json.dumps(
                    {"from_memory_id": memory.id, "from_type": "mid"}
                ),
            )

            logger.info(f"Promoted mid-term memory {memory.id} to long-term {long_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to promote mid-term memory {memory.id}: {e}")
            return False

    def cleanup_old_analytics(self, days: int = 30) -> int:
        """Clean up old analytics entries and return count of deleted entries."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            from database_models import get_connection

            with get_connection() as conn:
                cursor = conn.execute(
                    "DELETE FROM memory_analytics WHERE timestamp < ?", (cutoff_date,)
                )
                deleted_count = cursor.rowcount
                conn.commit()
            logger.info(f"Cleaned up {deleted_count} old analytics entries")
            return deleted_count
        except Exception as e:
            logger.error(f"Failed to cleanup analytics: {e}")
            return 0


# -----------------------------------------------------------------------------
# Background Tasks
# -----------------------------------------------------------------------------


class MemoryMaintenanceTask:
    """Background task for memory maintenance."""

    def __init__(self, memory_manager: MemoryManager):
        self.memory_manager = memory_manager
        self.running = False

    async def start_maintenance_loop(self, interval_minutes: int = 30):
        """Start the maintenance loop."""
        if self.running:
            return

        self.running = True
        logger.info(
            f"Starting memory maintenance loop (interval: {interval_minutes} minutes)"
        )

        try:
            while self.running:
                await self._run_maintenance()
                await asyncio.sleep(interval_minutes * 60)
        except asyncio.CancelledError:
            logger.info("Memory maintenance loop cancelled")
        finally:
            self.running = False

    def stop_maintenance_loop(self):
        """Stop the maintenance loop."""
        self.running = False

    async def _run_maintenance(self):
        """Run maintenance tasks."""
        try:
            # Promote memories
            results = self.memory_manager.promote_memories()

            # Clean expired memories
            cleaned = self.memory_manager.cleanup_expired_memories()

            if results["short_to_mid"] > 0 or results["mid_to_long"] > 0 or cleaned > 0:
                logger.info(
                    f"Memory maintenance: promoted {results['short_to_mid']} to mid, "
                    f"{results['mid_to_long']} to long, cleaned {cleaned} expired"
                )

        except Exception as e:
            logger.error(f"Error in memory maintenance: {e}")


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

# Global instances
_memory_manager = None
_maintenance_task = None


def get_memory_manager() -> MemoryManager:
    """Get global memory manager instance."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager


def start_memory_maintenance(interval_minutes: int = 30):
    """Start memory maintenance background task."""
    global _maintenance_task
    if _maintenance_task is None or not _maintenance_task.running:
        _maintenance_task = MemoryMaintenanceTask(get_memory_manager())
        # Note: This needs to be started in an event loop
        # asyncio.create_task(_maintenance_task.start_maintenance_loop(interval_minutes))


def stop_memory_maintenance():
    """Stop memory maintenance background task."""
    global _maintenance_task
    if _maintenance_task:
        _maintenance_task.stop_maintenance_loop()


# Background task functions for testing compatibility
async def memory_maintenance_loop(interval_minutes: int = 30):
    """Memory maintenance loop function."""
    manager = get_memory_manager()
    task = MemoryMaintenanceTask(manager)
    await task.start_maintenance_loop(interval_minutes)


async def analytics_cleanup_loop(interval_hours: int = 24):
    """Analytics cleanup loop function."""
    while True:
        try:
            # Clean up old analytics entries using memory manager
            manager = get_memory_manager()
            cleaned_count = manager.cleanup_old_analytics()
            logger.info(f"Analytics cleanup completed, removed {cleaned_count} entries")
        except Exception as e:
            logger.error(f"Analytics cleanup error: {e}")

        await asyncio.sleep(interval_hours * 3600)


# Compatibility functions for existing code
def add_memory_advanced(
    content: str,
    user: str = "assistant",
    memory_type: MemoryType = None,
    context: dict[str, Any] = None,
) -> tuple[str, bool]:
    """Add memory with advanced analysis - returns (message, success) for compatibility."""
    try:
        manager = get_memory_manager()
        entry = manager.add_memory(
            content, user, memory_type=memory_type, context=context
        )
        return f"Zapamiętałem ({entry.memory_type.value}): {content}", True
    except Exception as e:
        logger.error(f"Failed to add advanced memory: {e}")
        return "Wystąpił błąd przy zapisie do pamięci.", False


def search_memories_advanced(
    query: str = "", user: str = None, limit: int = 50
) -> list:
    """Search memories with advanced engine - returns list of memories for compatibility."""
    try:
        manager = get_memory_manager()
        result = manager.get_memories(query=query, user=user, limit=limit)
        return result.memories

    except Exception as e:
        logger.error(f"Failed to search advanced memories: {e}")
        return "Wystąpił błąd przy przeszukiwaniu pamięci.", False


# Global functions for function calling system
async def store_advanced_memory(
    user_id: str, key: str, content: str, category: str = "general", importance: int = 1
) -> dict[str, Any]:
    """Zapisz wpis do zaawansowanej pamięci długoterminowej."""
    try:
        manager = get_memory_manager()

        # Convert importance to memory type
        memory_type = MemoryType.LONG_TERM if importance >= 3 else MemoryType.MID_TERM

        # Create context with category and key
        context = {
            "category": category,
            "key": key,
            "importance": importance,
            "user_id": user_id,
        }

        entry = manager.add_memory(
            content, user_id, memory_type=memory_type, context=context
        )

        return {
            "success": True,
            "entry_id": entry.id,
            "key": key,
            "category": category,
            "memory_type": entry.memory_type.value,
            "message": f"Zapamiętałem: {key}",
        }
    except Exception as e:
        logger.error(f"Error in store_advanced_memory: {e}")
        return {"success": False, "error": str(e)}


async def search_advanced_memory(
    user_id: str, query: str, category: str = None, limit: int = 10
) -> dict[str, Any]:
    """Wyszukaj wpisy w zaawansowanej pamięci."""
    try:
        manager = get_memory_manager()
        result = manager.get_memories(query=query, user=user_id, limit=limit)

        # Filter by category if specified
        filtered_memories = []
        for memory in result.memories:
            if category and memory.context.get("category") != category:
                continue
            filtered_memories.append(
                {
                    "id": memory.id,
                    "content": memory.content,
                    "key": memory.context.get("key", ""),
                    "category": memory.context.get("category", "general"),
                    "importance": memory.context.get("importance", 1),
                    "memory_type": memory.memory_type.value,
                    "created_at": memory.timestamp.isoformat(),
                    "relevance_score": getattr(memory, "relevance_score", 0),
                }
            )

        return {
            "success": True,
            "results": filtered_memories[:limit],
            "query": query,
            "total": len(filtered_memories),
        }
    except Exception as e:
        logger.error(f"Error in search_advanced_memory: {e}")
        return {"success": False, "error": str(e)}


async def get_memory_statistics(user_id: str = "1") -> dict[str, Any]:
    """Pobierz statystyki pamięci użytkownika."""
    try:
        manager = get_memory_manager()

        # Get all memories for user
        result = manager.get_memories(user=user_id, limit=1000)
        memories = result.memories

        # Calculate statistics
        total_memories = len(memories)
        categories = set()
        memory_types = defaultdict(int)
        importance_dist = defaultdict(int)

        for memory in memories:
            category = memory.context.get("category", "general")
            importance = memory.context.get("importance", 1)

            categories.add(category)
            memory_types[memory.memory_type.value] += 1
            importance_dist[importance] += 1

        # Get analytics summary
        analytics_summary = manager.get_analytics_summary()

        return {
            "success": True,
            "general": {
                "total_entries": total_memories,
                "categories": len(categories),
                "avg_importance": sum(
                    memory.context.get("importance", 1) for memory in memories
                )
                / max(total_memories, 1),
                "memory_types": dict(memory_types),
            },
            "categories": list(categories),
            "importance_distribution": dict(importance_dist),
            "analytics": analytics_summary,
            "recent_entries": [
                {
                    "content": (
                        memory.content[:100] + "..."
                        if len(memory.content) > 100
                        else memory.content
                    ),
                    "category": memory.context.get("category", "general"),
                    "created_at": memory.timestamp.isoformat(),
                }
                for memory in memories[:10]
            ],
        }
    except Exception as e:
        logger.error(f"Error in get_memory_statistics: {e}")
        return {"success": False, "error": str(e)}


async def get_advanced_memory(
    user_id: str, key: str = None, category: str = None, limit: int = 10
) -> dict[str, Any]:
    """Pobierz wpisy z zaawansowanej pamięci."""
    try:
        manager = get_memory_manager()
        result = manager.get_memories(user=user_id, limit=100)

        # Filter by key and/or category
        filtered_memories = []
        for memory in result.memories:
            memory_key = memory.context.get("key", "")
            memory_category = memory.context.get("category", "general")

            if key and memory_key != key:
                continue
            if category and memory_category != category:
                continue

            filtered_memories.append(
                {
                    "id": memory.id,
                    "content": memory.content,
                    "key": memory_key,
                    "category": memory_category,
                    "importance": memory.context.get("importance", 1),
                    "memory_type": memory.memory_type.value,
                    "created_at": memory.timestamp.isoformat(),
                }
            )

        return {
            "success": True,
            "memories": filtered_memories[:limit],
            "total": len(filtered_memories),
        }
    except Exception as e:
        logger.error(f"Error in get_advanced_memory: {e}")
        return {"success": False, "error": str(e)}


def get_functions():
    """Zwróć listę funkcji dostępnych w zaawansowanym systemie pamięci."""
    return [
        {
            "name": "store_advanced_memory",
            "description": "Zapisz informację do zaawansowanej pamięci długoterminowej z kategoriami i poziomami ważności",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "ID użytkownika"},
                    "key": {
                        "type": "string",
                        "description": "Klucz/nazwa wpisu do łatwego odnalezienia",
                    },
                    "content": {
                        "type": "string",
                        "description": "Treść do zapamiętania",
                    },
                    "category": {
                        "type": "string",
                        "description": "Kategoria (personal, work, preferences, facts, etc.)",
                        "default": "general",
                    },
                    "importance": {
                        "type": "integer",
                        "description": "Poziom ważności 1-5 (3+ trafia do pamięci długoterminowej)",
                        "default": 1,
                    },
                },
                "required": ["user_id", "key", "content"],
            },
        },
        {
            "name": "search_advanced_memory",
            "description": "Wyszukaj informacje w zaawansowanej pamięci długoterminowej używając inteligentnego wyszukiwania",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "ID użytkownika"},
                    "query": {
                        "type": "string",
                        "description": "Zapytanie wyszukiwania (słowa kluczowe, frazy)",
                    },
                    "category": {
                        "type": "string",
                        "description": "Kategoria do przeszukania (opcjonalne)",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maksymalna liczba wyników",
                        "default": 10,
                    },
                },
                "required": ["user_id", "query"],
            },
        },
        {
            "name": "get_memory_statistics",
            "description": "Pobierz szczegółowe statystyki pamięci użytkownika (liczba wpisów, kategorie, analityki)",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "ID użytkownika",
                        "default": "1",
                    }
                },
                "required": [],
            },
        },
        {
            "name": "get_advanced_memory",
            "description": "Pobierz konkretne wpisy z zaawansowanej pamięci według klucza lub kategorii",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "ID użytkownika"},
                    "key": {
                        "type": "string",
                        "description": "Konkretny klucz wpisu (opcjonalne)",
                    },
                    "category": {
                        "type": "string",
                        "description": "Kategoria wpisów (opcjonalne)",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maksymalna liczba wyników",
                        "default": 10,
                    },
                },
                "required": ["user_id"],
            },
        },
    ]
