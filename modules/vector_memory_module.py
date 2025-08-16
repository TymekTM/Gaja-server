"""Vector (semantic) memory module.

Provides semantic memory storage and search using embedding model
`text-embedding-nomic-embed-text-v1.5`. Falls back to a deterministic local
embedding if OpenAI (or compatible) API key / network is unavailable so that
tests remain stable offline.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any, List, Optional

import numpy as np

try:  # Optional dependency; openai is in requirements
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

from config.config_manager import get_database_manager

logger = logging.getLogger(__name__)

EMBED_MODEL = "text-embedding-nomic-embed-text-v1.5"
FALLBACK_DIM = 768


class VectorMemoryModule:
    """Semantic vector memory service.

    Public methods are used indirectly through function calling system via
    `execute_function`.
    """

    def __init__(self):
        self.db = get_database_manager()
        self._client = None
        if OpenAI and os.getenv("OPENAI_API_KEY"):
            try:
                self._client = OpenAI()
            except Exception as e:  # pragma: no cover
                logger.warning(f"Failed to init OpenAI client, using fallback embeddings: {e}")

    # ---------------- Embedding generation -----------------
    def _get_embedding(self, text: str) -> List[float]:
        text = (text or "").strip()
        if not text:
            return [0.0] * FALLBACK_DIM
        # Try remote embedding first
        if self._client:
            try:
                resp = self._client.embeddings.create(model=EMBED_MODEL, input=text)
                emb = resp.data[0].embedding  # type: ignore[attr-defined]
                # Ensure it's a list[float]
                return list(map(float, emb))
            except Exception as e:  # pragma: no cover - network / API issues
                logger.warning(f"Embedding API failed, fallback vector used: {e}")
        # Deterministic fallback embedding
        return self._fallback_embedding(text)

    @staticmethod
    @lru_cache(maxsize=1024)
    def _fallback_embedding(text: str) -> List[float]:
        # Simple hashed bag-of-words into fixed dimension then L2 normalize
        import numpy as np  # local import for tests
        dim = FALLBACK_DIM
        vec = np.zeros(dim, dtype=np.float32)
        for token in text.lower().split():
            h = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16)
            idx = h % dim
            vec[idx] += 1.0
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec.tolist()

    # ---------------- Core operations -----------------
    def add_vector_memory(
        self,
        user_id: int,
        content: str,
        key: Optional[str] = None,
        persistent: bool = True,
        ttl_minutes: Optional[int] = None,
    ) -> dict[str, Any]:
        if not content or not content.strip():
            return {"success": False, "error": "Empty content"}
        embedding = self._get_embedding(content)
        expires_at = None
        if not persistent and ttl_minutes:
            expires_at = datetime.utcnow() + timedelta(minutes=ttl_minutes)
        vector_id = self.db.add_memory_vector(
            user_id=user_id,
            content=content.strip(),
            embedding=embedding,
            key=key,
            is_persistent=persistent,
            expires_at=expires_at,
        )
        return {"success": vector_id > 0, "id": vector_id}

    def search_vector_memory(
        self,
        user_id: int,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.25,
    ) -> dict[str, Any]:
        if not query:
            return {"success": False, "error": "Empty query"}
        query_emb = self._get_embedding(query)
        stored = self.db.get_memory_vectors(user_id=user_id, limit=500)
        if not stored:
            return {"success": True, "results": []}
        import numpy as np  # local import
        q = np.array(query_emb, dtype=np.float32)
        if np.linalg.norm(q) == 0:
            return {"success": True, "results": []}
        results = []
        for row in stored:
            emb = row.get("embedding") or []
            if not emb:
                continue
            v = np.array(emb, dtype=np.float32)
            denom = float(np.linalg.norm(q) * np.linalg.norm(v))
            if denom == 0:
                sim = 0.0
            else:
                sim = float(np.dot(q, v) / denom)
            if sim >= min_similarity:
                results.append(
                    {
                        "id": row.get("id"),
                        "content": row.get("content"),
                        "key": row.get("key"),
                        "similarity": sim,
                        "created_at": row.get("created_at"),
                    }
                )
        results.sort(key=lambda r: r["similarity"], reverse=True)
        return {"success": True, "results": results[: top_k or 5]}

    # --------------- Function calling integration ---------------
    async def execute_function(self, func_name: str, arguments: dict[str, Any], user_id: int = 1):
        if func_name == "add":
            return self.add_vector_memory(
                user_id=user_id,
                content=arguments.get("content", ""),
                key=arguments.get("key"),
                persistent=arguments.get("persistent", True),
                ttl_minutes=arguments.get("ttl_minutes"),
            )
        if func_name == "search":
            return self.search_vector_memory(
                user_id=user_id,
                query=arguments.get("query", ""),
                top_k=arguments.get("top_k", 5),
                min_similarity=arguments.get("min_similarity", 0.25),
            )
        return {"success": False, "error": f"Unknown function {func_name}"}


def get_functions() -> list[dict[str, Any]]:  # Used by function_calling_system
    return [
        {
            "name": "add",
            "description": "Add semantic memory content (vector stored). Use when user wants to remember or store information for later semantic recall.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "Content to store."},
                    "key": {"type": "string", "description": "Optional grouping key.", "nullable": True},
                    "persistent": {"type": "boolean", "description": "Keep permanently (True) or allow expiry.", "default": True},
                    "ttl_minutes": {"type": "integer", "description": "If not persistent, number of minutes before expiry.", "nullable": True},
                },
                "required": ["content"],
            },
        },
        {
            "name": "search",
            "description": "Semantic search across stored vector memories and return the most similar entries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "User query to match."},
                    "top_k": {"type": "integer", "description": "Max results to return", "default": 5},
                    "min_similarity": {"type": "number", "description": "Minimum cosine similarity threshold (0-1).", "default": 0.25},
                },
                "required": ["query"],
            },
        },
    ]


__all__ = ["VectorMemoryModule", "get_functions"]
