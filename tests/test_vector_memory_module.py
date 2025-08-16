import os
import pytest

from config.config_manager import get_database_manager, initialize_database_manager
from modules.vector_memory_module import VectorMemoryModule


@pytest.fixture(scope="function")
def fresh_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test_vectors.db"
    # Reinitialize database manager with temp path
    initialize_database_manager(str(db_path))
    yield get_database_manager()


@pytest.fixture()
def vector_module(fresh_db):
    return VectorMemoryModule()


def test_add_vector_memory_basic(vector_module):
    res = vector_module.add_vector_memory(user_id=1, content="John likes pizza", key="prefs")
    assert res["success"] is True
    assert res["id"] > 0


def test_search_vector_memory_similarity_order(vector_module):
    vector_module.add_vector_memory(1, "The cat sits on the mat")
    vector_module.add_vector_memory(1, "A dog runs in the park")
    vector_module.add_vector_memory(1, "Quantum mechanics explains particles")

    res = vector_module.search_vector_memory(1, query="cat on mat", top_k=2)
    assert res["success"] is True
    sims = [r["similarity"] for r in res["results"]]
    assert len(sims) > 0
    assert sims == sorted(sims, reverse=True)
    assert sims[0] >= 0.0  # fallback may yield low but non-negative similarity


def test_empty_query(vector_module):
    res = vector_module.search_vector_memory(1, query="")
    assert res["success"] is False
    assert "error" in res


def test_min_similarity_filter(vector_module):
    vector_module.add_vector_memory(1, "oranges and apples are fruits")
    res_low = vector_module.search_vector_memory(1, query="fruits", min_similarity=0.05)
    res_high = vector_module.search_vector_memory(1, query="fruits", min_similarity=0.90)
    assert len(res_low["results"]) >= len(res_high["results"])


def test_temporary_memory_expiry(vector_module):
    res = vector_module.add_vector_memory(1, "temp note", persistent=False, ttl_minutes=0)
    assert res["success"]
    search_res = vector_module.search_vector_memory(1, query="temp note")
    assert search_res["success"]


def test_fallback_embedding_determinism(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    vm = VectorMemoryModule()
    e1 = vm._fallback_embedding("Consistent Text")
    e2 = vm._fallback_embedding("Consistent Text")
    assert e1 == e2
    e3 = vm._fallback_embedding("Different Text")
    assert e1 != e3
