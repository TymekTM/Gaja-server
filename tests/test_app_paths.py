import os
from pathlib import Path

import pytest

from core import app_paths


@pytest.fixture(autouse=True)
def clear_cache():
    app_paths.get_data_root.cache_clear()
    yield
    app_paths.get_data_root.cache_clear()


def test_get_data_root_respects_env(tmp_path, monkeypatch):
    custom_dir = tmp_path / "data_root"
    monkeypatch.setenv("GAJA_DATA_DIR", str(custom_dir))

    root = app_paths.get_data_root()

    assert root == custom_dir
    assert root.exists()

    resolved = app_paths.resolve_data_path("nested", "file.txt")
    assert resolved == root / "nested" / "file.txt"
    # Parent directories should not be created until requested
    assert not resolved.exists()

    ensured = app_paths.resolve_data_path("nested", "file.txt", create_parents=True)
    assert ensured == resolved
    assert ensured.parent.exists()


def test_migrate_legacy_file_moves_when_present(tmp_path, monkeypatch):
    # Point data dir to temp root
    data_dir = tmp_path / "persistent"
    monkeypatch.setenv("GAJA_DATA_DIR", str(data_dir))

    target = app_paths.resolve_data_path("migrated.txt", create_parents=True)

    repo_root = Path(__file__).resolve().parent.parent
    legacy_path = repo_root / "legacy_temp_app_paths.txt"
    legacy_path.write_text("legacy", encoding="utf-8")

    try:
        migrated = app_paths.migrate_legacy_file("legacy_temp_app_paths.txt", target)
        assert migrated is True
        assert target.exists()
        assert target.read_text(encoding="utf-8") == "legacy"
    finally:
        if legacy_path.exists():
            legacy_path.unlink()
        if target.exists():
            target.unlink()
        parent = target.parent
        # Clean up empty parent directories inside data root
        while parent != data_dir.parent and parent.exists():
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent

    # Subsequent migration should return False because file no longer exists at legacy path
    assert app_paths.migrate_legacy_file("legacy_temp_app_paths.txt", target) is False
