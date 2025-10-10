"""Utilities for resolving persistent data directories for the GAJA server."""
from __future__ import annotations

import logging
import os
import sys
from functools import lru_cache
from pathlib import Path
import shutil
from typing import Iterable

logger = logging.getLogger(__name__)

_ENV_PRIORITY = (
    "GAJA_DATA_DIR",
    "GAJA_USER_DATA_DIR",
    "GAJA_STORAGE_DIR",
)

_APP_NAME = "GAJA"


def _iter_candidate_roots() -> Iterable[Path]:
    """Yield possible data root directories in preference order."""
    for env_key in _ENV_PRIORITY:
        env_value = os.getenv(env_key)
        if env_value:
            yield Path(env_value).expanduser()

    # Platform defaults
    home = Path.home()
    if sys.platform.startswith("win"):  # Windows
        appdata = os.getenv("APPDATA")
        if appdata:
            yield Path(appdata) / _APP_NAME
        yield home / "AppData" / "Roaming" / _APP_NAME
    elif sys.platform == "darwin":  # macOS
        yield home / "Library" / "Application Support" / _APP_NAME
    else:  # Linux / Unix
        xdg_data = os.getenv("XDG_DATA_HOME")
        if xdg_data:
            yield Path(xdg_data) / _APP_NAME.lower()
        yield home / f".{_APP_NAME.lower()}"

    # Repository-local fallbacks
    repo_root = Path(__file__).resolve().parent.parent
    yield repo_root / "data" / "persistent"
    yield repo_root / "data"


@lru_cache(maxsize=1)
def get_data_root() -> Path:
    """Return the resolved persistent data directory, ensuring it exists."""
    for candidate in _iter_candidate_roots():
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            logger.debug("Using GAJA data directory: %s", candidate)
            return candidate
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Cannot use data directory %s: %s", candidate, exc)

    # As a final fallback use current working directory
    fallback = Path.cwd() / "data"
    fallback.mkdir(parents=True, exist_ok=True)
    logger.warning("Falling back to working directory data path: %s", fallback)
    return fallback


def resolve_data_path(*parts: str | os.PathLike[str], create_parents: bool = False) -> Path:
    """Resolve a path inside the persistent data directory."""
    base = get_data_root()
    path = base.joinpath(*parts)
    if create_parents:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def migrate_legacy_file(legacy_relative_path: str | os.PathLike[str], target_path: Path) -> bool:
    """Move a legacy file into the persistent location if needed.

    Returns True if a migration occurred.
    """
    legacy_path = Path(__file__).resolve().parent.parent / legacy_relative_path
    if legacy_path.resolve() == target_path.resolve():
        return False
    if not legacy_path.exists() or target_path.exists():
        return False
    try:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(legacy_path), str(target_path))
        logger.info("Migrated legacy data file from %s to %s", legacy_path, target_path)
        return True
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to migrate legacy data file %s -> %s: %s", legacy_path, target_path, exc)
        return False
