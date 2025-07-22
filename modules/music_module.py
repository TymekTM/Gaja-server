"""
Refactored music_module for GAJA Assistant - AGENTS.md Compliant

This module provides music control functionality including:
- Spotify playback control
- Generic media key simulation
- Cross-platform music player integration

All functions are async/await compatible and properly handle external dependencies.
"""

import asyncio
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# Optional third-party libraries with proper async wrappers
_spotify_client = None
_keyboard_available = False

try:
    import spotipy
    from spotipy.oauth2 import SpotifyOAuth

    SPOTIFY_AVAILABLE = True
except ImportError:
    spotipy = None
    SpotifyOAuth = None
    SPOTIFY_AVAILABLE = False
    logger.info("Spotify support disabled: spotipy not installed")

try:
    import keyboard

    _keyboard_available = True
except ImportError:
    keyboard = None
    _keyboard_available = False
    logger.info("Keyboard support disabled: keyboard library not installed")

# Constants
SUPPORTED_ACTIONS = {
    "play": ["play", "resume"],
    "pause": ["pause", "stop"],
    "next": ["next", "skip", "forward"],
    "prev": ["prev", "previous", "back"],
}

# Reverse lookup for quick normalization
NORMALISE = {
    alias: canonical
    for canonical, aliases in SUPPORTED_ACTIONS.items()
    for alias in aliases
}

PLATFORM_ALIASES = {
    "spotify": ["spotify", "spo"],
    "ytmusic": ["ytmusic", "youtube", "youtube music", "yt"],
    "applemusic": ["applemusic", "itunes", "music", "apple"],
    "tidal": ["tidal"],
    "deezer": ["deezer"],
    "auto": ["auto", "default"],
}

SPOTIFY_SCOPE = "user-modify-playback-state user-read-playback-state"


# Required plugin functions
def get_functions() -> list[dict[str, Any]]:
    """Returns list of available functions in the plugin."""
    return [
        {
            "name": "control_music",
            "description": "Control music playback (play, pause, next, prev)",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "Action to perform (play, pause, next, prev)",
                        "enum": ["play", "pause", "next", "prev"],
                    },
                    "platform": {
                        "type": "string",
                        "description": "Music platform (spotify, auto)",
                        "default": "auto",
                    },
                    "test_mode": {
                        "type": "boolean",
                        "description": "Test mode (returns mock success without actual control)",
                        "default": False,
                    },
                },
                "required": ["action"],
            },
        },
        {
            "name": "get_spotify_status",
            "description": "Get current Spotify playback status",
            "parameters": {
                "type": "object",
                "properties": {
                    "test_mode": {
                        "type": "boolean",
                        "description": "Test mode (returns mock data)",
                        "default": False,
                    },
                },
            },
        },
    ]


async def execute_function(
    function_name: str, parameters: dict[str, Any], user_id: int
) -> dict[str, Any]:
    """Executes plugin function asynchronously."""
    try:
        if function_name == "control_music":
            action = parameters.get("action", "").lower()
            platform = parameters.get("platform", "auto").lower()
            test_mode = parameters.get("test_mode", False)

            if not action:
                return {
                    "success": False,
                    "message": "Action parameter is required",
                    "error": "Missing action parameter",
                }

            # Normalize action
            normalized_action = NORMALISE.get(action, action)
            if normalized_action not in SUPPORTED_ACTIONS:
                return {
                    "success": False,
                    "message": f"Unsupported action: {action}",
                    "error": f"Action '{action}' not supported",
                }

            if test_mode:
                return {
                    "success": True,
                    "message": f"Would perform {normalized_action} on {platform} (test mode)",
                    "test_mode": True,
                    "action": normalized_action,
                    "platform": platform,
                }

            # Execute the music control
            result = await _control_music_async(normalized_action, platform)
            return result

        elif function_name == "get_spotify_status":
            test_mode = parameters.get("test_mode", False)

            if test_mode:
                return {
                    "success": True,
                    "message": "Mock Spotify status",
                    "test_mode": True,
                    "status": {
                        "is_playing": True,
                        "track": "Test Track",
                        "artist": "Test Artist",
                    },
                }

            result = await _get_spotify_status_async()
            return result

        else:
            return {
                "success": False,
                "message": f"Unknown function: {function_name}",
                "error": f"Function '{function_name}' not supported",
            }

    except Exception as e:
        logger.error(f"Error in music module function {function_name}: {e}")
        return {
            "success": False,
            "message": f"Error executing {function_name}: {str(e)}",
            "error": str(e),
        }


# Async music control functions
async def _control_music_async(action: str, platform: str) -> dict[str, Any]:
    """Control music playback asynchronously."""
    try:
        # Normalize platform
        normalized_platform = _normalize_platform(platform)

        if normalized_platform == "spotify" or normalized_platform == "auto":
            # Try Spotify first
            if SPOTIFY_AVAILABLE:
                result = await _spotify_action_async(action)
                if result["success"]:
                    return result

                # If Spotify failed and platform was specifically "spotify", return error
                if normalized_platform == "spotify":
                    return result

        # Fallback to system media keys
        if _keyboard_available:
            result = await _system_media_key_async(action)
            return result
        else:
            return {
                "success": False,
                "message": "No music control methods available",
                "error": "Neither Spotify nor keyboard control is available",
            }

    except Exception as e:
        logger.error(f"Error controlling music: {e}")
        return {
            "success": False,
            "message": f"Error controlling music: {e}",
            "error": str(e),
        }


async def _spotify_action_async(action: str) -> dict[str, Any]:
    """Control Spotify asynchronously."""
    try:
        # Get Spotify client in executor to avoid blocking
        loop = asyncio.get_event_loop()
        sp = await loop.run_in_executor(None, _get_spotify_client)

        if sp is None:
            return {
                "success": False,
                "message": "Spotify support unavailable (spotipy not installed or auth failed)",
                "error": "Spotify client unavailable",
            }

        # Get active device
        devices_response = await loop.run_in_executor(None, sp.devices)
        devices = devices_response.get("devices", [])

        if not devices:
            return {
                "success": False,
                "message": "Spotify: no active device found. Start playback on a device first.",
                "error": "No active Spotify device",
            }

        device_id = devices[0]["id"]

        # Execute action in executor
        if action == "play":
            await loop.run_in_executor(None, sp.start_playback, device_id)
        elif action == "pause":
            await loop.run_in_executor(None, sp.pause_playback, device_id)
        elif action == "next":
            await loop.run_in_executor(None, sp.next_track, device_id)
        elif action == "prev":
            await loop.run_in_executor(None, sp.previous_track, device_id)
        else:
            return {
                "success": False,
                "message": f"Spotify: unknown action '{action}'",
                "error": f"Unsupported action: {action}",
            }

        return {
            "success": True,
            "message": f"Spotify → {action} ✓",
            "platform": "spotify",
            "action": action,
        }

    except Exception as e:
        if (
            spotipy
            and hasattr(spotipy, "SpotifyException")
            and isinstance(e, spotipy.SpotifyException)
        ):
            logger.error(f"Spotify API error: {e}")
            return {
                "success": False,
                "message": f"Spotify API error: {e}",
                "error": str(e),
            }
        else:
            logger.error(f"Error in Spotify control: {e}")
            return {
                "success": False,
                "message": f"Error controlling Spotify: {e}",
                "error": str(e),
            }


async def _get_spotify_status_async() -> dict[str, Any]:
    """Get Spotify playback status asynchronously."""
    try:
        if not SPOTIFY_AVAILABLE:
            return {
                "success": False,
                "message": "Spotify support unavailable",
                "error": "spotipy not installed",
            }

        # Get Spotify client in executor
        loop = asyncio.get_event_loop()
        sp = await loop.run_in_executor(None, _get_spotify_client)

        if sp is None:
            return {
                "success": False,
                "message": "Spotify client unavailable",
                "error": "Authentication failed",
            }

        # Get current playback in executor
        playback = await loop.run_in_executor(None, sp.current_playback)

        if not playback:
            return {
                "success": True,
                "message": "No active Spotify playback",
                "status": {"is_playing": False, "track": None, "artist": None},
            }

        track = playback.get("item", {})
        artists = track.get("artists", [])
        artist_names = [artist["name"] for artist in artists]

        status = {
            "is_playing": playback.get("is_playing", False),
            "track": track.get("name", "Unknown"),
            "artist": ", ".join(artist_names) if artist_names else "Unknown",
            "progress_ms": playback.get("progress_ms", 0),
            "duration_ms": track.get("duration_ms", 0),
        }

        return {
            "success": True,
            "message": f"Now {'playing' if status['is_playing'] else 'paused'}: {status['track']} by {status['artist']}",
            "status": status,
        }

    except Exception as e:
        logger.error(f"Error getting Spotify status: {e}")
        return {
            "success": False,
            "message": f"Error getting Spotify status: {e}",
            "error": str(e),
        }


async def _system_media_key_async(action: str) -> dict[str, Any]:
    """Send system media key asynchronously."""
    try:
        if not _keyboard_available:
            return {
                "success": False,
                "message": "Keyboard control unavailable",
                "error": "keyboard library not installed",
            }

        # Media key mapping
        media_keys = {
            "play": "play/pause media",
            "pause": "play/pause media",
            "next": "next track",
            "prev": "previous track",
        }

        if action not in media_keys:
            return {
                "success": False,
                "message": f"Unsupported media key action: {action}",
                "error": f"Action '{action}' not supported",
            }

        # Execute keyboard action in executor to avoid blocking
        loop = asyncio.get_event_loop()

        # Note: keyboard.send() might not work on all systems/environments
        # This is a best-effort implementation
        try:
            if action in ["play", "pause"]:
                await loop.run_in_executor(None, keyboard.send, "play/pause media")
            elif action == "next":
                await loop.run_in_executor(None, keyboard.send, "next track")
            elif action == "prev":
                await loop.run_in_executor(None, keyboard.send, "previous track")

            return {
                "success": True,
                "message": f"System media key → {action} ✓",
                "platform": "system",
                "action": action,
            }

        except Exception as keyboard_error:
            logger.warning(f"Keyboard media key failed: {keyboard_error}")
            return {
                "success": False,
                "message": f"Media key control failed: {keyboard_error}",
                "error": str(keyboard_error),
            }

    except Exception as e:
        logger.error(f"Error in system media key control: {e}")
        return {
            "success": False,
            "message": f"Error with system media keys: {e}",
            "error": str(e),
        }


# Helper functions
def _get_spotify_client() -> Any | None:
    """Get an authenticated Spotify client or None if not available."""
    global _spotify_client

    if not SPOTIFY_AVAILABLE:
        return None

    # Return cached client if available
    if _spotify_client is not None:
        return _spotify_client

    try:
        auth_manager = SpotifyOAuth(
            scope=SPOTIFY_SCOPE, cache_path=os.path.expanduser("~/.cache-music-control")
        )
        _spotify_client = spotipy.Spotify(auth_manager=auth_manager)
        return _spotify_client
    except Exception as exc:
        logger.warning(f"Spotify authentication failed: {exc}")
        return None


def _normalize_platform(platform: str) -> str:
    """Normalize platform name."""
    platform_lower = platform.lower()
    for canonical, aliases in PLATFORM_ALIASES.items():
        if platform_lower in aliases:
            return canonical
    return "auto"


def _normalize_action(action: str) -> str:
    """Normalize action name."""
    return NORMALISE.get(action.lower(), "unknown")


# Legacy handler for backward compatibility
async def process_input(params: str) -> str:
    """Legacy function for backward compatibility."""
    if not params:
        return "👉 Specify action: play / pause / next / prev."

    tokens = params.strip().lower().split()

    if not tokens:
        return "👉 Specify action: play / pause / next / prev."

    # Parse platform and action
    if len(tokens) == 1:
        # Only action provided
        action = tokens[0]
        platform = "auto"
    else:
        # Platform and action provided
        platform = tokens[0]
        action = tokens[1]

    # Use the new async execute_function
    result = await execute_function(
        "control_music", {"action": action, "platform": platform}, user_id=0
    )

    return result["message"]


def register():
    """Register the music module for backward compatibility."""
    return {
        "command": "music",
        "aliases": ["music", "play", "pause", "next", "prev", "spotify"],
        "description": "Control music playback (Spotify, system media keys)",
        "handler": process_input,
        "sub_commands": {
            "control": {
                "description": "Control music playback",
                "parameters": {
                    "action": {
                        "type": "string",
                        "description": "Action (play, pause, next, prev)",
                        "required": True,
                    },
                    "platform": {
                        "type": "string",
                        "description": "Platform (spotify, auto)",
                        "default": "auto",
                    },
                },
            }
        },
    }


# Module status functions
def get_module_status() -> dict[str, Any]:
    """Get the status of music module dependencies."""
    return {
        "spotify_available": SPOTIFY_AVAILABLE,
        "keyboard_available": _keyboard_available,
        "spotify_client_ready": _spotify_client is not None,
    }


class MusicModule:
    """Music module wrapper class for function calling system."""

    def __init__(self):
        """Initialize the music module."""
        logger.info("MusicModule initialized")

    def get_functions(self):
        """Return list of available functions."""
        return [
            {
                "name": "play_music",
                "description": "Play music or a specific song",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "song": {
                            "type": "string",
                            "description": "Song name or query to play",
                        },
                        "platform": {
                            "type": "string",
                            "description": "Platform to use (spotify, auto)",
                            "default": "auto",
                        },
                    },
                    "required": ["song"],
                },
            },
            {
                "name": "control_music",
                "description": "Control music playback (play, pause, next, previous)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "description": "Action to perform (play, pause, next, prev)",
                        },
                        "platform": {
                            "type": "string",
                            "description": "Platform to use (spotify, auto)",
                            "default": "auto",
                        },
                    },
                    "required": ["action"],
                },
            },
        ]
