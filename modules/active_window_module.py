import abc
import dataclasses
import logging
import sys
from typing import Any

import psutil

"""Active-window helper.

*The original public API (`get_active_window_title`) is **preserved**.*  A new
`get_active_window_context` function adds rich (but non-invasive!) metadata that
helps an LLM understand what the user is doing – e.g. *“site = YouTube”* or
*“match_state = searching”* – **without** OCR, packet sniffing, or any other
technique that might conflict with anti-cheats.

Only two information sources are used:
1. **Process name / PID** (via `psutil`).
2. **Window title** (Win32 API or X11 `_NET_WM_NAME`).

Those are low-impact, universally allowed, and identical to what the original
implementation relied on.
"""

logger = logging.getLogger(__name__)

###############################################################################
# Public data structures (NEW, but optional to consume)                     ###
###############################################################################


@dataclasses.dataclass
class ActiveWindowContext:
    """Rich context describing the foreground window.

    Attributes
    ----------
    app_name      Process name (e.g. ``chrome.exe``)
    window_title  Localised title string shown in the OS title-bar
    extra         Provider-specific key/value pairs with deeper context
    """

    app_name: str | None
    window_title: str | None
    extra: dict[str, Any]

    def __bool__(self):  # Truthiness helper
        return self.app_name is not None or self.window_title is not None


###############################################################################
# Provider plug-in system (safe – no memory hacks, OCR, etc.)               ###
###############################################################################


class ContextProvider(abc.ABC):
    """Base-class for per-application inspectors.

    Each subclass declares which *process names* it understands and may return a dict
    with extra insight based only on *window title* heuristics (zero intrusion, no anti-
    cheat issues).
    """

    HANDLED_PROCESSES: set[str] = set()

    @abc.abstractmethod
    def get_context(self, title: str) -> dict[str, Any]:
        """Return additional metadata extracted from *title*.

        Must never raise – return an empty dict if nothing recognised.
        """


# Registry populated automatically via metaclass
_PROVIDERS: list[ContextProvider] = []


class _ProviderMeta(abc.ABCMeta):
    def __init__(cls, name, bases, ns):
        super().__init__(name, bases, ns)
        if bases and ContextProvider in bases[0].__mro__:
            _PROVIDERS.append(cls())


###############################################################################
# Helper: get (process, title) using OS APIs already used in legacy code    ###
###############################################################################


def _foreground_info() -> tuple[psutil.Process, str] | None:
    try:
        if sys.platform == "win32":
            import win32gui
            import win32process  # type: ignore

            hwnd = win32gui.GetForegroundWindow()
            if not hwnd:
                return None
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            proc = psutil.Process(pid)
            title = win32gui.GetWindowText(hwnd)
            return proc, title

        elif sys.platform.startswith("linux"):
            from Xlib import X, display  # type: ignore

            d = display.Display()
            root = d.screen().root
            aw_id = root.get_full_property(
                d.intern_atom("_NET_ACTIVE_WINDOW"), X.AnyPropertyType
            ).value[0]
            aw = d.create_resource_object("window", aw_id)

            net_wm_name = aw.get_full_property(
                d.intern_atom("_NET_WM_NAME"), d.intern_atom("UTF8_STRING")
            )
            if net_wm_name and net_wm_name.value:
                title = net_wm_name.value.decode("utf-8", "ignore")
            else:
                raw = aw.get_wm_name() or ""
                title = (
                    raw.decode("latin-1", "ignore") if isinstance(raw, bytes) else raw
                )

            pid_prop = aw.get_full_property(
                d.intern_atom("_NET_WM_PID"), X.AnyPropertyType
            )
            if not pid_prop:
                return None
            proc = psutil.Process(int(pid_prop.value[0]))
            return proc, title
    except Exception as e:
        logger.debug(f"Foreground info failed: {e}")
    return None


###############################################################################
# Concrete non-invasive providers                                           ###
###############################################################################


class YouTubeTitleProvider(ContextProvider, metaclass=_ProviderMeta):
    HANDLED_PROCESSES = {
        "chrome.exe",
        "msedge.exe",
        "brave.exe",
        "opera.exe",
        "vivaldi.exe",
        "firefox.exe",
        "chrome",
        "chromium",
        "brave",
        "firefox",
    }

    _SUFFIX = " - YouTube"

    def get_context(self, title: str) -> dict[str, Any]:
        if title.endswith(self._SUFFIX):
            return {
                "site": "YouTube",
                "media_title": title[: -len(self._SUFFIX)].strip(),
            }
        return {}


class ValorantQueueProvider(ContextProvider, metaclass=_ProviderMeta):
    HANDLED_PROCESSES = {"valorant.exe"}

    def get_context(self, title: str) -> dict[str, Any]:
        t = title.lower()
        if "match found" in t:
            return {"match_state": "found"}
        if "queue" in t or "finding match" in t:
            return {"match_state": "searching"}
        return {}


###############################################################################
# Public helpers                                                            ###
###############################################################################


def get_active_window_context() -> ActiveWindowContext | None:
    """Return rich context without breaking anti-cheats or privacy."""

    info = _foreground_info()
    if not info:
        return None

    proc, title = info
    app = proc.name()
    extra: dict[str, Any] = {}

    for p in _PROVIDERS:
        try:
            if app.lower() in p.HANDLED_PROCESSES:
                extra.update(p.get_context(title))
        except Exception as e:
            logger.debug(f"Provider {p.__class__.__name__} failed: {e}")

    return ActiveWindowContext(app_name=app, window_title=title, extra=extra)


# ------------------------------------------------------------------------- #
# Legacy API – UNCHANGED SIGNATURE                                          #
# ------------------------------------------------------------------------- #


def get_active_window_title() -> str | None:
    """Backward-compatible helper.

    Returns exactly what the original function promised: **a string window
    title** or ``None``.  Internally we delegate to the richer context helper
    but throw away the extra details so upstream code keeps working unchanged.
    """

    ctx = get_active_window_context()
    return ctx.window_title if ctx else None


###############################################################################
# Example CLI usage (unchanged behaviour)                                   ###
###############################################################################

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    logger.info("Attempting to get active window title…")

    title = get_active_window_title()
    if title:
        logger.info(f"Currently active window: {title}")
    else:
        logger.warning("Could not determine active window title.")
