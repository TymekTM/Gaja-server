import asyncio
import logging
import webbrowser
from typing import Any

logger = logging.getLogger(__name__)


# Required plugin functions
def get_functions() -> list[dict[str, Any]]:
    """Returns list of available functions in the plugin."""
    return [
        {
            "name": "open_web",
            "description": "Opens a web page in the default browser",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL of the page to open (e.g. https://www.google.com)",
                    },
                    "test_mode": {
                        "type": "boolean",
                        "description": "Test mode (returns mock success without opening browser)",
                        "default": False,
                    },
                },
                "required": ["url"],
            },
        },
    ]


async def execute_function(
    function_name: str, parameters: dict[str, Any], user_id: int
) -> dict[str, Any]:
    """Executes plugin function."""
    try:
        if function_name == "open_web":
            url = parameters.get("url", "").strip()
            test_mode = parameters.get("test_mode", False)

            if not url:
                return {
                    "success": False,
                    "message": "URL parameter is required",
                    "error": "Missing URL parameter",
                }

            # Ensure URL has proper scheme
            if not url.startswith(("http://", "https://")):
                url = "https://" + url

            if test_mode:
                return {
                    "success": True,
                    "message": f"Would open page: {url} (test mode)",
                    "test_mode": True,
                    "url": url,
                }

            # Use run_in_executor for blocking webbrowser operation
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(None, webbrowser.open, url)

            if success:
                return {
                    "success": True,
                    "message": f"Successfully opened page: {url}",
                    "url": url,
                }
            else:
                logger.error("Failed to open page: %s", url)
                return {
                    "success": False,
                    "message": f"Failed to open page: {url}",
                    "error": "Browser failed to open URL",
                }

        else:
            return {
                "success": False,
                "message": f"Unknown function: {function_name}",
                "error": f"Function '{function_name}' not supported",
            }

    except Exception as e:
        logger.error("Error in open_web module: %s", e, exc_info=True)
        return {
            "success": False,
            "message": f"Error opening web page: {str(e)}",
            "error": str(e),
        }


# Legacy handler for backward compatibility
async def open_web_handler(params: str | dict = "", conversation_history=None) -> str:
    """Legacy handler for backward compatibility."""
    # Convert legacy params to new format
    if isinstance(params, dict):
        url = params.get("url", "").strip()
    else:
        url = str(params).strip()

    # Use the new execute_function
    result = await execute_function("open_web", {"url": url}, user_id=0)

    if result["success"]:
        return result["message"]
    else:
        return result["message"]


def register():
    """Registers the web page opening module for backward compatibility."""
    return {
        "command": "open",
        "aliases": ["open", "url", "browser", "open_web"],
        "description": "Open a web page in the default browser",
        "handler": open_web_handler,  # Legacy async handler
        "sub_commands": {
            "open": {
                "description": "Open a web page in the browser",
                "parameters": {
                    "url": {
                        "type": "string",
                        "description": "URL of the page to open (e.g. https://www.google.com)",
                        "required": True,
                    }
                },
            }
        },
    }


class WebModule:
    """Web module wrapper class for function calling system."""

    def __init__(self):
        """Initialize the web module."""
        logger.info("WebModule initialized")

    def get_functions(self):
        """Return list of available functions."""
        return [
            {
                "name": "open_web",
                "description": "Open a web page in the default browser",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "URL of the page to open (e.g. https://www.google.com)",
                        }
                    },
                    "required": ["url"],
                },
            }
        ]
