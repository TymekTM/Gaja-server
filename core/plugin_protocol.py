from typing import Any, Protocol


class PluginProtocol(Protocol):
    PLUGIN_NAME: str
    PLUGIN_DESCRIPTION: str
    PLUGIN_VERSION: str
    PLUGIN_AUTHOR: str
    PLUGIN_DEPENDENCIES: list[str]

    def get_functions() -> list[dict[str, Any]]:
        ...

    async def execute_function(
        self, function_name: str, parameters: dict[str, Any], user_id: int
    ) -> dict[str, Any]:
        ...
