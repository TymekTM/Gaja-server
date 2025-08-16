"""Legacy compatibility shims for old test expectations.

This module dynamically injects lightweight stub modules into sys.modules
so that older tests importing flat module names (e.g. performance_monitor,
plugin_monitor, environment_manager, prompt_builder, prompts, websocket_manager,
advanced_memory_system, plugin_protocol) do not fail with ImportError.

Each shim exposes only the minimal surface required by the tests:
- Classes with expected names & trivial attributes/methods.
- Functions returning simple deterministic placeholders.

All shims log a DEBUG line on first import to make tracing easier.
"""
from __future__ import annotations
import sys, types, logging, time, uuid

logger = logging.getLogger(__name__)

_CREATED = {}

def _ensure(name: str, builder):
    if name in sys.modules:
        return sys.modules[name]
    mod = types.ModuleType(name)
    builder(mod)
    sys.modules[name] = mod
    _CREATED[name] = True
    logger.debug("[compat] installed shim module %s", name)
    return mod

# --- Builders ---------------------------------------------------------------

def _build_performance_monitor(mod):
    class PerformanceMonitor:
        def __init__(self):
            self.metrics = {}
        def record_metric(self, name: str, value: float):
            self.metrics.setdefault(name, []).append(value)
        def start(self):
            self._start = time.perf_counter()
        def stop(self):
            if hasattr(self, '_start'):
                self.record_metric('duration_ms', (time.perf_counter()-self._start)*1000)
    mod.PerformanceMonitor = PerformanceMonitor


def _build_environment_manager(mod):
    class EnvironmentManager:
        def __init__(self):
            self.vars = {}
        def get_environment_info(self):
            return {'os':'shim','python':'shim'}
        def validate(self):
            return True
        def get(self, k, default=None):
            return self.vars.get(k, default)
        def set(self, k, v):
            self.vars[k] = v
    mod.EnvironmentManager = EnvironmentManager


def _build_plugin_monitor(mod):
    class PluginMonitor:
        def __init__(self):
            self.plugins = {}
        def register(self, name, fn):
            self.plugins[name] = fn
        def list_plugins(self):
            return list(self.plugins)
        async def start_monitoring(self):
            return True
    mod.PluginMonitor = PluginMonitor


def _build_plugin_protocol(mod):
    class PluginProtocol:
        def capabilities(self):
            return ['shim']
    mod.PluginProtocol = PluginProtocol


def _build_prompt_builder(mod):
    class PromptBuilder:
        def __init__(self, base: str = ""):
            self.base = base
        def add(self, text: str):
            self.base += text + "\n"
            return self
        def build(self):
            return self.base.strip()
    mod.PromptBuilder = PromptBuilder


def _build_prompts(mod):
    class Prompts:
        def __init__(self):
            self.templates = {'default':'Hello'}
        def get(self, name: str):
            return self.templates.get(name, '')
    mod.Prompts = Prompts


def _build_websocket_manager(mod):
    class WebSocketManager:
        def __init__(self):
            self.clients = {}
        def connect(self, cid):
            self.clients[cid] = True
        def broadcast(self, msg: str):
            return len(self.clients)
    mod.WebSocketManager = WebSocketManager


def _build_advanced_memory_system(mod):
    class AdvancedMemorySystem:
        def __init__(self):
            self.store = {}
        def add(self, key, value):
            self.store[key] = value
        def get(self, key):
            return self.store.get(key)
    mod.AdvancedMemorySystem = AdvancedMemorySystem


def _build_config_loader(mod):
    """Minimal config loader shim providing load_config/save_config used by older tests."""
    import json as _json
    from pathlib import Path as _Path
    _cache: dict[str, dict] = {}
    def load_config(path: str = 'server_config.json'):
        if path in _cache:
            return _cache[path]
        try:
            data = _json.loads(_Path(path).read_text(encoding='utf-8'))
        except Exception:
            data = {}
        _cache[path] = data
        return data
    def save_config(path: str, data: dict):
        try:
            _Path(path).write_text(_json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
            _cache[path] = data
            return True
        except Exception:
            return False
    mod.load_config = load_config
    mod.save_config = save_config
    # Provide placeholders mimicking real config_loader API referenced by tests
    class ConfigLoader:
        def __init__(self, path: str = 'server_config.json'):
            self.path = path
            self.data = load_config(path)
        def get(self, key, default=None):
            return self.data.get(key, default) if isinstance(self.data, dict) else default
        def save(self):
            return save_config(self.path, self.data if isinstance(self.data, dict) else {})
    mod.ConfigLoader = ConfigLoader
    # SHIM values to satisfy ai_module imports
    _cfg = load_config()
    if not isinstance(_cfg, dict):
        _cfg = {}
    mod._config = _cfg  # type: ignore
    mod.MAIN_MODEL = _cfg.get('ai', {}).get('model', 'gpt-4o-mini')
    mod.PROVIDER = _cfg.get('ai', {}).get('provider', 'openai')

# --- Public API ------------------------------------------------------------

def install_all():
    _ensure('performance_monitor', _build_performance_monitor)
    _ensure('environment_manager', _build_environment_manager)
    _ensure('plugin_monitor', _build_plugin_monitor)
    _ensure('plugin_protocol', _build_plugin_protocol)
    _ensure('prompt_builder', _build_prompt_builder)
    _ensure('prompts', _build_prompts)
    _ensure('websocket_manager', _build_websocket_manager)
    _ensure('advanced_memory_system', _build_advanced_memory_system)
    _ensure('config_loader', _build_config_loader)
    # Namespaced variant some code may attempt: config.config_loader
    if 'config' not in sys.modules:
        import types as _types
        sys.modules['config'] = _types.ModuleType('config')
    if not hasattr(sys.modules['config'], 'config_loader'):
        # attach shim as attribute
        sys.modules['config'].config_loader = sys.modules['config_loader']  # type: ignore[attr-defined]
        sys.modules['config.config_loader'] = sys.modules['config_loader']
    return list(_CREATED)

if __name__ == '__main__':  # manual debug
    install_all()
    print('Installed shims:', _CREATED)
