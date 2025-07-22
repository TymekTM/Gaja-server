"""Extended Web UI for GAJA Assistant Server Rozszerzony panel Web UI z Flask i pełną
funkcjonalnością."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from config_loader import ConfigLoader
from config_manager import DatabaseManager
from flask import Flask, jsonify, redirect, render_template, request, send_file
from flask_cors import CORS

logger = logging.getLogger(__name__)


class ExtendedWebUI:
    """Rozszerzony panel Web UI dla serwera GAJA."""

    def __init__(self, config_loader: ConfigLoader, db_manager: DatabaseManager):
        self.config_loader = config_loader
        self.db_manager = db_manager
        self.app = Flask(__name__, template_folder="templates", static_folder="static")
        CORS(self.app)

        # Server reference - will be set when starting
        self.server_app = None

        self._setup_routes()
        logger.info("ExtendedWebUI initialized")

    def set_server_app(self, server_app):
        """Ustaw referencję do głównej aplikacji serwera."""
        self.server_app = server_app

    def _setup_routes(self):
        """Skonfiguruj routing Flask."""

        # Strona główna - dashboard
        @self.app.route("/")
        def dashboard():
            try:
                config = self.config_loader.get_config()

                # Pobierz podstawowe statystyki
                pm = getattr(self.server_app, "plugin_manager", None)
                loaded_count = 0
                total_functions = 0
                if pm:
                    loaded_count = sum(1 for p in pm.plugins.values() if p.loaded)
                    total_functions = len(pm.function_registry)

                stats = {
                    "server_status": "running",
                    "uptime": getattr(
                        self.server_app, "start_time", datetime.now()
                    ).isoformat(),
                    "loaded_plugins": loaded_count,
                    "total_functions": total_functions,
                    "user_name": config.get("USER_NAME", "User"),
                    "first_run": config.get("FIRST_RUN", True),
                }

                return render_template("dashboard.html", stats=stats, config=config)
            except Exception as e:
                logger.error(f"Error loading dashboard: {e}")
                return f"Error loading dashboard: {e}", 500

        # Onboarding
        @self.app.route("/onboarding")
        def onboarding():
            try:
                config = self.config_loader.get_config()
                first_run = config.get("FIRST_RUN", True)

                if not first_run:
                    return redirect("/")

                # Pobierz szablon konfiguracji
                if hasattr(self.server_app, "onboarding_module"):
                    template = asyncio.run(
                        self.server_app.onboarding_module.get_default_config_template()
                    )
                else:
                    template = {}

                return render_template(
                    "onboarding.html", config_template=template, current_config=config
                )
            except Exception as e:
                logger.error(f"Error loading onboarding: {e}")
                return f"Error loading onboarding: {e}", 500

        # Konfiguracja
        @self.app.route("/config")
        def config_page():
            try:
                config = self.config_loader.get_config()
                return render_template("config.html", config=config)
            except Exception as e:
                logger.error(f"Error loading config page: {e}")
                return f"Error loading config page: {e}", 500

        # Pluginy
        @self.app.route("/plugins")
        def plugins_page():
            try:
                plugin_data = {}
                if hasattr(self.server_app, "plugin_manager"):
                    pm = self.server_app.plugin_manager
                    loaded = [name for name, info in pm.plugins.items() if info.loaded]
                    plugin_data = {
                        "loaded_plugins": loaded,
                        "function_registry": pm.function_registry,
                        "plugin_paths": getattr(pm, "plugin_paths", []),
                    }

                # Status monitorowania
                monitoring_status = {}
                if hasattr(self.server_app, "plugin_monitor"):
                    import asyncio

                    monitoring_status = asyncio.run(
                        self.server_app.plugin_monitor.get_monitoring_status()
                    )

                return render_template(
                    "plugins.html", plugins=plugin_data, monitoring=monitoring_status
                )
            except Exception as e:
                logger.error(f"Error loading plugins page: {e}")
                return f"Error loading plugins page: {e}", 500

        # Pamięć
        @self.app.route("/memory")
        def memory_page():
            try:
                memory_stats = {}
                if hasattr(self.server_app, "advanced_memory"):
                    import asyncio

                    memory_stats = asyncio.run(
                        self.server_app.advanced_memory.get_memory_statistics("1")
                    )

                return render_template("memory.html", memory_stats=memory_stats)
            except Exception as e:
                logger.error(f"Error loading memory page: {e}")
                return f"Error loading memory page: {e}", 500

        # Logi
        @self.app.route("/logs")
        def logs_page():
            try:
                # Pobierz ostatnie logi
                log_files = []
                logs_dir = Path("logs")
                if logs_dir.exists():
                    log_files = sorted(
                        logs_dir.glob("*.log"),
                        key=lambda x: x.stat().st_mtime,
                        reverse=True,
                    )[:10]

                return render_template("logs.html", log_files=log_files)
            except Exception as e:
                logger.error(f"Error loading logs page: {e}")
                return f"Error loading logs page: {e}", 500

        # API endpoints
        @self.app.route("/api/status")
        def api_status():
            try:
                pm = getattr(self.server_app, "plugin_manager", None)
                loaded_count = 0
                functions_available = 0
                if pm:
                    loaded_count = sum(1 for p in pm.plugins.values() if p.loaded)
                    functions_available = len(pm.function_registry)
                status = {
                    "server_running": True,
                    "timestamp": datetime.now().isoformat(),
                    "plugins_loaded": loaded_count,
                    "functions_available": functions_available,
                }
                return jsonify(status)
            except Exception as e:
                logger.error(f"Error getting status: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/config", methods=["GET", "POST"])
        def api_config():
            try:
                if request.method == "GET":
                    config = self.config_loader.get_config()
                    return jsonify(config)

                elif request.method == "POST":
                    new_config = request.json
                    if not new_config:
                        return jsonify({"error": "No configuration data provided"}), 400

                    # Waliduj i zapisz konfigurację
                    current_config = self.config_loader.get_config()
                    current_config.update(new_config)
                    self.config_loader.save_config(current_config)

                    return jsonify(
                        {
                            "success": True,
                            "message": "Configuration updated successfully",
                        }
                    )
            except Exception as e:
                logger.error(f"Error handling config API: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/plugins")
        def api_plugins():
            try:
                plugin_data = {
                    "loaded_plugins": [],
                    "function_registry": {},
                    "monitoring_status": {},
                }

                if hasattr(self.server_app, "plugin_manager"):
                    pm = self.server_app.plugin_manager
                    plugin_data["loaded_plugins"] = [
                        name for name, info in pm.plugins.items() if info.loaded
                    ]
                    plugin_data["function_registry"] = pm.function_registry

                if hasattr(self.server_app, "plugin_monitor"):
                    import asyncio

                    plugin_data["monitoring_status"] = asyncio.run(
                        self.server_app.plugin_monitor.get_monitoring_status()
                    )

                return jsonify(plugin_data)
            except Exception as e:
                logger.error(f"Error getting plugins data: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/plugins/reload/<plugin_name>", methods=["POST"])
        def api_reload_plugin(plugin_name):
            try:
                if hasattr(self.server_app, "plugin_monitor"):
                    import asyncio

                    result = asyncio.run(
                        self.server_app.plugin_monitor.reload_plugin_manually(
                            plugin_name
                        )
                    )
                    return jsonify(result)
                else:
                    return (
                        jsonify(
                            {
                                "success": False,
                                "error": "Plugin monitoring not available",
                            }
                        ),
                        400,
                    )
            except Exception as e:
                logger.error(f"Error reloading plugin {plugin_name}: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/memory/search", methods=["POST"])
        def api_memory_search():
            try:
                data = request.json
                query = data.get("query", "")
                user_id = data.get("user_id", "1")
                category = data.get("category")
                limit = data.get("limit", 10)

                if hasattr(self.server_app, "advanced_memory"):
                    import asyncio

                    result = asyncio.run(
                        self.server_app.advanced_memory.search_memory(
                            user_id, query, category, limit
                        )
                    )
                    return jsonify(result)
                else:
                    return (
                        jsonify(
                            {
                                "success": False,
                                "error": "Advanced memory system not available",
                            }
                        ),
                        400,
                    )
            except Exception as e:
                logger.error(f"Error searching memory: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/memory/add", methods=["POST"])
        def api_memory_add():
            try:
                data = request.json
                user_id = data.get("user_id", "1")
                key = data.get("key")
                content = data.get("content")
                category = data.get("category", "general")
                importance = data.get("importance", 1)

                if not key or not content:
                    return (
                        jsonify(
                            {"success": False, "error": "Key and content are required"}
                        ),
                        400,
                    )

                if hasattr(self.server_app, "advanced_memory"):
                    import asyncio

                    result = asyncio.run(
                        self.server_app.advanced_memory.store_memory(
                            user_id, key, content, category, importance
                        )
                    )
                    return jsonify(result)
                else:
                    return (
                        jsonify(
                            {
                                "success": False,
                                "error": "Advanced memory system not available",
                            }
                        ),
                        400,
                    )
            except Exception as e:
                logger.error(f"Error adding memory: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/logs/<log_file>")
        def api_get_log(log_file):
            try:
                log_path = Path("logs") / log_file
                if not log_path.exists() or not log_path.is_file():
                    return jsonify({"error": "Log file not found"}), 404

                # Bezpieczeństwo - sprawdź czy plik jest w katalogu logs
                if not str(log_path.resolve()).startswith(str(Path("logs").resolve())):
                    return jsonify({"error": "Access denied"}), 403

                with open(log_path, encoding="utf-8") as f:
                    lines = f.readlines()
                    # Zwróć ostatnie 1000 linii
                    recent_lines = lines[-1000:] if len(lines) > 1000 else lines

                return jsonify(
                    {"file": log_file, "lines": recent_lines, "total_lines": len(lines)}
                )
            except Exception as e:
                logger.error(f"Error reading log file {log_file}: {e}")
                return jsonify({"error": str(e)}), 500

        # Obsługa plików statycznych
        @self.app.route("/static/<path:filename>")
        def static_files(filename):
            return send_file(f"static/{filename}")

    def create_default_templates(self):
        """Stwórz domyślne templates jeśli nie istnieją."""
        templates_dir = Path("templates")
        templates_dir.mkdir(exist_ok=True)

        # Base template
        base_template = """<!DOCTYPE html>
<html lang="pl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}GAJA Assistant{% endblock %}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        .sidebar { min-height: 100vh; background-color: #f8f9fa; }
        .main-content { min-height: 100vh; }
        .status-badge { font-size: 0.75rem; }
    </style>
</head>
<body>
    <div class="container-fluid">
        <div class="row">
            <nav class="col-md-2 d-md-block sidebar collapse">
                <div class="position-sticky pt-3">
                    <h5 class="text-center mb-3">GAJA Assistant</h5>
                    <ul class="nav flex-column">
                        <li class="nav-item">
                            <a class="nav-link" href="/"><i class="fas fa-tachometer-alt"></i> Dashboard</a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link" href="/config"><i class="fas fa-cog"></i> Konfiguracja</a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link" href="/plugins"><i class="fas fa-puzzle-piece"></i> Pluginy</a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link" href="/memory"><i class="fas fa-brain"></i> Pamięć</a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link" href="/logs"><i class="fas fa-file-alt"></i> Logi</a>
                        </li>
                    </ul>
                </div>
            </nav>

            <main class="col-md-10 ms-sm-auto px-md-4 main-content">
                <div class="pt-3">
                    {% block content %}{% endblock %}
                </div>
            </main>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    {% block scripts %}{% endblock %}
</body>
</html>"""

        with open(templates_dir / "base.html", "w", encoding="utf-8") as f:
            f.write(base_template)

        # Dashboard template
        dashboard_template = """{% extends "base.html" %}

{% block title %}Dashboard - GAJA Assistant{% endblock %}

{% block content %}
<div class="d-flex justify-content-between flex-wrap flex-md-nowrap align-items-center pt-3 pb-2 mb-3 border-bottom">
    <h1 class="h2">Dashboard</h1>
    <div class="btn-toolbar mb-2 mb-md-0">
        <span class="badge bg-success status-badge">Serwer działa</span>
    </div>
</div>

<div class="row">
    <div class="col-md-3">
        <div class="card">
            <div class="card-body">
                <h5 class="card-title">Pluginy</h5>
                <h3 class="text-primary">{{ stats.loaded_plugins }}</h3>
                <small class="text-muted">załadowanych</small>
            </div>
        </div>
    </div>
    <div class="col-md-3">
        <div class="card">
            <div class="card-body">
                <h5 class="card-title">Funkcje</h5>
                <h3 class="text-info">{{ stats.total_functions }}</h3>
                <small class="text-muted">dostępnych</small>
            </div>
        </div>
    </div>
    <div class="col-md-3">
        <div class="card">
            <div class="card-body">
                <h5 class="card-title">Status</h5>
                <h3 class="text-success">Online</h3>
                <small class="text-muted">{{ stats.server_status }}</small>
            </div>
        </div>
    </div>
    <div class="col-md-3">
        <div class="card">
            <div class="card-body">
                <h5 class="card-title">Użytkownik</h5>
                <h3 class="text-secondary">{{ stats.user_name }}</h3>
                <small class="text-muted">aktywny</small>
            </div>
        </div>
    </div>
</div>

{% if stats.first_run %}
<div class="alert alert-info mt-4">
    <h5><i class="fas fa-info-circle"></i> Pierwsze uruchomienie</h5>
    <p>Wygląda na to, że to Twoje pierwsze uruchomienie GAJA Assistant. Przejdź przez proces konfiguracji.</p>
    <a href="/onboarding" class="btn btn-primary">Rozpocznij konfigurację</a>
</div>
{% endif %}

<div class="mt-4">
    <h3>Szybkie akcje</h3>
    <div class="row">
        <div class="col-md-6">
            <div class="list-group">
                <a href="/config" class="list-group-item list-group-item-action">
                    <i class="fas fa-cog"></i> Zmień konfigurację
                </a>
                <a href="/plugins" class="list-group-item list-group-item-action">
                    <i class="fas fa-puzzle-piece"></i> Zarządzaj pluginami
                </a>
                <a href="/memory" class="list-group-item list-group-item-action">
                    <i class="fas fa-brain"></i> Przeglądaj pamięć
                </a>
            </div>
        </div>
        <div class="col-md-6">
            <div class="card">
                <div class="card-header">
                    <h5>Informacje o systemie</h5>
                </div>
                <div class="card-body">
                    <p><strong>Czas uruchomienia:</strong> {{ stats.uptime }}</p>
                    <p><strong>Załadowane pluginy:</strong> {{ stats.loaded_plugins }}</p>
                    <p><strong>Dostępne funkcje:</strong> {{ stats.total_functions }}</p>
                </div>
            </div>
        </div>
    </div>
</div>
{% endblock %}"""

        with open(templates_dir / "dashboard.html", "w", encoding="utf-8") as f:
            f.write(dashboard_template)

        logger.info("Created default templates")

    def run(self, host: str = "localhost", port: int = 5001, debug: bool = False):
        """Uruchom serwer Web UI."""
        try:
            # Stwórz domyślne templates
            self.create_default_templates()

            logger.info(f"Starting Extended Web UI on http://{host}:{port}")
            self.app.run(host=host, port=port, debug=debug, threaded=True)
        except Exception as e:
            logger.error(f"Error starting Web UI: {e}")


# Global functions for function calling system
async def get_webui_status() -> dict[str, Any]:
    """Pobierz status Web UI."""
    # Import locally to avoid circular imports
    from server_main import server_app

    try:
        if hasattr(server_app, "web_ui"):
            return {"success": True, "running": True, "message": "Web UI is running"}
        else:
            return {
                "success": False,
                "running": False,
                "message": "Web UI not initialized",
            }
    except Exception as e:
        logger.error(f"Error getting Web UI status: {e}")
        return {"success": False, "error": str(e)}


def get_functions():
    """Zwróć listę funkcji dostępnych w Web UI."""
    return [
        {
            "name": "get_webui_status",
            "description": "Pobierz status panelu Web UI",
            "parameters": {"type": "object", "properties": {}, "required": []},
        }
    ]
