#!/usr/bin/env python3
"""
GAJA Server - Plug & Play Starter
Standalone server launcher with automatic dependency management and configuration.
"""

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional

# Minimal imports for startup - full imports happen after dependency check
try:
    import requests
except ImportError:
    requests = None


def load_env_file(env_path: Path = None):
    """Load environment variables from .env file."""
    if env_path is None:
        env_path = Path(__file__).parent / ".env"
    
    if not env_path.exists():
        return
    
    try:
        with open(env_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue
                
                # Parse variable
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    
                    # Only set if not already in environment
                    if key not in os.environ:
                        os.environ[key] = value
                        
    except Exception as e:
        print(f"Warning: Could not load .env file: {e}")


class GajaServerStarter:
    """Main server starter with plug & play functionality."""
    
    def __init__(self):
        # Load environment variables first
        load_env_file()
        
        self.server_root = Path(__file__).parent
        self.config_file = self.server_root / "server_config.json"
        self.log_dir = self.server_root / "logs"
        self.data_dir = self.server_root / "data"
        
        # Setup basic logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
    def _setup_logging(self):
        """Setup basic logging for startup process."""
        self.log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_dir / "server_startup.log"),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def check_python_version(self) -> bool:
        """Check if Python version is compatible."""
        if sys.version_info < (3, 11):
            self.logger.error(f"Python 3.11+ required, found {sys.version}")
            return False
        return True
    
    def install_dependencies(self) -> bool:
        """Install required dependencies automatically."""
        self.logger.info("Checking and installing dependencies...")
        
        requirements_file = self.server_root / "requirements_server.txt"
        if not requirements_file.exists():
            # Create minimal requirements if file doesn't exist
            minimal_deps = [
                "fastapi>=0.104.1",
                "uvicorn[standard]>=0.24.0",
                "aiohttp>=3.11.0",
                "openai>=1.70.0",
                "sqlalchemy>=2.0.0",
                "python-jose[cryptography]>=3.3.0",
                "python-multipart>=0.0.6",
                "bcrypt>=4.0.0",
                "python-dotenv>=1.0.0"
            ]
            
            with open(requirements_file, 'w') as f:
                f.write('\n'.join(minimal_deps))
            
            self.logger.info("Created minimal requirements file")
        
        try:
            # Install requirements
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
            ])
            self.logger.info("Dependencies installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to install dependencies: {e}")
            return False
    
    def create_default_config(self) -> Dict:
        """Create default configuration if none exists."""
        default_config = {
            "server": {
                "host": "0.0.0.0",
                "port": 8001,
                "debug": False,
                "workers": 1,
                "reload": False
            },
            "database": {
                "url": "sqlite:///./server_data.db",
                "echo": False
            },
            "ai": {
                "provider": "openai",
                "model": "gpt-4",
                "api_key": "",
                "max_tokens": 1500,
                "temperature": 0.7
            },
            "auth": {
                "secret_key": "your-secret-key-change-this",
                "algorithm": "HS256",
                "access_token_expire_minutes": 60
            },
            "features": {
                "memory_enabled": True,
                "function_calling": True,
                "proactive_assistant": False,
                "web_search": True
            },
            "paths": {
                "data_dir": "./data",
                "logs_dir": "./logs",
                "cache_dir": "./cache"
            }
        }
        
        if not self.config_file.exists():
            with open(self.config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
            self.logger.info(f"Created default config: {self.config_file}")
        
        return default_config
    
    def load_config(self) -> Dict:
        """Load server configuration."""
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            self.logger.info("Configuration loaded successfully")
            return config
        except Exception as e:
            self.logger.warning(f"Failed to load config: {e}, using defaults")
            return self.create_default_config()
    
    def create_directories(self, config: Dict):
        """Create necessary directories."""
        directories = [
            self.log_dir,
            self.data_dir,
            Path(config.get("paths", {}).get("cache_dir", "./cache"))
        ]
        
        for directory in directories:
            directory.mkdir(exist_ok=True)
            self.logger.debug(f"Created directory: {directory}")
    
    def check_api_keys(self, config: Dict) -> bool:
        """Check if API keys are configured via environment variables or config."""
        # Check environment variables first
        env_api_keys = {
            "openai": os.getenv("OPENAI_API_KEY"),
            "anthropic": os.getenv("ANTHROPIC_API_KEY"), 
            "deepseek": os.getenv("DEEPSEEK_API_KEY")
        }
        
        # Check if any API key is configured
        has_env_key = any(key for key in env_api_keys.values() if key and not key.startswith("your_"))
        
        # Fallback to config file
        config_api_key = config.get("ai", {}).get("api_key", "")
        has_config_key = config_api_key and config_api_key != ""
        
        if not has_env_key and not has_config_key:
            self.logger.warning("No AI API key configured!")
            self.logger.info("Please set API key in .env file or server_config.json")
            self.logger.info("Example: OPENAI_API_KEY=sk-your-key-here")
            self.logger.info("The server will start but AI features will be limited")
            return False
        
        if has_env_key:
            active_keys = [provider for provider, key in env_api_keys.items() if key and not key.startswith("your_")]
            self.logger.info(f"API keys configured via environment: {', '.join(active_keys)}")
        else:
            self.logger.info("API key configured via config file")
        
        return True
    
    def test_server_health(self, host: str, port: int, timeout: int = 10) -> bool:
        """Test if server is responding."""
        if not requests:
            self.logger.warning("Requests not available, skipping health check")
            return True
        
        url = f"http://{host}:{port}/health"
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(url, timeout=2)
                if response.status_code == 200:
                    self.logger.info("Server health check passed")
                    return True
            except Exception:
                pass
            time.sleep(1)
        
        self.logger.warning("Server health check failed")
        return False
    
    async def start_server(self, config: Dict, development: bool = False):
        """Start the GAJA server."""
        self.logger.info("Starting GAJA Server...")
        
        # Import server modules after dependencies are installed
        try:
            import uvicorn
            from server_main import app
        except ImportError as e:
            self.logger.error(f"Failed to import server modules: {e}")
            self.logger.error("Please check if all dependencies are installed")
            return False
        
        server_config = config.get("server", {})
        host = server_config.get("host", "0.0.0.0")
        port = server_config.get("port", 8001)
        
        self.logger.info(f"Server starting on {host}:{port}")
        
        # Configure uvicorn
        uvicorn_config = {
            "app": "server_main:app",
            "host": host,
            "port": port,
            "log_level": "info" if not development else "debug",
            "reload": development,
            "workers": 1 if development else server_config.get("workers", 1),
            "access_log": True
        }
        
        try:
            await uvicorn.run(**uvicorn_config)
        except Exception as e:
            self.logger.error(f"Server startup failed: {e}")
            return False
        
        return True
    
    def print_startup_info(self, config: Dict):
        """Print startup information."""
        server_config = config.get("server", {})
        host = server_config.get("host", "0.0.0.0")
        port = server_config.get("port", 8001)
        
        print("\n" + "="*60)
        print("🤖 GAJA Server - Beta Release")
        print("="*60)
        print(f"Server URL: http://{host}:{port}")
        print(f"Health Check: http://{host}:{port}/health")
        print(f"API Docs: http://{host}:{port}/docs")
        print(f"WebSocket: ws://{host}:{port}/ws")
        print("="*60)
        print("Configuration:")
        print(f"  - Config file: {self.config_file}")
        print(f"  - Data directory: {self.data_dir}")
        print(f"  - Logs directory: {self.log_dir}")
        print(f"  - AI Provider: {config.get('ai', {}).get('provider', 'openai')}")
        print(f"  - Debug mode: {server_config.get('debug', False)}")
        print("="*60)
        print("Press Ctrl+C to stop the server")
        print()
    
    async def run(self, args):
        """Main run method."""
        self.logger.info("GAJA Server Starter - Beta Release")
        
        # Step 1: Check Python version
        if not self.check_python_version():
            return 1
        
        # Step 2: Install dependencies if needed
        if args.install_deps and not self.install_dependencies():
            return 1
        
        # Step 3: Load/create configuration
        config = self.load_config()
        
        # Step 4: Create necessary directories
        self.create_directories(config)
        
        # Step 5: Check API keys (warning only)
        self.check_api_keys(config)
        
        # Step 6: Print startup info
        self.print_startup_info(config)
        
        # Step 7: Start server
        try:
            await self.start_server(config, development=args.dev)
            return 0
        except KeyboardInterrupt:
            self.logger.info("Server stopped by user")
            return 0
        except Exception as e:
            self.logger.error(f"Server failed: {e}")
            return 1


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="GAJA Server - Plug & Play Starter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python start.py                    # Start server with auto-setup
  python start.py --dev              # Start in development mode
  python start.py --install-deps     # Force dependency installation
  python start.py --config custom.json  # Use custom config file

First Run:
  1. python start.py --install-deps  # Install dependencies
  2. Edit server_config.json         # Add your API keys
  3. python start.py                 # Start server

For production:
  python start.py --install-deps
  # Edit server_config.json for production settings
  python start.py
        """
    )
    
    parser.add_argument(
        "--dev", 
        action="store_true",
        help="Start in development mode (auto-reload, debug)"
    )
    
    parser.add_argument(
        "--install-deps",
        action="store_true", 
        help="Force installation of dependencies"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file (default: server_config.json)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        help="Override server port"
    )
    
    parser.add_argument(
        "--host",
        type=str,
        help="Override server host"
    )
    
    args = parser.parse_args()
    
    # Create starter instance
    starter = GajaServerStarter()
    
    # Override config file if specified
    if args.config:
        starter.config_file = Path(args.config)
    
    # Run the server
    try:
        exit_code = asyncio.run(starter.run(args))
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nServer stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
