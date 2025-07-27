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

try:
    import docker
except ImportError:
    docker = None


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
        self.dockerfile_path = self.server_root / "Dockerfile"
        self.docker_image_name = "gaja-server"
        self.docker_container_name = "gaja-server-container"
        
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
    
    def is_docker_available(self) -> bool:
        """Check if Docker is installed and running."""
        try:
            result = subprocess.run(
                ["docker", "--version"], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            if result.returncode == 0:
                self.logger.info(f"Docker found: {result.stdout.strip()}")
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError):
            pass
        
        self.logger.info("Docker not available or not running")
        return False
    
    def docker_image_exists(self) -> bool:
        """Check if Docker image exists."""
        try:
            result = subprocess.run(
                ["docker", "images", "-q", self.docker_image_name],
                capture_output=True,
                text=True,
                timeout=10
            )
            return bool(result.stdout.strip())
        except Exception:
            return False
    
    def build_docker_image(self) -> bool:
        """Build Docker image if Dockerfile exists."""
        if not self.dockerfile_path.exists():
            self.logger.warning("Dockerfile not found, cannot build image")
            return False
        
        self.logger.info("Building Docker image...")
        try:
            result = subprocess.run(
                ["docker", "build", "-t", self.docker_image_name, "."],
                cwd=self.server_root,
                check=True,
                capture_output=True,
                text=True
            )
            self.logger.info("Docker image built successfully")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to build Docker image: {e}")
            self.logger.error(f"Build output: {e.stderr}")
            return False
    
    def stop_existing_container(self):
        """Stop and remove existing container if running."""
        try:
            # Stop container if running
            result = subprocess.run(
                ["docker", "stop", self.docker_container_name],
                capture_output=True,
                timeout=30,
                text=True
            )
            if result.returncode == 0:
                self.logger.info(f"Stopped container: {self.docker_container_name}")
            
            # Remove container
            result = subprocess.run(
                ["docker", "rm", self.docker_container_name],
                capture_output=True,
                timeout=10,
                text=True
            )
            if result.returncode == 0:
                self.logger.info(f"Removed container: {self.docker_container_name}")
        except Exception as e:
            self.logger.warning(f"Error stopping container: {e}")
    
    def get_container_status(self) -> str:
        """Get current container status."""
        try:
            result = subprocess.run(
                ["docker", "ps", "-a", "--filter", f"name={self.docker_container_name}", "--format", "table {{.Status}}"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0 and result.stdout.strip():
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:  # First line is header
                    return lines[1].strip()
            return "Not found"
        except Exception:
            return "Unknown"
    
    def start_docker_container(self, config: Dict) -> bool:
        """Start server in Docker container."""
        server_config = config.get("server", {})
        port = server_config.get("port", 8001)
        
        # Stop existing container
        self.stop_existing_container()
        
        # Prepare volume mounts
        volumes = [
            f"{self.config_file.absolute()}:/app/server_config.json:ro",
            f"{self.log_dir.absolute()}:/app/logs",
            f"{self.data_dir.absolute()}:/app/data"
        ]
        
        # Prepare environment variables
        env_vars = []
        if os.getenv("OPENAI_API_KEY"):
            env_vars.extend(["-e", f"OPENAI_API_KEY={os.getenv('OPENAI_API_KEY')}"])
        if os.getenv("ANTHROPIC_API_KEY"):
            env_vars.extend(["-e", f"ANTHROPIC_API_KEY={os.getenv('ANTHROPIC_API_KEY')}"])
        if os.getenv("DEEPSEEK_API_KEY"):
            env_vars.extend(["-e", f"DEEPSEEK_API_KEY={os.getenv('DEEPSEEK_API_KEY')}"])
        
        # Build Docker command
        docker_cmd = [
            "docker", "run", "-d",
            "--name", self.docker_container_name,
            "-p", f"{port}:{port}",
            "--restart", "unless-stopped"
        ]
        
        # Add volume mounts
        for volume in volumes:
            docker_cmd.extend(["-v", volume])
        
        # Add environment variables
        docker_cmd.extend(env_vars)
        
        # Add image name
        docker_cmd.append(self.docker_image_name)
        
        try:
            result = subprocess.run(docker_cmd, check=True, capture_output=True, text=True)
            container_id = result.stdout.strip()
            self.logger.info(f"Docker container started: {container_id[:12]}")
            
            # Wait a moment and check if container is running
            time.sleep(2)
            result = subprocess.run(
                ["docker", "ps", "-q", "-f", f"name={self.docker_container_name}"],
                capture_output=True,
                text=True
            )
            
            if result.stdout.strip():
                self.logger.info("Container is running successfully")
                return True
            else:
                self.logger.error("Container failed to start")
                # Show container logs
                subprocess.run(["docker", "logs", self.docker_container_name])
                return False
                
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to start Docker container: {e}")
            self.logger.error(f"Error output: {e.stderr}")
            return False
    
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
    
    def handle_docker_command(self, command: str) -> int:
        """Handle Docker-specific commands."""
        if not self.is_docker_available():
            self.logger.error("Docker is not available")
            return 1
            
        if command == "build-server":
            self.logger.info("Building Docker image...")
            if self.build_docker_image():
                self.logger.info("Docker image built successfully")
                return 0
            else:
                self.logger.error("Failed to build Docker image")
                return 1
                
        elif command == "start-server":
            self.logger.info("Starting Docker container...")
            config = self.load_config()
            
            # Check if image exists
            if not self.docker_image_exists():
                self.logger.info("Docker image not found, building first...")
                if not self.build_docker_image():
                    self.logger.error("Failed to build Docker image")
                    return 1
            
            # Start container
            if self.start_docker_container(config):
                self.logger.info("Docker container started successfully")
                
                # Show container info
                status = self.get_container_status()
                self.logger.info(f"Container status: {status}")
                
                # Show useful commands
                print("\n" + "="*60)
                print("Docker Container Management")
                print("="*60)
                print(f"Container name: {self.docker_container_name}")
                print(f"View logs: docker logs {self.docker_container_name}")
                print(f"Follow logs: docker logs -f {self.docker_container_name}")
                print(f"Stop server: python start.py stop-server")
                print(f"Container status: docker ps")
                print("="*60)
                return 0
            else:
                self.logger.error("Failed to start Docker container")
                return 1
                
        elif command == "stop-server":
            self.logger.info("Stopping Docker container...")
            status = self.get_container_status()
            
            if "Not found" in status:
                self.logger.warning("No container found to stop")
                return 0
            
            self.stop_existing_container()
            self.logger.info("Docker container stopped successfully")
            return 0
            
        else:
            self.logger.error(f"Unknown Docker command: {command}")
            return 1
    
    async def start_server(self, config: Dict, development: bool = False, use_docker: bool = False):
        """Start the GAJA server."""
        self.logger.info("Starting GAJA Server...")
        
        # If using Docker
        if use_docker and not development:
            return self.start_docker_container(config)
        
        # Console mode (development or no Docker)
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
        
        self.logger.info(f"Server starting on {host}:{port} (console mode)")
        
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
            config_obj = uvicorn.Config(
                app="server_main:app",
                host=host,
                port=port,
                log_level="info" if not development else "debug",
                reload=development,
                workers=1 if development else server_config.get("workers", 1),
                access_log=True
            )
            server = uvicorn.Server(config_obj)
            await server.serve()
        except Exception as e:
            self.logger.error(f"Server startup failed: {e}")
            return False
        
        return True
    
    def print_startup_info(self, config: Dict, use_docker: bool = False, development: bool = False):
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
        print(f"  - Runtime: {'Docker' if use_docker and not development else 'Console'}")
        print(f"  - Mode: {'Development' if development else 'Production'}")
        print("="*60)
        if use_docker and not development:
            print("Docker container starting...")
            print(f"Use 'docker logs {self.docker_container_name}' to view logs")
            print(f"Use 'docker stop {self.docker_container_name}' to stop server")
        else:
            print("Press Ctrl+C to stop the server")
        print()
    
    async def run(self, args):
        """Main run method."""
        self.logger.info("GAJA Server Starter - Beta Release")
        
        # Handle Docker-specific commands first
        if hasattr(args, 'command') and args.command in ['build-server', 'start-server', 'stop-server']:
            return self.handle_docker_command(args.command)
        
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
        
        # Step 6: Determine runtime mode
        use_docker = False
        if not args.dev:  # Only use Docker in production mode
            # Check force flags
            if args.no_docker:
                self.logger.info("Docker disabled by --no-docker flag")
            elif args.docker:
                if self.is_docker_available():
                    use_docker = True
                    self.logger.info("Docker forced by --docker flag")
                    # Check and build image if needed
                    if not self.docker_image_exists():
                        self.logger.info("Docker image not found, building...")
                        if not self.build_docker_image():
                            self.logger.error("Docker build failed")
                            return 1
                    else:
                        self.logger.info("Docker image found")
                else:
                    self.logger.error("Docker forced but not available")
                    return 1
            else:
                # Auto-detect Docker
                if self.is_docker_available():
                    use_docker = True
                    self.logger.info("Docker available, checking image...")
                    if not self.docker_image_exists():
                        self.logger.info("Docker image not found, building...")
                        if not self.build_docker_image():
                            self.logger.warning("Docker build failed, falling back to console mode")
                            use_docker = False
                        else:
                            self.logger.info("Docker image built successfully")
                    else:
                        self.logger.info("Docker image found")
                else:
                    self.logger.info("Docker not available, using console mode")
        else:
            self.logger.info("Development mode: using console mode")
        
        # Step 7: Print startup info
        self.print_startup_info(config, use_docker, args.dev)
        
        # Step 8: Start server
        try:
            result = await self.start_server(config, development=args.dev, use_docker=use_docker)
            if use_docker and not args.dev:
                # For Docker mode, just return success if container started
                return 0 if result else 1
            else:
                # For console mode, this will block until server stops
                return 0
        except KeyboardInterrupt:
            self.logger.info("Server stopped by user")
            if use_docker:
                self.logger.info("Stopping Docker container...")
                self.stop_existing_container()
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
  python start.py                    # Start server with auto-setup (Docker if available)
  python start.py --dev              # Start in development mode (always console)
  python start.py --install-deps     # Force dependency installation
  python start.py --config custom.json  # Use custom config file
  python start.py --docker           # Force Docker mode
  python start.py --no-docker        # Force console mode

Docker Management:
  python start.py build-server       # Build Docker image only
  python start.py start-server       # Start Docker container (builds if needed)
  python start.py stop-server        # Stop and remove Docker container

First Run:
  1. python start.py --install-deps  # Install dependencies
  2. Edit server_config.json         # Add your API keys
  3. python start.py                 # Start server (Docker or console)

For production:
  python start.py --install-deps
  # Edit server_config.json for production settings
  python start.py                    # Will use Docker if available

For development:
  python start.py --dev              # Always console mode with auto-reload

Docker modes:
  python start.py                    # Auto-detect Docker
  python start.py --docker           # Force Docker (fail if not available)
  python start.py --no-docker        # Force console mode
        """
    )
    
    # Add positional argument for commands
    parser.add_argument(
        'command',
        nargs='?',
        choices=['build-server', 'start-server', 'stop-server'],
        help='Docker management commands'
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
    
    parser.add_argument(
        "--docker",
        action="store_true",
        help="Force Docker mode (if Docker available)"
    )
    
    parser.add_argument(
        "--no-docker",
        action="store_true", 
        help="Force console mode (disable Docker)"
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
