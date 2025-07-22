# 🧠 GAJA Server

**AI-powered server component for GAJA Assistant - Beta Release**

GAJA Server is a FastAPI-based backend that provides AI processing, memory management, and WebSocket communication for the GAJA Assistant ecosystem.

## 🚀 Quick Start

```bash
# Clone or download this server folder
git clone <repo-url> gaja-server
cd gaja-server

# One command to install dependencies and start
python start.py --install-deps

# Edit configuration (add your API keys)
nano server_config.json

# Start server
python start.py
```

## 📋 Requirements

- **Python 3.11+** (Required)
- **OpenAI API Key** (for AI features)
- **512 MB RAM** minimum for server to run 
- **1GB disk space**
- **Docker installed** - Gaja server was designed to run in docker container

## ⚙️ Configuration

The server uses `server_config.json` for configuration:

```json
{
    "server": {
        "host": "0.0.0.0",
        "port": 8001,
        "debug": false
    },
    "database": {
        "url": "sqlite:///./server_data.db",
        "echo": false
    },
    "ai": {
        "provider": "openai",
        //recomended model for it price and speed, models must support function calling
        "model": "gpt-4.1-nano", 
        "temperature": 0.7,
        "max_tokens": 1000
    },
    "api_keys": {
        "openai": "YOUR_OPENAI_API_KEY_HERE",
        "anthropic": "YOUR_ANTHROPIC_API_KEY_HERE"
    },
    "plugins": {
        "auto_load": true,
        "default_enabled": [
            "weather_module",
            "search_module"
        ]
    },
    "logging": {
        "level": "WARNING",
        "file": "logs/server_{time:YYYY-MM-DD}.log"
    }
}

```

### Essential Settings

1. **AI API Key**: Add your OpenAI API key to `ai.api_key`
2. **Server Port**: Default is 8001, change if needed
3. **Host**: Use `0.0.0.0` for external access, `127.0.0.1` for local only

## 🛠️ CLI Usage

The `start.py` script provides a convenient CLI interface:

```bash
# Start server with auto-setup (Docker if available)
python start.py

# Development mode (always console mode)
python start.py --dev

# Force install dependencies
python start.py --install-deps

# Custom config file
python start.py --config my_config.json

# Override port
python start.py --port 9000

# Override host
python start.py --host 127.0.0.1

# Force Docker mode
python start.py --docker

# Force console mode (disable Docker)
python start.py --no-docker
```

## 🌐 API Endpoints

Once running, the server provides:

- **Health Check**: `GET /health`
- **API Documentation**: `GET /docs` (Swagger UI)
- **WebSocket**: `WS /ws` (for real-time communication)
- **Chat API**: `POST /chat`
- **Memory API**: `GET/POST /memory`
- **Config API**: `GET/PUT /config`

Example health check:
```bash
curl http://localhost:8001/health
# Response: {"status": "healthy", "version": "1.0.0-beta"}
```

## 🔌 Integration

### With GAJA Client

The client connects via WebSocket:
```python
import websockets

async def connect_to_server():
    uri = "ws://localhost:8001/ws"
    async with websockets.connect(uri) as websocket:
        await websocket.send('{"type": "chat", "message": "Hello"}')
        response = await websocket.recv()
        print(response)
```


## 📁 Project Structure

```
gaja-server/
├── start.py                 # 🚀 Plug & Play starter
├── server_main.py          # 🧠 Main FastAPI application
├── server_config.json      # ⚙️ Configuration
├── requirements_server.txt # 📦 Dependencies
├── README.md              # 📖 This file
├── 
├── api/                   # 🌐 API routes
│   ├── __init__.py
│   └── routes.py
├── 
├── modules/               # 🧩 Feature modules
│   ├── ai_module.py       # AI processing
│   ├── memory_module.py   # Memory management
│   ├── websocket_manager.py # WebSocket handling
│   └── ...
├── 
├── database/              # 🗄️ Database management
│   ├── models.py
│   └── database.py
├── 
├── logs/                  # 📝 Server logs
├── data/                  # 💾 Application data
└── tests/                 # 🧪 Unit tests
```

## Core Modules

### AI & Processing
- `ai_module.py` - AI integration and processing
- `function_calling_system.py` - Function calling and execution
- `prompt_builder.py` - AI prompt construction
- `prompts.py` - Predefined prompts and templates

### Memory & Data
- `advanced_memory_system.py` - Advanced memory management
- `database_models.py` - Database schema definitions
- `config_loader.py` - Configuration utilities

### Features
- `daily_briefing_module.py` - Daily briefing functionality
- `onboarding_module.py` - User onboarding system
- `performance_monitor.py` - System monitoring
- `plugin_manager.py` - Plugin system management

### Web Interface
- `webui.html` - Web-based user interface
- `extended_webui.py` - Extended web UI functionality

## 🧪 Testing

```bash
# Run unit tests
python -m pytest tests/ -v

# Test server health
curl http://localhost:8001/health

# Test WebSocket connection
python -c "
import asyncio
import websockets

async def test():
    async with websockets.connect('ws://localhost:8001/ws') as ws:
        await ws.send('{\"type\": \"ping\"}')
        response = await ws.recv()
        print(f'Server response: {response}')

asyncio.run(test())
"
```

## 🔧 Development

### Development Mode

```bash
# Start with auto-reload
python start.py --dev

# Monitor logs
tail -f logs/server_startup.log
```

### Adding Features

1. Create module in `modules/`
2. Add routes in `api/routes.py`
3. Update `server_main.py` to include new routes
4. Add tests in `tests/`

### Database

The server uses SQLite by default. For production:

```json
{
  "database": {
    "url": "postgresql://user:pass@host:port/dbname"
  }
}
```

## 🐳 Docker Support

The server automatically uses Docker if available. You can control this behavior:

```bash
# Auto-detect (use Docker if available, fall back to console)
python start.py

# Force Docker mode (will fail if Docker not available)
python start.py --docker

# Force console mode (disable Docker)
python start.py --no-docker

# Development mode (always console, auto-reload)
python start.py --dev
```

**Docker Image Management:**
- If Docker is available, the server will automatically check for existing image
- If image doesn't exist, it will build it from Dockerfile
- Container runs with proper volume mounts for config, data, and logs
- Environment variables are passed through for API keys

**Manual Docker Usage:**

```dockerfile
# Dockerfile included for containerization
FROM python:3.11-slim

WORKDIR /app
COPY requirements_server.txt .
RUN pip install -r requirements_server.txt

COPY . .
EXPOSE 8001

CMD ["python", "start.py"]
```

Build and run:
```bash
docker build -t gaja-server .
docker run -p 8001:8001 -v $(pwd)/server_config.json:/app/server_config.json gaja-server
```

## 🚨 Troubleshooting

### Common Issues

1. **Port already in use**
   ```bash
   python start.py --port 8002
   ```

2. **Missing dependencies**
   ```bash
   python start.py --install-deps
   ```

3. **API key not working**
   - Check `server_config.json`
   - Verify API key is valid
   - Check network connectivity

4. **Permission errors**
   ```bash
   # On Linux/Mac
   chmod +x start.py
   ```

### Debug Mode

```bash
# Enable debug logging
python start.py --dev

# Check logs
cat logs/server_startup.log
```

### Health Checks

```bash
# Basic health
curl http://localhost:8001/health

# Detailed status
curl http://localhost:8001/status
```

### API Key Security

```bash
# Use environment variable instead of config file
export OPENAI_API_KEY="your-key-here"
python start.py
```

## 📝 License

MPL V2.0 License - see LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📞 Support

- **Issues**: Create GitHub issue
- **Discussions**: GitHub Discussions

**Last Updated: July 22, 2025**
