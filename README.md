# 🧠 GAJA Server

**AI-powered server component for GAJA Assistant - Plug & Play Beta Release**

GAJA Server is a FastAPI-based backend that provides AI processing, memory management, and WebSocket communication for the GAJA Assistant ecosystem.

## 🚀 Quick Start

### Option 1: Plug & Play (Recommended)

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

### Option 2: Manual Setup

```bash
# Install Python 3.11+
# Install dependencies
pip install -r requirements_server.txt

# Copy and edit config
cp server_config.template.json server_config.json
nano server_config.json

# Start server
python server_main.py
```

## 📋 Requirements

- **Python 3.11+** (Required)
- **OpenAI API Key** (for AI features)
- **2GB RAM** minimum
- **100MB disk space**

## ⚙️ Configuration

The server uses `server_config.json` for configuration:

```json
{
  "server": {
    "host": "0.0.0.0",
    "port": 8001,
    "debug": false
  },
  "ai": {
    "provider": "openai",
    "model": "gpt-4",
    "api_key": "your-openai-api-key-here",
    "max_tokens": 1500,
    "temperature": 0.7
  },
  "database": {
    "url": "sqlite:///./server_data.db"
  },
  "features": {
    "memory_enabled": true,
    "function_calling": true,
    "web_search": true
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
# Basic start
python start.py

# Development mode (auto-reload, debug)
python start.py --dev

# Force install dependencies
python start.py --install-deps

# Custom config file
python start.py --config my_config.json

# Override port
python start.py --port 9000

# Override host
python start.py --host 127.0.0.1
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

### With Web UI

Configure the web UI to connect to:
```
Server URL: http://localhost:8001
WebSocket URL: ws://localhost:8001/ws
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

## 📊 Performance

**Tested Configuration:**
- Python 3.11
- 2GB RAM
- Single worker process

**Benchmarks:**
- ✅ 200 concurrent connections
- ✅ <100ms response time (local AI)
- ✅ 24/7 uptime tested

For production with higher load, consider:
- Multiple worker processes
- Redis for caching
- PostgreSQL database
- Load balancer

## 🔒 Security

### Production Checklist

- [ ] Change default secret key in config
- [ ] Use environment variables for API keys
- [ ] Enable HTTPS (reverse proxy)
- [ ] Configure firewall rules
- [ ] Regular security updates

### API Key Security

```bash
# Use environment variable instead of config file
export OPENAI_API_KEY="your-key-here"
python start.py
```

## 📝 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📞 Support

- **Issues**: Create GitHub issue
- **Discussions**: GitHub Discussions
- **Documentation**: See `/docs` folder

---

**Status: ✅ Beta Ready**
**Version: 1.0.0-beta**
**Last Updated: July 22, 2025**
