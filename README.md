# 🚀 Omni-Agent Hub

**Advanced Multi-Agent Orchestration System with ReAct, MCP, and Agentic RAG**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🌟 Overview

Omni-Agent Hub is a cutting-edge multi-agent orchestration platform that combines the latest AI technologies to create a powerful, scalable, and intelligent system. Inspired by the vision of a digital metropolis where AI agents work in perfect harmony, this system brings together:

- **🧠 ReAct Framework**: Reasoning and Acting for intelligent decision-making
- **🔧 MCP Protocol**: Model Context Protocol for standardized tool integration
- **📚 Agentic RAG**: Context-aware knowledge retrieval and generation
- **🤖 Multi-Agent Workflows**: Specialized agents working in coordination
- **🔍 Self-Reflection**: Quality assurance and continuous improvement
- **⚡ High Performance**: Built for scale with modern async architecture

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    🌐 API Gateway (Spring Cloud Gateway)        │
│                     Traffic Control & Rate Limiting             │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│                🧠 Orchestrator Service (ReAct Core)            │
│              Chain-of-Thought Reasoning Engine                 │
└─────────┬───────────────────────────────────────┬─────────────┘
          │                                       │
┌─────────▼─────────┐                   ┌─────────▼─────────────┐
│  🔧 ToolHub       │                   │  💻 CodeAct Runner    │
│  (MCP Protocol)   │                   │  (Secure Execution)   │
│  External APIs    │                   │  Docker Containers    │
└───────────────────┘                   └───────────────────────┘
          │                                       │
          └─────────┬───────────────────────────────┘
                    │
┌─────────────────▼─────────────────────────────────────────────┐
│                🔍 Self-Reflection Unit                        │
│              Quality Assurance & Improvement                 │
└─────────┬─────────────────────────────────────────────┬─────┘
          │                                             │
┌─────────▼─────────┐                         ┌─────────▼─────────┐
│  🤖 Multi-Agent   │                         │  📚 Agentic RAG   │
│  Workflow Engine  │                         │  Vector Database  │
│  Specialized Agents│                         │  Milvus + OpenAI  │
└───────────────────┘                         └───────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+**
- **Docker & Docker Compose**
- **Git**

### 1. Clone the Repository

```bash
git clone https://github.com/neuraparse/omni-agent-hub.git
cd omni-agent-hub
```

### 2. Environment Setup

```bash
# Copy environment template
cp .env.example .env

# Edit configuration (add your API keys)
nano .env
```

### 3. Start Infrastructure

```bash
# Start all services (PostgreSQL, Redis, Milvus, Kafka)
docker-compose up -d

# Wait for services to be ready (check with)
docker-compose ps
```

### 4. Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

### 5. Initialize Database

```bash
# Run database migrations
omni-hub db --create-tables

# Initialize vector database
omni-hub vector --recreate
```

### 6. Start the Application

```bash
# Development mode
omni-hub serve --reload

# Production mode
omni-hub serve --workers 4
```

The API will be available at `http://localhost:8000`

## 📖 API Documentation

Once the server is running, visit:

- **Interactive API Docs**: `http://localhost:8000/docs`
- **ReDoc Documentation**: `http://localhost:8000/redoc`

### Key Endpoints

```bash
# Health Check
GET /api/v1/health

# Create Session
POST /api/v1/sessions

# Chat with Agent
POST /api/v1/agents/chat

# Execute Task
POST /api/v1/agents/task

# Start Workflow
POST /api/v1/workflows/start

# Search Knowledge
POST /api/v1/knowledge/search
```

## 🛠️ CLI Usage

The Omni-Agent Hub comes with a powerful CLI for management:

```bash
# Start server
omni-hub serve --host 0.0.0.0 --port 8000

# Health checks
omni-hub health --all

# Database operations
omni-hub db --create-tables

# Vector database management
omni-hub vector --collection omni_embeddings

# Redis operations
omni-hub redis --flush

# View configuration
omni-hub config

# System status
omni-hub status
```

## 🔧 Configuration

### Environment Variables

Key configuration options in `.env`:

```bash
# Application
APP_NAME=Omni-Agent Hub
DEBUG=false
LOG_LEVEL=INFO

# AI Models
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key

# Databases
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/omni_hub
REDIS_URL=redis://localhost:6379/0
MILVUS_HOST=localhost
MILVUS_PORT=19530

# External APIs
KAGI_API_KEY=your_kagi_key
SLACK_BOT_TOKEN=your_slack_token
AWS_ACCESS_KEY_ID=your_aws_key
```

### Advanced Configuration

For production deployments, consider:

- **Load Balancing**: Use multiple worker processes
- **Monitoring**: Enable Prometheus metrics
- **Security**: Configure proper authentication
- **Scaling**: Use Kubernetes for container orchestration

## 🤖 Agent Types

### Core Agents

1. **🎯 Orchestrator Agent**: Main coordination and ReAct reasoning
2. **💻 Code Agent**: Python code generation and execution
3. **🔍 Search Agent**: Web search and information retrieval
4. **📊 Analysis Agent**: Data analysis and visualization
5. **🏗️ Architect Agent**: System design and planning
6. **✅ QA Agent**: Quality assurance and testing

### Specialized Workflows

- **Research Workflow**: Multi-step research with validation
- **Development Workflow**: Code generation, testing, deployment
- **Analysis Workflow**: Data processing and insights
- **Content Workflow**: Content creation and optimization

## 🔌 Tool Integration (MCP)

Supported tools and integrations:

### Search & Information
- **Kagi Search**: Advanced web search
- **Wikipedia**: Knowledge base queries
- **ArXiv**: Academic paper search

### Development
- **GitHub**: Repository management
- **Docker**: Container operations
- **AWS Lambda**: Serverless execution

### Communication
- **Slack**: Team notifications
- **Email**: SMTP integration
- **Discord**: Bot interactions

### Data & Analytics
- **dbt Cloud**: Data transformation
- **PostgreSQL**: Database queries
- **Pandas**: Data manipulation

## 📊 Monitoring & Observability

### Metrics

- **Performance**: Response times, throughput
- **Quality**: Success rates, error tracking
- **Usage**: API calls, resource utilization
- **Agents**: Task completion, efficiency

### Logging

Structured logging with correlation IDs:

```json
{
  "timestamp": "2025-01-05T12:00:00Z",
  "level": "INFO",
  "correlation_id": "req_123",
  "agent_name": "orchestrator",
  "action": "task_completed",
  "duration_ms": 250
}
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=omni_agent_hub

# Run specific test categories
pytest -m unit
pytest -m integration
```

## 🚀 Deployment

### Docker Production

```bash
# Build production image
docker build -t omni-agent-hub:latest .

# Run with docker-compose
docker-compose -f docker-compose.prod.yml up -d
```

### Kubernetes

```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -l app=omni-agent-hub
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run code formatting
black src/
isort src/

# Run type checking
mypy src/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAI** for GPT models and embeddings
- **Anthropic** for Claude models and MCP protocol
- **Milvus** for vector database technology
- **FastAPI** for the excellent web framework
- **The open-source community** for amazing tools and libraries

## 📞 Support

- **Documentation**: [docs.omni-agent.com](https://docs.omni-agent.com)
- **Issues**: [GitHub Issues](https://github.com/your-org/omni-agent-hub/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/omni-agent-hub/discussions)
- **Email**: support@omni-agent.com

---

**Built with ❤️ by the Omni-Agent Team**

*Transforming the future of AI agent orchestration, one intelligent decision at a time.*
