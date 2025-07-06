# Omni-Agent Hub Makefile
# Advanced Multi-Agent Orchestration System

.PHONY: help install dev-install clean test lint format type-check security
.PHONY: docker-build docker-up docker-down docker-logs docker-clean
.PHONY: db-init db-migrate db-reset vector-init redis-flush
.PHONY: serve serve-dev serve-prod health status
.PHONY: docs docs-serve release

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := python
PIP := pip
DOCKER_COMPOSE := docker-compose
PROJECT_NAME := omni-agent-hub
VERSION := $(shell grep version pyproject.toml | head -1 | cut -d'"' -f2)

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
PURPLE := \033[0;35m
CYAN := \033[0;36m
WHITE := \033[0;37m
RESET := \033[0m

help: ## Show this help message
	@echo "$(CYAN)🚀 Omni-Agent Hub - Development Commands$(RESET)"
	@echo ""
	@echo "$(YELLOW)📦 Installation:$(RESET)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(GREEN)%-20s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST) | grep -E "(install|clean)"
	@echo ""
	@echo "$(YELLOW)🔧 Development:$(RESET)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(GREEN)%-20s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST) | grep -E "(dev|test|lint|format|type|security)"
	@echo ""
	@echo "$(YELLOW)🐳 Docker:$(RESET)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(GREEN)%-20s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST) | grep -E "docker"
	@echo ""
	@echo "$(YELLOW)🗄️ Database:$(RESET)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(GREEN)%-20s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST) | grep -E "(db|vector|redis)"
	@echo ""
	@echo "$(YELLOW)🚀 Server:$(RESET)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(GREEN)%-20s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST) | grep -E "(serve|health|status)"
	@echo ""
	@echo "$(YELLOW)📚 Documentation:$(RESET)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(GREEN)%-20s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST) | grep -E "(docs|release)"

# Installation
install: ## Install production dependencies
	@echo "$(BLUE)📦 Installing production dependencies...$(RESET)"
	$(PIP) install -e .

dev-install: ## Install development dependencies
	@echo "$(BLUE)📦 Installing development dependencies...$(RESET)"
	$(PIP) install -e ".[dev,monitoring,security]"
	pre-commit install

clean: ## Clean up build artifacts and cache
	@echo "$(YELLOW)🧹 Cleaning up...$(RESET)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf build/ dist/ .coverage htmlcov/

# Development
test: ## Run all tests
	@echo "$(BLUE)🧪 Running tests...$(RESET)"
	pytest -v --cov=omni_agent_hub --cov-report=term-missing --cov-report=html

test-unit: ## Run unit tests only
	@echo "$(BLUE)🧪 Running unit tests...$(RESET)"
	pytest -v -m unit

test-integration: ## Run integration tests only
	@echo "$(BLUE)🧪 Running integration tests...$(RESET)"
	pytest -v -m integration

lint: ## Run linting checks
	@echo "$(BLUE)🔍 Running linting checks...$(RESET)"
	flake8 src/ tests/
	black --check src/ tests/
	isort --check-only src/ tests/

format: ## Format code with black and isort
	@echo "$(BLUE)✨ Formatting code...$(RESET)"
	black src/ tests/
	isort src/ tests/

type-check: ## Run type checking with mypy
	@echo "$(BLUE)🔍 Running type checks...$(RESET)"
	mypy src/

security: ## Run security checks
	@echo "$(BLUE)🔒 Running security checks...$(RESET)"
	bandit -r src/
	safety check

# Docker
docker-build: ## Build Docker image
	@echo "$(BLUE)🐳 Building Docker image...$(RESET)"
	docker build -t $(PROJECT_NAME):$(VERSION) .
	docker build -t $(PROJECT_NAME):latest .

docker-up: ## Start all Docker services
	@echo "$(BLUE)🐳 Starting Docker services...$(RESET)"
	$(DOCKER_COMPOSE) up -d

docker-down: ## Stop all Docker services
	@echo "$(YELLOW)🐳 Stopping Docker services...$(RESET)"
	$(DOCKER_COMPOSE) down

docker-logs: ## Show Docker logs
	@echo "$(BLUE)🐳 Showing Docker logs...$(RESET)"
	$(DOCKER_COMPOSE) logs -f

docker-clean: ## Clean up Docker resources
	@echo "$(YELLOW)🐳 Cleaning Docker resources...$(RESET)"
	$(DOCKER_COMPOSE) down -v --remove-orphans
	docker system prune -f

# Database
db-init: ## Initialize database with tables
	@echo "$(BLUE)🗄️ Initializing database...$(RESET)"
	omni-hub db --create-tables

db-migrate: ## Run database migrations
	@echo "$(BLUE)🗄️ Running database migrations...$(RESET)"
	omni-hub db --create-tables

db-reset: ## Reset database (DANGEROUS)
	@echo "$(RED)⚠️ Resetting database...$(RESET)"
	omni-hub db --drop-tables --create-tables

vector-init: ## Initialize vector database
	@echo "$(BLUE)📊 Initializing vector database...$(RESET)"
	omni-hub vector --recreate

redis-flush: ## Flush Redis cache
	@echo "$(YELLOW)🗑️ Flushing Redis cache...$(RESET)"
	omni-hub redis --flush

# Server
serve: ## Start production server
	@echo "$(GREEN)🚀 Starting production server...$(RESET)"
	omni-hub serve --workers 4

serve-dev: ## Start development server with auto-reload
	@echo "$(GREEN)🚀 Starting development server...$(RESET)"
	omni-hub serve --reload

serve-prod: ## Start production server with optimizations
	@echo "$(GREEN)🚀 Starting production server...$(RESET)"
	omni-hub serve --workers 8 --host 0.0.0.0 --port 8000

health: ## Check system health
	@echo "$(BLUE)🏥 Checking system health...$(RESET)"
	omni-hub health --all

status: ## Show system status
	@echo "$(BLUE)📊 Showing system status...$(RESET)"
	omni-hub status

# Documentation
docs: ## Generate documentation
	@echo "$(BLUE)📚 Generating documentation...$(RESET)"
	cd docs && make html

docs-serve: ## Serve documentation locally
	@echo "$(BLUE)📚 Serving documentation...$(RESET)"
	cd docs && make livehtml

# Release
release: ## Create a new release
	@echo "$(PURPLE)🚀 Creating release $(VERSION)...$(RESET)"
	@echo "$(YELLOW)Running pre-release checks...$(RESET)"
	$(MAKE) test
	$(MAKE) lint
	$(MAKE) type-check
	$(MAKE) security
	@echo "$(GREEN)✅ All checks passed!$(RESET)"
	@echo "$(BLUE)Building distribution...$(RESET)"
	$(PYTHON) -m build
	@echo "$(GREEN)🎉 Release $(VERSION) ready!$(RESET)"

# Quick setup for new developers
setup: ## Complete setup for new developers
	@echo "$(CYAN)🚀 Setting up Omni-Agent Hub development environment...$(RESET)"
	@echo "$(BLUE)1. Installing dependencies...$(RESET)"
	$(MAKE) dev-install
	@echo "$(BLUE)2. Starting Docker services...$(RESET)"
	$(MAKE) docker-up
	@echo "$(BLUE)3. Waiting for services to be ready...$(RESET)"
	sleep 30
	@echo "$(BLUE)4. Initializing databases...$(RESET)"
	$(MAKE) db-init
	$(MAKE) vector-init
	@echo "$(BLUE)5. Running health checks...$(RESET)"
	$(MAKE) health
	@echo "$(GREEN)✅ Setup complete! Run 'make serve-dev' to start development server.$(RESET)"

# CI/CD targets
ci-test: ## Run tests in CI environment
	@echo "$(BLUE)🤖 Running CI tests...$(RESET)"
	pytest --cov=omni_agent_hub --cov-report=xml --cov-report=term

ci-lint: ## Run linting in CI environment
	@echo "$(BLUE)🤖 Running CI linting...$(RESET)"
	flake8 src/ tests/ --format=github
	black --check src/ tests/
	isort --check-only src/ tests/

ci-security: ## Run security checks in CI environment
	@echo "$(BLUE)🤖 Running CI security checks...$(RESET)"
	bandit -r src/ -f json -o bandit-report.json
	safety check --json --output safety-report.json

# Performance testing
perf-test: ## Run performance tests
	@echo "$(BLUE)⚡ Running performance tests...$(RESET)"
	locust -f tests/performance/locustfile.py --headless -u 10 -r 2 -t 60s --host http://localhost:8000

# Monitoring
monitor: ## Start monitoring stack
	@echo "$(BLUE)📊 Starting monitoring stack...$(RESET)"
	$(DOCKER_COMPOSE) -f docker-compose.monitoring.yml up -d

# Backup
backup: ## Backup databases
	@echo "$(BLUE)💾 Creating database backup...$(RESET)"
	mkdir -p backups
	$(DOCKER_COMPOSE) exec postgres pg_dump -U omni_user omni_hub > backups/postgres_$(shell date +%Y%m%d_%H%M%S).sql
	@echo "$(GREEN)✅ Backup created in backups/ directory$(RESET)"

# Development utilities
shell: ## Open Python shell with app context
	@echo "$(BLUE)🐍 Opening Python shell...$(RESET)"
	$(PYTHON) -c "from omni_agent_hub import *; import asyncio"

logs: ## Show application logs
	@echo "$(BLUE)📋 Showing application logs...$(RESET)"
	tail -f logs/omni-agent-hub.log

# Version management
version: ## Show current version
	@echo "$(CYAN)Current version: $(VERSION)$(RESET)"

bump-patch: ## Bump patch version
	@echo "$(BLUE)📈 Bumping patch version...$(RESET)"
	bump2version patch

bump-minor: ## Bump minor version
	@echo "$(BLUE)📈 Bumping minor version...$(RESET)"
	bump2version minor

bump-major: ## Bump major version
	@echo "$(BLUE)📈 Bumping major version...$(RESET)"
	bump2version major
