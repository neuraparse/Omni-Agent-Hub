# Contributing to Omni-Agent Hub

Thank you for your interest in contributing to Omni-Agent Hub! This document provides guidelines and information for contributors.

## 🎯 How to Contribute

### 🐛 Reporting Bugs

1. **Search existing issues** to avoid duplicates
2. **Use the bug report template** when creating new issues
3. **Provide detailed information** including:
   - System environment
   - Steps to reproduce
   - Expected vs actual behavior
   - Error logs and screenshots

### 💡 Suggesting Features

1. **Check existing feature requests** to avoid duplicates
2. **Use the feature request template**
3. **Explain the motivation** and use cases
4. **Provide implementation ideas** if possible

### 🔧 Code Contributions

1. **Fork the repository**
2. **Create a feature branch** from `main`
3. **Make your changes** following our coding standards
4. **Add tests** for new functionality
5. **Update documentation** as needed
6. **Submit a pull request** using our template

## 🏗️ Development Setup

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- Git

### Local Development

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/Omni-Agent-Hub.git
cd Omni-Agent-Hub

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install development dependencies
pip install -e ".[dev]"

# Start infrastructure services
docker-compose up -d

# Run the application
omni-hub serve --reload
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=omni_agent_hub

# Run specific test file
pytest tests/test_agents.py

# Run with verbose output
pytest tests/ -v
```

## 📋 Coding Standards

### Python Code Style

- **PEP 8** compliance
- **Type hints** for all functions
- **Docstrings** for all public methods
- **Maximum line length**: 100 characters

### Code Organization

```python
# Import order
import standard_library
import third_party_packages
import local_modules

# Function structure
async def function_name(
    param1: str,
    param2: Optional[int] = None
) -> ReturnType:
    """
    Brief description.
    
    Args:
        param1: Description
        param2: Description
        
    Returns:
        Description of return value
        
    Raises:
        ExceptionType: Description
    """
    # Implementation
```

### Error Handling

```python
# Use custom exceptions
from omni_agent_hub.core.exceptions import OmniAgentException

try:
    # Code that might fail
    result = risky_operation()
except SpecificException as e:
    logger.error("Operation failed", error=str(e))
    raise OmniAgentException(f"Failed to perform operation: {e}")
```

### Logging

```python
# Use structured logging
from omni_agent_hub.core.logging import get_logger

logger = get_logger(__name__)

# Log with context
logger.info("Operation completed", 
           user_id=user_id, 
           operation="data_processing",
           duration_ms=duration)
```

## 🧪 Testing Guidelines

### Test Structure

```python
import pytest
from unittest.mock import AsyncMock, patch

class TestAgentOrchestrator:
    """Test cases for ReAct Orchestrator."""
    
    @pytest.fixture
    async def orchestrator(self):
        """Create test orchestrator instance."""
        # Setup code
        
    async def test_successful_execution(self, orchestrator):
        """Test successful task execution."""
        # Test implementation
        
    async def test_error_handling(self, orchestrator):
        """Test error handling scenarios."""
        # Test implementation
```

### Test Categories

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test component interactions
- **End-to-End Tests**: Test complete workflows
- **Performance Tests**: Test system performance

## 📚 Documentation

### Code Documentation

- **Docstrings** for all public APIs
- **Type hints** for better IDE support
- **Inline comments** for complex logic
- **README updates** for new features

### API Documentation

- **OpenAPI/Swagger** annotations for endpoints
- **Request/response examples**
- **Error code documentation**
- **Authentication requirements**

## 🔒 Security Guidelines

### Secure Coding Practices

- **Input validation** for all user inputs
- **SQL injection prevention** using parameterized queries
- **XSS prevention** in web interfaces
- **Secure secret management**

### Security Review

- **No hardcoded secrets** in code
- **Proper authentication** and authorization
- **Secure communication** (HTTPS/TLS)
- **Regular dependency updates**

## 🚀 Release Process

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Checklist

- [ ] All tests pass
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version bumped
- [ ] Security review completed
- [ ] Performance testing completed

## 🤝 Community Guidelines

### Code of Conduct

- **Be respectful** and inclusive
- **Provide constructive feedback**
- **Help others learn and grow**
- **Focus on the issue, not the person**

### Communication

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and ideas
- **Pull Requests**: Code contributions and reviews

## 📋 Pull Request Process

### Before Submitting

1. **Ensure tests pass** locally
2. **Update documentation** as needed
3. **Follow coding standards**
4. **Add appropriate tests**
5. **Update CHANGELOG.md**

### Review Process

1. **Automated checks** must pass
2. **Code review** by maintainers
3. **Testing** in development environment
4. **Documentation review**
5. **Final approval** and merge

### After Merge

1. **Monitor** for any issues
2. **Update** related documentation
3. **Communicate** changes to users
4. **Plan** follow-up improvements

## 🆘 Getting Help

- **Documentation**: Check README.md and API docs
- **Issues**: Search existing issues first
- **Discussions**: Ask questions in GitHub Discussions
- **Code Review**: Request feedback on draft PRs

## 🙏 Recognition

Contributors will be recognized in:
- **CONTRIBUTORS.md** file
- **Release notes** for significant contributions
- **GitHub contributors** page

Thank you for contributing to Omni-Agent Hub! 🚀
