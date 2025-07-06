"""
Pytest configuration and fixtures for Omni-Agent Hub tests.

This module provides shared fixtures and configuration for all tests.
"""

import asyncio
import pytest
import pytest_asyncio
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient
from httpx import AsyncClient

from omni_agent_hub.main import create_app
from omni_agent_hub.core.config import get_settings
from omni_agent_hub.services.database import DatabaseManager
from omni_agent_hub.services.redis_manager import RedisManager
from omni_agent_hub.services.vector_db import VectorDatabaseManager


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_settings():
    """Override settings for testing."""
    settings = get_settings()
    # Override with test values
    settings.debug = True
    settings.database_url = "sqlite+aiosqlite:///test.db"
    settings.redis_url = "redis://localhost:6379/15"  # Use test database
    settings.milvus_collection_name = "test_embeddings"
    return settings


@pytest.fixture
def app(test_settings):
    """Create FastAPI app for testing."""
    return create_app()


@pytest.fixture
def client(app) -> TestClient:
    """Create test client."""
    return TestClient(app)


@pytest_asyncio.fixture
async def async_client(app) -> AsyncGenerator[AsyncClient, None]:
    """Create async test client."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def mock_db_manager():
    """Mock database manager."""
    mock = AsyncMock(spec=DatabaseManager)
    mock.initialize = AsyncMock()
    mock.close = AsyncMock()
    mock.create_session = AsyncMock(return_value={
        "id": "test-id",
        "session_id": "test-session",
        "user_id": "test-user",
        "status": "active",
        "created_at": "2025-01-05T00:00:00Z"
    })
    mock.get_session = AsyncMock(return_value={
        "id": "test-id",
        "session_id": "test-session",
        "user_id": "test-user",
        "status": "active",
        "created_at": "2025-01-05T00:00:00Z"
    })
    mock.create_task = AsyncMock(return_value={
        "id": "test-task-id",
        "task_type": "chat",
        "status": "pending",
        "created_at": "2025-01-05T00:00:00Z"
    })
    return mock


@pytest.fixture
def mock_redis_manager():
    """Mock Redis manager."""
    mock = AsyncMock(spec=RedisManager)
    mock.initialize = AsyncMock()
    mock.close = AsyncMock()
    mock.set = AsyncMock(return_value=True)
    mock.get = AsyncMock(return_value=None)
    mock.delete = AsyncMock(return_value=1)
    mock.create_session = AsyncMock(return_value=True)
    mock.get_session = AsyncMock(return_value={
        "user_id": "test-user",
        "status": "active"
    })
    return mock


@pytest.fixture
def mock_vector_db_manager():
    """Mock vector database manager."""
    mock = AsyncMock(spec=VectorDatabaseManager)
    mock.initialize = AsyncMock()
    mock.close = AsyncMock()
    mock.insert_embeddings = AsyncMock(return_value=["test-id"])
    mock.search_similar = AsyncMock(return_value=[
        {
            "id": "test-id",
            "content": "test content",
            "similarity": 0.95,
            "metadata": {"source": "test"}
        }
    ])
    mock.get_collection_stats = AsyncMock(return_value={
        "collection_name": "test_embeddings",
        "total_entities": 100,
        "dimension": 1536
    })
    return mock


@pytest.fixture
def sample_agent_request():
    """Sample agent request data."""
    return {
        "message": "Hello, how can you help me?",
        "session_id": "test-session-123",
        "agent_type": "orchestrator",
        "context": {"user_preference": "detailed"},
        "max_iterations": 3
    }


@pytest.fixture
def sample_session_request():
    """Sample session request data."""
    return {
        "user_id": "test-user-123",
        "context": {"language": "en", "timezone": "UTC"},
        "preferences": {"response_style": "detailed"}
    }


@pytest.fixture
def sample_task_request():
    """Sample task request data."""
    return {
        "task_type": "code_generation",
        "session_id": "test-session-123",
        "parameters": {
            "language": "python",
            "description": "Create a function to calculate fibonacci numbers"
        },
        "priority": 5,
        "timeout_seconds": 300
    }


@pytest.fixture
def sample_embeddings_data():
    """Sample embeddings data for testing."""
    return [
        {
            "id": "test-1",
            "embedding": [0.1] * 1536,  # Mock embedding vector
            "content": "This is test content 1",
            "metadata": {"source": "test", "type": "document"},
            "timestamp": 1704067200,  # 2025-01-01
            "session_id": "test-session"
        },
        {
            "id": "test-2",
            "embedding": [0.2] * 1536,
            "content": "This is test content 2",
            "metadata": {"source": "test", "type": "conversation"},
            "timestamp": 1704067260,
            "session_id": "test-session"
        }
    ]


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client."""
    mock = MagicMock()
    
    # Mock chat completion
    mock.chat.completions.create.return_value = MagicMock(
        choices=[
            MagicMock(
                message=MagicMock(
                    content="This is a test response from the AI assistant."
                )
            )
        ]
    )
    
    # Mock embeddings
    mock.embeddings.create.return_value = MagicMock(
        data=[
            MagicMock(embedding=[0.1] * 1536)
        ]
    )
    
    return mock


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client."""
    mock = MagicMock()
    
    # Mock message creation
    mock.messages.create.return_value = MagicMock(
        content=[
            MagicMock(
                text="This is a test response from Claude."
            )
        ]
    )
    
    return mock


# Markers for different test types
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.slow = pytest.mark.slow


# Test utilities
class TestUtils:
    """Utility functions for tests."""
    
    @staticmethod
    def create_mock_response(status_code: int = 200, json_data: dict = None):
        """Create a mock HTTP response."""
        mock_response = MagicMock()
        mock_response.status_code = status_code
        mock_response.json.return_value = json_data or {}
        return mock_response
    
    @staticmethod
    def assert_response_structure(response_data: dict, expected_keys: list):
        """Assert that response has expected structure."""
        for key in expected_keys:
            assert key in response_data, f"Missing key: {key}"
    
    @staticmethod
    def assert_valid_uuid(uuid_string: str):
        """Assert that string is a valid UUID."""
        import uuid
        try:
            uuid.UUID(uuid_string)
        except ValueError:
            pytest.fail(f"Invalid UUID: {uuid_string}")


@pytest.fixture
def test_utils():
    """Provide test utilities."""
    return TestUtils
