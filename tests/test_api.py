"""
Tests for API endpoints.

This module contains tests for all API endpoints in the Omni-Agent Hub.
"""

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient


class TestHealthEndpoints:
    """Test health check endpoints."""
    
    def test_health_check(self, client: TestClient):
        """Test basic health check endpoint."""
        response = client.get("/api/v1/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "version" in data
        assert "timestamp" in data
        assert data["status"] == "healthy"
    
    def test_detailed_health_check(self, client: TestClient):
        """Test detailed health check endpoint."""
        response = client.get("/api/v1/health/detailed")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "version" in data
        assert "services" in data
        assert "metrics" in data
        
        # Check services structure
        services = data["services"]
        expected_services = ["database", "redis", "vector_db", "kafka"]
        for service in expected_services:
            assert service in services
        
        # Check metrics structure
        metrics = data["metrics"]
        expected_metrics = ["active_sessions", "pending_tasks", "active_agents"]
        for metric in expected_metrics:
            assert metric in metrics


class TestSessionEndpoints:
    """Test session management endpoints."""
    
    def test_create_session(self, client: TestClient, sample_session_request):
        """Test session creation."""
        response = client.post("/api/v1/sessions", json=sample_session_request)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "session_id" in data
        assert "user_id" in data
        assert "status" in data
        assert "created_at" in data
        assert data["user_id"] == sample_session_request["user_id"]
        assert data["status"] == "active"
    
    def test_get_session(self, client: TestClient):
        """Test session retrieval."""
        session_id = "test-session-123"
        response = client.get(f"/api/v1/sessions/{session_id}")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "session_id" in data
        assert "user_id" in data
        assert "status" in data
        assert "created_at" in data
        assert data["session_id"] == session_id
    
    def test_delete_session(self, client: TestClient):
        """Test session deletion."""
        session_id = "test-session-123"
        response = client.delete(f"/api/v1/sessions/{session_id}")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "message" in data
        assert "deleted" in data["message"].lower()


class TestAgentEndpoints:
    """Test agent interaction endpoints."""
    
    def test_chat_with_agent(self, client: TestClient, sample_agent_request):
        """Test agent chat endpoint."""
        response = client.post("/api/v1/agents/chat", json=sample_agent_request)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "response" in data
        assert "session_id" in data
        assert "agent_type" in data
        assert "confidence" in data
        assert "metadata" in data
        
        assert data["session_id"] == sample_agent_request["session_id"]
        assert data["agent_type"] == "orchestrator"
        assert isinstance(data["confidence"], float)
        assert 0 <= data["confidence"] <= 1
    
    def test_execute_task(self, client: TestClient, sample_task_request):
        """Test task execution endpoint."""
        response = client.post("/api/v1/agents/task", json=sample_task_request)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "task_id" in data
        assert "status" in data
        assert "result" in data
        assert "session_id" in data
        assert "created_at" in data
        
        assert data["session_id"] == sample_task_request["session_id"]
        assert data["status"] in ["pending", "running", "completed", "failed"]
    
    def test_chat_validation_error(self, client: TestClient):
        """Test chat endpoint with invalid data."""
        invalid_request = {
            "message": "",  # Empty message should fail validation
            "session_id": "test-session"
        }
        
        response = client.post("/api/v1/agents/chat", json=invalid_request)
        assert response.status_code == 422  # Validation error


class TestWorkflowEndpoints:
    """Test workflow management endpoints."""
    
    def test_start_workflow(self, client: TestClient):
        """Test workflow start endpoint."""
        workflow_data = {
            "workflow_name": "test_workflow",
            "session_id": "test-session-123",
            "parameters": {"param1": "value1"}
        }
        
        response = client.post(
            "/api/v1/workflows/start",
            params=workflow_data
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "workflow_id" in data
        assert "status" in data
        assert "steps" in data
        assert data["status"] == "running"
        assert isinstance(data["steps"], list)
    
    def test_get_workflow_status(self, client: TestClient):
        """Test workflow status endpoint."""
        workflow_id = "test-workflow-123"
        response = client.get(f"/api/v1/workflows/{workflow_id}")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "workflow_id" in data
        assert "status" in data
        assert "current_step" in data
        assert "progress" in data
        assert "steps" in data
        
        assert data["workflow_id"] == workflow_id
        assert isinstance(data["progress"], float)
        assert 0 <= data["progress"] <= 1


class TestToolEndpoints:
    """Test tool integration endpoints."""
    
    def test_list_tools(self, client: TestClient):
        """Test tools listing endpoint."""
        response = client.get("/api/v1/tools")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "tools" in data
        assert isinstance(data["tools"], list)
        
        if data["tools"]:  # If tools exist, check structure
            tool = data["tools"][0]
            assert "name" in tool
            assert "description" in tool
            assert "parameters" in tool
            assert "category" in tool
    
    def test_execute_tool(self, client: TestClient):
        """Test tool execution endpoint."""
        tool_name = "web_search"
        parameters = {"query": "test query", "max_results": 5}
        
        response = client.post(
            f"/api/v1/tools/{tool_name}/execute",
            json=parameters,
            params={"session_id": "test-session"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "tool_name" in data
        assert "status" in data
        assert "result" in data
        assert "execution_time_ms" in data
        
        assert data["tool_name"] == tool_name
        assert data["status"] == "success"


class TestKnowledgeEndpoints:
    """Test knowledge and RAG endpoints."""
    
    def test_search_knowledge(self, client: TestClient):
        """Test knowledge search endpoint."""
        search_data = {
            "query": "test search query",
            "session_id": "test-session",
            "max_results": 5
        }
        
        response = client.post("/api/v1/knowledge/search", json=search_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "query" in data
        assert "results" in data
        assert "total_results" in data
        
        assert data["query"] == search_data["query"]
        assert isinstance(data["results"], list)
        assert isinstance(data["total_results"], int)
    
    def test_add_knowledge(self, client: TestClient):
        """Test knowledge addition endpoint."""
        knowledge_data = {
            "content": "This is test knowledge content",
            "metadata": {"source": "test", "type": "document"},
            "session_id": "test-session"
        }
        
        response = client.post("/api/v1/knowledge/add", json=knowledge_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "document_id" in data
        assert "embedding_id" in data
        
        assert data["status"] == "success"


class TestMetricsEndpoints:
    """Test metrics and monitoring endpoints."""
    
    def test_get_metrics(self, client: TestClient):
        """Test metrics endpoint."""
        response = client.get("/api/v1/metrics")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "system" in data
        assert "agents" in data
        assert "performance" in data
        
        # Check system metrics
        system = data["system"]
        assert "uptime_seconds" in system
        assert "memory_usage_mb" in system
        assert "cpu_usage_percent" in system
        
        # Check agent metrics
        agents = data["agents"]
        assert "total_sessions" in agents
        assert "active_sessions" in agents
        assert "completed_tasks" in agents
        assert "failed_tasks" in agents
        
        # Check performance metrics
        performance = data["performance"]
        assert "avg_response_time_ms" in performance
        assert "requests_per_minute" in performance
        assert "success_rate" in performance


@pytest.mark.asyncio
class TestAsyncEndpoints:
    """Test endpoints using async client."""
    
    async def test_async_health_check(self, async_client: AsyncClient):
        """Test health check with async client."""
        response = await async_client.get("/api/v1/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    async def test_async_chat(self, async_client: AsyncClient, sample_agent_request):
        """Test chat endpoint with async client."""
        response = await async_client.post(
            "/api/v1/agents/chat",
            json=sample_agent_request
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert "session_id" in data


@pytest.mark.integration
class TestEndpointIntegration:
    """Integration tests for endpoint workflows."""
    
    def test_full_session_workflow(self, client: TestClient, sample_session_request):
        """Test complete session workflow."""
        # 1. Create session
        response = client.post("/api/v1/sessions", json=sample_session_request)
        assert response.status_code == 200
        session_data = response.json()
        session_id = session_data["session_id"]
        
        # 2. Chat with agent
        chat_request = {
            "message": "Hello, test message",
            "session_id": session_id
        }
        response = client.post("/api/v1/agents/chat", json=chat_request)
        assert response.status_code == 200
        
        # 3. Get session details
        response = client.get(f"/api/v1/sessions/{session_id}")
        assert response.status_code == 200
        
        # 4. Delete session
        response = client.delete(f"/api/v1/sessions/{session_id}")
        assert response.status_code == 200
