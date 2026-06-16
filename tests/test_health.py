# tests/test_health.py
"""Tests for the /health endpoint."""
import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_health_returns_200(client: AsyncClient):
    response = await client.get("/health")
    assert response.status_code == 200

@pytest.mark.asyncio
async def test_health_response_structure(client: AsyncClient):
    response = await client.get("/health")
    data = response.json()
    assert set(data.keys()) == {"status", "model"}

@pytest.mark.asyncio
async def test_health_status_value(client: AsyncClient):
    response = await client.get("/health")
    data = response.json()
    assert data["status"] == "ok"

@pytest.mark.asyncio
async def test_health_does_not_call_ollama(client: AsyncClient, mock_ollama_generate):
    await client.get("/health")
    mock_ollama_generate.assert_not_called()
