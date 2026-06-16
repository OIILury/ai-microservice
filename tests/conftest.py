# tests/conftest.py
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch
from httpx import AsyncClient, ASGITransport

from app.main import app
from app.services.ollama_client import ollama_client

pytest_plugins = ('pytest_asyncio',)

@pytest.fixture(scope="function")
def mock_ollama_generate():
    """
    Fixture to patch the OllamaClient.generate method with an AsyncMock.
    """
    with patch.object(ollama_client, "generate", new_callable=AsyncMock) as mock:
        yield mock

@pytest_asyncio.fixture(scope="function")
async def client():
    """
    Fixture to provide an AsyncClient for testing the FastAPI application.
    Manages the lifespan of the app.
    """
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver"
    ) as ac:
        yield ac
