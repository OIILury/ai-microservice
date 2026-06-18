"""Tests for the /api/reformuler endpoint."""
import pytest
from fastapi import HTTPException
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_reformulation_returns_200(client: AsyncClient, mock_ollama_sauvegarder):
    mock_ollama_sauvegarder.return_value = "Texte reformulé."
    response = await client.post("/api/reformuler", json={"texte": "Texte à reformuler."})
    assert response.status_code == 200

@pytest.mark.asyncio
async def test_reformulation_response_contains_reformulated_text(client: AsyncClient, mock_ollama_sauvegarder):
    mock_ollama_sauvegarder.return_value = "Texte reformulé."
    response = await client.post("/api/reformuler", json={"texte": "Texte à reformuler."})
    data = response.json()
    assert data["texte_reformule"] == "Texte reformulé."

@pytest.mark.asyncio
async def test_reformulation_response_contains_model_name(client: AsyncClient, mock_ollama_sauvegarder):
    mock_ollama_sauvegarder.return_value = "Texte reformulé."
    response = await client.post("/api/reformuler", json={"texte": "Texte à reformuler."})
    data = response.json()
    assert "modele_utilise" in data
    assert isinstance(data["modele_utilise"], str)

@pytest.mark.asyncio
async def test_reformulation_calls_ollama_once(client: AsyncClient, mock_ollama_sauvegarder):
    mock_ollama_sauvegarder.return_value = "Texte reformulé."
    await client.post("/api/reformuler", json={"texte": "Texte à reformuler."})
    mock_ollama_sauvegarder.assert_called_once()

# Pydantic validation cases
@pytest.mark.asyncio
async def test_reformulation_empty_string_returns_422(client: AsyncClient):
    response = await client.post("/api/reformuler", json={"texte": ""})
    assert response.status_code == 422

# Ollama error cases
@pytest.mark.asyncio
async def test_reformulation_ollama_timeout_returns_504(client: AsyncClient, mock_ollama_sauvegarder):
    mock_ollama_sauvegarder.side_effect = HTTPException(status_code=504, detail="Timeout")
    response = await client.post("/api/reformuler", json={"texte": "test"})
    assert response.status_code == 504
