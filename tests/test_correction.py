# tests/test_correction.py
"""Tests for the /api/corriger endpoint."""
import pytest
from fastapi import HTTPException
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_correction_returns_200(client: AsyncClient, mock_ollama_generate):
    mock_ollama_generate.return_value = "Texte corrigé."
    response = await client.post("/api/corriger", json={"texte": "Texte a corriger."})
    assert response.status_code == 200

@pytest.mark.asyncio
async def test_correction_response_contains_corrected_text(client: AsyncClient, mock_ollama_generate):
    mock_ollama_generate.return_value = "Texte corrigé."
    response = await client.post("/api/corriger", json={"texte": "Texte a corriger."})
    data = response.json()
    assert data["texte_corrige"] == "Texte corrigé."

@pytest.mark.asyncio
async def test_correction_response_contains_model_name(client: AsyncClient, mock_ollama_generate):
    mock_ollama_generate.return_value = "Texte corrigé."
    response = await client.post("/api/corriger", json={"texte": "Texte a corriger."})
    data = response.json()
    assert "modele_utilise" in data
    assert isinstance(data["modele_utilise"], str)
    assert len(data["modele_utilise"]) > 0

@pytest.mark.asyncio
async def test_correction_calls_ollama_once(client: AsyncClient, mock_ollama_generate):
    mock_ollama_generate.return_value = "Texte corrigé."
    await client.post("/api/corriger", json={"texte": "Texte a corriger."})
    mock_ollama_generate.assert_called_once()

# Pydantic validation cases
@pytest.mark.asyncio
async def test_correction_empty_string_returns_422(client: AsyncClient):
    response = await client.post("/api/corriger", json={"texte": ""})
    assert response.status_code == 422

@pytest.mark.asyncio
async def test_correction_whitespace_only_returns_422(client: AsyncClient):
    response = await client.post("/api/corriger", json={"texte": "   "})
    assert response.status_code == 422

@pytest.mark.asyncio
async def test_correction_missing_field_returns_422(client: AsyncClient):
    response = await client.post("/api/corriger", json={})
    assert response.status_code == 422

@pytest.mark.asyncio
async def test_correction_text_too_long_returns_422(client: AsyncClient):
    response = await client.post("/api/corriger", json={"texte": "a" * 10001})
    assert response.status_code == 422

# Ollama error cases
@pytest.mark.asyncio
async def test_correction_ollama_timeout_returns_504(client: AsyncClient, mock_ollama_generate):
    mock_ollama_generate.side_effect = HTTPException(status_code=504, detail="Timeout")
    response = await client.post("/api/corriger", json={"texte": "test"})
    assert response.status_code == 504

@pytest.mark.asyncio
async def test_correction_ollama_unavailable_returns_503(client: AsyncClient, mock_ollama_generate):
    mock_ollama_generate.side_effect = HTTPException(status_code=503, detail="Unavailable")
    response = await client.post("/api/corriger", json={"texte": "test"})
    assert response.status_code == 503

@pytest.mark.asyncio
async def test_correction_empty_llm_response_returns_502(client: AsyncClient, mock_ollama_generate):
    mock_ollama_generate.return_value = ""
    response = await client.post("/api/corriger", json={"texte": "test"})
    assert response.status_code == 502

@pytest.mark.asyncio
async def test_correction_whitespace_llm_response_returns_502(client: AsyncClient, mock_ollama_generate):
    mock_ollama_generate.return_value = "   "
    response = await client.post("/api/corriger", json={"texte": "test"})
    assert response.status_code == 502
