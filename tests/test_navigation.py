# tests/test_navigation.py
"""Tests for the /api/trouver-page endpoint with RAG."""
import pytest
from unittest.mock import patch
from fastapi import HTTPException
from httpx import AsyncClient
from app.services.embedding_store import embedding_store

@pytest.fixture
def mock_embedding_search():
    with patch.object(embedding_store, "search") as mock:
        mock.return_value = ["Ceci est un contexte de test."]
        yield mock

@pytest.mark.asyncio
async def test_navigation_rag_success_with_link(client: AsyncClient, mock_ollama_generate, mock_embedding_search):
    # Mock Ollama pour retourner une réponse avec un lien
    mock_ollama_generate.return_value = "Voici les tarifs.\nLIEN: /tarifs"
    
    response = await client.post("/api/trouver-page", json={"requete": "Quels sont les tarifs ?"})
    
    assert response.status_code == 200
    data = response.json()
    assert data["reponse"] == "Voici les tarifs."
    assert data["page"] == "/tarifs"
    assert data["page_trouvee"] is True
    mock_embedding_search.assert_called_once()

@pytest.mark.asyncio
async def test_navigation_rag_no_relevant_link(client: AsyncClient, mock_ollama_generate, mock_embedding_search):
    # Mock Ollama pour retourner une réponse sans lien
    mock_ollama_generate.return_value = "Je n'ai pas trouvé cette information dans ma base de connaissances."
    
    response = await client.post("/api/trouver-page", json={"requete": "Qui est le président ?"})
    
    assert response.status_code == 200
    data = response.json()
    assert data["reponse"] == "Je n'ai pas trouvé cette information dans ma base de connaissances."
    assert data["page"] is None
    assert data["page_trouvee"] is False

@pytest.mark.asyncio
async def test_navigation_rag_invalid_link_ignored(client: AsyncClient, mock_ollama_generate, mock_embedding_search):
    # Mock Ollama pour retourner un lien invalide
    mock_ollama_generate.return_value = "Voici une info.\nLIEN: /page-inexistante"
    
    response = await client.post("/api/trouver-page", json={"requete": "test"})
    
    assert response.status_code == 200
    data = response.json()
    assert data["reponse"] == "Voici une info."
    assert data["page"] is None
    assert data["page_trouvee"] is False

@pytest.mark.asyncio
async def test_navigation_empty_string_returns_422(client: AsyncClient):
    response = await client.post("/api/trouver-page", json={"requete": ""})
    assert response.status_code == 422

@pytest.mark.asyncio
async def test_navigation_ollama_timeout_returns_504(client: AsyncClient, mock_ollama_generate, mock_embedding_search):
    mock_ollama_generate.side_effect = HTTPException(status_code=504, detail="Timeout")
    response = await client.post("/api/trouver-page", json={"requete": "test"})
    assert response.status_code == 504
