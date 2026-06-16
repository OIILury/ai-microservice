# tests/test_navigation.py
"""Tests for the /api/trouver-page endpoint."""
import pytest
from fastapi import HTTPException
from httpx import AsyncClient

@pytest.mark.asyncio
@pytest.mark.parametrize("return_value_mock, expected_page", [
    ("/contact", "/contact"),
    ("/tarifs", "/tarifs"),
    ("/produits", "/produits"),
    ("/faq", "/faq"),
    ("/a-propos", "/a-propos"),
    ("/accueil", "/accueil"),
    ("/inconnu", "/inconnu"),
])
async def test_navigation_valid_pages(client: AsyncClient, mock_ollama_generate, return_value_mock, expected_page):
    mock_ollama_generate.return_value = return_value_mock
    response = await client.post("/api/trouver-page", json={"requete": "test"})
    assert response.status_code == 200
    assert response.json()["page"] == expected_page

@pytest.mark.asyncio
async def test_navigation_page_trouvee_true(client: AsyncClient, mock_ollama_generate):
    mock_ollama_generate.return_value = "/contact"
    response = await client.post("/api/trouver-page", json={"requete": "test"})
    assert response.json()["page_trouvee"] is True

@pytest.mark.asyncio
async def test_navigation_page_trouvee_false(client: AsyncClient, mock_ollama_generate):
    mock_ollama_generate.return_value = "/inconnu"
    response = await client.post("/api/trouver-page", json={"requete": "test"})
    assert response.json()["page_trouvee"] is False

# Sanitization cases
@pytest.mark.asyncio
async def test_navigation_llm_returns_url_with_extra_text(client: AsyncClient, mock_ollama_generate):
    mock_ollama_generate.return_value = "/contact voici la page"
    response = await client.post("/api/trouver-page", json={"requete": "test"})
    # It takes the first line and then checks if it's in PAGES_VALIDES. 
    # "/contact voici la page" is not in PAGES_VALIDES.
    assert response.json()["page"] == "/inconnu"

@pytest.mark.asyncio
async def test_navigation_llm_returns_multiline(client: AsyncClient, mock_ollama_generate):
    mock_ollama_generate.return_value = "/tarifs\nVoici les tarifs"
    response = await client.post("/api/trouver-page", json={"requete": "test"})
    assert response.json()["page"] == "/tarifs"

@pytest.mark.asyncio
async def test_navigation_llm_returns_unknown_url(client: AsyncClient, mock_ollama_generate):
    mock_ollama_generate.return_value = "/une-page-inexistante"
    response = await client.post("/api/trouver-page", json={"requete": "test"})
    assert response.json()["page"] == "/inconnu"

@pytest.mark.asyncio
async def test_navigation_llm_returns_empty(client: AsyncClient, mock_ollama_generate):
    mock_ollama_generate.return_value = ""
    response = await client.post("/api/trouver-page", json={"requete": "test"})
    assert response.json()["page"] == "/inconnu"

# Pydantic validation cases
@pytest.mark.asyncio
async def test_navigation_empty_string_returns_422(client: AsyncClient):
    response = await client.post("/api/trouver-page", json={"requete": ""})
    assert response.status_code == 422

@pytest.mark.asyncio
async def test_navigation_whitespace_only_returns_422(client: AsyncClient):
    response = await client.post("/api/trouver-page", json={"requete": "  "})
    assert response.status_code == 422

@pytest.mark.asyncio
async def test_navigation_missing_field_returns_422(client: AsyncClient):
    response = await client.post("/api/trouver-page", json={})
    assert response.status_code == 422

# Ollama error cases
@pytest.mark.asyncio
async def test_navigation_ollama_timeout_returns_504(client: AsyncClient, mock_ollama_generate):
    mock_ollama_generate.side_effect = HTTPException(status_code=504, detail="Timeout")
    response = await client.post("/api/trouver-page", json={"requete": "test"})
    assert response.status_code == 504

@pytest.mark.asyncio
async def test_navigation_ollama_unavailable_returns_503(client: AsyncClient, mock_ollama_generate):
    mock_ollama_generate.side_effect = HTTPException(status_code=503, detail="Unavailable")
    response = await client.post("/api/trouver-page", json={"requete": "test"})
    assert response.status_code == 503
