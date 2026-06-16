# app/services/ollama_client.py
import asyncio
import httpx
from fastapi import HTTPException, status
from app.config import settings

class OllamaClient:
    def __init__(self):
        self.client: httpx.AsyncClient | None = None
        self._lock = asyncio.Lock()

    async def start(self):
        """Initializes the httpx client with a lock to prevent race conditions."""
        async with self._lock:
            if self.client is None:
                self.client = httpx.AsyncClient(
                    base_url=settings.OLLAMA_BASE_URL,
                    timeout=httpx.Timeout(settings.OLLAMA_TIMEOUT_SECONDS)
                )

    async def stop(self):
        """Closes the httpx client."""
        if self.client:
            await self.client.aclose()
            self.client = None

    async def generate(self, messages: list[dict], temperature: float) -> str:
        """
        Sends a chat request to Ollama and returns the content of the response.
        """
        if self.client is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Ollama client is not initialized."
            )

        payload = {
            "model": settings.OLLAMA_MODEL,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }

        try:
            response = await self.client.post("/api/chat", json=payload)
            response.raise_for_status()
            
            data = response.json()
            content = data["message"]["content"].strip()
            return content

        except httpx.TimeoutException:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Le modèle LLM n'a pas répondu dans le délai imparti."
            )
        except httpx.ConnectError:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Impossible de joindre le service Ollama. Vérifiez qu'il est en cours d'exécution."
            )
        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Erreur reçue d'Ollama (Statut {e.response.status_code})."
            )
        except (KeyError, IndexError):
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Format de réponse inattendu reçu du modèle LLM."
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Erreur interne lors du traitement par le modèle: {str(e)}"
            )

# Singleton instance
ollama_client = OllamaClient()
