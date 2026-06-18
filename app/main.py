import logging
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.config import settings
from app.services.ollama_client import ollama_client
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
import ipaddress
import time

from app.services.embedding_store import embedding_store
from app.services.metrics_store import metrics_store
from app.routers import correction, navigation, metrics

# Configuration du logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Démarrage : initialisation de la base de données, du client Ollama et de l'index RAG
    await asyncio.to_thread(metrics_store.init_db)
    logger.info("Base de métriques SQLite initialisée.")
    
    logger.info(f"Démarrage du microservice. Modèle configuré : {settings.OLLAMA_MODEL}")
    await ollama_client.start()
    
    start_time = time.time()
    await asyncio.to_thread(embedding_store.build_index)
    duration = time.time() - start_time
    logger.info(f"Index RAG initialisé en {duration:.2f}s ({len(embedding_store.chunks)} chunks).")
    
    yield
    # Arrêt : fermeture du client Ollama
    logger.info("Arrêt du microservice.")
    await ollama_client.stop()

app = FastAPI(
    title="AI Microservice",
    version="1.0.0",
    description="Microservice de production pour la correction de texte et le routage de navigation via Ollama local.",
    lifespan=lifespan
)

class LocalNetworkMiddleware(BaseHTTPMiddleware):
    ALLOWED_NETWORKS = [
        ipaddress.ip_network("127.0.0.0/8"),
        ipaddress.ip_network("192.168.0.0/16"),
        ipaddress.ip_network("10.0.0.0/8"),
        ipaddress.ip_network("172.16.0.0/12"),
    ]

    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host if request.client else None
        if client_ip:
            try:
                ip = ipaddress.ip_address(client_ip)
                if not any(ip in net for net in self.ALLOWED_NETWORKS):
                    return Response("Accès refusé : réseau non autorisé.", status_code=403)
            except ValueError:
                return Response("Accès refusé : IP invalide.", status_code=403)
        return await call_next(request)

app.add_middleware(LocalNetworkMiddleware)

# Restriction aux hôtes du réseau local uniquement (privacy-first)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=[
        "localhost",
        "127.0.0.1",
        "testserver",      # requis pour les tests httpx
        "*.local",
        "192.168.82.169",
        "192.168.82.169:8000",
        "192.168.82.117",
        "192.168.82.117:8000",
    ]
)

# Enregistrement des routers
app.include_router(correction.router)
app.include_router(navigation.router)
app.include_router(metrics.router)

# Service des fichiers statiques
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/", include_in_schema=False)
async def serve_frontend():
    return FileResponse("app/static/index.html")

@app.get("/health", tags=["Health"])
async def health_check():
    """
    Vérifie l'état de santé du service sans appeler le LLM.
    """
    return {
        "status": "ok",
        "model": settings.OLLAMA_MODEL
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=True if settings.LOG_LEVEL.lower() == "debug" else False
    )
