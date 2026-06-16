# app/main.py
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.config import settings
from app.services.ollama_client import ollama_client
from app.routers import correction, navigation

# Configuration du logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Démarrage : initialisation du client Ollama
    logger.info(f"Démarrage du microservice. Modèle configuré : {settings.OLLAMA_MODEL}")
    await ollama_client.start()
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

# Restriction aux hôtes du réseau local uniquement (privacy-first)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=[
        "localhost",
        "127.0.0.1",
        "*.local",
        "192.168.*",
        "10.*",
        "172.16.*",
        "172.17.*",
        "172.18.*",
        "172.19.*",
        "172.20.*",
        "172.21.*",
        "172.22.*",
        "172.23.*",
        "172.24.*",
        "172.25.*",
        "172.26.*",
        "172.27.*",
        "172.28.*",
        "172.29.*",
        "172.30.*",
        "172.31.*",
    ]
)

# Enregistrement des routers
app.include_router(correction.router)
app.include_router(navigation.router)

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
