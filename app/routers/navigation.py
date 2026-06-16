# app/routers/navigation.py
import logging
from fastapi import APIRouter
from app.models.navigation import NavigationRequest, NavigationResponse
from app.services import prompt_builder
from app.services.ollama_client import ollama_client
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Navigation"])

PAGES_VALIDES = {"/accueil", "/contact", "/tarifs", "/produits", "/faq", "/a-propos", "/inconnu"}

@router.post("/trouver-page", response_model=NavigationResponse)
async def trouver_page(request: NavigationRequest):
    """
    Endpoint pour mapper une requête utilisateur vers une URL de page.
    """
    messages = prompt_builder.build_navigation_prompt(request.requete)
    
    resultat_brut = await ollama_client.generate(
        messages=messages,
        temperature=settings.NAVIGATION_TEMPERATURE
    )
    
    # Sanitization: extraire uniquement la première ligne et nettoyer
    page = resultat_brut.split("\n")[0].strip()
    
    if page not in PAGES_VALIDES:
        logger.warning(
            "Réponse hors périmètre reçue d'Ollama. Longueur brute : %d caractères. Mapping vers /inconnu.",
            len(resultat_brut)
        )
        page = "/inconnu"
    
    return NavigationResponse(
        page=page,
        modele_utilise=settings.OLLAMA_MODEL,
        page_trouvee=(page != "/inconnu")
    )
