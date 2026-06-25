# app/routers/navigation.py
import logging
from fastapi import APIRouter
from app.models.navigation import NavigationRequest, NavigationResponse
from app.services.rag_engine import build_rag_prompt, post_traiter_reponse, MESSAGE_REFUS_FINAL
from app.services.embedding_store import embedding_store
from app.services.ollama_client import ollama_client
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Navigation"])

PAGES_VALIDES = {
    "https://portal.fluidexpert.com/contact",
    "https://fluidexpert.fr/",
    "https://www.option-automatismes.fr/",
}

@router.post("/trouver-page", response_model=NavigationResponse)
async def trouver_page(request: NavigationRequest):
    """
    Endpoint conversationnel avec RAG pour répondre aux questions et proposer une page.
    """
    # 1. Recherche des chunks pertinents
    chunks = embedding_store.search(request.requete, top_k=3)

    # 2. Construction du prompt RAG
    messages = build_rag_prompt(request.requete, chunks)

    # 2bis. Contexte vide : court-circuit déterministe, pas d'appel au LLM.
    # (cf. rag_engine.build_rag_prompt : un contexte vide laisse au LLM l'opportunité
    # de répondre avec ses connaissances générales au lieu de respecter la consigne de refus.)
    if messages is None:
        return NavigationResponse(
            reponse=MESSAGE_REFUS_FINAL,
            page=None,
            modele_utilise=settings.OLLAMA_MODEL,
            page_trouvee=False
        )

    # 3. Appel au LLM
    resultat_brut = await ollama_client.generate(
        messages=messages,
        temperature=settings.NAVIGATION_TEMPERATURE
    )

    # 3bis. Substitution déterministe si le LLM a refusé (cf. rag_engine._est_un_refus).
    # Appliquée systématiquement avant le parsing, qu'il y ait digression ou non.
    resultat_traite = post_traiter_reponse(resultat_brut)

    # 3ter. Si la réponse a été remplacée par le message de refus fixe, on s'arrête là :
    # ce message ne contient jamais de ligne LIEN, le parsing est inutile et risquerait
    # de mal interpréter le contenu fixe.
    if resultat_traite == MESSAGE_REFUS_FINAL:
        return NavigationResponse(
            reponse=MESSAGE_REFUS_FINAL,
            page=None,
            modele_utilise=settings.OLLAMA_MODEL,
            page_trouvee=False
        )

    # 4. Parsing de la réponse
    lignes = resultat_traite.strip().split("\n")
    reponse_nettoyee = []
    url_extraite = None

    for ligne in lignes:
        ligne_normalisee = ligne.strip().upper().replace(" ", "")
        if ligne_normalisee.startswith("LIEN:"):
            url = ligne.split(":", 1)[1].strip() if ":" in ligne else ""
            if url in PAGES_VALIDES:
                url_extraite = url
            else:
                logger.warning("Ligne LIEN détectée mais URL extraite non valide. Ignorée.")
        else:
            reponse_nettoyee.append(ligne)

    texte_final = "\n".join(reponse_nettoyee).strip()

    return NavigationResponse(
        reponse=texte_final,
        page=url_extraite,
        modele_utilise=settings.OLLAMA_MODEL,
        page_trouvee=(url_extraite is not None)
    )