from fastapi import APIRouter, HTTPException, Request, status
from app.models.correction import CorrectionRequest, CorrectionResponse
from app.services import prompt_builder
from app.services.ollama_metrics_service import appeler_ollama_et_sauvegarder
from app.services.rate_limiter import limiter
from app.config import settings

router = APIRouter(prefix="/api", tags=["Text Processing"])

@router.post("/corriger", response_model=CorrectionResponse)
@limiter.limit("10/minute")
async def corriger_texte(request: Request, payload: CorrectionRequest):
    """
    Pipeline interne : corrige le texte, puis le reformule automatiquement.
    Seul le résultat final reformulé est retourné au client.
    Les deux étapes sont mesurées et enregistrées séparément dans les métriques.
    """
    # Étape 1 : correction
    messages_correction = prompt_builder.build_correction_prompt(payload.texte)
    texte_corrigee = await appeler_ollama_et_sauvegarder(
        messages=messages_correction,
        temperature=settings.CORRECTION_TEMPERATURE,
        type_tache="correction",
        texte_entree=payload.texte
    )

    if not texte_corrigee.strip():
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, 
            detail="Le modèle a retourné une réponse vide à l'étape de correction."
        )
    """
    # Étape 2 : reformulation du texte déjà corrigé
    messages_reformulation = prompt_builder.build_reformulation_prompt(texte_corrige)
    texte_final = await appeler_ollama_et_sauvegarder(
        messages=messages_reformulation,
        temperature=settings.REFORMULATION_TEMPERATURE,
        type_tache="reformulation",
        texte_entree=texte_corrige
    )

    if not texte_final.strip():
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, 
            detail="Le modèle a retourné une réponse vide à l'étape de reformulation."
        )
    """
    return CorrectionResponse(
        # texte_corrige=texte_final,
        texte_corrige=texte_corrigee,
        modele_utilise=settings.OLLAMA_MODEL
    )