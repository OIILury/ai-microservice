# app/routers/correction.py
from fastapi import APIRouter, HTTPException, status
from app.models.correction import CorrectionRequest, CorrectionResponse
from app.models.reformulation import ReformulationRequest, ReformulationResponse
from app.services import prompt_builder
from app.services.ollama_client import ollama_client
from app.config import settings

router = APIRouter(prefix="/api", tags=["Text Processing"])

@router.post("/corriger", response_model=CorrectionResponse)
async def corriger_texte(request: CorrectionRequest):
    """
    Endpoint pour corriger l'orthographe et la grammaire d'un texte.
    """
    messages = prompt_builder.build_correction_prompt(request.texte)
    
    resultat = await ollama_client.appeler_ollama_et_sauvegarder(
        modele=settings.OLLAMA_MODEL,
        messages=messages,
        texte_entree=request.texte,
        endpoint_nom="correction"
    )
    
    if not resultat.strip():
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Le modèle a retourné une réponse vide."
        )
    
    return CorrectionResponse(
        texte_corrige=resultat,
        modele_utilise=settings.OLLAMA_MODEL
    )

@router.post("/reformuler", response_model=ReformulationResponse)
async def reformuler_texte(request: ReformulationRequest):
    """
    Endpoint pour reformuler un texte de manière professionnelle.
    """
    messages = prompt_builder.build_reformulation_prompt(request.texte)
    
    resultat = await ollama_client.appeler_ollama_et_sauvegarder(
        modele=settings.OLLAMA_MODEL,
        messages=messages,
        texte_entree=request.texte,
        endpoint_nom="reformulation"
    )
    
    if not resultat.strip():
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Le modèle a retourné une réponse vide."
        )
    
    return ReformulationResponse(
        texte_reformule=resultat,
        modele_utilise=settings.OLLAMA_MODEL
    )
