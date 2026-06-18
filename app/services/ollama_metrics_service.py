import time
import asyncio
from datetime import datetime
from fastapi import HTTPException
from app.services.ollama_client import ollama_client
from app.services.metrics_store import metrics_store
from app.services import text_metrics
from app.config import settings

async def appeler_ollama_et_sauvegarder(
    messages: list[dict],
    temperature: float,
    type_tache: str,
    texte_entree: str
) -> str:
    """
    Appelle Ollama en streaming, calcule toutes les métriques, les sauvegarde en base
    de manière non-bloquante, et retourne le texte généré.
    """
    timestamp_debut = time.time()
    timestamp_iso = datetime.now().isoformat()
    
    try:
        # Appel au client en streaming
        result = await ollama_client.generate_with_metrics(messages, temperature)
        texte_genere = result["texte"]
        
        # Calcul des métriques
        temps_reponse_ms = (time.time() - timestamp_debut) * 1000
        
        eval_count = result["eval_count"]
        eval_duration_ns = result["eval_duration_ns"]
        tokens_par_seconde = 0.0
        if eval_duration_ns > 0:
            tokens_par_seconde = (eval_count / (eval_duration_ns / 1e9))
            
        # Métriques de texte
        nb_mots_entree = text_metrics.compter_mots(texte_entree)
        nb_mots_sortie = text_metrics.compter_mots(texte_genere)
        distance_levenshtein = text_metrics.calculer_distance_levenshtein(texte_entree, texte_genere)
        ratio_taille = text_metrics.calculer_ratio_taille(texte_entree, texte_genere)
        
        # Sauvegarde non-bloquante
        metriques = {
            "timestamp": timestamp_iso,
            "modele": settings.OLLAMA_MODEL,
            "type_tache": type_tache,
            "texte_entree": texte_entree[:500] + "..." if len(texte_entree) > 500 else texte_entree,
            "nb_caracteres_entree": len(texte_entree),
            "nb_mots_entree": nb_mots_entree,
            "texte_sortie": texte_genere[:500] + "..." if len(texte_genere) > 500 else texte_genere,
            "nb_caracteres_sortie": len(texte_genere),
            "nb_mots_sortie": nb_mots_sortie,
            "temps_reponse_ms": temps_reponse_ms,
            "time_to_first_token_ms": result["ttft_ms"],
            "tokens_entree": result["prompt_eval_count"],
            "tokens_sortie": eval_count,
            "tokens_par_seconde": tokens_par_seconde,
            "ratio_taille_sortie_entree": ratio_taille,
            "distance_levenshtein": distance_levenshtein,
            "statut": "succes"
        }
        
        await asyncio.to_thread(metrics_store.enregistrer_metrique, **metriques)
        
        return texte_genere

    except HTTPException as e:
        # Enregistrement de l'erreur
        metriques_erreur = {
            "timestamp": timestamp_iso,
            "modele": settings.OLLAMA_MODEL,
            "type_tache": type_tache,
            "texte_entree": texte_entree[:500] + "..." if len(texte_entree) > 500 else texte_entree,
            "nb_caracteres_entree": len(texte_entree),
            "statut": "erreur",
            "notes": str(e.detail)
        }
        await asyncio.to_thread(metrics_store.enregistrer_metrique, **metriques_erreur)
        raise e
    except Exception as e:
        metriques_erreur = {
            "timestamp": timestamp_iso,
            "modele": settings.OLLAMA_MODEL,
            "type_tache": type_tache,
            "texte_entree": texte_entree[:500] + "..." if len(texte_entree) > 500 else texte_entree,
            "nb_caracteres_entree": len(texte_entree),
            "statut": "erreur",
            "notes": str(e)
        }
        await asyncio.to_thread(metrics_store.enregistrer_metrique, **metriques_erreur)
        raise e
