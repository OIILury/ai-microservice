"""
Test end-to-end (option 3) : le retrieval seul ne garantit pas une bonne réponse finale,
donc on teste ici la VRAIE chaîne complète : EmbeddingStore.search() -> build_rag_prompt()
-> Ollama/Mistral, pour vérifier si le system prompt compense les imperfections du retrieval
qu'on a mesurées dans calibrate_v3.py.

Prérequis : Ollama doit tourner localement avec le modèle Mistral déjà pull (ollama pull mistral).
À lancer depuis la racine du projet, ou ajuster MD_PATH / les imports selon ta structure
(cf. la même remarque que pour calibrate_v3.py : adapte l'import EmbeddingStore à ton
arbo réelle, ex. 'from app.services.embedding_store import EmbeddingStore').

Usage: python3 test_e2e_mistral.py
"""
import sys
import os
import time
import httpx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app.services.embedding_store import EmbeddingStore
from app.services.rag_engine import build_rag_prompt, post_traiter_reponse, MESSAGE_REFUS_FINAL

import app.services.rag_engine as rag_engine_module
print(f"[DEBUG] rag_engine chargé depuis : {rag_engine_module.__file__}")
print(f"[DEBUG] test_e2e_mistral exécuté depuis : {__file__}")

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "mistral"
KB_DIR = "knowledge_base"  # doit contenir RAGComGroupeFluidexpert.md

REPONSE_REJET = MESSAGE_REFUS_FINAL  # le test vérifie le message final, après post-traitement

# Mêmes questions que calibrate_v3.py, pour rester comparable aux distances déjà mesurées.
pertinentes = [
    "Depuis quand existe Fluidexpert ?",
    "Quel est l'effectif de l'entreprise ?",
    "Avez-vous travaillé pour le secteur de la pâtisserie ?",
    "Avez-vous un projet réalisé à Versailles ?",
    "Comment fonctionne votre système anticollision ?",
    "Le RGPD est-il respecté chez vous ?",
    "c'est qui votre boite sœur ?",
    "vous avez deja fait des machines a gateaux ?",
]

# Cas identifiés dans calibrate_v3.py comme les PIRES (distance basse malgré hors-sujet réel).
# C'est le test le plus dur et le plus important : si Mistral résiste ici, l'option 2+3 tient.
hors_sujet_difficiles = [
    "Comment réparer une fuite d'eau sous l'évier ?",      # 12.93 - pire cas mesuré
    "Aide-moi à réviser mon bac de philosophie.",            # 17.16
    "Quelle est la meilleure recette de ratatouille ?",      # 18.47
    "Donne-moi un programme de musculation pour débutant.",  # 20.62
    "Comment savoir si un avocat est mûr ?",                 # 20.77
]

hors_sujet_faciles = [
    "Quelle est la capitale du Canada ?",
    "Qui a remporté le dernier Tour de France ?",
]


def appeler_mistral(messages: list[dict]) -> str:
    """
    Appelle Ollama en mode non-streaming pour simplifier ce script de test.

    temperature=0 : rend les réponses déterministes (sampling glouton). Sans ça,
    deux runs avec le même code et le même corpus peuvent donner des résultats
    différents sur les cas limites (cf. "anticollision" qui a basculé OK/ECHEC
    entre deux runs identiques), ce qui rend toute comparaison entre versions
    du prompt peu fiable. En production, ce déterminisme est aussi souhaitable
    pour un assistant d'entreprise (réponses cohérentes à une même question).
    """
    response = httpx.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0},
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["message"]["content"].strip()


def tester_question(question: str, store: EmbeddingStore, attendu_rejet: bool):
    chunks = store.search(question, top_k=3)
    distances_info = f"{len(chunks)} chunk(s) retenu(s) (seuil={store.max_distance})"

    messages = build_rag_prompt(question, chunks)

    if messages is None:
        # Contexte vide : court-circuit, pas d'appel LLM (cf. docstring de build_rag_prompt).
        reponse_brute = "(contexte vide, LLM non appelé)"
        reponse_finale = MESSAGE_REFUS_FINAL
        duree = 0.0
    else:
        t0 = time.time()
        try:
            reponse_brute = appeler_mistral(messages)
        except Exception as e:
            print(f"  [ERREUR OLLAMA] {e}")
            return None
        duree = time.time() - t0
        reponse_finale = post_traiter_reponse(reponse_brute)

    a_rejete = reponse_finale == REPONSE_REJET

    if attendu_rejet:
        ok = a_rejete
        statut = "OK (rejeté)" if ok else "ECHEC (a répondu alors qu'il ne devait pas)"
    else:
        ok = not a_rejete
        statut = "OK (a répondu)" if ok else "ECHEC (rejeté alors qu'il devait répondre)"

    print(f"  [{statut}] ({duree:.1f}s, {distances_info})")
    print(f"    Q: {question}")
    print(f"    R (brute Mistral) : {reponse_brute}")
    if reponse_finale != reponse_brute:
        print(f"    R (après post-traitement) : {reponse_finale}")
    print()
    return ok


def run(label: str, questions: list[str], store: EmbeddingStore, attendu_rejet: bool):
    print(f"=== {label} ===")
    resultats = [tester_question(q, store, attendu_rejet) for q in questions]
    resultats = [r for r in resultats if r is not None]
    if resultats:
        taux = sum(resultats) / len(resultats) * 100
        print(f"  -- Taux de succès : {taux:.0f}% ({sum(resultats)}/{len(resultats)})")
    print()


if __name__ == "__main__":
    store = EmbeddingStore(KB_DIR, max_distance=30.0)
    store.build_index()
    print(f"Index chargé : {len(store.chunks)} chunks.\n")

    print("Warm-up Ollama (chargement du modèle en mémoire)...")
    try:
        appeler_mistral([{"role": "user", "content": "Bonjour"}])
        print("Modèle chargé.\n")
    except Exception as e:
        print(f"  [ATTENTION] Warm-up échoué : {e}\n")

    run("PERTINENTES (Mistral doit répondre)", pertinentes, store, attendu_rejet=False)
    run("HORS-SUJET DIFFICILES (Mistral doit rejeter malgré un contexte trompeur)",
        hors_sujet_difficiles, store, attendu_rejet=True)
    run("HORS-SUJET FACILES (cas de contrôle, sans ambiguïté)",
        hors_sujet_faciles, store, attendu_rejet=True)

    print("=> Si le taux de succès sur 'HORS-SUJET DIFFICILES' est élevé (>80%),")
    print("   le system prompt compense bien l'imperfection du retrieval : l'option 2+3 est validée.")
    print("=> Si Mistral répond quand même à partir d'un contexte trompeur, il faut soit")
    print("   durcir encore le prompt, soit revenir à un seuil plus bas pour ces cas précis.")