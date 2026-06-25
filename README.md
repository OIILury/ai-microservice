# AI Microservice

Ce microservice Python (FastAPI) expose des capacités d'Intelligence Artificielle **entièrement locales** via [Ollama](https://ollama.com/). Il est conçu pour être déployé sur des systèmes embarqués comme le Jetson Orin NX, garantissant une confidentialité totale (privacy-first).

Le projet couvre deux usages complémentaires :

1. **Correction et reformulation de texte** — accès technique pour corriger et reformuler des contenus rédigés par des utilisateurs.
2. **Chatbot d'entreprise (RAG)** — assistant conversationnel alimenté par une base de connaissances locale, déployé par exemple sur les **plaquettes de présentation** de l'entreprise **Fluidexpert** pour répondre aux visiteurs sur l'activité, les produits et les coordonnées du groupe.

## Fonctionnalités

### Correction et reformulation (accès technique)
- **Correction orthographique et grammaticale** stricte, suivie d'une **reformulation** pour améliorer la clarté du texte.
- Endpoint : `POST /api/corriger`
- Les deux étapes sont exécutées en pipeline ; seul le texte final reformulé est retourné au client.

### Chatbot d'entreprise (RAG)
- **Réponses conversationnelles** basées uniquement sur la base de connaissances (`knowledge_base/`), sans envoi de données vers le cloud.
- **Recherche sémantique** (embeddings locaux) pour retrouver les passages pertinents avant génération par le LLM.
- **Navigation contextuelle** : proposition d'un lien vers une page web lorsque c'est pertinent (site Fluidexpert, portail contact, Option Automatismes, etc.).
- Endpoint : `POST /api/trouver-page`
- Interface web de démonstration accessible à la racine : `http://localhost:8000/`

### Monitoring
- Endpoint `/health` pour vérifier l'état du service.
- Métriques d'utilisation via `/api/metrics/*`.

## Prérequis
- Python 3.10+
- [Ollama](https://ollama.com/) installé et en cours d'exécution.
- Modèle Llama 3 téléchargé :
  ```bash
  ollama pull llama3:8b
  ```

## Installation

1. Cloner le répertoire.
2. Installer les dépendances :
   ```bash
   pip install -r requirements.txt
   ```
3. Configurer l'environnement :
   ```bash
   cp .env.example .env
   # Modifier .env si nécessaire (ex: OLLAMA_BASE_URL)
   ```

## Lancement
```bash
python -m app.main
```
Le service sera accessible sur `http://0.0.0.0:8000`.

Au démarrage, l'index RAG est construit automatiquement à partir des fichiers de `knowledge_base/`.

## Utilisation (Exemples curl)

### Health Check
```bash
curl http://localhost:8000/health
```

### Correction et reformulation de texte
```bash
curl -X POST http://localhost:8000/api/corriger \
     -H "Content-Type: application/json" \
     -d '{"texte": "Je mangeais une pomme hier soir, cétait tres bon."}'
```

### Chatbot (question sur l'entreprise)
```bash
curl -X POST http://localhost:8000/api/trouver-page \
     -H "Content-Type: application/json" \
     -d '{"requete": "Depuis combien de temps Fluidexpert existe-t-elle ?"}'
```

## Architecture
- **FastAPI** : framework web asynchrone performant.
- **Pydantic** : validation stricte des données d'entrée/sortie.
- **RAG** : index d'embeddings locaux + prompts contraints pour limiter les hallucinations.
- **Httpx** : client HTTP asynchrone pour la communication avec Ollama.
- **Ollama** : moteur d'inférence LLM local (aucune donnée ne quitte la machine).
