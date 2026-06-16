# AI Microservice

Ce microservice Python (FastAPI) expose des capacités d'Intelligence Artificielle localement via Ollama. Il est conçu pour être déployé sur des systèmes embarqués comme le Jetson Orin NX, garantissant une confidentialité totale (privacy-first).

## Fonctionnalités
- **Correction de texte** : Correction orthographique et grammaticale stricte.
- **Navigation** : Routage intelligent des requêtes utilisateur vers des URLs prédéfinies.
- **Santé** : Endpoint `/health` pour monitoring.

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

## Utilisation (Exemples curl)

### Health Check
```bash
curl http://localhost:8000/health
```

### Correction de texte
```bash
curl -X POST http://localhost:8000/api/corriger \
     -H "Content-Type: application/json" \
     -d '{"texte": "Je mangeais une pomme hier soir, cétait tres bon."}'
```

### Navigation
```bash
curl -X POST http://localhost:8000/api/trouver-page \
     -H "Content-Type: application/json" \
     -d '{"requete": "Je voudrais voir les tarifs sil vous plaît."}'
```

## Architecture
- **FastAPI** : Framework web asynchrone performant.
- **Pydantic** : Validation stricte des données d'entrée/sortie.
- **Httpx** : Client HTTP asynchrone pour la communication avec Ollama.
- **Ollama** : Moteur d'inférence LLM local.
