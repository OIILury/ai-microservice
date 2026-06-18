import sqlite3
import logging

logger = logging.getLogger(__name__)
DB_PATH = "metrics.db"

def init_db():
    """Initialise la base de données SQLite et crée la table ollama_metrics si elle n'existe pas."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ollama_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    modele TEXT,
                    endpoint TEXT,
                    input_chars INTEGER,
                    output_chars INTEGER,
                    total_duration_ns INTEGER,
                    eval_count INTEGER,
                    tokens_per_sec REAL
                )
            """)
        logger.info("Base de données initialisée avec succès.")
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation de la base de données : {e}")

def save_metrics(modele, endpoint, input_chars, output_chars, total_duration_ns, eval_count, tokens_per_sec):
    """Sauvegarde les métriques d'inférence dans la base de données."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                INSERT INTO ollama_metrics (
                    modele, endpoint, input_chars, output_chars, total_duration_ns, eval_count, tokens_per_sec
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (modele, endpoint, input_chars, output_chars, total_duration_ns, eval_count, tokens_per_sec))
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde des métriques : {e}")
