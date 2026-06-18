import sqlite3
import logging
from datetime import datetime
from typing import List, Dict

logger = logging.getLogger(__name__)

class MetricsStore:
    def __init__(self, db_path: str = "metrics.db"):
        self.db_path = db_path

    def init_db(self):
        """Initialise la base de données et crée la table ollama_metrics enrichie."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS ollama_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        modele TEXT NOT NULL,
                        type_tache TEXT NOT NULL,
                        texte_entree TEXT,
                        nb_caracteres_entree INTEGER,
                        nb_mots_entree INTEGER,
                        texte_sortie TEXT,
                        nb_caracteres_sortie INTEGER,
                        nb_mots_sortie INTEGER,
                        temps_reponse_ms REAL,
                        time_to_first_token_ms REAL,
                        tokens_entree INTEGER,
                        tokens_sortie INTEGER,
                        tokens_par_seconde REAL,
                        ratio_taille_sortie_entree REAL,
                        distance_levenshtein INTEGER,
                        statut TEXT NOT NULL,
                        notes TEXT
                    )
                """)
            logger.info("Base de données de métriques initialisée.")
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation de la base de données : {e}")

    def enregistrer_metrique(self, **kwargs):
        """Insère une ligne de métrique dans la base de données."""
        keys = kwargs.keys()
        placeholders = ', '.join(['?'] * len(keys))
        columns = ', '.join(keys)
        values = tuple(kwargs.values())
        
        sql = f"INSERT INTO ollama_metrics ({columns}) VALUES ({placeholders})"
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(sql, values)
        except Exception as e:
            logger.error(f"Erreur lors de l'enregistrement de la métrique : {e}")

    def get_all_metrics(self, limit: int = 200) -> List[Dict]:
        """Retourne les limit dernières métriques."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    "SELECT * FROM ollama_metrics ORDER BY timestamp DESC LIMIT ?", 
                    (limit,)
                )
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des métriques : {e}")
            return []

    def get_aggregated_stats(self) -> Dict:
        """Retourne des statistiques agrégées par modèle, par type de tâche et évolution temporelle."""
        stats = {
            "par_modele": [],
            "par_type_tache": [],
            "evolution_temporelle": []
        }
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                # Stats par modèle
                cursor = conn.execute("""
                    SELECT 
                        modele, 
                        COUNT(*) as nb_appels, 
                        AVG(tokens_par_seconde) as tokens_par_seconde_moyen,
                        AVG(temps_reponse_ms) as temps_reponse_ms_moyen,
                        AVG(time_to_first_token_ms) as ttft_ms_moyen
                    FROM ollama_metrics
                    WHERE statut = 'succes'
                    GROUP BY modele
                """)
                stats["par_modele"] = [dict(row) for row in cursor.fetchall()]
                
                # Stats par type de tâche
                cursor = conn.execute("""
                    SELECT 
                        type_tache, 
                        COUNT(*) as nb_appels, 
                        AVG(tokens_par_seconde) as tokens_par_seconde_moyen,
                        AVG(ratio_taille_sortie_entree) as ratio_taille_moyen
                    FROM ollama_metrics
                    WHERE statut = 'succes'
                    GROUP BY type_tache
                """)
                stats["par_type_tache"] = [dict(row) for row in cursor.fetchall()]
                
                # Évolution temporelle (30 derniers jours)
                cursor = conn.execute("""
                    SELECT 
                        substr(timestamp, 1, 10) as date, 
                        COUNT(*) as nb_appels, 
                        AVG(tokens_par_seconde) as tokens_par_seconde_moyen
                    FROM ollama_metrics
                    WHERE statut = 'succes'
                    GROUP BY date
                    ORDER BY date DESC
                    LIMIT 30
                """)
                stats["evolution_temporelle"] = [dict(row) for row in cursor.fetchall()]
                
        except Exception as e:
            logger.error(f"Erreur lors de l'agrégation des statistiques : {e}")
            
        return stats

# Singleton instance
metrics_store = MetricsStore()
