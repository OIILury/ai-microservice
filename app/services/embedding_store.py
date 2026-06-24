import os
import hashlib
import logging
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class EmbeddingStore:
    def __init__(self, knowledge_base_dir: str, max_distance: float = 30.0):
        # max_distance n'est PAS calibré pour distinguer finement pertinent/hors-sujet
        # (calibration empirique : le chevauchement entre les deux catégories est
        # structurel, ~9-11 points, et aucun seuil unique ne les sépare proprement).
        # Ce seuil sert uniquement de filet de sécurité grossier pour écarter le bruit
        # complètement aberrant (cf. calibrate_v3.py : hors-sujet va de ~13 à ~40).
        # Le tri fin pertinent/hors-sujet est délégué au system prompt de rag_engine.py
        # (règle 1 : répondre "je n'ai pas trouvé cette information" si absent du contexte).
        self.knowledge_base_dir = knowledge_base_dir
        self.model = None
        self.index = None
        self.chunks = []
        self.max_distance = max_distance
        self.hash_file = os.path.join(knowledge_base_dir, ".index_hash")
        self.index_path = os.path.join(knowledge_base_dir, ".index.faiss")
        self.chunks_path = os.path.join(knowledge_base_dir, ".chunks.pkl")

    def load_model(self):
        """Charge le modèle SentenceTransformer en mémoire. Appelé une seule fois au démarrage de l'application."""
        if self.model is None:
            logger.info("Chargement du modèle d'embeddings 'paraphrase-multilingual-MiniLM-L12-v2'...")
            self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            logger.info("Modèle d'embeddings chargé en mémoire.")

    def _calculate_hash(self) -> str:
        """Calcule un hash MD5 du contenu concaténé des fichiers source."""
        hasher = hashlib.md5()
        files = sorted([
            f for f in os.listdir(self.knowledge_base_dir) 
            if f.endswith(('.md', '.txt'))
        ])
        
        for filename in files:
            file_path = os.path.join(self.knowledge_base_dir, filename)
            with open(file_path, 'rb') as f:
                buf = f.read()
                hasher.update(buf)
        
        return hasher.hexdigest()

    def _chunk_text(self, text: str, max_size: int = 500, overlap: int = 50) -> list[str]:
        """Découpe un texte en chunks avec chevauchement."""
        if len(text) <= max_size:
            return [text]
        
        chunks = []
        start = 0
        while start < len(text):
            end = start + max_size
            chunks.append(text[start:end])
            start += max_size - overlap
            if start >= len(text):
                break
        return chunks

    def _split_synonyms(self, section_text: str) -> tuple[str, str]:
        """
        Sépare le texte "officiel" (destiné au LLM) des synonymes/reformulations
        familières (destinés uniquement à enrichir l'embedding, jamais injectés
        dans le prompt Mistral).

        Convention dans les .md : une ligne '%%SYN%%' suivie de variantes
        séparées par '|' marque le début du bloc synonymes.
        Retourne (texte_officiel, texte_synonymes). texte_synonymes == "" si absent.
        """
        if "%%SYN%%" not in section_text:
            return section_text, ""

        texte_officiel, _, bloc_syn = section_text.partition("%%SYN%%")
        texte_officiel = texte_officiel.strip()
        # Les variantes peuvent être séparées par '|' et/ou des retours à la ligne.
        # On filtre aussi les résidus de séparateurs markdown ('---') qui peuvent
        # se glisser en fin de bloc si une ligne '---' suit immédiatement le %%SYN%%.
        variantes = [v.strip() for v in bloc_syn.replace("\n", "|").split("|")]
        variantes = [v for v in variantes if v and v.strip("-") != ""]
        texte_synonymes = " ".join(variantes)
        return texte_officiel, texte_synonymes

    def build_index(self):
        """Construit l'index FAISS à partir des fichiers de la base de connaissances, ou le recharge depuis le cache disque si rien n'a changé."""
        if not os.path.exists(self.knowledge_base_dir):
            os.makedirs(self.knowledge_base_dir)
            logger.warning(f"Dossier {self.knowledge_base_dir} créé car inexistant.")
            return

        current_hash = self._calculate_hash()

        # Tentative de chargement depuis le cache disque
        cache_complet = (
            os.path.exists(self.hash_file)
            and os.path.exists(self.index_path)
            and os.path.exists(self.chunks_path)
        )
        if cache_complet:
            with open(self.hash_file, 'r') as f:
                cached_hash = f.read().strip()
            if cached_hash == current_hash:
                logger.info("Index à jour trouvé sur disque. Chargement depuis le cache.")
                if self.model is None:
                    self.load_model()
                self.index = faiss.read_index(self.index_path)
                with open(self.chunks_path, 'rb') as f:
                    self.chunks = pickle.load(f)
                logger.info(f"Index chargé depuis le cache avec {len(self.chunks)} chunks.")
                return

        if self.model is None:
            self.load_model()

        logger.info("Construction de l'index vectoriel...")
        all_chunks = []        # Texte "officiel", celui qui sera injecté dans le prompt Mistral.
        all_embed_texts = []   # Texte utilisé pour l'embedding (officiel + synonymes), jamais montré au LLM.
        files = [
            f for f in os.listdir(self.knowledge_base_dir)
            if f.endswith(('.md', '.txt'))
        ]

        for filename in files:
            file_path = os.path.join(self.knowledge_base_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

                sections = content.split('##')
                for i, section in enumerate(sections):
                    section_text = section.strip()
                    if not section_text:
                        continue

                    if i > 0:
                        section_text = "## " + section_text

                    texte_officiel, texte_synonymes = self._split_synonyms(section_text)

                    for sous_chunk in self._chunk_text(texte_officiel):
                        all_chunks.append(sous_chunk)
                        if texte_synonymes:
                            all_embed_texts.append(f"{sous_chunk}\n{texte_synonymes}")
                        else:
                            all_embed_texts.append(sous_chunk)

        if not all_chunks:
            logger.warning("Aucun document trouvé dans la base de connaissances.")
            self.chunks = []
            self.index = None
            return

        self.chunks = all_chunks
        embeddings = self.model.encode(all_embed_texts)

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype('float32'))

        # Sauvegarde du cache complet sur disque
        faiss.write_index(self.index, self.index_path)
        with open(self.chunks_path, 'wb') as f:
            pickle.dump(self.chunks, f)
        with open(self.hash_file, 'w') as f:
            f.write(current_hash)

        logger.info(f"Index construit et sauvegardé avec {len(self.chunks)} chunks.")

    def search(self, query: str, top_k: int = 3) -> list[str]:
        """Recherche les chunks les plus pertinents pour une requête, en filtrant ceux trop éloignés (distance L2 > max_distance)."""
        if self.model is None or self.index is None or not self.chunks:
            return []

        query_vector = self.model.encode([query])
        distances, indices = self.index.search(np.array(query_vector).astype('float32'), top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1 or idx >= len(self.chunks):
                continue
            if dist > self.max_distance:
                logger.debug(f"Chunk {idx} écarté (distance {dist:.3f} > seuil {self.max_distance}).")
                continue
            results.append(self.chunks[idx])

        if not results:
            logger.info(f"Aucun chunk sous le seuil de distance {self.max_distance} pour la requête : '{query}'.")

        return results

# Instance singleton
embedding_store = EmbeddingStore("knowledge_base")