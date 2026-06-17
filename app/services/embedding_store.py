import os
import hashlib
import logging
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class EmbeddingStore:
    def __init__(self, knowledge_base_dir: str):
        self.knowledge_base_dir = knowledge_base_dir
        self.model = None
        self.index = None
        self.chunks = []
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
        all_chunks = []
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

                    all_chunks.extend(self._chunk_text(section_text))

        if not all_chunks:
            logger.warning("Aucun document trouvé dans la base de connaissances.")
            self.chunks = []
            self.index = None
            return

        self.chunks = all_chunks
        embeddings = self.model.encode(self.chunks)

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
        """Recherche les chunks les plus pertinents pour une requête."""
        if self.model is None or self.index is None or not self.chunks:
            return []

        query_vector = self.model.encode([query])
        distances, indices = self.index.search(np.array(query_vector).astype('float32'), top_k)
        
        results = []
        for idx in indices[0]:
            if idx != -1 and idx < len(self.chunks):
                results.append(self.chunks[idx])
        
        return results

# Instance singleton
embedding_store = EmbeddingStore("knowledge_base")
