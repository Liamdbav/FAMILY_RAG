"""Moteur RAG avec FAISS et LangChain."""

import logging
import re
import shutil
import time
import base64
import httpx
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass

from fastapi import HTTPException
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate

from config import get_settings
from document_loader import DocumentLoader
from structure_aware_splitter import StructureAwareSplitter

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Résultat d'une requête RAG."""
    answer: str
    sources: List[dict]
    query_time_ms: float
    chunks_found: int
    model_used: str


@dataclass
class IndexStats:
    """Statistiques de l'index."""
    total_documents: int
    total_chunks: int
    index_exists: bool
    last_updated: Optional[str] = None


class RAGEngine:
    """Moteur RAG principal."""

    def __init__(self):
        self.settings = get_settings()
        self.index_path = Path(self.settings.index_dir) / "faiss_index"
        self.vectorstore: Optional[FAISS] = None
        self.doc_loader = DocumentLoader()
        self.llm: Optional[ChatOllama] = None

        # Auto-détecter le modèle LLM si non spécifié
        if not self.settings.llm_model:
            self.settings.llm_model = self._detect_llm_model()

        # Initialiser les composants Ollama
        self.embeddings = OllamaEmbeddings(
            model=self.settings.embedding_model,
            base_url=self.settings.ollama_base_url
        )

        # Initialiser le LLM seulement si un modèle est disponible
        if self.settings.llm_model:
            self.llm = ChatOllama(
                model=self.settings.llm_model,
                base_url=self.settings.ollama_base_url,
                temperature=self.settings.temperature
            )

        # Charger l'index existant si disponible
        self._load_index()

    def _detect_llm_model(self) -> str:
        """Détecte automatiquement un modèle LLM disponible dans Ollama."""
        try:
            response = httpx.get(
                f"{self.settings.ollama_base_url}/api/tags",
                timeout=5.0
            )
            if response.status_code == 200:
                models = response.json().get("models", [])
                # Exclure les modèles d'embedding connus
                embedding_models = {"nomic-embed-text", "mxbai-embed-large", "all-minilm", "snowflake-arctic-embed"}
                for model in models:
                    name = model.get("name", "")
                    # Vérifier que ce n'est pas un modèle d'embedding
                    base_name = name.split(":")[0]
                    if base_name not in embedding_models:
                        logger.info("Modèle LLM auto-détecté : %s", name)
                        return name
        except Exception as e:
            logger.warning("Impossible de détecter les modèles Ollama : %s", e)

        logger.warning("Aucun modèle LLM trouvé. Installez un modèle avec 'ollama pull <modele>'")
        return ""

    def _load_index(self) -> bool:
        """Charge l'index FAISS depuis le disque."""
        if self.index_path.exists():
            try:
                logger.info("Chargement de l'index depuis %s...", self.index_path)
                self.vectorstore = FAISS.load_local(
                    str(self.index_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info("Index chargé avec succès")
                return True
            except Exception as e:
                logger.error("Erreur chargement index: %s", e)
                logger.warning("Suppression de l'index corrompu...")
                try:
                    shutil.rmtree(str(self.index_path))
                    logger.info("Index corrompu supprimé. Veuillez ré-indexer vos documents.")
                except Exception as cleanup_error:
                    logger.error("Impossible de supprimer l'index corrompu: %s", cleanup_error)
        return False

    def _save_index(self):
        """Sauvegarde l'index FAISS sur le disque."""
        if self.vectorstore:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            self.vectorstore.save_local(str(self.index_path))

    def index_documents(self, selected_files: list = None) -> dict:
        """Indexe les documents sélectionnés ou tous les documents.

        Args:
            selected_files: Liste des chemins de fichiers à indexer (None = tous)
        """
        start_time = time.time()

        # Charger les documents (tous ou sélectionnés)
        if selected_files:
            logger.info("Indexation sélective de %d fichiers", len(selected_files))
            documents = self.doc_loader.load_specific(selected_files)
        else:
            logger.info("Indexation de tous les documents")
            documents = self.doc_loader.load_all()

        if not documents:
            return {
                'success': False,
                'error': 'Aucun document trouvé dans /data',
                'documents': 0,
                'chunks': 0
            }

        # Découper en chunks avec le splitter intelligent
        logger.info("Utilisation du StructureAwareSplitter pour préserver la structure des documents")
        splitter = StructureAwareSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            min_chunk_size=100,
            max_chunk_size=1500  # Réduit à 1500 pour éviter dépassement limite embedding
        )
        chunks = splitter.split_documents(documents)
        logger.info("%d documents → %d chunks intelligents", len(documents), len(chunks))

        # Vérifier et limiter la taille des chunks pour l'embedding (sécurité)
        # nomic-embed-text a une limite de ~2000 chars, on limite à 1500 par sécurité
        safe_chunks = []
        for chunk in chunks:
            if len(chunk.page_content) > 1500:
                logger.warning("Chunk trop grand (%d chars), découpage...", len(chunk.page_content))
                # Re-découper ce chunk avec le splitter fallback
                fallback_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    separators=["\n\n", "\n", ". ", " ", ""]
                )
                sub_chunks = fallback_splitter.split_documents([chunk])
                safe_chunks.extend(sub_chunks)
            else:
                safe_chunks.append(chunk)

        chunks = safe_chunks
        logger.info("Après vérification : %d chunks finaux", len(chunks))

        if not chunks:
            return {
                'success': False,
                'error': 'Aucun contenu extractible des documents',
                'documents': len(documents),
                'chunks': 0
            }

        # Créer l'index vectoriel
        try:
            self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
            self._save_index()
        except Exception as e:
            error_msg = str(e)
            logger.error("Erreur création index FAISS : %s", error_msg)

            # Message plus clair pour l'utilisateur
            if "exceeds the context length" in error_msg:
                return {
                    'success': False,
                    'error': 'Certains documents sont trop longs pour le modèle d\'embedding. Essayez de diviser vos documents en fichiers plus petits.',
                    'documents': len(documents),
                    'chunks': len(chunks)
                }
            else:
                return {
                    'success': False,
                    'error': f'Erreur lors de la création de l\'index : {error_msg}',
                    'documents': len(documents),
                    'chunks': len(chunks)
                }

        elapsed = time.time() - start_time

        return {
            'success': True,
            'documents': len(documents),
            'chunks': len(chunks),
            'time_seconds': round(elapsed, 2),
            'indexed_files': [doc.metadata.get('source', 'unknown') for doc in documents]
        }

    def get_index_details(self) -> dict:
        """Retourne les détails complets de l'index FAISS."""
        if not self.vectorstore:
            return {
                'exists': False,
                'total_vectors': 0,
                'indexed_files': [],
                'index_path': str(self.index_path)
            }

        # Compter les vecteurs dans l'index
        total_vectors = self.vectorstore.index.ntotal if hasattr(self.vectorstore, 'index') else 0

        # Extraire la liste des fichiers indexés depuis les métadonnées
        indexed_files = set()
        if hasattr(self.vectorstore, 'docstore') and hasattr(self.vectorstore.docstore, '_dict'):
            for doc in self.vectorstore.docstore._dict.values():
                if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                    indexed_files.add(doc.metadata['source'])

        return {
            'exists': True,
            'total_vectors': total_vectors,
            'indexed_files': sorted(list(indexed_files)),
            'index_path': str(self.index_path),
            'embedding_model': self.settings.embedding_model
        }

    def _rerank_chunks(self, question: str, docs_with_scores: List, target_k: int) -> List:
        """Re-classe les chunks avec le LLM pour améliorer la pertinence sémantique.

        Args:
            question: La question de l'utilisateur
            docs_with_scores: Liste de tuples (Document, score) depuis FAISS
            target_k: Nombre final de chunks à retourner après re-ranking

        Returns:
            Liste de tuples (Document, nouveau_score) re-classés par pertinence
        """
        if not docs_with_scores or not self.llm:
            return docs_with_scores

        # Préparer le prompt de scoring
        rerank_prompt = ChatPromptTemplate.from_template("""Tu es un expert en évaluation de pertinence de documents.

Ta tâche : Évaluer la pertinence de chaque extrait de document par rapport à la question.

QUESTION : {question}

EXTRAIT {index} :
{chunk}

INSTRUCTIONS :
1. Lis attentivement l'extrait
2. Détermine s'il contient des informations DIRECTEMENT utiles pour répondre à la question
3. Donne un score de 0 à 10 :
   - 10 = Répond EXACTEMENT à la question avec détails précis
   - 7-9 = Informations très pertinentes et utiles
   - 4-6 = Informations partiellement pertinentes
   - 1-3 = Vaguement lié au sujet
   - 0 = Hors-sujet complet

Réponds UNIQUEMENT avec un nombre entre 0 et 10, sans explication.""")

        # Scorer chaque chunk
        reranked = []
        for idx, (doc, original_score) in enumerate(docs_with_scores):
            try:
                # Limiter la taille du chunk pour le scoring (économie de tokens)
                chunk_preview = doc.page_content[:1500] if len(doc.page_content) > 1500 else doc.page_content

                # Demander au LLM de scorer (timeout 30 s pour éviter le blocage)
                chain = rerank_prompt | self.llm
                with ThreadPoolExecutor(max_workers=1) as ex:
                    fut = ex.submit(chain.invoke, {
                        "question": question,
                        "chunk": chunk_preview,
                        "index": idx + 1,
                    })
                    response = fut.result(timeout=30)

                # Extraire le score (parser la réponse)
                score_text = response.content.strip()
                # Tenter d'extraire un nombre
                match = re.search(r'(\d+(?:\.\d+)?)', score_text)
                if match:
                    llm_score = float(match.group(1))
                    # Normaliser entre 0 et 1
                    llm_score = min(10.0, max(0.0, llm_score)) / 10.0
                else:
                    # Si parsing échoue, garder score FAISS normalisé
                    llm_score = 1.0 / (1.0 + original_score)  # Convertir distance en similarité

                reranked.append((doc, llm_score))

            except Exception as e:
                # En cas d'erreur, garder le score FAISS
                logger.warning("Re-ranking : erreur sur chunk %d : %s", idx, e)
                reranked.append((doc, 1.0 / (1.0 + original_score)))

        # Trier par score décroissant
        reranked.sort(key=lambda x: x[1], reverse=True)

        # Retourner les top_k meilleurs
        return reranked[:target_k]

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        enable_reranking: bool = True,
        filter_metadata: Optional[dict] = None
    ) -> QueryResult:
        """Exécute une requête RAG."""
        if not self.llm or not self.settings.llm_model:
            return QueryResult(
                answer="Aucun modèle LLM disponible. Installez un modèle avec 'ollama pull <modele>' puis sélectionnez-le dans les paramètres.",
                sources=[],
                query_time_ms=0,
                chunks_found=0,
                model_used="Aucun"
            )

        if not self.vectorstore:
            return QueryResult(
                answer="Aucun document n'a été indexé. Veuillez d'abord indexer vos documents.",
                sources=[],
                query_time_ms=0,
                chunks_found=0,
                model_used=self.settings.llm_model
            )

        start_time = time.time()
        k = top_k or self.settings.top_k

        # Recherche des chunks pertinents avec FAISS
        # Si filtrage activé, récupérer plus de résultats pour compenser le filtrage
        # Si re-ranking activé, récupérer 2x plus de chunks pour avoir plus de choix
        if filter_metadata:
            initial_k = k * 4  # Plus de résultats pour le filtrage
        elif enable_reranking:
            initial_k = k * 2
        else:
            initial_k = k

        docs_with_scores = self.vectorstore.similarity_search_with_score(question, k=initial_k)

        if not docs_with_scores:
            return QueryResult(
                answer="Aucun document pertinent trouvé pour cette question.",
                sources=[],
                query_time_ms=0,
                chunks_found=0,
                model_used=self.settings.llm_model
            )

        # Filtrage par métadonnées si spécifié
        if filter_metadata:
            filtered_docs = []
            for doc, score in docs_with_scores:
                # Vérifier si le document correspond aux critères
                matches = True
                for key, value in filter_metadata.items():
                    doc_value = doc.metadata.get(key)
                    if doc_value is None:
                        matches = False
                        break
                    # Support pour plusieurs types de comparaisons
                    if isinstance(value, list):  # Valeur dans une liste
                        if doc_value not in value:
                            matches = False
                            break
                    elif isinstance(value, dict):  # Comparaisons avancées (ex: year >= 2020)
                        if 'gte' in value and doc_value < value['gte']:
                            matches = False
                            break
                        if 'lte' in value and doc_value > value['lte']:
                            matches = False
                            break
                        if 'eq' in value and doc_value != value['eq']:
                            matches = False
                            break
                    else:  # Égalité simple
                        if doc_value != value:
                            matches = False
                            break

                if matches:
                    filtered_docs.append((doc, score))

            docs_with_scores = filtered_docs[:initial_k]
            logger.debug(
                "Filtrage métadonnées : %d chunks correspondent aux critères %s",
                len(filtered_docs), filter_metadata
            )

            if not docs_with_scores:
                return QueryResult(
                    answer=f"Aucun document trouvé correspondant aux critères : {filter_metadata}",
                    sources=[],
                    query_time_ms=0,
                    chunks_found=0,
                    model_used=self.settings.llm_model
                )

        # Re-ranking avec LLM si activé
        if enable_reranking and len(docs_with_scores) > k:
            logger.info("Re-ranking %d chunks → top %d", len(docs_with_scores), k)
            docs_with_scores = self._rerank_chunks(question, docs_with_scores, k)
        else:
            # Limiter au top_k si pas de re-ranking
            docs_with_scores = docs_with_scores[:k]
            logger.debug("Utilisation de %d chunks", len(docs_with_scores))

        # Préparer le contexte
        context_parts = []
        sources = []

        for doc, score in docs_with_scores:
            context_parts.append(doc.page_content)
            sources.append({
                'source': doc.metadata.get('source', 'Unknown'),
                'score': round(float(score), 4),
                'preview': doc.page_content[:200] + '...' if len(doc.page_content) > 200 else doc.page_content
            })

        context = "\n\n---\n\n".join(context_parts)

        # Prompt RAG amélioré
        prompt = ChatPromptTemplate.from_template("""Tu es un assistant familial qui répond aux questions en te basant UNIQUEMENT sur les extraits de documents fournis.

PRINCIPES DE RÉPONSE :

1. PRÉCISION ET SOURCES
- Base-toi UNIQUEMENT sur les informations présentes dans le contexte
- Si une information N'EST PAS dans le contexte, dis clairement "Cette information n'est pas présente dans les documents"
- Cite la source entre crochets [nom_fichier] pour chaque affirmation factuelle

2. EFFICACITÉ
- Réponds de manière concise et précise
- Va droit au but sans sur-développer
- Structure ta réponse clairement (listes à puces si pertinent)

3. VÉRIFICATION
- Relis le contexte avant de répondre
- Vérifie que chaque fait mentionné est bien sourcé
- N'invente RIEN, n'extrapole PAS

4. CLARTÉ
- Utilise un langage simple et direct
- Si plusieurs sources mentionnent des informations différentes, note-le clairement
- En cas d'ambiguïté, demande des précisions

CONTEXTE (Extraits de documents) :
{context}

QUESTION : {question}

RÉPONSE (avec citations des sources) :""")

        # Générer la réponse
        chain = prompt | self.llm
        response = chain.invoke({"context": context, "question": question})

        elapsed_ms = (time.time() - start_time) * 1000

        return QueryResult(
            answer=response.content,
            sources=sources,
            query_time_ms=round(elapsed_ms, 2),
            chunks_found=len(docs_with_scores),
            model_used=self.settings.llm_model
        )

    def get_stats(self) -> IndexStats:
        """Retourne les statistiques de l'index."""
        if not self.vectorstore:
            return IndexStats(
                total_documents=0,
                total_chunks=0,
                index_exists=False
            )

        # FAISS ne stocke pas directement le nombre de documents originaux
        # On compte les chunks dans l'index
        try:
            total_chunks = self.vectorstore.index.ntotal
        except Exception:
            total_chunks = 0

        return IndexStats(
            total_documents=len(self.doc_loader.list_files()),
            total_chunks=total_chunks,
            index_exists=self.index_path.exists()
        )

    def update_settings(
        self,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        llm_model: Optional[str] = None
    ):
        """Met à jour les paramètres du moteur."""
        rebuild_llm = False

        if temperature is not None:
            self.settings.temperature = temperature
            rebuild_llm = True

        if llm_model is not None:
            self.settings.llm_model = llm_model
            rebuild_llm = True

        if rebuild_llm and self.settings.llm_model:
            self.llm = ChatOllama(
                model=self.settings.llm_model,
                base_url=self.settings.ollama_base_url,
                temperature=self.settings.temperature
            )

        if top_k is not None:
            self.settings.top_k = top_k

    def update_embedding_model(self, embedding_model: str) -> bool:
        """
        Met à jour le modèle d'embedding.
        Retourne True si une ré-indexation est nécessaire.
        """
        # Vérifier si le modèle a vraiment changé
        if embedding_model == self.settings.embedding_model:
            return False  # Pas besoin de ré-indexer

        logger.info(
            "Changement de modèle d'embedding : %s → %s",
            self.settings.embedding_model, embedding_model
        )

        # Mettre à jour le modèle dans les settings
        self.settings.embedding_model = embedding_model

        # Reconstruire l'objet OllamaEmbeddings avec le nouveau modèle
        self.embeddings = OllamaEmbeddings(
            model=embedding_model,
            base_url=self.settings.ollama_base_url
        )

        # Supprimer l'index existant car il n'est plus valide
        # (les embeddings ont changé de dimension/sémantique)
        if self.index_path.exists():
            logger.info("Suppression de l'ancien index (incompatible avec le nouveau modèle d'embedding)")
            try:
                shutil.rmtree(self.index_path)
            except Exception as e:
                logger.error("Erreur lors de la suppression de l'index : %s", e)

        # Réinitialiser le vectorstore
        self.vectorstore = None

        # Retourner True pour indiquer qu'une ré-indexation est nécessaire
        return True

    def analyze_image_with_vision(self, image_path: str, question: str) -> dict:
        """Analyse une image avec un modèle vision (Ministral 3)."""
        start_time = time.time()

        # Construire le chemin complet et vérifier qu'il reste dans data_dir
        base_dir = Path(self.settings.data_dir).resolve()
        full_path = (Path(self.settings.data_dir) / image_path).resolve()
        if not full_path.is_relative_to(base_dir):
            raise HTTPException(status_code=403, detail="Accès refusé : chemin hors de /data")

        if not full_path.exists():
            return {
                "success": False,
                "error": f"Image non trouvée: {image_path}",
                "analysis": None,
                "time_ms": 0
            }

        # Lire et encoder l'image en base64
        with open(full_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        # Détecter le modèle vision disponible
        vision_model = self._detect_vision_model()
        if not vision_model:
            return {
                "success": False,
                "error": "Aucun modèle vision disponible. Installez ministral-3 avec 'ollama pull ministral-3:latest'",
                "analysis": None,
                "time_ms": 0
            }

        # Appel API Ollama avec image
        try:
            with httpx.Client(timeout=120.0) as client:
                response = client.post(
                    f"{self.settings.ollama_base_url}/api/chat",
                    json={
                        "model": vision_model,
                        "messages": [
                            {
                                "role": "user",
                                "content": question,
                                "images": [image_data]
                            }
                        ],
                        "stream": False
                    }
                )
                response.raise_for_status()
                data = response.json()

                elapsed_ms = (time.time() - start_time) * 1000

                return {
                    "success": True,
                    "analysis": data.get("message", {}).get("content", ""),
                    "model": vision_model,
                    "time_ms": round(elapsed_ms, 2)
                }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "analysis": None,
                "time_ms": 0
            }

    def _detect_vision_model(self) -> Optional[str]:
        """Détecte un modèle vision disponible dans Ollama."""
        vision_models = ["ministral-3:latest", "ministral-3:8b", "ministral-3:14b", "ministral-3:3b", "llava:latest", "moondream:latest"]

        try:
            response = httpx.get(
                f"{self.settings.ollama_base_url}/api/tags",
                timeout=5.0
            )
            if response.status_code == 200:
                models = response.json().get("models", [])
                available = [m.get("name", "") for m in models]

                for vm in vision_models:
                    if vm in available:
                        return vm
                    # Check without tag
                    base = vm.split(":")[0]
                    for a in available:
                        if a.startswith(base):
                            return a
        except Exception:
            pass

        return None

    async def analyze_image_base64(self, image_data: str, question: str) -> dict:
        """Analyse une image encodée en base64 avec un modèle vision."""
        start_time = time.time()

        # Nettoyer le base64 si nécessaire (retirer le préfixe data:image/...)
        if ',' in image_data:
            image_data = image_data.split(',')[1]

        # Détecter le modèle vision disponible
        vision_model = self._detect_vision_model()
        if not vision_model:
            return {
                "success": False,
                "error": "Aucun modèle vision disponible. Installez ministral-3 avec 'ollama pull ministral-3:latest'",
                "analysis": None,
                "time_ms": 0
            }

        # Appel API Ollama avec image base64
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{self.settings.ollama_base_url}/api/chat",
                    json={
                        "model": vision_model,
                        "messages": [
                            {
                                "role": "user",
                                "content": question,
                                "images": [image_data]
                            }
                        ],
                        "stream": False
                    }
                )
                response.raise_for_status()
                data = response.json()

                elapsed_ms = (time.time() - start_time) * 1000

                return {
                    "success": True,
                    "analysis": data.get("message", {}).get("content", ""),
                    "model": vision_model,
                    "time_ms": round(elapsed_ms, 2)
                }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "analysis": None,
                "time_ms": 0
            }
