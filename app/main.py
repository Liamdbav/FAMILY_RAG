"""API FastAPI pour le système RAG."""

import logging
import httpx
import psutil
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional, List

from starlette.concurrency import run_in_threadpool

from config import get_settings
from rag_engine import RAGEngine

logger = logging.getLogger(__name__)

# Instance globale du moteur RAG
rag_engine: Optional[RAGEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialisation et nettoyage de l'application."""
    global rag_engine
    rag_engine = RAGEngine()
    yield


app = FastAPI(
    title="Family RAG",
    description="Système RAG local pour la famille",
    version="2.7.0",
    lifespan=lifespan
)

# Templates et fichiers statiques
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


# --- Modèles Pydantic ---

class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = None
    image_data: Optional[str] = None  # Base64 encoded image
    selected_sources: Optional[list[str]] = None  # Filtrage par sources


class SettingsUpdate(BaseModel):
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    llm_model: Optional[str] = None
    embedding_model: Optional[str] = None


class VisionRequest(BaseModel):
    image_path: str
    question: str = "Décris cette image en détail et extrait tout le texte visible."


class VisionBase64Request(BaseModel):
    image_data: str  # Base64 encoded
    question: str = "Décris cette image en détail et extrait tout le texte visible."


# --- Routes API ---

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Page d'accueil avec l'interface web."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health():
    """Endpoint de health check."""
    settings = get_settings()

    # Vérifier la connexion Ollama
    ollama_ok = False
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{settings.ollama_base_url}/api/tags",
                timeout=5.0
            )
            ollama_ok = response.status_code == 200
    except Exception:
        pass

    return {
        "status": "healthy" if ollama_ok else "degraded",
        "ollama_connected": ollama_ok,
        "ollama_host": settings.ollama_host
    }


@app.get("/api/files")
async def list_files():
    """Liste les fichiers disponibles dans le dossier data."""
    return {"files": rag_engine.doc_loader.list_files()}


class IndexRequest(BaseModel):
    """Requête d'indexation avec fichiers sélectionnés."""
    selected_files: Optional[List[str]] = None  # Liste des chemins de fichiers à indexer


@app.post("/api/index")
async def index_documents(request: IndexRequest = None):
    """Indexe les documents sélectionnés (ou tous si aucune sélection)."""
    try:
        selected_files = request.selected_files if request else None
        result = rag_engine.index_documents(selected_files=selected_files)
        if not result['success']:
            raise HTTPException(status_code=400, detail=result.get('error', 'Indexation échouée'))
        return result
    except HTTPException:
        raise
    except Exception as e:
        error_message = str(e)
        logger.error("[API] Erreur indexation : %s", error_message)
        return {
            'success': False,
            'error': f"Erreur lors de l'indexation : {error_message}",
            'documents': 0,
            'chunks': 0
        }


@app.get("/api/index/details")
async def get_index_details():
    """Retourne les détails de l'index FAISS (fichiers indexés, nombre de vecteurs, etc.)."""
    return rag_engine.get_index_details()


@app.post("/api/query")
async def query(request: QueryRequest):
    """Exécute une requête RAG."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="La question ne peut pas être vide")

    # Créer le filtre de métadonnées si des sources sont sélectionnées
    filter_metadata = None
    if request.selected_sources and len(request.selected_sources) > 0:
        filter_metadata = {'source': request.selected_sources}

    result = await run_in_threadpool(
        rag_engine.query,
        request.question,
        request.top_k,
        filter_metadata=filter_metadata,
    )

    # Ajouter les chunks détaillés pour debug
    chunks_preview = []
    if result.sources:
        for source in result.sources:
            chunks_preview.append({
                'source': source['source'],
                'score': source['score'],
                'content': source['preview']  # Déjà limité à 200 chars
            })

    return {
        "answer": result.answer,
        "sources": result.sources,
        "chunks_preview": chunks_preview,  # Pour debug
        "metrics": {
            "query_time_ms": result.query_time_ms,
            "chunks_found": result.chunks_found,
            "model": result.model_used
        }
    }


@app.get("/api/stats")
async def get_stats():
    """Retourne les statistiques de l'index."""
    stats = rag_engine.get_stats()
    settings = get_settings()

    return {
        "index": {
            "exists": stats.index_exists,
            "total_documents": stats.total_documents,
            "total_chunks": stats.total_chunks
        },
        "settings": {
            "embedding_model": settings.embedding_model,
            "llm_model": rag_engine.settings.llm_model,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "temperature": rag_engine.settings.temperature,
            "top_k": rag_engine.settings.top_k
        }
    }


@app.put("/api/settings")
async def update_settings(settings: SettingsUpdate):
    """Met à jour les paramètres du moteur RAG."""
    needs_reindex = False

    # Vérifier si on change le modèle d'embedding
    if settings.embedding_model is not None:
        needs_reindex = rag_engine.update_embedding_model(settings.embedding_model)

    # Mettre à jour les autres paramètres
    rag_engine.update_settings(
        temperature=settings.temperature,
        top_k=settings.top_k,
        llm_model=settings.llm_model
    )

    return {
        "status": "ok",
        "llm_model": rag_engine.settings.llm_model,
        "embedding_model": rag_engine.settings.embedding_model,
        "temperature": rag_engine.settings.temperature,
        "top_k": rag_engine.settings.top_k,
        "needs_reindex": needs_reindex  # Indique si une ré-indexation est nécessaire
    }


@app.get("/api/ollama/models")
async def get_ollama_models():
    """Liste les modèles LLM disponibles sur Ollama (exclut les modèles d'embedding)."""
    settings = get_settings()

    # Modèles d'embedding à exclure
    EMBEDDING_MODELS = {
        "nomic-embed-text", "nomic",
        "mxbai-embed-large", "mxbai",
        "snowflake-arctic-embed", "snowflake",
        "all-minilm", "minilm",
        "bge-", "gte-"
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{settings.ollama_base_url}/api/tags",
                timeout=10.0
            )
            response.raise_for_status()
            data = response.json()

            # Filtrer pour exclure les modèles d'embedding
            llm_models = []
            for m in data.get("models", []):
                name = m["name"].lower()
                # Vérifier que ce n'est PAS un modèle d'embedding
                if not any(embed_name in name for embed_name in EMBEDDING_MODELS):
                    llm_models.append({
                        "name": m["name"],
                        "size": m.get("size", 0),
                        "modified": m.get("modified_at", "")
                    })

            return {
                "models": llm_models
            }
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Impossible de contacter Ollama: {str(e)}"
        )


@app.get("/api/ollama/embedding-models")
async def get_embedding_models():
    """Liste les modèles d'embedding disponibles sur Ollama."""
    settings = get_settings()

    # Modèles d'embedding connus
    EMBEDDING_MODELS = {
        "nomic-embed-text", "nomic",
        "mxbai-embed-large", "mxbai",
        "snowflake-arctic-embed", "snowflake",
        "all-minilm", "minilm",
        "bge-", "gte-"
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{settings.ollama_base_url}/api/tags",
                timeout=10.0
            )
            response.raise_for_status()
            data = response.json()

            # Filtrer uniquement les modèles d'embedding
            embedding_models = []
            for m in data.get("models", []):
                name = m["name"].lower()
                if any(embed_name in name for embed_name in EMBEDDING_MODELS):
                    embedding_models.append({
                        "name": m["name"],
                        "size": m.get("size", 0),
                        "modified": m.get("modified_at", "")
                    })

            return {
                "models": embedding_models
            }
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Impossible de contacter Ollama: {str(e)}"
        )


@app.get("/api/system/metrics")
async def get_system_metrics():
    """Retourne les métriques système en temps réel (CPU, RAM, disque)."""
    try:
        # CPU
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_count = psutil.cpu_count()

        # RAM
        memory = psutil.virtual_memory()
        memory_total_gb = memory.total / (1024 ** 3)
        memory_used_gb = memory.used / (1024 ** 3)
        memory_available_gb = memory.available / (1024 ** 3)
        memory_percent = memory.percent

        # Disque (partition racine)
        disk = psutil.disk_usage('/')
        disk_total_gb = disk.total / (1024 ** 3)
        disk_used_gb = disk.used / (1024 ** 3)
        disk_free_gb = disk.free / (1024 ** 3)
        disk_percent = disk.percent

        return {
            "cpu": {
                "percent": round(cpu_percent, 1),
                "count": cpu_count,
                "available": round(100 - cpu_percent, 1)
            },
            "memory": {
                "total_gb": round(memory_total_gb, 2),
                "used_gb": round(memory_used_gb, 2),
                "available_gb": round(memory_available_gb, 2),
                "percent": round(memory_percent, 1)
            },
            "disk": {
                "total_gb": round(disk_total_gb, 2),
                "used_gb": round(disk_used_gb, 2),
                "free_gb": round(disk_free_gb, 2),
                "percent": round(disk_percent, 1)
            }
        }
    except Exception as e:
        return {
            "cpu": {"percent": 0, "count": 0, "available": 100},
            "memory": {"total_gb": 0, "used_gb": 0, "available_gb": 0, "percent": 0},
            "disk": {"total_gb": 0, "used_gb": 0, "free_gb": 0, "percent": 0},
            "error": str(e)
        }


@app.post("/api/vision")
async def analyze_image(request: VisionRequest):
    """Analyse une image avec un modèle vision (Ministral 3)."""
    result = await run_in_threadpool(
        rag_engine.analyze_image_with_vision,
        request.image_path,
        request.question,
    )
    return result


@app.post("/api/vision/base64")
async def analyze_image_base64(request: VisionBase64Request):
    """Analyse une image encodée en base64 avec un modèle vision."""
    result = await rag_engine.analyze_image_base64(
        request.image_data,
        request.question
    )
    return result
