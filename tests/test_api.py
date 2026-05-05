"""Tests d'intégration — routes principales de l'API FastAPI.

Le RAGEngine est remplacé par un double de test (MagicMock) afin que
les tests ne nécessitent ni Ollama ni FAISS sur la machine de CI.

Routes couvertes :
  GET  /                     → HTML 200
  GET  /health               → JSON structuré (Ollama peut être absent)
  GET  /api/files            → liste des fichiers
  POST /api/query            → réponse RAG correctement formée
  POST /api/index            → succès et erreur
  GET  /api/index/details    → détails FAISS
  GET  /api/stats            → statistiques index + settings
  PUT  /api/settings         → mise à jour temperature / top_k / llm_model
  GET  /api/ollama/models    → 503 gracieux si Ollama absent
  GET  /api/system/metrics   → métriques CPU / RAM / disque
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

def _make_mock_rag():
    """Retourne un RAGEngine factice prêt à l'emploi."""
    from rag_engine import QueryResult, IndexStats

    mock = MagicMock()

    mock.doc_loader.list_files.return_value = [
        {"name": "contrat.pdf", "path": "contrat.pdf", "type": "pdf", "size": 20480},
        {"name": "recette.txt", "path": "recette.txt", "type": "txt", "size": 1024},
    ]

    mock.query.return_value = QueryResult(
        answer="La réponse se trouve dans [contrat.pdf].",
        sources=[
            {"source": "contrat.pdf", "score": 0.92, "preview": "Article 1 — Les parties..."}
        ],
        query_time_ms=312.5,
        chunks_found=1,
        model_used="ministral-3:latest",
    )

    mock.get_stats.return_value = IndexStats(
        total_documents=2,
        total_chunks=18,
        index_exists=True,
    )

    mock.get_index_details.return_value = {
        "exists": True,
        "total_vectors": 18,
        "indexed_files": ["contrat.pdf", "recette.txt"],
        "embedding_model": "nomic-embed-text",
        "index_path": "/app/index/faiss_index",
    }

    mock.index_documents.return_value = {
        "success": True,
        "documents": 2,
        "chunks": 18,
        "time_seconds": 4.2,
        "indexed_files": ["contrat.pdf", "recette.txt"],
    }

    mock.settings.llm_model = "ministral-3:latest"
    mock.settings.embedding_model = "nomic-embed-text"
    mock.settings.temperature = 0.7
    mock.settings.top_k = 12

    mock.update_settings.return_value = None
    mock.update_embedding_model.return_value = False

    return mock


@pytest.fixture
def client():
    """Client de test avec RAGEngine mocké — aucune connexion externe requise."""
    mock_rag = _make_mock_rag()
    with patch("main.RAGEngine", return_value=mock_rag):
        import main
        with TestClient(main.app) as test_client:
            yield test_client, mock_rag


# ---------------------------------------------------------------------------
# Page d'accueil
# ---------------------------------------------------------------------------

class TestHomePage:
    def test_returns_html(self, client):
        c, _ = client
        r = c.get("/")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]
        assert "Family RAG" in r.text


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

class TestHealth:
    def test_returns_200_with_expected_keys(self, client):
        c, _ = client
        r = c.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert "status" in data
        assert "ollama_connected" in data
        assert "ollama_host" in data

    def test_status_is_string(self, client):
        c, _ = client
        data = c.get("/health").json()
        assert data["status"] in ("healthy", "degraded")


# ---------------------------------------------------------------------------
# Fichiers
# ---------------------------------------------------------------------------

class TestFiles:
    def test_list_files_structure(self, client):
        c, mock_rag = client
        r = c.get("/api/files")
        assert r.status_code == 200
        data = r.json()
        assert "files" in data
        assert isinstance(data["files"], list)

    def test_file_entries_have_required_fields(self, client):
        c, _ = client
        files = c.get("/api/files").json()["files"]
        assert len(files) == 2
        for f in files:
            assert "name" in f
            assert "path" in f
            assert "type" in f
            assert "size" in f


# ---------------------------------------------------------------------------
# Requête RAG
# ---------------------------------------------------------------------------

class TestQuery:
    def test_successful_query(self, client):
        c, _ = client
        r = c.post("/api/query", json={"question": "Quelles sont les parties du contrat ?"})
        assert r.status_code == 200
        data = r.json()
        assert "answer" in data
        assert "sources" in data
        assert "metrics" in data

    def test_answer_is_string(self, client):
        c, _ = client
        data = c.post("/api/query", json={"question": "Test"}).json()
        assert isinstance(data["answer"], str)
        assert len(data["answer"]) > 0

    def test_metrics_fields(self, client):
        c, _ = client
        metrics = c.post("/api/query", json={"question": "Test"}).json()["metrics"]
        assert "query_time_ms" in metrics
        assert "chunks_found" in metrics
        assert "model" in metrics

    def test_empty_question_returns_400(self, client):
        c, _ = client
        r = c.post("/api/query", json={"question": "   "})
        assert r.status_code == 400

    def test_source_filter_forwarded(self, client):
        c, mock_rag = client
        c.post("/api/query", json={
            "question": "Question",
            "selected_sources": ["contrat.pdf"]
        })
        call_kwargs = mock_rag.query.call_args
        # filter_metadata doit contenir les sources sélectionnées
        assert call_kwargs.kwargs.get("filter_metadata") == {"source": ["contrat.pdf"]}

    def test_no_filter_when_sources_empty(self, client):
        c, mock_rag = client
        c.post("/api/query", json={"question": "Question", "selected_sources": []})
        call_kwargs = mock_rag.query.call_args
        assert call_kwargs.kwargs.get("filter_metadata") is None


# ---------------------------------------------------------------------------
# Indexation
# ---------------------------------------------------------------------------

class TestIndex:
    def test_index_success(self, client):
        c, _ = client
        r = c.post("/api/index", json={"selected_files": ["contrat.pdf"]})
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is True
        assert data["documents"] > 0
        assert data["chunks"] > 0

    def test_index_failure_returns_json_error(self, client):
        c, mock_rag = client
        mock_rag.index_documents.return_value = {
            "success": False,
            "error": "Aucun document trouvé",
            "documents": 0,
            "chunks": 0,
        }
        r = c.post("/api/index", json={})
        assert r.status_code == 400
        assert "Aucun document" in r.json()["detail"]

    def test_index_exception_returns_json_not_500(self, client):
        """Une exception interne est absorbée et retourne un JSON d'erreur propre."""
        c, mock_rag = client
        mock_rag.index_documents.side_effect = RuntimeError("crash inattendu")
        r = c.post("/api/index", json={})
        assert r.status_code == 200  # géré par le catch-all
        data = r.json()
        assert data["success"] is False
        assert "crash inattendu" in data["error"]


# ---------------------------------------------------------------------------
# Détails de l'index
# ---------------------------------------------------------------------------

class TestIndexDetails:
    def test_structure(self, client):
        c, _ = client
        r = c.get("/api/index/details")
        assert r.status_code == 200
        data = r.json()
        assert "exists" in data
        assert "total_vectors" in data
        assert "indexed_files" in data


# ---------------------------------------------------------------------------
# Statistiques
# ---------------------------------------------------------------------------

class TestStats:
    def test_returns_index_and_settings(self, client):
        c, _ = client
        r = c.get("/api/stats")
        assert r.status_code == 200
        data = r.json()
        assert "index" in data
        assert "settings" in data

    def test_index_section_fields(self, client):
        c, _ = client
        idx = c.get("/api/stats").json()["index"]
        assert "exists" in idx
        assert "total_documents" in idx
        assert "total_chunks" in idx
        assert idx["exists"] is True
        assert idx["total_chunks"] == 18


# ---------------------------------------------------------------------------
# Mise à jour des paramètres
# ---------------------------------------------------------------------------

class TestSettings:
    def test_update_temperature(self, client):
        c, mock_rag = client
        r = c.put("/api/settings", json={"temperature": 0.3})
        assert r.status_code == 200
        mock_rag.update_settings.assert_called_once()

    def test_update_top_k(self, client):
        c, mock_rag = client
        c.put("/api/settings", json={"top_k": 5})
        mock_rag.update_settings.assert_called()

    def test_response_contains_model_info(self, client):
        c, _ = client
        data = c.put("/api/settings", json={"temperature": 0.5}).json()
        assert "llm_model" in data
        assert "embedding_model" in data
        assert "needs_reindex" in data

    def test_embedding_model_change_triggers_update(self, client):
        c, mock_rag = client
        mock_rag.update_embedding_model.return_value = True
        data = c.put("/api/settings", json={"embedding_model": "mxbai-embed-large"}).json()
        mock_rag.update_embedding_model.assert_called_once_with("mxbai-embed-large")
        assert data["needs_reindex"] is True


# ---------------------------------------------------------------------------
# Modèles Ollama (erreur gracieuse si Ollama absent)
# ---------------------------------------------------------------------------

class TestOllamaModels:
    def test_llm_models_returns_503_or_200(self, client):
        """Sans Ollama, l'endpoint retourne 503 — pas de crash non géré."""
        c, _ = client
        r = c.get("/api/ollama/models")
        assert r.status_code in (200, 503)

    def test_embedding_models_returns_503_or_200(self, client):
        c, _ = client
        r = c.get("/api/ollama/embedding-models")
        assert r.status_code in (200, 503)

    def test_503_has_detail_field(self, client):
        """Le 503 doit contenir un message d'erreur exploitable."""
        c, _ = client
        r = c.get("/api/ollama/models")
        if r.status_code == 503:
            assert "detail" in r.json()


# ---------------------------------------------------------------------------
# Métriques système
# ---------------------------------------------------------------------------

class TestSystemMetrics:
    def test_returns_cpu_memory_disk(self, client):
        c, _ = client
        r = c.get("/api/system/metrics")
        assert r.status_code == 200
        data = r.json()
        assert "cpu" in data
        assert "memory" in data
        assert "disk" in data

    def test_cpu_has_percent(self, client):
        c, _ = client
        cpu = c.get("/api/system/metrics").json()["cpu"]
        assert "percent" in cpu
        assert 0 <= cpu["percent"] <= 100

    def test_memory_has_gb_fields(self, client):
        c, _ = client
        mem = c.get("/api/system/metrics").json()["memory"]
        assert "total_gb" in mem
        assert "used_gb" in mem
        assert "available_gb" in mem
