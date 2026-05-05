"""Tests de sécurité — couverture des vulnérabilités LFI (Local File Inclusion).

Scénarios couverts :
  - Traversée de répertoire via ../.. dans DocumentLoader.load_specific
  - Traversée de répertoire via ../.. dans RAGEngine.analyze_image_with_vision
  - Chemin absolu hors data_dir
  - Traversée imbriquée (subdir/../../)
  - Symlink pointant hors de data_dir
  - Chemin légitime — doit passer sans 403
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi import HTTPException
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers de construction sans dépendances externes
# ---------------------------------------------------------------------------

def _make_loader(data_dir: Path):
    """DocumentLoader pointant sur data_dir, sans connexion Ollama ni LangChain."""
    with patch("document_loader.get_settings") as mock_cfg:
        mock_cfg.return_value = MagicMock(data_dir=str(data_dir))
        from document_loader import DocumentLoader
        loader = DocumentLoader()
    return loader


def _make_engine(data_dir: Path):
    """RAGEngine minimal (sans Ollama) avec data_dir contrôlé."""
    from rag_engine import RAGEngine
    with patch.object(RAGEngine, "__init__", lambda self: None):
        engine = RAGEngine()
    engine.settings = MagicMock(
        data_dir=str(data_dir),
        ollama_base_url="http://localhost:11434",
    )
    return engine


# ---------------------------------------------------------------------------
# DocumentLoader — load_specific
# ---------------------------------------------------------------------------

class TestDocumentLoaderLFI:
    """Vérifie que load_specific rejette tout chemin hors de data_dir."""

    def test_double_dot_traversal(self, tmp_path):
        loader = _make_loader(tmp_path)
        with pytest.raises(HTTPException) as exc:
            loader.load_specific(["../../etc/passwd"])
        assert exc.value.status_code == 403

    def test_absolute_path_outside(self, tmp_path):
        loader = _make_loader(tmp_path)
        with pytest.raises(HTTPException) as exc:
            loader.load_specific(["/etc/passwd"])
        assert exc.value.status_code == 403

    def test_nested_traversal(self, tmp_path):
        """Traversée depuis un sous-dossier légitime."""
        loader = _make_loader(tmp_path)
        with pytest.raises(HTTPException) as exc:
            loader.load_specific(["docs/../../etc/shadow"])
        assert exc.value.status_code == 403

    def test_symlink_outside_data_dir(self, tmp_path):
        """Symlink résolu hors de data_dir → 403."""
        evil = tmp_path / "evil"
        try:
            evil.symlink_to("/etc")
        except (OSError, NotImplementedError):
            pytest.skip("Symlinks non supportés sur cette plateforme")
        loader = _make_loader(tmp_path)
        with pytest.raises(HTTPException) as exc:
            loader.load_specific(["evil/passwd"])
        assert exc.value.status_code == 403

    def test_error_detail_contains_path(self, tmp_path):
        """Le message 403 reproduit le chemin suspect pour aider au diagnostic."""
        loader = _make_loader(tmp_path)
        with pytest.raises(HTTPException) as exc:
            loader.load_specific(["../../etc/passwd"])
        assert "../../etc/passwd" in exc.value.detail

    def test_valid_txt_file_allowed(self, tmp_path):
        """Un fichier .txt dans data_dir doit être chargé sans lever 403."""
        doc = tmp_path / "note.txt"
        doc.write_text("contenu de test", encoding="utf-8")
        loader = _make_loader(tmp_path)
        result = loader.load_specific(["note.txt"])
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].page_content == "contenu de test"
        assert result[0].metadata["source"] == "note.txt"

    def test_multiple_paths_one_malicious(self, tmp_path):
        """Un seul chemin malveillant dans une liste suffit à lever 403."""
        good = tmp_path / "bon.txt"
        good.write_text("ok")
        loader = _make_loader(tmp_path)
        with pytest.raises(HTTPException) as exc:
            loader.load_specific(["bon.txt", "../../etc/passwd"])
        assert exc.value.status_code == 403


# ---------------------------------------------------------------------------
# RAGEngine — analyze_image_with_vision
# ---------------------------------------------------------------------------

class TestRAGEngineVisionLFI:
    """Vérifie que analyze_image_with_vision rejette tout chemin hors de data_dir."""

    def test_double_dot_traversal(self, tmp_path):
        engine = _make_engine(tmp_path)
        with pytest.raises(HTTPException) as exc:
            engine.analyze_image_with_vision("../../etc/passwd", "Décris")
        assert exc.value.status_code == 403

    def test_absolute_path_outside(self, tmp_path):
        engine = _make_engine(tmp_path)
        with pytest.raises(HTTPException) as exc:
            engine.analyze_image_with_vision("/etc/passwd", "Décris")
        assert exc.value.status_code == 403

    def test_nested_traversal(self, tmp_path):
        engine = _make_engine(tmp_path)
        with pytest.raises(HTTPException) as exc:
            engine.analyze_image_with_vision("img/../../etc/shadow", "Décris")
        assert exc.value.status_code == 403

    def test_symlink_outside_data_dir(self, tmp_path):
        """Un symlink résolu hors de data_dir doit être bloqué (resolve() inclus)."""
        evil = tmp_path / "evil.jpg"
        try:
            evil.symlink_to("/etc/passwd")
        except (OSError, NotImplementedError):
            pytest.skip("Symlinks non supportés sur cette plateforme")
        engine = _make_engine(tmp_path)
        with pytest.raises(HTTPException) as exc:
            engine.analyze_image_with_vision("evil.jpg", "Décris")
        assert exc.value.status_code == 403

    def test_valid_image_passes_guard(self, tmp_path):
        """Une image dans data_dir passe la garde — l'échec vient de l'absence de modèle."""
        img = tmp_path / "photo.jpg"
        img.write_bytes(b"\xff\xd8\xff")  # magic bytes JPEG, contenu invalide mais suffisant
        engine = _make_engine(tmp_path)
        with patch.object(engine, "_detect_vision_model", return_value=None):
            result = engine.analyze_image_with_vision("photo.jpg", "Décris")
        # La garde est passée → pas de HTTPException ; l'erreur est l'absence de modèle
        assert result["success"] is False
        assert "Aucun modèle vision" in result["error"]

    def test_403_detail_message(self, tmp_path):
        """Le détail de l'exception 403 indique clairement le refus d'accès."""
        engine = _make_engine(tmp_path)
        with pytest.raises(HTTPException) as exc:
            engine.analyze_image_with_vision("../../etc/passwd", "Décris")
        assert "Accès refusé" in exc.value.detail
