<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/Docker-24.0-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker"/>
  <img src="https://img.shields.io/badge/Ollama-0.13+-000000?style=for-the-badge&logo=ollama&logoColor=white" alt="Ollama"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/LangChain-0.3-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white" alt="LangChain"/>
  <img src="https://img.shields.io/badge/FAISS-Vector_DB-0467DF?style=for-the-badge&logo=meta&logoColor=white" alt="FAISS"/>
  <img src="https://img.shields.io/badge/Ministral_3-Vision-FF6B6B?style=for-the-badge&logo=mistral&logoColor=white" alt="Ministral 3"/>
  <img src="https://img.shields.io/badge/Tesseract-OCR-5A5A5A?style=for-the-badge&logo=google&logoColor=white" alt="Tesseract"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Apple_Silicon-M1%2FM2%2FM3-000000?style=for-the-badge&logo=apple&logoColor=white" alt="Apple Silicon"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License"/>
</p>

---

<h1 align="center">📚 FamilyRAG 2.7</h1>

<h3 align="center">
  <em>Votre bibliothèque numérique privée — dernière release 2025</em>
</h3>

<p align="center">
  Posez des questions à vos documents en langage naturel.<br/>
  Sans cloud. Sans abonnement. Sans compromis sur la vie privée.
</p>

---

## 🏠 Qu'est-ce que FamilyRAG ?

**FamilyRAG** est un système RAG (Retrieval-Augmented Generation) entièrement local, conçu pour les familles qui souhaitent exploiter la puissance de l'IA générative tout en gardant le contrôle total sur leurs données.

Imaginez pouvoir interroger en langage naturel :
- 📄 Les documents administratifs de la famille
- 📖 Les recettes de grand-mère numérisées
- 🖼️ Les photos de documents et textes scannés (OCR + Vision IA)
- 📝 Les cours et devoirs des enfants
- 📧 Les archives emails importantes
- 📚 Votre bibliothèque personnelle de livres et articles

**Le tout sans qu'une seule donnée ne quitte votre domicile.**

---

## ✨ Nouveautés v2.7

### 📊 Dashboard système en temps réel

- **Monitoring live** — CPU, RAM et disque affichés directement dans l'interface
- **Statistiques de précision** — Score de confiance et probabilité des résultats pour chaque réponse
- **Endpoint dédié** — `/api/system/metrics` pour surveiller les ressources

### 🎨 3 nouveaux thèmes premium

Refonte graphique complète avec 3 thèmes plus poussés remplaçant les 6 précédents :

- **Midnight** — Bleu nuit profond, design épuré et professionnel
- **Cyber** — Noir intense avec accents cyan néon, ambiance futuriste
- **Tactical** — Vert militaire avec accents rouges, style opérationnel

### 📁 Indexation sélective

- **Choix précis des fichiers** — Sélectionnez exactement quels documents indexer
- **Barre de progression intelligente** — Suivi en temps réel de l'avancement de l'indexation
- **Contrôle granulaire** — Plus besoin d'indexer l'intégralité du dossier

### 💬 Gestion avancée des conversations

- **Onglet dédié** — Interface complète pour gérer l'historique des conversations
- **Sélection de sources par conversation** — Limitez la recherche à des documents spécifiques pour chaque discussion
- **Persistance** — Reprenez vos conversations là où vous les avez laissées

### 🔒 Sécurité & Qualité (post-audit)

- **Gardes LFI** — Toute tentative de path traversal (`../../etc/passwd`) est bloquée avec HTTP 403 sur les endpoints `/api/index` et `/api/vision`
- **Interface 100 % hors-ligne** — Tailwind CSS et Alpine.js servis localement (`/static/`) ; aucune requête CDN externe
- **Suite de tests** — 40 tests pytest couvrant les routes API, les gardes de sécurité et les scénarios d'erreur
- **Logs structurés** — Tous les `print()` remplacés par `logging` avec niveaux INFO / WARNING / ERROR

---

## 🎯 Rappel v2.6 — Précision RAG +65%

La v2.6 avait apporté une refonte majeure du pipeline de recherche :

- **Re-ranking LLM** — Chaque chunk réévalué sémantiquement, éliminant les faux positifs
- **Métadonnées enrichies** — Date, année, type de document et auteur extraits automatiquement
- **Chunking intelligent** — Préserve la structure (articles, listes, tableaux)
- **Citations obligatoires** — Chaque réponse cite ses sources `[document.pdf]`
- **Sélection des modèles** — LLM et embedding modifiables en un clic depuis l'interface
- **Debug chunks** — Visualisation des passages récupérés et leurs scores

### 📊 Gains de précision v2.5 → v2.6

| Métrique | v2.5 | v2.6+ |
|----------|------|-------|
| Chunks récupérés | 4 | 12 |
| Réponses avec citations | ~30% | ~85% |
| Faux positifs | ~40% | ~10% |
| **Précision globale** | **~45%** | **~75%** |

---

## 🗂️ Formats supportés

<p align="center">
  <img src="https://img.shields.io/badge/PDF-Documents-EC1C24?style=for-the-badge&logo=adobe-acrobat-reader&logoColor=white" alt="PDF"/>
  <img src="https://img.shields.io/badge/TXT-Texte-4A4A4A?style=for-the-badge&logo=textpattern&logoColor=white" alt="TXT"/>
  <img src="https://img.shields.io/badge/MD-Markdown-000000?style=for-the-badge&logo=markdown&logoColor=white" alt="MD"/>
  <img src="https://img.shields.io/badge/DOCX-Word-2B579A?style=for-the-badge&logo=microsoft-word&logoColor=white" alt="DOCX"/>
  <img src="https://img.shields.io/badge/EML-Email-005FF9?style=for-the-badge&logo=mail.ru&logoColor=white" alt="EML"/>
  <img src="https://img.shields.io/badge/JPG-Image-FFD700?style=for-the-badge&logo=image&logoColor=black" alt="JPG"/>
  <img src="https://img.shields.io/badge/PNG-Image-FFD700?style=for-the-badge&logo=image&logoColor=black" alt="PNG"/>
</p>

**Images** : OCR Tesseract (indexation) + Vision Ministral 3 (analyse à la demande)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Votre machine (Mac/Linux/Windows)          │
│                                                                 │
│  ┌─────────────────┐        ┌────────────────────────────────┐  │
│  │     Ollama      │◄──────►│     Docker Container           │  │
│  │    (natif)      │  API   │                                │  │
│  │                 │        │  ┌──────────────────────────┐  │  │
│  │ • ministral-3   │        │  │  FastAPI + FAISS         │  │  │
│  │ • nomic-embed   │        │  │  + LangChain + Tesseract │  │  │
│  │                 │        │  └──────────────────────────┘  │  │
│  └─────────────────┘        │                                │  │
│          │                  │  ┌──────────────────────────┐  │  │
│          │ GPU              │  │  WebUI (3 thèmes)        │  │  │
│          ▼                  │  │  + Dashboard metrics     │  │  │
│  ┌─────────────────┐        │  └──────────────────────────┘  │  │
│  │  Apple Silicon  │        │                                │  │
│  │   M1/M2/M3      │        │   127.0.0.1:8000 (local only)  │  │
│  └─────────────────┘        └────────────────────────────────┘  │
│                                        ▲                        │
│  ┌─────────────────┐                   │                        │
│  │  📁 Vos Docs    │───────────────────┘                        │
│  │  (RAG folder)   │  volume mount                              │
│  └─────────────────┘                                            │
└─────────────────────────────────────────────────────────────────┘
```

Par défaut, le port 8000 est lié à `127.0.0.1` — l'application n'est accessible que depuis la machine hôte. Pour l'exposer sur le réseau local, modifiez `docker-compose.yml` (voir [Guide d'administration](https://github.com/Liamdbav/FAMILY_RAG/blob/main/MANAGE.md)).

---

## Prérequis

- **Ollama 0.13.1+** (requis pour Ministral 3)
- Docker Desktop

```bash
# Vérifier la version Ollama
ollama --version
```

---

## Installation

### 1. Cloner et configurer

```bash
git clone https://github.com/Liamdbav/FAMILY_RAG.git
cd FAMILY_RAG
cp .env.example .env
```

Éditer `.env` pour définir le chemin vers vos documents :

```bash
nano .env
# Modifier HOST_DATA_PATH=/chemin/vers/vos/documents
```

### 2. Installer les modèles Ollama

```bash
# Modèle de génération + vision (recommandé)
ollama pull ministral-3:latest

# Ou autres modèles de génération
ollama pull qwen2.5:7b
ollama pull mistral:latest

# Modèle d'embedding (obligatoire)
ollama pull nomic-embed-text
```

### 3. Lancer

```bash
docker compose up -d --build
```

### 4. C'est prêt !

Ouvrez **http://localhost:8000** 🎉

---

## ⚙️ Variables d'environnement

Toutes les variables sont déclarées dans `.env` (copié depuis `.env.example`).

| Variable | Défaut | Obligatoire | Description |
|----------|--------|:-----------:|-------------|
| `HOST_DATA_PATH` | — | ✅ | Chemin absolu vers le dossier de documents sur l'hôte |
| `OLLAMA_HOST` | `host.docker.internal:11434` | | Adresse du serveur Ollama (`host:port`) — `localhost:11434` sur Linux avec `network_mode: host` |
| `EMBEDDING_MODEL` | `nomic-embed-text` | | Modèle Ollama utilisé pour la vectorisation des documents |
| `LLM_MODEL` | _(auto-détecté)_ | | Modèle de génération Ollama ; si vide, le premier modèle disponible est sélectionné automatiquement |
| `CHUNK_SIZE` | `1200` | | Taille maximale d'un chunk en caractères |
| `CHUNK_OVERLAP` | `300` | | Chevauchement entre chunks consécutifs (en caractères) |
| `TOP_K` | `12` | | Nombre de chunks récupérés par requête RAG avant re-ranking |
| `TEMPERATURE` | `0.7` | | Créativité du LLM (`0` = précis et factuel, `1` = créatif) |

> **Note Linux** : remplacez `OLLAMA_HOST=host.docker.internal:11434` par `OLLAMA_HOST=localhost:11434` et décommentez `network_mode: host` dans `docker-compose.yml`.

---

## 🧪 Tests

Les tests nécessitent `pytest` et les dépendances applicatives :

```bash
pip install pytest
# Depuis la racine du projet
pytest tests/ -v
```

Les tests ne requièrent **ni Ollama ni FAISS** — le moteur RAG est entièrement simulé.

| Fichier | Couverture |
|---------|------------|
| `tests/test_security.py` | Traversées de répertoire (LFI) sur `DocumentLoader` et `RAGEngine` — 13 tests |
| `tests/test_api.py` | Routes FastAPI principales + gestion des erreurs — 27 tests |

---

## 📖 Documentation

| Guide | Description |
|-------|-------------|
| [Administration](https://github.com/Liamdbav/FAMILY_RAG/blob/main/MANAGE.md) | Gestion, diagnostic et maintenance |
| [Changelog](CHANGELOG.md) | Historique complet des versions |

---

## Licence

MIT — voir [LICENSE](LICENSE)

---

<div align="center">

Fait avec soin par **Liam**

[![Follow on X](https://img.shields.io/badge/Follow-%40Liamdbav-000000?style=flat-square&logo=x&logoColor=white)](https://x.com/Liamdbav)

</div>
