# Rapport d'audit — FamilyRAG 2.7
**Date :** 2026-05-05
**Stack :** Python 3.11, FastAPI 0.115, LangChain 0.3, FAISS, Ollama, Tesseract, Docker, Alpine.js + Tailwind (CDN)
**Auditeur :** Claude Code — mode audit

---

## Résumé exécutif

FamilyRAG 2.7 est une application RAG locale fonctionnelle, bien structurée et globalement lisible. La proposition de valeur (privacy-first, 100% local) est tenue. **En revanche, le projet n'est pas livrable en l'état pour un client.** Trois familles de problèmes bloquent la livraison : (1) le fichier `.env` est versionné dans git sans `.gitignore` ni `.env.example`, (2) deux endpoints (`/api/vision` et `/api/index`) acceptent des chemins de fichiers non validés et exposent une vulnérabilité de **path traversal** (LFI) trivialement exploitable, (3) toute l'API est exposée sans authentification, sans CORS configuré, sans rate limiting — un poste compromis sur le LAN peut indexer/lire n'importe quel fichier du conteneur. Côté qualité, les routes async appellent du code synchrone bloquant (LLM, embeddings) qui gèle l'event loop FastAPI ; il n'existe aucun test, aucun logging structuré et le `CHANGELOG.md` référencé dans le README est manquant.

**Verdict : NON LIVRABLE en l'état. 1 à 2 jours de travail correctifs sont nécessaires avant remise au client.**

## Score de livraison : 4.5/10

| Dimension | Score | Statut |
|-----------|-------|--------|
| Sécurité | 3/10 | 🔴 |
| Stabilité | 5/10 | 🟠 |
| Performance | 5/10 | 🟠 |
| Qualité de code | 6/10 | 🟡 |
| Tests & Observabilité | 1/10 | 🔴 |
| Documentation | 6/10 | 🟡 |

---

## Problèmes identifiés

### 🔴 DANGERS (5 problèmes)

#### [DANGER-001] Path Traversal / LFI sur l'endpoint `/api/vision`
**Fichier :** `app/rag_engine.py:578-595` (et `app/main.py:367-374`)
**Observation :** Le paramètre `image_path` envoyé par le client est concaténé tel quel au répertoire `data_dir` :
```python
full_path = Path(self.settings.data_dir) / image_path
...
with open(full_path, "rb") as f:
    image_data = base64.b64encode(f.read()).decode("utf-8")
```
Aucune normalisation, aucune vérification que le chemin résolu reste dans `data_dir`. Une requête avec `image_path = "../../etc/passwd"` ou `../../../app/.env` lit n'importe quel fichier accessible par le process.
**Risque :** Lecture arbitraire de fichiers dans le conteneur (clés, configs, secrets, `/etc/passwd`, l'index FAISS, les autres documents privés). Exploitable depuis n'importe quel client réseau ayant accès au port 8000.
**Solution :**
```python
candidate = (Path(self.settings.data_dir) / image_path).resolve()
data_root = Path(self.settings.data_dir).resolve()
if not candidate.is_file() or data_root not in candidate.parents:
    return {"success": False, "error": "Chemin invalide", "analysis": None, "time_ms": 0}
```
Refuser également les chemins absolus en amont (`if Path(image_path).is_absolute(): raise HTTPException(400)`).

#### [DANGER-002] Path Traversal sur l'endpoint `/api/index` (indexation sélective)
**Fichier :** `app/document_loader.py:215-276`
**Observation :** `load_specific()` itère sur les chemins reçus et fait `file_path = self.data_dir / relative_path` sans validation. Le contrôle se fait uniquement sur `file_path.suffix.lower()` et `file_path.exists()`. Un attaquant peut donc demander l'indexation de `../../etc/hostname.txt` (renommage/symlink suffit) ou plus simplement abuser des extensions supportées présentes ailleurs sur le filesystem.
**Risque :** Lecture et indexation de fichiers hors `/data`, exfiltration via les réponses RAG ultérieures (le contenu sera retourné dans `/api/query` à toute personne y ayant accès).
**Solution :** Même pattern que DANGER-001 — résoudre le chemin et vérifier l'appartenance à `data_dir` :
```python
candidate = (self.data_dir / relative_path).resolve()
data_root = self.data_dir.resolve()
if data_root not in candidate.parents and candidate != data_root:
    print(f"[Loader] ⚠️ Tentative path traversal : {relative_path}")
    continue
```

#### [DANGER-003] Fichier `.env` versionné dans git, pas de `.gitignore`
**Fichier :** racine du projet
**Observation :** `git ls-files` montre que `.env` est tracké. Aucun `.gitignore` n'existe. Aucun `.env.example` n'existe alors que le README ligne 169 invite à `cp .env.example .env`. Le contenu actuel est anodin (`HOST_DATA_PATH=/home/user`, `OLLAMA_HOST=...`) mais le risque est structurel : à la première itération qui ajoute une clé API ou un token, il sera commité.
**Risque :** Fuite future de secrets. Instructions README mensongères. Pollution de l'historique git si quelqu'un édite `.env` localement.
**Solution :**
1. Créer `.gitignore` à la racine :
```
.env
.env.local
__pycache__/
*.pyc
*.pyo
.venv/
venv/
RAG/
backups/
.DS_Store
.idea/
.vscode/
```
2. Créer `.env.example` (copie sanitisée du `.env`).
3. Retirer `.env` du tracking : `git rm --cached .env && git commit -m "chore: untrack .env"`.

#### [DANGER-004] API totalement non authentifiée et exposable sur le réseau local
**Fichier :** `app/main.py` (toutes les routes), `MANAGE.md:330-341`
**Observation :** Aucune route n'a d'authentification, pas de middleware `CORS` configuré (donc tout site web peut appeler l'API depuis le navigateur de la victime via fetch cross-origin si CORS est laissé permissif par défaut FastAPI — en pratique FastAPI bloque, mais un visiteur sur `localhost:autre-port` pourra forger des requêtes simples). Le port 8000 est bindé sur `0.0.0.0` (Dockerfile ligne 34) et `MANAGE.md` l'annonce explicitement comme une feature « Accès réseau local ». N'importe quel appareil du LAN peut donc ré-indexer, supprimer l'index, lire tous les documents indexés et exploiter DANGER-001/002.
**Risque :** Exfiltration totale du contenu indexé (administratifs familiaux, scans, etc.) par toute personne ayant accès au réseau Wi-Fi (voisin, invité, IoT compromis).
**Solution minimale pour livraison :** binder par défaut sur `127.0.0.1` côté docker-compose et rendre l'exposition LAN explicite (variable `BIND_HOST` documentée). Pour une vraie protection : ajouter une auth simple Basic Auth ou un token bearer en variable d'env.
```yaml
# docker-compose.yml
ports:
  - "127.0.0.1:8000:8000"   # Au lieu de "8000:8000"
```
Documenter dans le README la procédure pour exposer sur le LAN avec authentification.

#### [DANGER-005] Re-ranking N+1 sans timeout — boucle event loop bloquée et timeout client garanti sur gros corpus
**Fichier :** `app/rag_engine.py:251-326` (couplé à `app/main.py:140-176`)
**Observation :** `_rerank_chunks` invoque le LLM **une fois par chunk** (jusqu'à `top_k * 4 = 48` chunks si filtrage activé). C'est un appel synchrone à `ChatOllama` sans timeout (le défaut LangChain est généralement infini). La méthode est appelée depuis une route `async def query` qui n'utilise pas `run_in_threadpool` ⇒ **toute la boucle event loop FastAPI est bloquée** pendant des dizaines de secondes voire minutes, gelant `/health`, `/api/system/metrics` (rafraîchis toutes les 2s côté front) et toute autre requête.
**Risque :** Healthcheck Docker en échec → restart en boucle, frontend qui se croit déconnecté, timeouts navigateur, expérience inutilisable dès qu'un document volumineux est indexé.
**Solution :**
1. Ajouter un timeout explicite à `ChatOllama(request_timeout=60.0)`.
2. Wrapper l'appel synchrone dans le route async :
```python
from fastapi.concurrency import run_in_threadpool
result = await run_in_threadpool(rag_engine.query, request.question, request.top_k, filter_metadata=filter_metadata)
```
3. Idem pour `index_documents`. À moyen terme, batcher le scoring (un seul prompt qui scorre N chunks d'un coup) ou utiliser un cross-encoder local plutôt que le LLM généraliste.

---

### 🟠 MANQUEMENTS (8 problèmes)

#### [MANQUE-001] Aucune suite de tests
**Fichier :** projet entier
**Observation :** Aucun fichier `test_*.py` ni dossier `tests/`. Aucun `pytest` dans `requirements.txt`. Aucune CI.
**Impact :** Impossible de vérifier les régressions, de valider les correctifs ci-dessus, ou de démontrer la qualité au client.
**Solution :** Ajouter au minimum des tests unitaires sur `DocumentLoader._extract_date/_classify_document`, `StructureAwareSplitter._has_sections/_has_table`, et un test d'intégration `httpx.AsyncClient` sur `/health` et `/api/files` avec data_dir mocké. Cible minimale : 30% de couverture sur le code critique.

#### [MANQUE-002] Aucun `.dockerignore`
**Fichier :** racine
**Observation :** Le `Dockerfile` fait `COPY app/ .` mais en l'absence de `.dockerignore`, des artefacts comme `.git`, `.env`, `__pycache__`, `RAG/` peuvent finir dans le contexte de build (ralentit les builds, peut fuiter des secrets dans les layers).
**Impact :** Builds plus lents, image plus grosse, surface d'exposition accrue.
**Solution :** Créer `.dockerignore` :
```
.git
.gitignore
.env
.env.example
__pycache__
*.pyc
.venv
venv
RAG/
backups/
AUDIT_REPORT.md
```

#### [MANQUE-003] `CHANGELOG.md` référencé mais inexistant
**Fichier :** `README.md:210` pointe vers `CHANGELOG.md` qui n'existe pas dans le repo.
**Impact :** Documentation mensongère, lien mort visible publiquement (le repo est public sur GitHub).
**Solution :** Créer `CHANGELOG.md` avec l'historique 2.5 → 2.6 → 2.7 ou retirer le lien.

#### [MANQUE-004] Aucun logging structuré, uniquement des `print()`
**Fichier :** tout `rag_engine.py`, `document_loader.py`, `main.py:125`
**Observation :** Plus de 20 `print(...)` éparpillés. Pas de niveau (INFO/WARN/ERROR), pas de timestamps fiables (Docker en ajoute mais les corrélations sont impossibles), pas de configuration via variable d'env.
**Impact :** Impossible de filtrer en prod, impossible d'agréger, impossible de désactiver la verbosité.
**Solution :** Remplacer par `logging` Python avec format JSON ou clé=valeur. À minima :
```python
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s %(message)s")
```

#### [MANQUE-005] Variables d'environnement non documentées de manière exhaustive
**Fichier :** `app/config.py` vs `README.md` / `MANAGE.md`
**Observation :** `Settings` expose `ollama_host`, `embedding_model`, `llm_model`, `chunk_size`, `chunk_overlap`, `top_k`, `temperature`, `data_dir`, `index_dir`. Le `.env` ne couvre que 2 variables. Le `docker-compose.yml` en injecte 4. Aucune liste consolidée n'existe.
**Impact :** Le client ne sait pas ce qu'il peut configurer.
**Solution :** Tableau dans le README listant chaque variable, valeur par défaut, plage acceptée, impact.

#### [MANQUE-006] Pas de `graceful shutdown` ni de fermeture explicite des ressources
**Fichier :** `app/main.py:24-29`
**Observation :** Le `lifespan` initialise le moteur RAG mais ne libère rien à l'arrêt (vectorstore, clients httpx). En cas de SIGTERM Docker, on perd silencieusement les requêtes en cours.
**Impact :** Risque de corruption d'index si arrêt pendant une indexation, requêtes utilisateur perdues sans message.
**Solution :** Ajouter dans `lifespan` un sémaphore `indexing_lock` pour bloquer le shutdown pendant une indexation, et logger l'arrêt.

#### [MANQUE-007] Le LLM autodétecté peut être un modèle vision/embedding inadéquat
**Fichier :** `app/rag_engine.py:73-95`
**Observation :** `_detect_llm_model` retourne le **premier** modèle non-embedding rencontré. Sur une machine où Ollama a `llava:latest`, `phi:1b`, `qwen-coder:0.5b`, on choisira au hasard, sans aucun garde-fou de qualité ou de taille.
**Impact :** Réponses RAG potentiellement incohérentes au premier démarrage, sans message clair pour l'utilisateur.
**Solution :** Préférer une liste blanche prioritaire (`mistral`, `qwen2.5`, `llama3.1`, `ministral-3`) et ne tomber sur le premier dispo qu'en dernier recours. Logger explicitement le modèle choisi à l'utilisateur dans l'UI au boot.

#### [MANQUE-008] Aucune validation de taille/MIME sur les uploads vision base64
**Fichier :** `app/main.py:377-384`, `app/rag_engine.py:670-723`
**Observation :** `image_data` peut être un blob base64 de plusieurs Mo, voire des Go, sans vérification. Pas de limite côté FastAPI (`max_request_size` non configuré). Pas de vérification MIME ou de validation que c'est bien une image.
**Impact :** DoS trivial (envoi de payloads géants) et risque que le modèle vision reçoive du non-image.
**Solution :** Limiter via Starlette middleware (`request.body()` avec garde) ou via Pydantic (`max_length` sur `image_data`). Vérifier le préfixe `data:image/(png|jpeg|jpg|webp)` ou décoder + magic-bytes.

---

### 🟡 OPTIMISATIONS (7 points)

#### [OPTI-001] Imports inutilisés dans `main.py` et `rag_engine.py`
**Fichier :** `app/main.py:4-9`, `app/rag_engine.py:3,16`
**Observation :** `import uuid`, `import shutil`, `UploadFile`, `File`, `Path` (jamais utilisés dans main.py). `import os` et `RunnablePassthrough` (jamais utilisés dans rag_engine.py).
**Gain estimé :** Lisibilité, lint propre.
**Solution :** Supprimer les imports morts.

#### [OPTI-002] Six blocs `except:` nus avalent toutes les exceptions
**Fichier :** `app/main.py:92`, `app/document_loader.py:85,99,121`, `app/rag_engine.py:506,665`
**Observation :** Bare `except:` capture aussi `KeyboardInterrupt` et `SystemExit`. Les erreurs réelles sont silencieuses.
**Gain estimé :** Debug plus facile, comportements de shutdown corrects.
**Solution :** Remplacer par `except Exception as e:` et au minimum logger `e`.

#### [OPTI-003] Polling `/api/system/metrics` toutes les 2 secondes en permanence
**Fichier :** `app/templates/index.html:870`
**Observation :** Le frontend appelle `psutil.cpu_percent` + `virtual_memory` + `disk_usage` toutes les 2s même quand l'onglet n'est pas visible.
**Gain estimé :** Réduction CPU continue, surtout sur Apple Silicon où `interval=0.1` cumule du temps CPU.
**Solution :** `setInterval` à 5s, et utiliser `document.visibilityState` pour suspendre quand l'onglet n'est pas visible.

#### [OPTI-004] Re-création complète de l'index FAISS à chaque indexation
**Fichier :** `app/rag_engine.py:191`
**Observation :** `FAISS.from_documents(chunks, ...)` est appelé même quand on ajoute juste un fichier — l'intégralité du corpus est ré-embeddée.
**Gain estimé :** Énorme sur les grosses bibliothèques (× 10-100 plus rapide pour ajouts incrémentaux).
**Solution :** Si l'index existe déjà, utiliser `self.vectorstore.add_documents(new_chunks)` puis `_save_index()`. Maintenir un manifeste des fichiers déjà indexés (hash + mtime) pour éviter les doublons.

#### [OPTI-005] Tailwind et Alpine.js chargés depuis un CDN public
**Fichier :** `app/templates/index.html:7-8`
**Observation :** Dépendances frontend depuis `cdn.tailwindcss.com` et `cdn.jsdelivr.net`. Cela contredit l'argument « 100% local » du projet.
**Gain estimé :** Cohérence avec la promesse privacy-first, fonctionnement hors-ligne.
**Solution :** Vendorer les libs dans `app/static/` et servir localement.

#### [OPTI-006] Métadonnées tronquées à `[:5000]` puis `[:1000]` pour extraction date/année
**Fichier :** `app/document_loader.py:89,112`
**Observation :** `_extract_date` analyse 5000 chars et `_extract_year` 1000 chars. Magique, non documenté, divergent.
**Gain estimé :** Code plus prévisible.
**Solution :** Constante de classe partagée (`METADATA_SCAN_LIMIT = 5000`) et docstring explicative.

#### [OPTI-007] Modèles d'embedding hardcodés dupliqués dans 3 endroits
**Fichier :** `app/main.py:234,279`, `app/rag_engine.py:83`
**Observation :** La liste `{"nomic-embed-text", "mxbai-embed-large", ...}` est répétée trois fois avec des variantes (parfois prefixes seulement, parfois noms complets).
**Gain estimé :** Cohérence, un seul endroit à maintenir.
**Solution :** Constante module-level dans `config.py` : `EMBEDDING_MODEL_PATTERNS: set[str]`.

---

## Plan d'action prioritaire

Ordre recommandé avant livraison demain :

1. **[DANGER-003]** Créer `.gitignore`, `.dockerignore`, `.env.example` ; retirer `.env` du tracking — ⏱️ < 30min
2. **[DANGER-001 + DANGER-002]** Patcher la validation de chemin sur `/api/vision`, `load_specific` et `analyze_image_with_vision` (3 endroits, même pattern `resolve()` + check parent) — 🕐 < 2h
3. **[DANGER-004]** Binder le port 8000 sur `127.0.0.1` par défaut dans `docker-compose.yml`, documenter l'opt-in LAN — ⏱️ < 30min
4. **[DANGER-005]** Wrapper `rag_engine.query()` et `rag_engine.index_documents()` dans `run_in_threadpool` ; ajouter `request_timeout` au `ChatOllama` — 🕐 < 2h
5. **[MANQUE-001 + MANQUE-003 + MANQUE-005]** Ajouter une suite minimale `tests/test_loader.py` + `test_splitter.py` + `test_health.py`, créer `CHANGELOG.md`, compléter le tableau des variables d'env dans le README — 📅 > 2h

Une fois ces cinq blocs traités, l'application passe en zone livrable (score estimé 7/10).

---

## Points positifs

Ce qui est bien fait et ne doit pas être touché :

- **Architecture claire et modulaire** : séparation propre `main.py` (API) / `rag_engine.py` (logique) / `document_loader.py` (I/O) / `structure_aware_splitter.py` (chunking) / `config.py` (settings).
- **Pydantic Settings** correctement utilisé pour la configuration centralisée avec `lru_cache`.
- **Healthcheck Docker** fonctionnel (`/health` testant Ollama).
- **Restart policy `unless-stopped`** correcte pour un déploiement local.
- **Volume nommé `family-rag-index`** persistant indépendamment du conteneur — bonne pratique.
- **`extra_hosts` + `host.docker.internal`** propres pour la connectivité Mac/Windows.
- **Détection automatique d'Ollama** via `entrypoint.sh` avec fallback ordonné — très ergonomique pour l'utilisateur final.
- **Fallback gracieux** dans `_load_index` qui supprime un index corrompu plutôt que de crasher.
- **Re-ranking LLM** : algorithme conceptuellement bon (même si N+1, voir DANGER-005), avec fallback de score FAISS si parsing échoue.
- **Splitter structure-aware** : effort qualitatif réel (sections, listes, tables), avec fallback systématique sur `RecursiveCharacterTextSplitter`.
- **Gestion encodage texte** via `chardet` plutôt que de présumer UTF-8 — correct pour des documents familiaux hétérogènes.
- **OCR multilingue** (`fra+eng`) installé dans le Dockerfile.
- **README** soigné, visuel, avec architecture ASCII et tableau de gains de précision — agréable à lire pour un client final.
- **Licence MIT** présente.
