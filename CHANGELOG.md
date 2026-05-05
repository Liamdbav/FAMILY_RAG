# Changelog

Toutes les modifications notables de ce projet sont documentées dans ce fichier.

Format basé sur [Keep a Changelog](https://keepachangelog.com/fr/1.0.0/),
versionnage selon [Semantic Versioning](https://semver.org/lang/fr/).

---

## [Unreleased]

### Ajouté
- `.gitignore` — exclusion des secrets, caches Python et index vectoriel
- `.dockerignore` — réduction du contexte de build Docker
- `.env.example` — gabarit documenté de toutes les variables d'environnement
- `CHANGELOG.md` — journal des modifications (ce fichier)

### Modifié
- `docker-compose.yml` — binding du port 8000 restreint à `127.0.0.1` (accès local uniquement)
