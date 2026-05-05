"""Configuration pytest — chemin et répertoire de travail."""

import os
import sys
from pathlib import Path

APP_DIR = Path(__file__).parent.parent / "app"

# Rend les modules de app/ importables sans modifier PYTHONPATH en dehors des tests
sys.path.insert(0, str(APP_DIR))

# StaticFiles et Jinja2Templates de Starlette cherchent leurs dossiers
# relativement au cwd — on pointe donc sur app/ avant tout import de main
os.chdir(APP_DIR)
