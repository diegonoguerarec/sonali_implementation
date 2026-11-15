"""Hace importables `procesamiento` e `investigacion` desde las pruebas de la investigación."""

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))
