"""Hace importable el paquete `procesamiento` y `sonali.py` desde las pruebas."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
