"""Pruebas de la conversión del clip limit de CLAHE."""

import numpy as np
import pytest

from procesamiento.realce import aplicar_clahe, clip_limit_matlab_a_opencv, pixeles_por_tesela


def test_pixeles_por_tesela_stare():
    assert pixeles_por_tesela(605, 700) == 6688


@pytest.mark.parametrize("clip, cuentas, opencv", [
    (0.0005, 30, 1.167464),
    (0.01, 94, 3.617225),
])
def test_conversion_clip_limit(clip, cuentas, opencv):
    datos = clip_limit_matlab_a_opencv(clip, 605, 700)
    assert datos["pixeles_por_tesela"] == 6688
    assert datos["clip_cuentas"] == cuentas
    assert datos["clip_limit_opencv"] == pytest.approx(opencv, abs=1e-6)
    assert int(datos["clip_limit_opencv"] * 6688 / 256) == cuentas


def test_aplicar_clahe_uint8():
    rng = np.random.default_rng(0)
    canal = rng.integers(0, 256, size=(64, 80), dtype=np.uint8)
    salida = aplicar_clahe(canal, 0.0005)
    assert salida.dtype == np.uint8 and salida.shape == canal.shape
