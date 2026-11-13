"""Pruebas del ruido gaussiano y de sal y pimienta."""

import numpy as np
import pytest

from procesamiento.ruido import a_uint8_matlab, agregar_ruido

GRIS = np.full((200, 200, 3), 128, dtype=np.uint8)


def test_sal_pimienta_fracciones():
    salida = agregar_ruido(GRIS, "sal_pimienta", 0.1, 0)
    assert np.mean(salida == 0) == pytest.approx(0.05, abs=0.005)
    assert np.mean(salida == 255) == pytest.approx(0.05, abs=0.005)
    assert set(np.unique(salida)) <= {0, 128, 255}


def test_gaussiano_varianza():
    salida = agregar_ruido(GRIS, "gaussiano", 0.01, 0)
    diferencia = salida.astype(np.float64) / 255 - 128 / 255
    assert np.var(diferencia) == pytest.approx(0.01, rel=0.05)
    assert salida.dtype == np.uint8


@pytest.mark.parametrize("tipo, valor", [("gaussiano", 0.01), ("sal_pimienta", 0.1)])
def test_semilla(tipo, valor):
    a = agregar_ruido(GRIS, tipo, valor, 3)
    b = agregar_ruido(GRIS, tipo, valor, 3)
    c = agregar_ruido(GRIS, tipo, valor, 4)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_redondeo_matlab_empates_hacia_arriba():
    # 0.5/255 y 2.5/255 son empates: MATLAB da 1 y 3 (np.round daría 0 y 2).
    x = np.array([0.5, 2.5, 254.5]) / 255
    np.testing.assert_array_equal(a_uint8_matlab(x), [1, 3, 255])
