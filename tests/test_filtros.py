"""Pruebas de la mediana ponderada."""

import numpy as np
import pytest

from procesamiento.filtros import FILTROS, mediana_ponderada, mediana_ponderada_referencia
from procesamiento.parametros import PESOS_MEDIANA_PONDERADA


@pytest.mark.parametrize("semilla", range(5))
def test_vectorizada_igual_a_referencia(semilla):
    rng = np.random.default_rng(semilla)
    canal = rng.integers(0, 256, size=(7, 9), dtype=np.uint8)
    np.testing.assert_array_equal(mediana_ponderada(canal), mediana_ponderada_referencia(canal))


@pytest.mark.parametrize("pesos", [
    [[1, 2, 1], [2, 3, 2], [1, 2, 1]],   # suma impar (15)
    [[2, 1, 2], [1, 2, 1], [2, 1, 2]],   # suma par (14)
    [[1, 1, 1], [1, 1, 1], [1, 1, 1]],   # suma impar (9)
])
def test_vectorizada_igual_a_referencia_con_otros_pesos(pesos):
    rng = np.random.default_rng(11)
    canal = rng.integers(0, 256, size=(7, 9), dtype=np.uint8)
    pesos = np.array(pesos)
    np.testing.assert_array_equal(mediana_ponderada(canal, pesos),
                                  mediana_ponderada_referencia(canal, pesos))


def test_caso_a_mano_empate_sin_fraccion():
    # Rodeo con ceros: el relleno de bordes también es cero, así que no cambia nada.
    canal = np.zeros((5, 5), dtype=np.uint8)
    canal[1:4, 1:4] = [[0, 0, 0], [0, 101, 0], [0, 101, 101]]
    assert mediana_ponderada(canal, PESOS_MEDIANA_PONDERADA)[2, 2] == 50
    assert mediana_ponderada_referencia(canal, PESOS_MEDIANA_PONDERADA)[2, 2] == 50


def test_empate_sin_desborde_uint8():
    canal = np.full((3, 3), 255, dtype=np.uint8)
    assert mediana_ponderada(canal)[1, 1] == 255


def test_pesos_uno_igual_a_mediana_comun():
    rng = np.random.default_rng(7)
    canal = rng.integers(0, 256, size=(7, 9), dtype=np.uint8)
    rellenado = np.pad(canal, 1, mode="constant", constant_values=0)
    esperado = np.empty_like(canal)
    for f in range(canal.shape[0]):
        for c in range(canal.shape[1]):
            esperado[f, c] = np.median(rellenado[f:f + 3, c:c + 3])
    np.testing.assert_array_equal(mediana_ponderada(canal, np.ones((3, 3), dtype=int)), esperado)


def test_salida_uint8_y_registro():
    canal = np.zeros((4, 4), dtype=np.uint8)
    assert mediana_ponderada(canal).dtype == np.uint8
    assert FILTROS["mediana_ponderada"] is mediana_ponderada


def test_pesos_invalidos():
    canal = np.zeros((4, 4), dtype=np.uint8)
    with pytest.raises(ValueError):
        mediana_ponderada(canal, np.array([[1, 1], [1, 1]]))
    with pytest.raises(ValueError):
        mediana_ponderada(canal, np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]]))
