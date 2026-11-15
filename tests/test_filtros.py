"""Pruebas de la mediana ponderada."""

import numpy as np
import pytest

from procesamiento.filtros import (FILTROS, gaussiano, media, mediana, mediana_ponderada,
                                   mediana_ponderada_referencia, nucleo_gaussiano, wiener)
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


# --- Mediana, media, gaussiano y Wiener ---------------------------------------

CANAL = np.random.default_rng(21).integers(0, 256, size=(9, 11), dtype=np.uint8)


def _por_ventanas(canal, tamano, funcion):
    """Aplica `funcion` a cada ventana tamano×tamano con relleno de ceros (bucles explícitos)."""
    radio = tamano // 2
    rellenada = np.pad(canal, radio, mode="constant").astype(np.float64)
    salida = np.empty(canal.shape)
    for f in range(canal.shape[0]):
        for c in range(canal.shape[1]):
            salida[f, c] = funcion(rellenada[f:f + tamano, c:c + tamano])
    return salida


def test_registro_tiene_los_cinco_filtros():
    assert set(FILTROS) == {"mediana", "media", "gaussiano", "wiener", "mediana_ponderada"}
    for funcion in FILTROS.values():
        salida = funcion(CANAL)
        assert salida.dtype == np.uint8 and salida.shape == CANAL.shape


def test_mediana_5x5_contra_bucles():
    np.testing.assert_array_equal(mediana(CANAL), _por_ventanas(CANAL, 5, np.median))


def test_media_5x5_contra_bucles():
    np.testing.assert_array_equal(media(CANAL), np.floor(_por_ventanas(CANAL, 5, np.mean) + 0.5))


def test_media_a_mano_con_relleno_de_ceros():
    # Imagen 5×5 toda en 100: en el centro entran los 25 valores (100); en la esquina solo
    # 9 de los 25 (los otros 16 son ceros de relleno): 900 / 25 = 36.
    canal = np.full((5, 5), 100, dtype=np.uint8)
    salida = media(canal)
    assert salida[2, 2] == 100 and salida[0, 0] == 36


def test_nucleo_gaussiano_3x3_sigma_05():
    # A mano: e^0 = 1, e^-2 = 0.1353, e^-4 = 0.0183; suma = 1 + 4·0.1353 + 4·0.0183 = 1.6146,
    # así que el centro vale 0.6193, los lados 0.0838 y las esquinas 0.0113.
    nucleo = nucleo_gaussiano(3, 0.5)
    np.testing.assert_allclose(nucleo, [[0.0113, 0.0838, 0.0113], [0.0838, 0.6193, 0.0838],
                                        [0.0113, 0.0838, 0.0113]], atol=5e-5)


def test_gaussiano_contra_bucles():
    nucleo = nucleo_gaussiano(3, 0.5)
    esperado = np.floor(_por_ventanas(CANAL, 3, lambda v: float((v * nucleo).sum())) + 0.5)
    np.testing.assert_array_equal(gaussiano(CANAL), esperado)


def test_wiener_contra_la_formula():
    g = CANAL.astype(np.float64) / 255
    media_local = _por_ventanas(g, 5, np.mean)
    varianza = _por_ventanas(g * g, 5, np.mean) - media_local ** 2
    ruido = varianza.mean()
    esperado = media_local + np.maximum(varianza - ruido, 0) / np.maximum(varianza, ruido) * (g - media_local)
    np.testing.assert_array_equal(wiener(CANAL), np.floor(np.clip(esperado, 0, 1) * 255 + 0.5))


def test_wiener_imagen_constante():
    canal = np.full((12, 12), 77, dtype=np.uint8)
    np.testing.assert_array_equal(wiener(canal)[3:-3, 3:-3], 77)
    assert wiener(np.zeros((6, 6), dtype=np.uint8)).max() == 0      # varianza 0 en todos lados
