"""Pruebas de los filtros candidatos."""

import numpy as np
import pytest

from investigacion import filtros_candidatos as fc
from procesamiento.filtros import mediana_ponderada

RNG = np.random.default_rng(0)
CANAL = RNG.integers(0, 256, size=(9, 11), dtype=np.uint8)


def _por_ventanas(canal, tamano, modo, funcion):
    radio = tamano // 2
    rellenada = np.pad(canal, radio, mode=modo).astype(np.float64)
    salida = np.empty(canal.shape)
    for f in range(canal.shape[0]):
        for c in range(canal.shape[1]):
            salida[f, c] = funcion(rellenada[f:f + tamano, c:c + tamano])
    return salida


@pytest.mark.parametrize("tamano", [3, 5, 7])
@pytest.mark.parametrize("bordes, modo", [("ceros", "constant"), ("symmetric", "symmetric")])
def test_mediana_contra_bucles(tamano, bordes, modo):
    esperado = _por_ventanas(CANAL, tamano, modo, np.median)
    np.testing.assert_array_equal(fc.mediana(CANAL, tamano, bordes), esperado)


@pytest.mark.parametrize("tamano", [3, 5, 7])
@pytest.mark.parametrize("bordes, modo", [("ceros", "constant"), ("replicate", "edge"), ("symmetric", "symmetric")])
def test_media_contra_bucles(tamano, bordes, modo):
    esperado = np.floor(_por_ventanas(CANAL, tamano, modo, np.mean) + 0.5)
    np.testing.assert_array_equal(fc.media(CANAL, tamano, bordes), esperado)


def test_media_a_mano_con_relleno_de_ceros():
    canal = np.full((3, 3), 90, dtype=np.uint8)
    salida = fc.media(canal, 3, "ceros")
    assert salida[1, 1] == 90 and salida[0, 0] == 40 and salida[0, 1] == 60


def test_nucleo_gaussiano_3x3_sigma_05():
    # A mano con la fórmula de fspecial: e^0 = 1, e^-2 = 0.1353, e^-4 = 0.0183;
    # suma = 1 + 4·0.1353 + 4·0.0183 = 1.6146 -> centro 0.6193, lados 0.0838, esquinas 0.0113.
    nucleo = fc.nucleo_gaussiano(3, 0.5)
    assert nucleo.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(nucleo, [[0.0113, 0.0838, 0.0113], [0.0838, 0.6193, 0.0838],
                                        [0.0113, 0.0838, 0.0113]], atol=5e-5)


def test_gaussiano_bordes():
    canal = np.full((8, 8), 200, dtype=np.uint8)
    assert np.all(fc.gaussiano(canal, "imgaussfilt", 5, 1.0) == 200)      # 'replicate' no oscurece
    con_ceros = fc.gaussiano(canal, "fspecial", 3, 1.0)
    assert con_ceros[4, 4] == 200 and con_ceros[0, 0] < 200               # el relleno de ceros sí


def test_wiener_constante_y_formula():
    plano = np.full((10, 10), 77, dtype=np.uint8)
    np.testing.assert_array_equal(fc.wiener(plano, 3)[2:-2, 2:-2], 77)
    g = CANAL.astype(np.float64) / 255
    media = _por_ventanas(g, 3, "constant", np.mean)
    varianza = _por_ventanas(g * g, 3, "constant", np.mean) - media ** 2
    ruido = varianza.mean()
    esperado = media + np.maximum(varianza - ruido, 0) / np.maximum(varianza, ruido) * (g - media)
    np.testing.assert_array_equal(fc.wiener(CANAL, 3), np.floor(np.clip(esperado, 0, 1) * 255 + 0.5))


def test_mediana_ponderada_c1_es_la_mediana_comun():
    np.testing.assert_array_equal(fc.mediana_ponderada_centro(CANAL, 1), fc.mediana(CANAL, 3, "ceros"))
    np.testing.assert_array_equal(fc.mediana_ponderada_centro(CANAL, 4), mediana_ponderada(CANAL))


def test_mediana_literal():
    canal = np.full((5, 5), 120, dtype=np.uint8)
    # W/12: 8 productos valen 10 y el central 40 -> mediana 10.
    assert fc.mediana_ponderada_literal(canal, "W/12")[2, 2] == 10
    # W sin dividir: 8 productos valen 120 y el central satura en 255 -> mediana 120.
    assert fc.mediana_ponderada_literal(canal, "W_saturada")[2, 2] == 120


def test_aplicar_filtro_rgb():
    imagen = RNG.integers(0, 256, size=(9, 11, 3), dtype=np.uint8)
    for filtro, candidatos in fc.CANDIDATOS.items():
        for parametros in candidatos:
            salida = fc.aplicar_filtro(imagen, filtro, parametros)
            assert salida.shape == imagen.shape and salida.dtype == np.uint8
