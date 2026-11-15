"""Pruebas de CLAHE al estilo de adapthisteq."""

import numpy as np
import pytest

from procesamiento.realce import aplicar_clahe, datos_clahe, lado_tesela, pixeles_por_tesela, recortar_histograma


def test_pixeles_por_tesela_stare():
    assert pixeles_por_tesela(605, 700) == 6688


def test_lado_de_tesela_par():
    assert lado_tesela(605, 8) == 76      # ceil(605/8) = 76, ya es par
    assert lado_tesela(700, 8) == 88      # ceil(700/8) = 88
    assert lado_tesela(600, 8) == 76      # 75 es impar: sube a 76
    assert lado_tesela(65, 8) == 10       # ceil(65/8) = 9, impar


@pytest.mark.parametrize("clip, cuentas", [(0.0005, 30), (0.01, 94)])
def test_clip_en_cuentas(clip, cuentas):
    datos = datos_clahe(clip, 605, 700)
    assert datos["lados_tesela"] == (76, 88)
    assert datos["pixeles_por_tesela"] == 6688
    assert datos["clip_cuentas"] == cuentas


@pytest.mark.parametrize("semilla", range(4))
def test_recorte_no_supera_el_clip_y_conserva_las_cuentas(semilla):
    rng = np.random.default_rng(semilla)
    valores = np.clip(rng.normal(30 + 40 * semilla, 5 + 8 * semilla, size=6688), 0, 255).astype(np.uint8)
    histograma = np.bincount(valores, minlength=256)
    for clip in (27, 30, 94):
        recortado = recortar_histograma(histograma, clip)
        assert recortado.max() <= clip
        assert recortado.sum() == histograma.sum()


def test_recorte_a_mano():
    # 4 bins, 20 cuentas, clip 6: el exceso es 8 (todo del primer bin) y el incremento, 8 // 4 = 2.
    # El primer bin queda en 6; el de 5 se llena hasta 6 (gasta 1); los de 1 y 0 suman 2 (gastan 4)
    # y quedan en 3 y 2. Sobran 3, que van de a 1 a los bins bajo el clip: 3 -> 4 -> 5 y 2 -> 3.
    np.testing.assert_array_equal(recortar_histograma(np.array([14, 5, 1, 0]), 6), [6, 6, 5, 3])


def test_aplicar_clahe_uint8():
    rng = np.random.default_rng(0)
    canal = rng.integers(0, 256, size=(64, 80), dtype=np.uint8)
    salida = aplicar_clahe(canal, 0.0005)
    assert salida.dtype == np.uint8 and salida.shape == canal.shape


def test_imagen_constante_queda_constante():
    canal = np.full((61, 70), 100, dtype=np.uint8)
    assert np.unique(aplicar_clahe(canal, 0.0005)).size == 1
