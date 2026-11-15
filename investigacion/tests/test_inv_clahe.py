"""Pruebas del `adapthisteq` reconstruido."""

import numpy as np
import pytest

from investigacion import clahe_matlab as cm


def test_teselas_y_clip_de_stare():
    datos = cm.datos_clahe(605, 700, 0.0005)
    assert (datos["alto_tesela"], datos["ancho_tesela"]) == (76, 88)
    assert datos["pixeles_por_tesela"] == 6688
    assert datos["clip_cuentas"] == 30
    assert cm.datos_clahe(605, 700, 0.01)["clip_cuentas"] == 94


def test_tesela_siempre_par():
    assert cm.tamano_tesela(600, 8) == 76      # 75 es impar: sube a 76
    assert cm.tamano_tesela(64, 8) == 8
    assert cm.tamano_tesela(65, 8) == 10       # ceil(65/8) = 9, impar


@pytest.mark.parametrize("semilla", range(6))
def test_recorte_cumple_el_tope_y_conserva_las_cuentas(semilla):
    rng = np.random.default_rng(semilla)
    # Histogramas concentrados, como los del canal azul de un fondo de ojo.
    valores = np.clip(rng.normal(20 + 30 * semilla, 4 + 6 * semilla, size=6688), 0, 255).astype(np.uint8)
    histograma = np.bincount(valores, minlength=256)
    for clip in (27, 30, 94):
        recortado = cm.recortar_histograma(histograma, clip)
        assert recortado.max() <= clip
        assert recortado.sum() == histograma.sum()
        np.testing.assert_array_equal(recortado, cm.recortar_histograma_referencia(histograma, clip))


def test_recorte_a_mano():
    # 4 bins, 20 cuentas, clip 6: exceso 8 (todo del bin 0); incremento 8 // 4 = 2.
    # Bin 0 queda en 6; el bin de 5 está a menos de 2 del clip y se llena (gasta 1);
    # los bins de 1 y 0 suman 2 cada uno (gastan 4) y quedan en 3 y 2. Sobran 3,
    # que se reparten de a 1 entre los bins que siguen bajo el clip.
    recortado = cm.recortar_histograma(np.array([14, 5, 1, 0]), 6)
    np.testing.assert_array_equal(recortado, cm.recortar_histograma_referencia(np.array([14, 5, 1, 0]), 6))
    assert recortado.sum() == 20 and recortado.max() <= 6


def test_imagen_constante_y_forma():
    canal = np.full((605, 700), 100, dtype=np.uint8)
    salida = cm.adapthisteq(canal, 0.0005)
    assert salida.shape == canal.shape and salida.dtype == np.uint8
    assert np.unique(salida).size == 1


def test_tabla_es_no_decreciente_y_llega_al_maximo():
    rng = np.random.default_rng(0)
    histograma = np.bincount(rng.integers(0, 256, size=6688), minlength=256)
    tabla = cm.tabla_de_tesela(cm.recortar_histograma(histograma, 30), 6688, "redondeada")
    assert np.all(np.diff(tabla) >= 0) and tabla[-1] == 255
    assert np.array_equal(tabla, np.floor(tabla))


@pytest.mark.parametrize("tabla", cm.VARIANTES_TABLA)
@pytest.mark.parametrize("final", cm.VARIANTES_FINAL)
@pytest.mark.parametrize("relleno", cm.VARIANTES_RELLENO)
def test_variantes_difieren_poco(tabla, final, relleno):
    rng = np.random.default_rng(1)
    canal = rng.integers(0, 256, size=(61, 70), dtype=np.uint8)
    base = cm.adapthisteq(canal, 0.01).astype(int)
    variante = cm.adapthisteq(canal, 0.01, tabla, final, relleno).astype(int)
    assert variante.shape == base.shape
    if relleno == "centrado":
        assert np.abs(variante - base).max() <= 2
