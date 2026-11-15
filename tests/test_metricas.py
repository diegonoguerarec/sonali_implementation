"""Pruebas de PSNR, SSIM, CoC y EPI."""

import math

import numpy as np
import pytest

from procesamiento.metricas import a_gris, calcular_metricas, coc, epi, psnr, ssim


def test_imagenes_identicas():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(32, 40, 3), dtype=np.uint8)
    m = calcular_metricas(img, img.copy())
    assert m["PSNR (dB)"] == math.inf
    assert m["SSIM"] == pytest.approx(1.0)
    assert m["CoC"] == pytest.approx(1.0)
    assert m["EPI"] == pytest.approx(1.0)


def test_gris_de_matlab():
    imagen = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255], [10, 20, 30]]], dtype=np.uint8)
    # 76.23 -> 76; 149.70 -> 150; 29.08 -> 29; 2.989 + 11.741 + 3.421 = 18.15 -> 18.
    np.testing.assert_array_equal(a_gris(imagen), [[76, 150, 29, 18]])


def test_psnr_en_gris_con_pico_por_rango():
    # Canales iguales: el gris es el mismo valor. Referencia con 0 y 200 (pico 200);
    # la otra está 10 niveles arriba (MSE = 100): 10·log10(200² / 100) = 26.0206 dB.
    ref = np.zeros((10, 10, 3), dtype=np.uint8)
    ref[:, 5:] = 200
    otra = ref + 10
    assert psnr(otra, ref) == pytest.approx(26.0206, abs=1e-4)


def test_psnr_con_referencia_constante_devuelve_nan():
    ref = np.full((5, 5, 3), 100, dtype=np.uint8)
    with pytest.warns(UserWarning):
        assert math.isnan(psnr(ref + 10, ref))


def test_epi_a_mano():
    ref = np.zeros((2, 3, 3), dtype=np.uint8)
    otra = np.zeros((2, 3, 3), dtype=np.uint8)
    ref[:, :, :] = np.array([0, 10, 20])[None, :, None]
    otra[:, :, :] = np.array([0, 20, 40])[None, :, None]
    assert epi(otra, ref) == pytest.approx(2.0)


def test_denominador_cero_devuelve_nan_con_aviso():
    constante = np.full((5, 5, 3), 7, dtype=np.uint8)
    with pytest.warns(UserWarning):
        assert math.isnan(coc(constante, constante))
    with pytest.warns(UserWarning):
        assert math.isnan(epi(constante, constante))


def test_ssim_menor_a_uno_con_ruido():
    rng = np.random.default_rng(1)
    ref = rng.integers(0, 256, size=(32, 40, 3), dtype=np.uint8)
    otra = np.clip(ref.astype(int) + rng.integers(-30, 31, size=ref.shape), 0, 255).astype(np.uint8)
    assert ssim(otra, ref) < 1.0
