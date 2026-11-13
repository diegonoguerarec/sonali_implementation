"""Pruebas de PSNR, SSIM, CoC y EPI."""

import math

import numpy as np
import pytest

from procesamiento.metricas import calcular_metricas, coc, epi, psnr, ssim


def test_imagenes_identicas():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(32, 40, 3), dtype=np.uint8)
    m = calcular_metricas(img, img.copy())
    assert m["PSNR (dB)"] == math.inf
    assert m["SSIM"] == pytest.approx(1.0)
    assert m["CoC"] == pytest.approx(1.0)
    assert m["EPI"] == pytest.approx(1.0)


def test_psnr_100_contra_110():
    ref = np.full((10, 10, 3), 100, dtype=np.uint8)
    otra = np.full((10, 10, 3), 110, dtype=np.uint8)
    assert psnr(otra, ref) == pytest.approx(28.1308, abs=1e-4)


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
