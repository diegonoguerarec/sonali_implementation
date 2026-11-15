"""Pruebas del ruido candidato, de las métricas candidatas y de la lectura de las tablas."""

import math

import numpy as np
import pytest

from investigacion import busqueda
from investigacion import metricas_candidatas as mc
from investigacion import ruido_candidatos as rc
from procesamiento import metricas as metricas_proyecto

RNG = np.random.default_rng(3)
REF = RNG.integers(0, 256, size=(40, 48, 3), dtype=np.uint8)
OTRA = np.clip(REF.astype(int) + RNG.integers(-25, 26, size=REF.shape), 0, 255).astype(np.uint8)


def test_gaussiano_con_media():
    gris = np.full((200, 200, 3), 100, dtype=np.uint8)
    salida = rc.agregar_ruido_candidato(gris, "gaussiano", 0.2, 0, "media").astype(np.float64) / 255
    assert salida.mean() - 100 / 255 == pytest.approx(0.2, abs=0.003)
    assert salida.var() == pytest.approx(0.01, rel=0.05)


def test_gaussiano_con_varianza():
    gris = np.full((200, 200, 3), 128, dtype=np.uint8)
    salida = rc.agregar_ruido_candidato(gris, "gaussiano", 0.01, 0, "varianza").astype(np.float64) / 255
    assert salida.mean() - 128 / 255 == pytest.approx(0.0, abs=0.003)
    assert salida.var() == pytest.approx(0.01, rel=0.05)


def test_el_proyecto_usa_la_configuracion_propuesta():
    from procesamiento.realce import aplicar_clahe
    from procesamiento.ruido import agregar_ruido
    from investigacion import clahe_matlab
    for tipo, nivel in (("gaussiano", 0.01), ("sal_pimienta", 0.1)):
        np.testing.assert_array_equal(rc.agregar_ruido_candidato(REF, tipo, nivel, 5, "media"),
                                      agregar_ruido(REF, tipo, nivel, 5))
    canal = REF[:, :, 1]
    np.testing.assert_array_equal(aplicar_clahe(canal, 0.0005), clahe_matlab.adapthisteq(canal, 0.0005))
    todas = mc.calcular_todas(OTRA, REF, OTRA)
    assert todas["PSNR", "gris|rango"] == pytest.approx(metricas_proyecto.psnr(OTRA, REF))
    assert todas["SSIM", "plano|gris"] == pytest.approx(metricas_proyecto.ssim(OTRA, REF))
    assert todas["CoC", "rgb|limpia"] == pytest.approx(metricas_proyecto.coc(OTRA, REF))
    assert todas["EPI", "rgb"] == pytest.approx(metricas_proyecto.epi(OTRA, REF))
    assert set(todas) == {(m, v) for m, variantes in mc.VARIANTES.items() for v in variantes}


def test_imagenes_identicas():
    todas = mc.calcular_todas(REF, REF, REF)
    for (metrica, _), valor in todas.items():
        assert valor == (math.inf if metrica == "PSNR" else pytest.approx(1.0))


def test_gris_de_matlab():
    imagen = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255], [10, 20, 30]]], dtype=np.uint8)
    # 76.23 -> 76; 149.70 -> 150; 29.08 -> 29; 2.989 + 11.741 + 3.421 = 18.15 -> 18.
    np.testing.assert_array_equal(mc.a_gris(imagen), [[76, 150, 29, 18]])


def test_psnr_con_pico_por_rango():
    ref = np.zeros((4, 4, 3), dtype=np.uint8)
    ref[0, 0] = 100
    otra = ref.copy()
    otra[1, 1] = 10                      # MSE = 3·100/48 = 6.25
    x, y = mc.planos(otra), mc.planos(ref)
    assert mc.psnr(x["rgb"], y["rgb"], "255") == pytest.approx(10 * math.log10(255 ** 2 / 6.25))
    assert mc.psnr(x["rgb"], y["rgb"], "rango") == pytest.approx(10 * math.log10(100 ** 2 / 6.25))


def test_ssim_wang_es_el_interior_del_mapa():
    x, y = mc.planos(OTRA), mc.planos(REF)
    mapa = mc.mapa_ssim(mc.momentos_ssim(x["rgb"][0], y["rgb"][0]))
    assert mc.variantes_ssim(x, y)["wang|canal"] == pytest.approx(mapa[5:-5, 5:-5].mean())


def test_ssim_volumen_mezcla_canales():
    mezcla = mc._mezcla_canales()
    np.testing.assert_allclose(mezcla.sum(axis=1), 1.0)
    np.testing.assert_allclose(mezcla, mezcla[::-1, ::-1])
    # Con los tres canales iguales, el volumen no cambia nada respecto del plano.
    gris3 = np.repeat(REF[:, :, :1], 3, axis=2)
    otra3 = np.repeat(OTRA[:, :, :1], 3, axis=2)
    variantes = mc.variantes_ssim(mc.planos(otra3), mc.planos(gris3))
    assert variantes["volumen|rgb"] == pytest.approx(variantes["plano|rgb"])


def test_coc_y_epi_por_dominio():
    x, y = mc.planos(OTRA), mc.planos(REF)
    por_canal = np.mean([np.corrcoef(OTRA[:, :, c].ravel(), REF[:, :, c].ravel())[0, 1] for c in range(3)])
    assert mc.coc(x["canal"], y["canal"]) == pytest.approx(por_canal)
    assert mc.coc(x["rgb"], y["rgb"]) == pytest.approx(np.corrcoef(OTRA.ravel(), REF.ravel())[0, 1])
    ref = np.zeros((2, 3, 3), dtype=np.uint8)
    otra = np.zeros((2, 3, 3), dtype=np.uint8)
    ref[:] = np.array([0, 10, 20])[None, :, None]
    otra[:] = np.array([0, 20, 40])[None, :, None]
    for dominio in mc.DOMINIOS:
        assert mc.epi(mc.planos(otra)[dominio], mc.planos(ref)[dominio]) == pytest.approx(2.0, rel=0.02)


def test_tablas_del_paper():
    filas = busqueda.cargar_tablas()
    assert len(filas) == 208
    assert sum(f["tabla"] in busqueda.TABLAS_CALIBRACION for f in filas) == 160
    desvios = busqueda.desvios_del_paper(filas)
    assert len(desvios) == 12 and all(d > 0 for d in desvios.values())
    distintas = busqueda.diferencias_csv_pdf(filas)
    if distintas is None:
        pytest.skip("pdftotext no está instalado")
    assert distintas == 0


def test_categoria():
    assert busqueda.categoria("PSNR", 0.02, 0.02) == "coincide"
    assert busqueda.categoria("PSNR", -0.4, 0.01) == "cerca"
    assert busqueda.categoria("SSIM", 0.02, 0.001) == "no coincide"


def test_tarea_completa_chica(monkeypatch):
    chica = RNG.integers(0, 256, size=(64, 80, 3), dtype=np.uint8)
    monkeypatch.setitem(busqueda._LIMPIAS, "prueba", chica)
    config = dict(busqueda.CONFIG_INICIAL)
    tarea = busqueda.crear_tarea(config, "mediana", "sal_pimienta", 0.01, imagen="prueba")
    misma, valores = busqueda.evaluar_tarea(tarea)
    assert misma == tarea and math.isfinite(valores["PSNR", "rgb|255"])
    sin_nada = busqueda.crear_tarea(config, "ninguno", "ninguno", 0.0, imagen="prueba", clahe=None)
    assert busqueda.evaluar_tarea(sin_nada)[1]["PSNR", "rgb|255"] == math.inf
