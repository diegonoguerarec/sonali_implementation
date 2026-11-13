"""Prueba completa: consola → procesamiento → archivos, en una carpeta temporal."""

import csv
import json

import cv2
import numpy as np
import pytest

import sonali


@pytest.fixture
def imagen_sintetica(tmp_path):
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, size=(24, 30, 3), dtype=np.uint8)
    ruta = tmp_path / "sintetica.png"
    cv2.imwrite(str(ruta), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return ruta, rgb


@pytest.mark.parametrize("tipo, valor", [("gaussiano", 0.001), ("sal_pimienta", 0.01)])
def test_corrida_completa(tmp_path, imagen_sintetica, tipo, valor):
    ruta, rgb = imagen_sintetica
    salida = tmp_path / "resultado"
    carpeta = sonali.main([str(ruta), tipo, str(valor), "mediana_ponderada"], dir_resultado=salida)

    esperado = f"sintetica__ruido-{tipo}__var-{valor}__filtro-mediana_ponderada__clahe-0.0005__semilla-0"
    assert carpeta == salida / esperado
    archivos = {"ruidosa.png", "solo_filtro.png", "final.png", "metricas.csv", "parametros.json"}
    assert {a.name for a in carpeta.iterdir()} == archivos

    for nombre in ("ruidosa.png", "solo_filtro.png", "final.png"):
        img = cv2.imread(str(carpeta / nombre), cv2.IMREAD_UNCHANGED)
        assert img.shape == rgb.shape and img.dtype == np.uint8

    with (carpeta / "metricas.csv").open(encoding="utf-8") as f:
        filas = list(csv.reader(f))
    assert filas[0] == ["metrica", "ruidosa", "solo_filtro", "final"]
    assert [fila[0] for fila in filas[1:]] == ["PSNR (dB)", "SSIM", "CoC", "EPI"]

    datos = json.loads((carpeta / "parametros.json").read_text(encoding="utf-8"))
    assert datos["ruido"]["tipo"] == tipo
    assert datos["filtro"]["pesos"] == [[1, 1, 1], [1, 4, 1], [1, 1, 1]]
    assert len(datos["procedencia"]) > 0


def test_ruidosa_guardada_en_rgb(tmp_path, imagen_sintetica):
    """La ruidosa con sal y pimienta conserva los valores no afectados en el orden RGB."""
    ruta, rgb = imagen_sintetica
    carpeta = sonali.main([str(ruta), "sal_pimienta", "0.01", "mediana_ponderada"],
                          dir_resultado=tmp_path / "r")
    leida = cv2.cvtColor(cv2.imread(str(carpeta / "ruidosa.png")), cv2.COLOR_BGR2RGB)
    intactos = (leida != 0) & (leida != 255)
    np.testing.assert_array_equal(leida[intactos], rgb[intactos])


@pytest.mark.parametrize("argumentos", [
    ["no_existe.ppm", "gaussiano", "0.001", "mediana_ponderada"],
    ["IMG", "gaussiano", "0", "mediana_ponderada"],
    ["IMG", "sal_pimienta", "1.5", "mediana_ponderada"],
    ["IMG", "gaussiano", "0.001", "mediana_ponderada", "--clip-limit", "0"],
    ["IMG", "gaussiano", "0.001", "mediana_ponderada", "--clip-limit", "1.5"],
    ["IMG", "gaussiano", "0.001", "media"],
])
def test_validaciones(tmp_path, imagen_sintetica, argumentos):
    ruta, _ = imagen_sintetica
    argumentos = [str(ruta) if a == "IMG" else a for a in argumentos]
    with pytest.raises(SystemExit):
        sonali.main(argumentos, dir_resultado=tmp_path / "r")


def test_imagen_gris_rechazada(tmp_path):
    ruta = tmp_path / "gris.png"
    cv2.imwrite(str(ruta), np.zeros((10, 10), dtype=np.uint8))
    with pytest.raises(SystemExit):
        sonali.main([str(ruta), "gaussiano", "0.001", "mediana_ponderada"], dir_resultado=tmp_path / "r")
