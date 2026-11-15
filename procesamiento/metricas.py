"""Métricas de calidad: PSNR, SSIM, CoC y EPI.

Todas se calculan en float64 contra la imagen limpia. PSNR y SSIM usan la
imagen en gris (`rgb2gray` de MATLAB); CoC y EPI, la imagen RGB completa
(alto × ancho × 3). El dominio de PSNR y SSIM y el pico del PSNR son los que
mejor reproducen las tablas del paper (investigacion/INFORME.md). Si algún
denominador da cero, devuelven NaN y avisan.
"""

import warnings

import cv2
import numpy as np

from procesamiento import parametros as p


def _nan_con_aviso(metrica: str) -> float:
    """Emite un aviso de denominador cero y devuelve NaN."""
    warnings.warn(f"{metrica}: el denominador da cero (imagen constante); se devuelve NaN.")
    return float("nan")


def a_gris(imagen: np.ndarray) -> np.ndarray:
    """`rgb2gray` de MATLAB: suma ponderada de R, G y B, redondeada a uint8.

    https://www.mathworks.com/help/matlab/ref/rgb2gray.html
    """
    suma = imagen.astype(np.float64) @ np.array(p.COEFICIENTES_GRIS)
    return np.floor(suma + p.DESPLAZAMIENTO_REDONDEO).astype(np.uint8)


def psnr(imagen: np.ndarray, referencia: np.ndarray) -> float:
    """PSNR = 10·log10(pico²/MSE) sobre la imagen en gris (ecuaciones 2 y 3).

    El pico es "the maximum difference in the input image" (texto de la
    ecuación 2): máximo − mínimo de la referencia en gris. Si MSE = 0, devuelve
    infinito; si la referencia es constante, NaN.
    """
    x = a_gris(imagen).astype(np.float64)
    y = a_gris(referencia).astype(np.float64)
    mse = float(np.mean((x - y) ** 2))
    if mse == 0:
        return float("inf")
    pico = float(y.max() - y.min())
    if pico == 0:
        return _nan_con_aviso("PSNR")
    return float(p.FACTOR_DECIBELES * np.log10(pico ** 2 / mse))


def _filtro_gaussiano_ssim(x: np.ndarray) -> np.ndarray:
    """Ventana gaussiana 11×11, σ = 1.5, bordes replicados."""
    tamano = (p.TAMANO_VENTANA_SSIM, p.TAMANO_VENTANA_SSIM)
    return cv2.GaussianBlur(x, tamano, p.SIGMA_SSIM, sigmaY=p.SIGMA_SSIM,
                            borderType=cv2.BORDER_REPLICATE)


def ssim(imagen: np.ndarray, referencia: np.ndarray) -> float:
    """SSIM con los valores por defecto de `ssim` de MATLAB (ecuaciones 4 a 7).

    C1 = (0.01·255)², C2 = (0.03·255)², C3 = C2/2 y exponentes 1, que dan
    ((2μxμy + C1)(2σxy + C2)) / ((μx² + μy² + C1)(σx² + σy² + C2)).
    Se calcula sobre la imagen en gris (`rgb2gray`); el resultado es el promedio
    del mapa alto × ancho.
    """
    x = a_gris(imagen).astype(np.float64)
    y = a_gris(referencia).astype(np.float64)
    c1 = (p.K1_SSIM * p.RANGO_DINAMICO_SSIM) ** 2
    c2 = (p.K2_SSIM * p.RANGO_DINAMICO_SSIM) ** 2
    mu_x = _filtro_gaussiano_ssim(x)
    mu_y = _filtro_gaussiano_ssim(y)
    var_x = _filtro_gaussiano_ssim(x * x) - mu_x ** 2
    var_y = _filtro_gaussiano_ssim(y * y) - mu_y ** 2
    cov_xy = _filtro_gaussiano_ssim(x * y) - mu_x * mu_y
    k = p.FACTOR_NUMERADOR_SSIM
    numerador = (k * mu_x * mu_y + c1) * (k * cov_xy + c2)
    denominador = (mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2)
    return float(np.mean(numerador / denominador))


def coc(imagen: np.ndarray, referencia: np.ndarray) -> float:
    """Coeficiente de correlación de Pearson entre todos los valores RGB (ecuación 8)."""
    x = imagen.astype(np.float64).ravel()
    y = referencia.astype(np.float64).ravel()
    dx = x - x.mean()
    dy = y - y.mean()
    denominador = np.sqrt(np.sum(dx * dx) * np.sum(dy * dy))
    if denominador == 0:
        return _nan_con_aviso("CoC")
    return float(np.sum(dx * dy) / denominador)


def _suma_diferencias_horizontales(imagen: np.ndarray) -> float:
    """Suma de |X(m, n+1) − X(m, n)| en todas las filas, columnas y canales."""
    x = imagen.astype(np.float64)
    return float(np.sum(np.abs(x[:, 1:] - x[:, :-1])))


def epi(imagen: np.ndarray, referencia: np.ndarray) -> float:
    """EPI de la ecuación 9: diferencias horizontales de la evaluada / de la limpia.

    Suma los 3 canales. Puede pasar de 1, por ejemplo si el ruido agrega bordes.
    """
    denominador = _suma_diferencias_horizontales(referencia)
    if denominador == 0:
        return _nan_con_aviso("EPI")
    return _suma_diferencias_horizontales(imagen) / denominador


# Orden y nombres de las métricas en la tabla y el CSV.
METRICAS = {
    "PSNR (dB)": psnr,
    "SSIM": ssim,
    "CoC": coc,
    "EPI": epi,
}


def calcular_metricas(imagen: np.ndarray, referencia: np.ndarray) -> dict[str, float]:
    """Calcula todas las métricas de `imagen` contra `referencia`."""
    return {nombre: funcion(imagen, referencia) for nombre, funcion in METRICAS.items()}
