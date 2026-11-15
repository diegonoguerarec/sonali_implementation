"""Candidatos para las métricas (sección 4.4 del prompt 01).

Dominios:
- "rgb":   todos los valores RGB juntos (la del proyecto).
- "canal": la métrica de cada canal, promediada.
- "gris":  `rgb2gray` de MATLAB: 0.298936021293775·R + 0.587043074451121·G +
           0.114020904255103·B, redondeado a uint8.
           https://www.mathworks.com/help/matlab/ref/rgb2gray.html

Variantes (clave de cada una entre comillas):
- PSNR "dominio|pico": pico 255 (por defecto en `psnr` para uint8,
  https://www.mathworks.com/help/images/ref/psnr.html) o "rango" = max − min de la
  referencia ("maximum difference in the input image", texto de la Ec. 2).
- SSIM:
  "plano|rgb"   gaussiana 11×11, σ = 1.5, bordes replicados, plano por plano (proyecto).
                Coincide con "canal": el promedio del mapa es el promedio de los planos.
  "plano|gris"  lo mismo sobre la imagen gris.
  "volumen|rgb" el arreglo M×N×3 tratado como volumen 3-D, que es lo que hace
                `ssim(A, ref)` sin DataFormat (equivale a "SSS"):
                https://www.mathworks.com/matlabcentral/answers/2142916-ssim-calculation-for-rgb-images
                La documentación actual dice que acepta "3-D grayscale volume"
                (https://www.mathworks.com/help/images/ref/ssim.html). NO VERIFICADO:
                la documentación de R2016a no está en línea; se asume gaussiana
                isotrópica 3-D de radio ceil(3σ) con bordes replicados.
  "wang|canal"  `ssim_index.m` de Wang et al. (2004): gaussiana 11×11, σ = 1.5,
                filtrado 'valid', K = [0.01 0.03], L = 255; por canal y promediado.
  "wang|gris"   lo mismo sobre la imagen gris.
- CoC "dominio|referencia": Pearson (Ec. 8; `corr2`,
  https://www.mathworks.com/help/images/ref/corr2.html) contra la imagen "limpia" o
  contra la "ruidosa" ("original degraded image", texto de la Ec. 8).
- EPI "dominio": Ec. 9 (con valor absoluto en numerador y denominador, según la
  página renderizada del PDF).
"""

import cv2
import numpy as np

from procesamiento import parametros as p

DOMINIOS: tuple[str, ...] = ("rgb", "canal", "gris")

# [MATLAB] rgb2gray (documentación, sección Algorithms).
COEFICIENTES_GRIS: tuple[float, float, float] = (0.298936021293775, 0.587043074451121, 0.114020904255103)

VARIANTES: dict[str, list[str]] = {
    "PSNR": [f"{d}|{pico}" for d in DOMINIOS for pico in ("255", "rango")],
    "SSIM": ["plano|rgb", "plano|gris", "volumen|rgb", "wang|canal", "wang|gris"],
    "CoC": [f"{d}|{ref}" for d in DOMINIOS for ref in ("limpia", "ruidosa")],
    "EPI": list(DOMINIOS),
}


def a_gris(imagen: np.ndarray) -> np.ndarray:
    """`rgb2gray` de una imagen RGB uint8 (suma ponderada redondeada)."""
    suma = imagen.astype(np.float64) @ np.array(COEFICIENTES_GRIS)
    return np.floor(suma + p.DESPLAZAMIENTO_REDONDEO).astype(np.uint8)


def planos(imagen: np.ndarray) -> dict[str, list[np.ndarray]]:
    """Arreglos float64 de cada dominio: {"rgb": [RGB], "canal": [R, G, B], "gris": [gris]}."""
    rgb = imagen.astype(np.float64)
    return {
        "rgb": [rgb],
        "canal": [rgb[:, :, c] for c in range(imagen.shape[2])],
        "gris": [a_gris(imagen).astype(np.float64)],
    }


def psnr(x: list[np.ndarray], y: list[np.ndarray], pico: str) -> float:
    """PSNR = 10·log10(pico²/MSE) (Ecs. 2 y 3), promediado sobre los planos del dominio."""
    valores = []
    for a, b in zip(x, y):
        mse = float(np.mean((a - b) ** 2))
        maximo = p.PICO_PSNR if pico == "255" else float(b.max() - b.min())
        valores.append(float("inf") if mse == 0 else p.FACTOR_DECIBELES * np.log10(maximo ** 2 / mse))
    return float(np.mean(valores))


def _mezcla_canales() -> np.ndarray:
    """Matriz 3×3 de la gaussiana (σ = 1.5, radio 5) a lo largo del eje de canales, con réplica."""
    radio = p.TAMANO_VENTANA_SSIM // 2
    nucleo = cv2.getGaussianKernel(p.TAMANO_VENTANA_SSIM, p.SIGMA_SSIM).ravel()
    mezcla = np.zeros((p.CANALES_RGB, p.CANALES_RGB))
    for c in range(p.CANALES_RGB):
        for k in range(-radio, radio + 1):
            mezcla[c, min(max(c + k, 0), p.CANALES_RGB - 1)] += nucleo[k + radio]
    return mezcla


def _suavizar(z: np.ndarray) -> np.ndarray:
    """Gaussiana 11×11, σ = 1.5, bordes replicados, plano por plano."""
    tamano = (p.TAMANO_VENTANA_SSIM, p.TAMANO_VENTANA_SSIM)
    return cv2.GaussianBlur(z, tamano, p.SIGMA_SSIM, sigmaY=p.SIGMA_SSIM, borderType=cv2.BORDER_REPLICATE)


def momentos_ssim(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, ...]:
    """Promedios locales de x, y, x², y² y x·y (los ingredientes del mapa SSIM)."""
    return _suavizar(x), _suavizar(y), _suavizar(x * x), _suavizar(y * y), _suavizar(x * y)


def mapa_ssim(momentos: tuple[np.ndarray, ...], volumen: bool = False) -> np.ndarray:
    """Mapa SSIM (Ecs. 4 a 7, C3 = C2/2, exponentes 1) a partir de los promedios locales.

    Con `volumen`, la gaussiana también promedia a lo largo del eje de canales
    (el filtro 3-D es separable, así que alcanza con mezclar los planos ya suavizados).
    """
    if volumen:
        mezcla = _mezcla_canales().T
        momentos = tuple(m @ mezcla for m in momentos)
    mu_x, mu_y, xx, yy, xy = momentos
    c1 = (p.K1_SSIM * p.RANGO_DINAMICO_SSIM) ** 2
    c2 = (p.K2_SSIM * p.RANGO_DINAMICO_SSIM) ** 2
    var_x, var_y, cov = xx - mu_x ** 2, yy - mu_y ** 2, xy - mu_x * mu_y
    return ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2))


def _solo_valido(mapa: np.ndarray) -> np.ndarray:
    """Filtrado 'valid' de Wang: solo las posiciones donde la ventana 11×11 entra completa."""
    radio = p.TAMANO_VENTANA_SSIM // 2
    return mapa[radio:-radio, radio:-radio]


def variantes_ssim(x: dict[str, list[np.ndarray]], y: dict[str, list[np.ndarray]]) -> dict[str, float]:
    """Las cinco variantes de SSIM a partir de los planos de la imagen y de la referencia."""
    momentos_rgb = momentos_ssim(x["rgb"][0], y["rgb"][0])
    mapa_rgb = mapa_ssim(momentos_rgb)
    mapa_gris = mapa_ssim(momentos_ssim(x["gris"][0], y["gris"][0]))
    return {
        "plano|rgb": float(np.mean(mapa_rgb)),
        "plano|gris": float(np.mean(mapa_gris)),
        "volumen|rgb": float(np.mean(mapa_ssim(momentos_rgb, volumen=True))),
        "wang|canal": float(np.mean(_solo_valido(mapa_rgb))),
        "wang|gris": float(np.mean(_solo_valido(mapa_gris))),
    }


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Coeficiente de correlación de Pearson entre dos arreglos (corr2)."""
    dx, dy = x - x.mean(), y - y.mean()
    denominador = np.sqrt(np.sum(dx * dx) * np.sum(dy * dy))
    return float("nan") if denominador == 0 else float(np.sum(dx * dy) / denominador)


def coc(x: list[np.ndarray], y: list[np.ndarray]) -> float:
    """CoC (Ec. 8) promediado sobre los planos del dominio."""
    return float(np.mean([_pearson(a, b) for a, b in zip(x, y)]))


def epi(x: list[np.ndarray], y: list[np.ndarray]) -> float:
    """EPI (Ec. 9): Σ|X'(m,n+1) − X'(m,n)| / Σ|X(m,n+1) − X(m,n)|, promediado sobre los planos."""
    valores = []
    for a, b in zip(x, y):
        denominador = float(np.sum(np.abs(b[:, 1:] - b[:, :-1])))
        numerador = float(np.sum(np.abs(a[:, 1:] - a[:, :-1])))
        valores.append(float("nan") if denominador == 0 else numerador / denominador)
    return float(np.mean(valores))


def calcular_todas(imagen: np.ndarray, limpia: np.ndarray, ruidosa: np.ndarray) -> dict[tuple[str, str], float]:
    """Todas las variantes de las cuatro métricas: {(métrica, variante): valor}."""
    x, y = planos(imagen), planos(limpia)
    referencias = {"limpia": y, "ruidosa": planos(ruidosa)}
    valores: dict[tuple[str, str], float] = {}
    for variante in VARIANTES["PSNR"]:
        dominio, pico = variante.split("|")
        valores["PSNR", variante] = psnr(x[dominio], y[dominio], pico)
    for variante, valor in variantes_ssim(x, y).items():
        valores["SSIM", variante] = valor
    for variante in VARIANTES["CoC"]:
        dominio, referencia = variante.split("|")
        valores["CoC", variante] = coc(x[dominio], referencias[referencia][dominio])
    for dominio in VARIANTES["EPI"]:
        valores["EPI", dominio] = epi(x[dominio], y[dominio])
    return valores
