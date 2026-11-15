"""Candidatos para los cinco filtros del paper (sección 4.2 del prompt 01).

Todos reciben un canal uint8 y devuelven un canal uint8, como haría MATLAB
con una imagen de 8 bits. Fuentes [MATLAB]:
- medfilt2 (3×3 por defecto; relleno 'zeros' por defecto o 'symmetric'):
  https://www.mathworks.com/help/images/ref/medfilt2.html
- fspecial ('average' 3×3 por defecto; 'gaussian' con sigma 0.5 por defecto,
  núcleo exp(−(x²+y²)/(2σ²)) normalizado por su suma):
  https://www.mathworks.com/help/images/ref/fspecial.html
- imfilter (correlación, relleno 0, salida 'same' y del mismo tipo; en enteros
  redondea y recorta al rango): https://www.mathworks.com/help/images/ref/imfilter.html
- imgaussfilt (sigma 0.5 por defecto, tamaño 2·ceil(2σ)+1, relleno 'replicate'):
  https://www.mathworks.com/help/images/ref/imgaussfilt.html
- wiener2 (3×3 por defecto; ruido = mean2(localVar);
  b = μ + (σ² − ν²)/σ² · (a − μ)): https://www.mathworks.com/help/images/ref/wiener2.html

No verificado en la documentación:
- el tamaño por defecto de fspecial('gaussian') (la página actual no lo dice; el
  prompt 01 indica 3; se prueban 3 y 5);
- que imgaussfilt construya el núcleo igual que fspecial (se asume);
- cómo trata wiener2 los bordes y la clase uint8 (se asume filter2 con relleno
  de ceros sobre la imagen en double y vuelta a uint8 con redondeo);
- los max(·, 0) y max(σ², ν²) de wiener2, que evitan ganancias negativas
  (están en la formulación de Lim, 1990, que cita la documentación).
"""

import cv2
import numpy as np

from procesamiento import parametros as p
from procesamiento.filtros import mediana_ponderada
from procesamiento.ruido import a_uint8_matlab

# Nombre de relleno de MATLAB -> modo de np.pad.
MODOS_RELLENO: dict[str, str] = {"ceros": "constant", "replicate": "edge", "symmetric": "symmetric"}

# [FILE EXCHANGE] #29386: W = [1 1 1; 1 4 1; 1 1 1] / 12.
DIVISOR_PESOS_FILE_EXCHANGE: int = 12


def _rellenar(canal: np.ndarray, radio: int, bordes: str) -> np.ndarray:
    """Agranda el canal `radio` píxeles por lado con el relleno pedido."""
    return np.pad(canal, radio, mode=MODOS_RELLENO[bordes])


def _a_uint8_imfilter(x: np.ndarray) -> np.ndarray:
    """Salida de imfilter para uint8: redondea (empates hacia arriba) y recorta."""
    return np.clip(np.floor(x + p.DESPLAZAMIENTO_REDONDEO), 0, p.MAX_INTENSIDAD).astype(np.uint8)


def _correlacion(canal: np.ndarray, nucleo: np.ndarray, bordes: str) -> np.ndarray:
    """Correlación 2-D 'same' en float64 con el relleno pedido (como imfilter)."""
    radio = nucleo.shape[0] // 2
    rellenada = _rellenar(canal, radio, bordes).astype(np.float64)
    filtrada = cv2.filter2D(rellenada, cv2.CV_64F, nucleo, borderType=cv2.BORDER_CONSTANT)
    return filtrada[radio:-radio, radio:-radio]


def mediana(canal: np.ndarray, tamano: int = 3, bordes: str = "ceros") -> np.ndarray:
    """`medfilt2(I, [t t], padopt)`: mediana de la ventana t×t (t impar)."""
    radio = tamano // 2
    filtrada = cv2.medianBlur(_rellenar(canal, radio, bordes), tamano)
    return np.ascontiguousarray(filtrada[radio:-radio, radio:-radio])


def media(canal: np.ndarray, tamano: int = 3, bordes: str = "ceros") -> np.ndarray:
    """`imfilter(I, fspecial('average', t), bordes)`."""
    nucleo = np.full((tamano, tamano), 1.0 / (tamano * tamano))
    return _a_uint8_imfilter(_correlacion(canal, nucleo, bordes))


def nucleo_gaussiano(tamano: int, sigma: float) -> np.ndarray:
    """`fspecial('gaussian', t, sigma)`: exp(−(x²+y²)/(2σ²)) dividido por su suma."""
    eje = np.arange(tamano) - (tamano - 1) / 2
    x, y = np.meshgrid(eje, eje)
    nucleo = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    return nucleo / nucleo.sum()


def gaussiano(canal: np.ndarray, funcion: str = "fspecial", tamano: int = 3, sigma: float = 1.0) -> np.ndarray:
    """Filtro gaussiano.

    - "fspecial": `imfilter(I, fspecial('gaussian', tamano, sigma))`, relleno de ceros.
    - "imgaussfilt": `imgaussfilt(I, sigma)`, tamaño 2·ceil(2σ)+1 y relleno 'replicate'
      (`tamano` se ignora).
    """
    if funcion == "imgaussfilt":
        tamano = 2 * int(np.ceil(2 * sigma)) + 1
        return _a_uint8_imfilter(_correlacion(canal, nucleo_gaussiano(tamano, sigma), "replicate"))
    return _a_uint8_imfilter(_correlacion(canal, nucleo_gaussiano(tamano, sigma), "ceros"))


def wiener(canal: np.ndarray, tamano: int = 3) -> np.ndarray:
    """`wiener2(I, [t t])` con el ruido estimado como el promedio de las varianzas locales."""
    g = canal.astype(np.float64) / p.MAX_INTENSIDAD
    nucleo = np.full((tamano, tamano), 1.0 / (tamano * tamano))
    radio = tamano // 2

    def promedio_local(x: np.ndarray) -> np.ndarray:
        rellenada = np.pad(x, radio, mode="constant")
        return cv2.filter2D(rellenada, cv2.CV_64F, nucleo,
                            borderType=cv2.BORDER_CONSTANT)[radio:-radio, radio:-radio]

    media_local = promedio_local(g)
    varianza_local = promedio_local(g * g) - media_local ** 2
    ruido = varianza_local.mean()
    ganancia = np.maximum(varianza_local - ruido, 0) / np.maximum(np.maximum(varianza_local, ruido), 1e-300)
    salida = media_local + ganancia * (g - media_local)
    return a_uint8_matlab(np.clip(salida, 0, 1))


def pesos_centro(c: int) -> np.ndarray:
    """Pesos [[1,1,1],[1,c,1],[1,1,1]]; c = 4 es el de File Exchange #29386."""
    pesos = np.ones((3, 3), dtype=np.int64)
    pesos[1, 1] = c
    return pesos


def mediana_ponderada_centro(canal: np.ndarray, c: int = 4) -> np.ndarray:
    """Mediana ponderada clásica del proyecto (Brownrigg 1984; Yin et al. 1996) con peso central c."""
    return mediana_ponderada(canal, pesos_centro(c))


def _ventanas_3x3(canal: np.ndarray) -> np.ndarray:
    """Ventanas 3×3 con relleno de ceros, como arreglo alto × ancho × 9 en float64."""
    rellenada = np.pad(canal, 1, mode="constant").astype(np.float64)
    vistas = np.lib.stride_tricks.sliding_window_view(rellenada, (3, 3))
    return vistas.reshape(canal.shape + (9,))


def mediana_ponderada_literal(canal: np.ndarray, variante: str) -> np.ndarray:
    """Variantes literales de la Tabla 1: mediana de (ventana · máscara).

    - "W/12": máscara [1 1 1; 1 4 1; 1 1 1]/12, como File Exchange #29386.
    - "W_saturada": máscara sin dividir; cada producto satura en 255 (uint8, como immultiply).
    """
    productos = _ventanas_3x3(canal) * pesos_centro(4).ravel()
    if variante == "W/12":
        productos = productos / DIVISOR_PESOS_FILE_EXCHANGE
    else:
        productos = np.minimum(productos, p.MAX_INTENSIDAD)
    return _a_uint8_imfilter(np.median(productos, axis=-1))


# Grillas cerradas de la sección 4.2: el primer candidato de cada filtro es el
# valor por defecto de MATLAB (o el del paper, en el gaussiano: σ = 1).
CANDIDATOS: dict[str, list[tuple]] = {
    "mediana": [(t, b) for t in (3, 5, 7) for b in ("ceros", "symmetric")],
    "media": [(t, b) for t in (3, 5, 7) for b in ("ceros", "replicate", "symmetric")],
    "gaussiano": [("fspecial", 3, 1.0), ("fspecial", 5, 1.0), ("imgaussfilt", 5, 1.0),
                  ("fspecial", 3, 0.5), ("fspecial", 5, 0.5), ("imgaussfilt", 3, 0.5)],
    "wiener": [(3,), (5,), (7,)],
    "mediana_ponderada": [(4,), (1,), (2,), (3,), (5,)],
}

FUNCIONES = {
    "mediana": mediana,
    "media": media,
    "gaussiano": gaussiano,
    "wiener": wiener,
    "mediana_ponderada": mediana_ponderada_centro,
    "mediana_ponderada_literal": mediana_ponderada_literal,
}


def aplicar_filtro(imagen: np.ndarray, filtro: str, parametros: tuple) -> np.ndarray:
    """Aplica el filtro a cada canal de una imagen RGB uint8 y vuelve a unirlos."""
    funcion = FUNCIONES[filtro]
    return np.stack([funcion(imagen[:, :, c], *parametros) for c in range(imagen.shape[2])], axis=2)
