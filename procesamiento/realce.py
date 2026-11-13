"""CLAHE con los valores por defecto de `adapthisteq` de MATLAB, usando OpenCV.

Fuente: https://www.mathworks.com/help/images/ref/adapthisteq.html

8×8 teselas, 256 bins y distribución uniforme. El clip limit es normalizado
(0 a 1, como en MATLAB) y se convierte a la escala de OpenCV.

Limitación: el resultado no es idéntico bit a bit al de adapthisteq, porque
MATLAB y OpenCV rellenan los bordes y redistribuyen el exceso del histograma
de forma algo distinta.
"""

import math

import cv2
import numpy as np

from procesamiento import parametros as p


def pixeles_por_tesela(alto: int, ancho: int) -> int:
    """Cantidad P de píxeles de una tesela, como la calcula OpenCV (clahe.cpp).

    Si alto y ancho son divisibles por la cantidad de teselas, la tesela mide
    (alto/8) × (ancho/8). Si no, OpenCV rellena hasta alto + (8 − alto % 8) y
    ancho + (8 − ancho % 8) y divide por 8.
    """
    teselas_y, teselas_x = p.TESELAS_CLAHE
    if alto % teselas_y == 0 and ancho % teselas_x == 0:
        return (alto // teselas_y) * (ancho // teselas_x)
    alto_ext = alto + (teselas_y - alto % teselas_y)
    ancho_ext = ancho + (teselas_x - ancho % teselas_x)
    return (alto_ext // teselas_y) * (ancho_ext // teselas_x)


def round_matlab(x: float) -> int:
    """Redondeo de MATLAB para x ≥ 0: floor(x + 0.5)."""
    return math.floor(x + p.DESPLAZAMIENTO_REDONDEO)


def clip_limit_matlab_a_opencv(clip_normalizado: float, alto: int, ancho: int) -> dict:
    """Convierte el clip limit normalizado de MATLAB al `clipLimit` de OpenCV.

    1. P: píxeles por tesela, igual que OpenCV.
    2. Cuentas, como MATLAB (ejemplo HDL de MathWorks que toma adapthisteq como
       referencia): min_clip = ceil(P/256);
       clip_cuentas = min_clip + round_matlab(c·(P − min_clip)).
    3. OpenCV usa int(clipLimit·P/256) con mínimo 1; para obtener exactamente
       clip_cuentas: clipLimit = (clip_cuentas + 0.5)·256/P.

    Devuelve un diccionario con P, las cuentas y el clipLimit de OpenCV.
    """
    pixeles = pixeles_por_tesela(alto, ancho)
    min_clip = math.ceil(pixeles / p.BINS_CLAHE)
    clip_cuentas = min_clip + round_matlab(clip_normalizado * (pixeles - min_clip))
    clip_opencv = (clip_cuentas + p.DESPLAZAMIENTO_CLIP_OPENCV) * p.BINS_CLAHE / pixeles
    return {
        "pixeles_por_tesela": pixeles,
        "clip_cuentas": clip_cuentas,
        "clip_limit_opencv": clip_opencv,
    }


def aplicar_clahe(canal: np.ndarray, clip_normalizado: float) -> np.ndarray:
    """Aplica CLAHE a un canal uint8 con el clip limit normalizado de MATLAB."""
    alto, ancho = canal.shape
    clip_opencv = clip_limit_matlab_a_opencv(clip_normalizado, alto, ancho)["clip_limit_opencv"]
    clahe = cv2.createCLAHE(clipLimit=clip_opencv, tileGridSize=p.TESELAS_CLAHE)
    return clahe.apply(canal)
