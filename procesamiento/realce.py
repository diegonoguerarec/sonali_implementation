"""CLAHE al estilo de `adapthisteq` de MATLAB, implementado con NumPy.

Fuentes:
- Documentación de adapthisteq (8×8 teselas, 256 bins, distribución uniforme,
  interpolación bilineal entre teselas):
  https://www.mathworks.com/help/images/ref/adapthisteq.html
- Ejemplo HDL de MathWorks, que declara coincidir con adapthisteq salvo "unos
  pocos píxeles" (teselas de lado par, relleno simétrico, fórmula del clip
  limit, reparto del exceso sin superar el clip y tabla round(255·cdf/P)):
  https://www.mathworks.com/help/visionhdl/ug/contrast-adaptive-histogram-equalization.html
- Zuiderveld, "Contrast Limited Adaptive Histogram Equalization", Graphics
  Gems IV, 1994.

No se usa el CLAHE de OpenCV porque reparte el exceso recortado sin tope y
realza mucho más que adapthisteq (ver investigacion/INFORME.md).

Limitación: no está verificado contra MATLAB. La documentación no dice de qué
lado va el relleno, cómo se redondea la salida ni los pesos exactos de la
interpolación; se usan el relleno centrado, el redondeo de MATLAB y los pesos
de Zuiderveld.
"""

import math

import numpy as np

from procesamiento import parametros as p


def round_matlab(x: float) -> int:
    """Redondeo de MATLAB para x ≥ 0: floor(x + 0.5)."""
    return math.floor(x + p.DESPLAZAMIENTO_REDONDEO)


def lado_tesela(n: int, teselas: int) -> int:
    """Lado de la tesela: ceil(n / teselas), subido al siguiente par."""
    lado = math.ceil(n / teselas)
    return lado + lado % p.PARIDAD_TESELA


def datos_clahe(clip_normalizado: float, alto: int, ancho: int) -> dict:
    """Píxeles por tesela (P) y clip limit en cuentas, como los calcula MATLAB.

    min_clip = ceil(P/256); clip_cuentas = min_clip + round(c·(P − min_clip)).
    """
    lados = (lado_tesela(alto, p.TESELAS_CLAHE[0]), lado_tesela(ancho, p.TESELAS_CLAHE[1]))
    pixeles = lados[0] * lados[1]
    min_clip = math.ceil(pixeles / p.BINS_CLAHE)
    clip_cuentas = min_clip + round_matlab(clip_normalizado * (pixeles - min_clip))
    return {"lados_tesela": lados, "pixeles_por_tesela": pixeles, "clip_cuentas": clip_cuentas}


def pixeles_por_tesela(alto: int, ancho: int) -> int:
    """Cantidad P de píxeles de una tesela (lados pares)."""
    return lado_tesela(alto, p.TESELAS_CLAHE[0]) * lado_tesela(ancho, p.TESELAS_CLAHE[1])


def recortar_histograma(histograma: np.ndarray, clip: int) -> np.ndarray:
    """Recorta el histograma en `clip` y reparte el exceso sin que ningún bin lo supere.

    Primero ("Distribute"): los bins sobre el clip quedan en el clip; los que
    están a menos de `incremento` del clip se llenan hasta el clip; el resto
    suma `incremento` = exceso // bins. Después ("Redistribute"): lo que sobra
    se reparte de a 1, con paso bins // exceso, salteando los bins llenos.
    """
    h = histograma.astype(np.int64).copy()
    bins = h.size
    exceso = int(np.maximum(h - clip, 0).sum())
    incremento = exceso // bins
    sobre = h > clip
    casi = (h > clip - incremento) & ~sobre
    bajo = ~sobre & ~casi
    exceso -= int((clip - h[casi]).sum()) + incremento * int(bajo.sum())
    h[sobre | casi] = clip
    h[bajo] += incremento
    inicio = 0
    while exceso > 0:
        paso = max(bins // exceso, 1)
        indices = np.arange(inicio, bins, paso)
        elegidos = indices[h[indices] < clip][:exceso]
        h[elegidos] += 1
        exceso -= elegidos.size
        inicio = (inicio + 1) % bins
    return h


def _tabla_de_tesela(tesela: np.ndarray, clip: int, pixeles: int) -> np.ndarray:
    """Tabla de valores de una tesela: round(255 · cdf del histograma recortado / P)."""
    histograma = recortar_histograma(np.bincount(tesela.ravel(), minlength=p.BINS_CLAHE), clip)
    valores = np.minimum(np.cumsum(histograma) * p.MAX_INTENSIDAD / pixeles, p.MAX_INTENSIDAD)
    return np.floor(valores + p.DESPLAZAMIENTO_REDONDEO)


def _regiones(teselas: int, lado: int) -> list[tuple[int, int, int, int]]:
    """Regiones de interpolación en un eje: (inicio, largo, tesela A, tesela B).

    Media tesela en cada borde (una sola tabla) y una tesela entera entre los
    centros de dos teselas vecinas.
    """
    regiones, posicion = [], 0
    for k in range(teselas + 1):
        if k == 0:
            largo, a, b = lado // 2, 0, 0
        elif k == teselas:
            largo, a, b = lado // 2, teselas - 1, teselas - 1
        else:
            largo, a, b = lado, k - 1, k
        regiones.append((posicion, largo, a, b))
        posicion += largo
    return regiones


def aplicar_clahe(canal: np.ndarray, clip_normalizado: float) -> np.ndarray:
    """Aplica CLAHE a un canal uint8 con el clip limit normalizado de MATLAB."""
    alto, ancho = canal.shape
    datos = datos_clahe(clip_normalizado, alto, ancho)
    lados, pixeles, clip = datos["lados_tesela"], datos["pixeles_por_tesela"], datos["clip_cuentas"]
    teselas_y, teselas_x = p.TESELAS_CLAHE

    sobrante = (lados[0] * teselas_y - alto, lados[1] * teselas_x - ancho)
    antes = (sobrante[0] // 2, sobrante[1] // 2)
    relleno = ((antes[0], sobrante[0] - antes[0]), (antes[1], sobrante[1] - antes[1]))
    rellenada = np.pad(canal, relleno, mode="symmetric")

    tablas = np.empty((teselas_y, teselas_x, p.BINS_CLAHE), dtype=np.float64)
    for i in range(teselas_y):
        for j in range(teselas_x):
            tesela = rellenada[i * lados[0]:(i + 1) * lados[0], j * lados[1]:(j + 1) * lados[1]]
            tablas[i, j] = _tabla_de_tesela(tesela, clip, pixeles)

    salida = np.empty(rellenada.shape, dtype=np.float64)
    for fila, nf, fa, fb in _regiones(teselas_y, lados[0]):
        peso_f = np.arange(nf)[:, None]
        for col, nc, ca, cb in _regiones(teselas_x, lados[1]):
            peso_c = np.arange(nc)[None, :]
            sub = rellenada[fila:fila + nf, col:col + nc]
            arriba = (nc - peso_c) * tablas[fa, ca][sub] + peso_c * tablas[fa, cb][sub]
            abajo = (nc - peso_c) * tablas[fb, ca][sub] + peso_c * tablas[fb, cb][sub]
            salida[fila:fila + nf, col:col + nc] = ((nf - peso_f) * arriba + peso_f * abajo) / (nf * nc)

    salida = np.floor(salida + p.DESPLAZAMIENTO_REDONDEO).astype(np.uint8)
    return salida[antes[0]:antes[0] + alto, antes[1]:antes[1] + ancho]
