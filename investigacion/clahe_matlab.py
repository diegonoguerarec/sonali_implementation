"""Reconstrucción de `adapthisteq` de MATLAB (CLAHE) con NumPy, para la investigación.

Fuentes:
- [MATLAB] Documentación de adapthisteq (8×8 teselas, 256 bins, 'uniform', 'full';
  las teselas vecinas se combinan con interpolación bilineal):
  https://www.mathworks.com/help/images/ref/adapthisteq.html
- [MATLAB] Ejemplo HDL de MathWorks, que declara coincidir con adapthisteq salvo
  "unos pocos píxeles": teselas de tamaño par, relleno simétrico, fórmula del clip
  limit, subsistemas "Distribute"/"Redistribute" y
  grayxform = round(255·cdf/numPixInTile):
  https://www.mathworks.com/help/visionhdl/ug/contrast-adaptive-histogram-equalization.html
- [LITERATURA] Zuiderveld, "Contrast Limited Adaptive Histogram Equalization",
  Graphics Gems IV, 1994 (recorte del histograma e interpolación).

No verificado (no figura en la documentación consultada; se exponen como variantes):
- en qué lado va el relleno simétrico (`relleno`);
- si la tabla de cada tesela se redondea a enteros antes de interpolar (`tabla`;
  el ejemplo HDL sí la redondea);
- el redondeo de la salida (`final`);
- los pesos exactos de la interpolación bilineal (se usan los de Zuiderveld).
"""

import math

import cv2
import numpy as np

# [MATLAB] adapthisteq: NumTiles por defecto [8 8].
TESELAS: tuple[int, int] = (8, 8)
# [MATLAB] adapthisteq: NBins por defecto 256.
BINS: int = 256
# [MATLAB] Máximo de uint8 (Range 'full').
MAX_INTENSIDAD: int = 255
# [MATLAB] round() redondea los empates hacia arriba para x ≥ 0: floor(x + 0.5).
MEDIO: float = 0.5

# Variantes de los detalles no verificados; la primera de cada una es la que
# describe el ejemplo HDL (o, si no dice nada, la del borrador).
VARIANTES_TABLA: tuple[str, ...] = ("redondeada", "continua")
VARIANTES_FINAL: tuple[str, ...] = ("redondear", "truncar")
VARIANTES_RELLENO: tuple[str, ...] = ("centrado", "final")


def tamano_tesela(n: int, teselas: int) -> int:
    """Lado de la tesela: ceil(n / teselas), subido al siguiente par (ejemplo HDL)."""
    lado = math.ceil(n / teselas)
    return lado + lado % 2


def clip_en_cuentas(clip_normalizado: float, pixeles: int, bins: int = BINS) -> int:
    """Clip limit en cuentas (ejemplo HDL):
    minClip = ceil(P/bins); clip = minClip + round(c·(P − minClip))."""
    min_clip = math.ceil(pixeles / bins)
    return min_clip + math.floor(clip_normalizado * (pixeles - min_clip) + MEDIO)


def recortar_histograma(histograma: np.ndarray, clip: int) -> np.ndarray:
    """Recorta el histograma y reparte el exceso sin que ningún bin pase el clip.

    Paso "Distribute": los bins sobre el clip quedan en el clip; los que están a
    menos de `incremento` del clip se llenan hasta el clip; el resto suma
    `incremento`. Paso "Redistribute": el exceso restante se reparte de a 1,
    con paso floor(bins/exceso), salteando los bins que ya están en el clip.
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


def recortar_histograma_referencia(histograma: np.ndarray, clip: int) -> np.ndarray:
    """Versión con bucles explícitos de `recortar_histograma`. Solo para las pruebas."""
    h = [int(v) for v in histograma]
    bins = len(h)
    exceso = sum(max(v - clip, 0) for v in h)
    incremento = exceso // bins
    for k in range(bins):
        if h[k] > clip:
            h[k] = clip
        elif h[k] > clip - incremento:
            exceso -= clip - h[k]
            h[k] = clip
        else:
            exceso -= incremento
            h[k] += incremento
    inicio = 0
    while exceso > 0:
        paso = max(bins // exceso, 1)
        for m in range(inicio, bins, paso):
            if h[m] < clip:
                h[m] += 1
                exceso -= 1
                if exceso == 0:
                    break
        inicio = (inicio + 1) % bins
    return np.array(h, dtype=np.int64)


def tabla_de_tesela(histograma: np.ndarray, pixeles: int, tabla: str) -> np.ndarray:
    """Tabla de valores de una tesela ('uniform', rango completo): 255·cdf/P, tope 255."""
    valores = np.minimum(np.cumsum(histograma) * MAX_INTENSIDAD / pixeles, MAX_INTENSIDAD)
    if tabla == "redondeada":
        return np.floor(valores + MEDIO)
    return valores


def _rellenar(canal: np.ndarray, lados: tuple[int, int], relleno: str) -> tuple[np.ndarray, int, int]:
    """Relleno simétrico hasta 8×8 teselas; devuelve además el desplazamiento inicial."""
    sobrante = [lado * t - n for lado, t, n in zip(lados, TESELAS, canal.shape)]
    antes = [s // 2 if relleno == "centrado" else 0 for s in sobrante]
    despues = [s - a for s, a in zip(sobrante, antes)]
    rellenada = np.pad(canal, tuple(zip(antes, despues)), mode="symmetric")
    return rellenada, antes[0], antes[1]


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


def adapthisteq(
    canal: np.ndarray,
    clip_normalizado: float,
    tabla: str = "redondeada",
    final: str = "redondear",
    relleno: str = "centrado",
) -> np.ndarray:
    """CLAHE de un canal uint8 al estilo de `adapthisteq(I, 'ClipLimit', c)`."""
    alto, ancho = canal.shape
    lados = (tamano_tesela(alto, TESELAS[0]), tamano_tesela(ancho, TESELAS[1]))
    rellenada, f0, c0 = _rellenar(canal, lados, relleno)
    pixeles = lados[0] * lados[1]
    clip = clip_en_cuentas(clip_normalizado, pixeles)

    tablas = np.empty(TESELAS + (BINS,), dtype=np.float64)
    for i in range(TESELAS[0]):
        for j in range(TESELAS[1]):
            tesela = rellenada[i * lados[0]:(i + 1) * lados[0], j * lados[1]:(j + 1) * lados[1]]
            histograma = np.bincount(tesela.ravel(), minlength=BINS)
            tablas[i, j] = tabla_de_tesela(recortar_histograma(histograma, clip), pixeles, tabla)

    salida = np.empty(rellenada.shape, dtype=np.float64)
    for fila, nf, fa, fb in _regiones(TESELAS[0], lados[0]):
        peso_f = np.arange(nf)[:, None]
        for col, nc, ca, cb in _regiones(TESELAS[1], lados[1]):
            peso_c = np.arange(nc)[None, :]
            sub = rellenada[fila:fila + nf, col:col + nc]
            arriba = (nc - peso_c) * tablas[fa, ca][sub] + peso_c * tablas[fa, cb][sub]
            abajo = (nc - peso_c) * tablas[fb, ca][sub] + peso_c * tablas[fb, cb][sub]
            salida[fila:fila + nf, col:col + nc] = ((nf - peso_f) * arriba + peso_f * abajo) / (nf * nc)

    if final == "redondear":
        salida = np.floor(salida + MEDIO)
    salida = np.clip(salida, 0, MAX_INTENSIDAD).astype(np.uint8)
    return salida[f0:f0 + alto, c0:c0 + ancho]


def datos_clahe(alto: int, ancho: int, clip_normalizado: float) -> dict[str, int]:
    """Lados de tesela, píxeles por tesela y clip en cuentas para una imagen."""
    lados = (tamano_tesela(alto, TESELAS[0]), tamano_tesela(ancho, TESELAS[1]))
    pixeles = lados[0] * lados[1]
    return {"alto_tesela": lados[0], "ancho_tesela": lados[1], "pixeles_por_tesela": pixeles,
            "clip_cuentas": clip_en_cuentas(clip_normalizado, pixeles)}


def clahe_opencv(canal: np.ndarray, clip_normalizado: float) -> np.ndarray:
    """CLAHE de OpenCV con el mismo clip en cuentas (la versión original del proyecto).

    Solo como referencia. [OPENCV] `modules/imgproc/src/clahe.cpp` (rama 5.x): si el
    tamaño no es divisible por 8, rellena hasta alto + (8 − alto % 8) y ancho + (8 − ancho % 8);
    usa int(clipLimit·P/256) cuentas, así que clipLimit = (cuentas + 0.5)·256/P da las mismas
    cuentas que MATLAB; el exceso recortado se reparte sin tope.
    """
    alto, ancho = canal.shape
    if alto % TESELAS[0] == 0 and ancho % TESELAS[1] == 0:
        pixeles = (alto // TESELAS[0]) * (ancho // TESELAS[1])
    else:
        pixeles = ((alto + TESELAS[0] - alto % TESELAS[0]) // TESELAS[0]) * (
            (ancho + TESELAS[1] - ancho % TESELAS[1]) // TESELAS[1])
    clip_opencv = (clip_en_cuentas(clip_normalizado, pixeles) + MEDIO) * BINS / pixeles
    return cv2.createCLAHE(clipLimit=clip_opencv, tileGridSize=TESELAS).apply(canal)
