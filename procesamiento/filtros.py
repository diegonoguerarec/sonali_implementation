"""Filtros de eliminación de ruido y su registro.

Los cinco filtros del paper: mediana, media, gaussiano, Wiener y mediana
ponderada. Cada uno recibe un canal uint8 y devuelve un canal uint8. Para
agregar otro alcanza con escribir su función y sumarla a `FILTROS`.

Los tamaños de ventana y la sigma salen de la búsqueda de
investigacion/INFORME.md (etiqueta [AJUSTE] en parametros.py); el resto del
comportamiento es el por defecto de MATLAB:
- medfilt2: https://www.mathworks.com/help/images/ref/medfilt2.html
- fspecial: https://www.mathworks.com/help/images/ref/fspecial.html
- imfilter: https://www.mathworks.com/help/images/ref/imfilter.html
- wiener2:  https://www.mathworks.com/help/images/ref/wiener2.html
"""

from collections.abc import Callable

import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from procesamiento import parametros as p


def _validar_pesos(pesos: np.ndarray) -> np.ndarray:
    """Comprueba que los pesos sean una matriz 3×3 de enteros positivos."""
    pesos = np.asarray(pesos)
    forma = (p.TAMANO_VENTANA_MEDIANA, p.TAMANO_VENTANA_MEDIANA)
    if pesos.shape != forma or not np.issubdtype(pesos.dtype, np.integer) or np.any(pesos <= 0):
        raise ValueError("Los pesos tienen que ser una matriz 3×3 de enteros positivos.")
    return pesos.astype(np.int64)


def _mediana_de_ordenados(ordenados: np.ndarray, total: int) -> np.ndarray:
    """Mediana a lo largo del último eje de valores ya ordenados (estilo medfilt2).

    Si `total` es impar, toma el valor central. Si es par, promedia los dos
    centrales con división entera (sin fracción), en int64 para no desbordar.
    """
    mitad = total // p.DIVISOR_EMPATE_MEDIANA
    if total % p.DIVISOR_EMPATE_MEDIANA == 1:
        return ordenados[..., mitad]
    suma = ordenados[..., mitad - 1].astype(np.int64) + ordenados[..., mitad].astype(np.int64)
    return suma // p.DIVISOR_EMPATE_MEDIANA


def mediana_ponderada(canal: np.ndarray, pesos: np.ndarray = p.PESOS_MEDIANA_PONDERADA) -> np.ndarray:
    """Mediana ponderada 3×3 de un canal uint8 (versión vectorizada).

    Cada valor de la ventana se repite tantas veces como su peso y la salida es
    la mediana de esos valores. Con suma de pesos par, es el promedio de los
    dos centrales sin la parte fraccionaria. Bordes con relleno de ceros.

    Procedencia:
    - MATLAB no trae un filtro de mediana ponderada, así que no hay valores por
      defecto que copiar.
    - La descripción de la Tabla 1 del paper es casi textual la del código de
      MATLAB File Exchange #29386 (M. K. Sharma, 2010):
      https://www.mathworks.com/matlabcentral/fileexchange/29386-weighted-median-filter
    - Ese código usa ventana 3×3 y W = [1 1 1; 1 4 1; 1 1 1]/12: multiplica la
      ventana por W y toma la mediana de los 9 productos, lo que divide la
      intensidad por unos 12. Por eso no se usa esa versión multiplicativa: se
      toman los pesos sin dividir con la definición clásica (Brownrigg, "The
      weighted median filter", Communications of the ACM 27(8), 1984; Yin,
      Yang, Gabbouj y Neuvo, "Weighted median filters: a tutorial", IEEE
      TCAS-II 43(3), 1996, doi:10.1109/82.486465).
    - Empate y bordes: https://www.mathworks.com/help/images/ref/medfilt2.html
    """
    pesos = _validar_pesos(pesos)
    radio = p.TAMANO_VENTANA_MEDIANA // 2
    rellenado = np.pad(canal, radio, mode="constant", constant_values=p.VALOR_RELLENO_MEDIANA)
    ventanas = sliding_window_view(rellenado, pesos.shape)
    ventanas = ventanas.reshape(canal.shape + (pesos.size,))
    repetidos = np.repeat(ventanas, pesos.ravel(), axis=-1)
    ordenados = np.sort(repetidos, axis=-1)
    return _mediana_de_ordenados(ordenados, int(pesos.sum())).astype(np.uint8)


def mediana_ponderada_referencia(
    canal: np.ndarray, pesos: np.ndarray = p.PESOS_MEDIANA_PONDERADA
) -> np.ndarray:
    """Mediana ponderada con bucles explícitos. Solo para verificar la vectorizada."""
    pesos = _validar_pesos(pesos)
    radio = p.TAMANO_VENTANA_MEDIANA // 2
    rellenado = np.pad(canal, radio, mode="constant", constant_values=p.VALOR_RELLENO_MEDIANA)
    salida = np.zeros_like(canal)
    alto, ancho = canal.shape
    for fila in range(alto):
        for columna in range(ancho):
            valores = []
            for i in range(p.TAMANO_VENTANA_MEDIANA):
                for j in range(p.TAMANO_VENTANA_MEDIANA):
                    valores += [int(rellenado[fila + i, columna + j])] * int(pesos[i, j])
            valores.sort()
            salida[fila, columna] = _mediana_de_ordenados(np.array(valores), len(valores))
    return salida


def _rellenar_con_ceros(canal: np.ndarray, radio: int) -> np.ndarray:
    """Agranda el canal `radio` píxeles por lado con ceros (relleno por defecto de MATLAB)."""
    return np.pad(canal, radio, mode="constant", constant_values=p.VALOR_RELLENO_IMFILTER)


def _promedio_local(x: np.ndarray, tamano: int) -> np.ndarray:
    """Promedio de la ventana tamano×tamano en float64, con relleno de ceros."""
    radio = tamano // 2
    nucleo = np.full((tamano, tamano), 1.0 / (tamano * tamano))
    filtrada = cv2.filter2D(_rellenar_con_ceros(x, radio).astype(np.float64), cv2.CV_64F, nucleo,
                            borderType=cv2.BORDER_CONSTANT)
    return filtrada[radio:-radio, radio:-radio]


def _a_uint8_imfilter(x: np.ndarray) -> np.ndarray:
    """Salida de imfilter para uint8: redondea (empates hacia arriba) y recorta al rango."""
    return np.clip(np.floor(x + p.DESPLAZAMIENTO_REDONDEO), 0, p.MAX_INTENSIDAD).astype(np.uint8)


def mediana(canal: np.ndarray) -> np.ndarray:
    """Mediana 5×5 con relleno de ceros: `medfilt2(I, [5 5])`."""
    radio = p.TAMANO_MEDIANA // 2
    filtrada = cv2.medianBlur(_rellenar_con_ceros(canal, radio), p.TAMANO_MEDIANA)
    return np.ascontiguousarray(filtrada[radio:-radio, radio:-radio])


def media(canal: np.ndarray) -> np.ndarray:
    """Media 5×5 con relleno de ceros: `imfilter(I, fspecial('average', 5))`."""
    return _a_uint8_imfilter(_promedio_local(canal, p.TAMANO_MEDIA))


def nucleo_gaussiano(tamano: int, sigma: float) -> np.ndarray:
    """`fspecial('gaussian', tamano, sigma)`: exp(−(x²+y²)/(2σ²)) dividido por su suma."""
    eje = np.arange(tamano) - (tamano - 1) / 2
    x, y = np.meshgrid(eje, eje)
    nucleo = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    return nucleo / nucleo.sum()


def gaussiano(canal: np.ndarray) -> np.ndarray:
    """Gaussiano 3×3 con sigma 0.5 y relleno de ceros: `imfilter(I, fspecial('gaussian', 3, 0.5))`."""
    radio = p.TAMANO_GAUSSIANO // 2
    nucleo = nucleo_gaussiano(p.TAMANO_GAUSSIANO, p.SIGMA_GAUSSIANO)
    filtrada = cv2.filter2D(_rellenar_con_ceros(canal, radio).astype(np.float64), cv2.CV_64F, nucleo,
                            borderType=cv2.BORDER_CONSTANT)
    return _a_uint8_imfilter(filtrada[radio:-radio, radio:-radio])


def wiener(canal: np.ndarray) -> np.ndarray:
    """Wiener adaptativo 5×5: `wiener2(I, [5 5])`.

    Con μ y σ² la media y la varianza locales y ν² el promedio de todas las
    varianzas locales: salida = μ + (σ² − ν²)/σ² · (I − μ), sin ganancias
    negativas. Se calcula con la imagen en [0, 1] y vuelve a uint8 con redondeo.
    No verificado en la documentación: el relleno de ceros en los bordes.
    """
    g = canal.astype(np.float64) / p.MAX_INTENSIDAD
    media_local = _promedio_local(g, p.TAMANO_WIENER)
    varianza_local = _promedio_local(g * g, p.TAMANO_WIENER) - media_local ** 2
    ruido = varianza_local.mean()
    denominador = np.maximum(varianza_local, ruido)
    ganancia = np.divide(np.maximum(varianza_local - ruido, 0), denominador,
                         out=np.zeros_like(denominador), where=denominador > 0)
    salida = np.clip(media_local + ganancia * (g - media_local),
                     p.INTENSIDAD_MIN_NORMALIZADA, p.INTENSIDAD_MAX_NORMALIZADA)
    return np.floor(salida * p.MAX_INTENSIDAD + p.DESPLAZAMIENTO_REDONDEO).astype(np.uint8)


# Registro de filtros: nombre de consola -> función(canal uint8) -> canal uint8.
FILTROS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "mediana": mediana,
    "media": media,
    "gaussiano": gaussiano,
    "wiener": wiener,
    "mediana_ponderada": mediana_ponderada,
}

# Descripción de cada filtro para parametros.json.
DESCRIPCION_FILTROS: dict[str, dict] = {
    "mediana": {"ventana": [p.TAMANO_MEDIANA, p.TAMANO_MEDIANA], "bordes": "relleno con ceros (medfilt2)"},
    "media": {"ventana": [p.TAMANO_MEDIA, p.TAMANO_MEDIA],
              "definicion": "imfilter con fspecial('average'): correlación, salida uint8 redondeada",
              "bordes": "relleno con ceros (imfilter)"},
    "gaussiano": {"ventana": [p.TAMANO_GAUSSIANO, p.TAMANO_GAUSSIANO], "sigma": p.SIGMA_GAUSSIANO,
                  "definicion": "imfilter con fspecial('gaussian'): correlación, salida uint8 redondeada",
                  "bordes": "relleno con ceros (imfilter)"},
    "wiener": {"ventana": [p.TAMANO_WIENER, p.TAMANO_WIENER],
               "definicion": "wiener2: media + (var − ruido)/var · (I − media), ruido = promedio de las varianzas locales",
               "bordes": "relleno con ceros"},
    "mediana_ponderada": {
        "ventana": [p.TAMANO_VENTANA_MEDIANA, p.TAMANO_VENTANA_MEDIANA],
        "pesos": p.PESOS_MEDIANA_PONDERADA.tolist(),
        "definicion": "repetir cada valor según su peso y tomar la mediana",
        "regla_empate": "suma de pesos par: promedio de los 2 centrales, sin fracción (medfilt2)",
        "bordes": "relleno con ceros (medfilt2)",
    },
}
