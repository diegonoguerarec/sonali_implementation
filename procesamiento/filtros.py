"""Filtros de eliminación de ruido y su registro.

Por ahora solo está la mediana ponderada. Para agregar un filtro (media,
mediana, Wiener, gaussiano) alcanza con escribir una función que reciba un
canal uint8 y devuelva un canal uint8, y sumarla a `FILTROS`.
"""

from collections.abc import Callable

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


# Registro de filtros: nombre de consola -> función(canal uint8) -> canal uint8.
FILTROS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "mediana_ponderada": mediana_ponderada,
}
