"""Ruido gaussiano y de sal y pimienta, como lo agrega `imnoise` de MATLAB.

Fuente: https://www.mathworks.com/help/images/ref/imnoise.html

El ruido se sortea por separado para cada valor (alto × ancho × 3), porque
imnoise trata el arreglo elemento a elemento. El generador es
`numpy.random.default_rng(semilla)`, distinto del de MATLAB: el ruido no es
idéntico al de MATLAB aunque la semilla sea la misma.
"""

from collections.abc import Callable

import numpy as np

from procesamiento import parametros as p


def a_uint8_matlab(x: np.ndarray) -> np.ndarray:
    """Pasa intensidades en [0, 1] a uint8 como `uint8(round(x*255))` de MATLAB.

    Usa floor(x·255 + 0.5): los empates se redondean hacia arriba, no al par.
    """
    escalado = np.floor(x * p.MAX_INTENSIDAD + p.DESPLAZAMIENTO_REDONDEO)
    return escalado.astype(np.uint8)


def ruido_gaussiano(imagen: np.ndarray, varianza: float, rng: np.random.Generator) -> np.ndarray:
    """Suma ruido normal de media 0 y la varianza pedida (intensidades en [0, 1])."""
    normalizada = imagen.astype(np.float64) / p.MAX_INTENSIDAD
    ruido = rng.normal(p.MEDIA_RUIDO_GAUSSIANO, np.sqrt(varianza), size=imagen.shape)
    con_ruido = np.clip(normalizada + ruido, p.INTENSIDAD_MIN_NORMALIZADA, p.INTENSIDAD_MAX_NORMALIZADA)
    return a_uint8_matlab(con_ruido)


def ruido_sal_pimienta(imagen: np.ndarray, densidad: float, rng: np.random.Generator) -> np.ndarray:
    """Con u uniforme por valor: u < d/2 → 0; d/2 ≤ u < d → 255; si no, queda igual."""
    u = rng.uniform(size=imagen.shape)
    salida = imagen.copy()
    salida[u < densidad * p.FRACCION_PIMIENTA] = p.VALOR_PIMIENTA
    salida[(u >= densidad * p.FRACCION_PIMIENTA) & (u < densidad)] = p.VALOR_SAL
    return salida


# Registro de tipos de ruido: nombre de consola -> función.
RUIDOS: dict[str, Callable[[np.ndarray, float, np.random.Generator], np.ndarray]] = {
    "gaussiano": ruido_gaussiano,
    "sal_pimienta": ruido_sal_pimienta,
}


def agregar_ruido(imagen: np.ndarray, tipo: str, valor: float, semilla: int) -> np.ndarray:
    """Agrega ruido de tipo `tipo` a la imagen RGB uint8 completa.

    `valor` es la varianza (gaussiano) o la densidad d (sal_pimienta).
    """
    rng = np.random.default_rng(semilla)
    return RUIDOS[tipo](imagen, valor, rng)
