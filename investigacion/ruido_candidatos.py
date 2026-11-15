"""Candidatos para el ruido (sección 4.1 del prompt 01).

Fuente [MATLAB]: https://www.mathworks.com/help/images/ref/imnoise.html
- `imnoise(I,'gaussian',m,var_gauss)`: media m y varianza var_gauss.
- `imnoise(I,'gaussian',m)`: media m y varianza 0.01 (valor por defecto).
- Media y varianza se expresan como si la imagen fuera double en [0, 1]; la
  imagen se pasa a double, se recorta a [0, 1] y vuelve a la clase original.
- `imnoise(I,'salt & pepper',d)`: densidad d.

Candidatos de la llamada gaussiana para un "nivel" v del paper:
- "varianza": imnoise(I,'gaussian',0,v)  → media 0, varianza v (la del proyecto
              antes de aplicar la propuesta del informe).
- "media":    imnoise(I,'gaussian',v)    → media v, varianza 0.01 (hipótesis 1.7;
              la del proyecto desde que se aplicó la propuesta).

Las dos llamadas se implementan acá, sin depender de `procesamiento.ruido`,
para que la búsqueda dé lo mismo aunque el proyecto cambie.
"""

import numpy as np

from procesamiento import parametros as p
from procesamiento.ruido import a_uint8_matlab, ruido_sal_pimienta

# [MATLAB] imnoise: media y varianza por defecto del ruido gaussiano.
MEDIA_GAUSS_POR_DEFECTO: float = 0.0
VARIANZA_GAUSS_POR_DEFECTO: float = 0.01

LLAMADAS_GAUSS: tuple[str, ...] = ("varianza", "media")


def ruido_gaussiano(imagen: np.ndarray, media: float, varianza: float, semilla: int) -> np.ndarray:
    """`imnoise(I,'gaussian',media,varianza)` sobre [0, 1], con recorte y vuelta a uint8."""
    rng = np.random.default_rng(semilla)
    normalizada = imagen.astype(np.float64) / p.MAX_INTENSIDAD
    ruido = rng.normal(media, np.sqrt(varianza), size=imagen.shape)
    con_ruido = np.clip(normalizada + ruido, p.INTENSIDAD_MIN_NORMALIZADA, p.INTENSIDAD_MAX_NORMALIZADA)
    return a_uint8_matlab(con_ruido)


def agregar_ruido_candidato(
    imagen: np.ndarray, tipo: str, nivel: float, semilla: int, llamada_gauss: str = "varianza",
) -> np.ndarray:
    """Imagen RGB uint8 con ruido. `llamada_gauss` solo se usa si tipo == 'gaussiano'."""
    if tipo == "sal_pimienta":
        return ruido_sal_pimienta(imagen, nivel, np.random.default_rng(semilla))
    if llamada_gauss == "media":
        return ruido_gaussiano(imagen, nivel, VARIANZA_GAUSS_POR_DEFECTO, semilla)
    return ruido_gaussiano(imagen, MEDIA_GAUSS_POR_DEFECTO, nivel, semilla)
