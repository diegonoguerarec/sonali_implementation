"""Método de Sonali et al. (2019): ruido → separar RGB → filtrar → CLAHE → unir."""

import numpy as np

from procesamiento.filtros import FILTROS
from procesamiento.metricas import calcular_metricas
from procesamiento.realce import aplicar_clahe, datos_clahe
from procesamiento.ruido import agregar_ruido


def _por_canal(imagen: np.ndarray, funcion) -> np.ndarray:
    """Separa los canales R, G y B, aplica `funcion` a cada uno y los vuelve a unir."""
    canales = [funcion(imagen[:, :, c]) for c in range(imagen.shape[2])]
    return np.stack(canales, axis=2)


def procesar_imagen(
    limpia: np.ndarray,
    tipo_ruido: str,
    valor_ruido: float,
    tipo_filtro: str,
    clip_limit: float,
    semilla: int,
) -> dict:
    """Aplica el método a una imagen RGB uint8 limpia, sin leer ni escribir archivos.

    Devuelve las imágenes "ruidosa", "solo_filtro" y "final", sus métricas contra
    la limpia y los datos de CLAHE (píxeles por tesela y clip en cuentas).
    """
    ruidosa = agregar_ruido(limpia, tipo_ruido, valor_ruido, semilla)
    solo_filtro = _por_canal(ruidosa, FILTROS[tipo_filtro])
    final = _por_canal(solo_filtro, lambda canal: aplicar_clahe(canal, clip_limit))
    imagenes = {"ruidosa": ruidosa, "solo_filtro": solo_filtro, "final": final}
    alto, ancho = limpia.shape[:2]
    return {
        "imagenes": imagenes,
        "metricas": {nombre: calcular_metricas(img, limpia) for nombre, img in imagenes.items()},
        "clahe": datos_clahe(clip_limit, alto, ancho),
    }
