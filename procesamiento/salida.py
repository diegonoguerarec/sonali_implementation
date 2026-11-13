"""Carpeta de resultados, archivos de salida y tabla en consola."""

import csv
import json
import platform
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from procesamiento import parametros as p
from procesamiento.metricas import METRICAS

COLUMNAS = ("ruidosa", "solo_filtro", "final")
TITULOS_COLUMNAS = ("Ruidosa", "Solo filtro", "Final")
ANCHO_COLUMNA_METRICA = 13
ANCHOS_COLUMNAS = (11, 15, 0)
DECIMALES_TABLA = 4


def nombre_carpeta(
    ruta_imagen: Path, tipo_ruido: str, valor_ruido: float, tipo_filtro: str,
    clip_limit: float, semilla: int,
) -> str:
    """Nombre de la carpeta de resultados; los números con format(x, "g")."""
    campos = [
        ruta_imagen.stem,
        f"ruido-{tipo_ruido}",
        f"var-{format(valor_ruido, 'g')}",
        f"filtro-{tipo_filtro}",
        f"clahe-{format(clip_limit, 'g')}",
        f"semilla-{semilla}",
    ]
    return "__".join(campos)


def _guardar_png(ruta: Path, imagen_rgb: np.ndarray) -> None:
    """Guarda una imagen RGB como PNG sin pérdida (OpenCV espera BGR)."""
    if not cv2.imwrite(str(ruta), cv2.cvtColor(imagen_rgb, cv2.COLOR_RGB2BGR)):
        raise OSError(f"No se pudo escribir {ruta}")


def _guardar_csv(ruta: Path, metricas: dict[str, dict[str, float]]) -> None:
    """Escribe metricas.csv con precisión completa."""
    with ruta.open("w", newline="", encoding="utf-8") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(["metrica", *COLUMNAS])
        for nombre in METRICAS:
            escritor.writerow([nombre, *(repr(metricas[col][nombre]) for col in COLUMNAS)])


def datos_parametros(
    comando: str, ruta_imagen: Path, limpia: np.ndarray, tipo_ruido: str,
    valor_ruido: float, tipo_filtro: str, clip_limit: float, semilla: int, clahe: dict,
) -> dict:
    """Arma el contenido de parametros.json."""
    alto, ancho = limpia.shape[:2]
    ruido = {"tipo": tipo_ruido, "semilla": semilla, "generador": "numpy.random.default_rng"}
    if tipo_ruido == "gaussiano":
        ruido |= {"varianza": valor_ruido, "media": p.MEDIA_RUIDO_GAUSSIANO,
                  "escala": "intensidades en [0, 1]; recorte a [0, 1]; uint8 con floor(x*255 + 0.5)"}
    else:
        ruido |= {"densidad": valor_ruido,
                  "regla": "u < d/2 -> 0; d/2 <= u < d -> 255; si no, queda igual"}
    return {
        "comando": comando,
        "fecha_hora": datetime.now().isoformat(timespec="seconds"),
        "imagen": {"ruta": str(ruta_imagen.resolve()), "alto": alto, "ancho": ancho,
                   "canales": int(limpia.shape[2])},
        "ruido": ruido,
        "filtro": {
            "tipo": tipo_filtro,
            "ventana": [p.TAMANO_VENTANA_MEDIANA, p.TAMANO_VENTANA_MEDIANA],
            "pesos": p.PESOS_MEDIANA_PONDERADA.tolist(),
            "definicion": "repetir cada valor según su peso y tomar la mediana",
            "regla_empate": "suma de pesos par: promedio de los 2 centrales, sin fracción (medfilt2)",
            "bordes": "relleno con ceros (medfilt2)",
        },
        "clahe": {
            "clip_limit_normalizado": clip_limit,
            "teselas": list(p.TESELAS_CLAHE),
            "bins": p.BINS_CLAHE,
            "distribucion": p.DISTRIBUCION_CLAHE,
            "pixeles_por_tesela": clahe["pixeles_por_tesela"],
            "clip_cuentas": clahe["clip_cuentas"],
            "clip_limit_opencv": clahe["clip_limit_opencv"],
        },
        "metricas": {
            "referencia": "imagen limpia; RGB completo; float64",
            "PSNR": "10*log10(255^2 / MSE), MSE sobre todos los valores RGB; inf si MSE = 0",
            "SSIM": ("gaussiana 11x11, sigma 1.5, bordes replicados, por plano; "
                     "C1=(0.01*255)^2, C2=(0.03*255)^2, C3=C2/2, exponentes 1; promedio del mapa HxWx3"),
            "CoC": "Pearson entre todos los valores RGB",
            "EPI": "sum|X'(m,n+1)-X'(m,n)| / sum|X(m,n+1)-X(m,n)|, 3 canales",
        },
        "versiones": {"python": platform.python_version(), "numpy": np.__version__,
                      "opencv": cv2.__version__},
        "procedencia": p.TABLA_PROCEDENCIA,
    }


def guardar_resultados(
    dir_resultado: Path, nombre: str, resultado: dict, datos: dict,
) -> Path:
    """Crea (o sobrescribe) la carpeta y guarda los 3 PNG, el CSV y el JSON."""
    carpeta = Path(dir_resultado) / nombre
    if carpeta.exists():
        shutil.rmtree(carpeta)
    carpeta.mkdir(parents=True)
    for columna in COLUMNAS:
        _guardar_png(carpeta / f"{columna}.png", resultado["imagenes"][columna])
    _guardar_csv(carpeta / "metricas.csv", resultado["metricas"])
    with (carpeta / "parametros.json").open("w", encoding="utf-8") as archivo:
        json.dump(datos, archivo, ensure_ascii=False, indent=2)
    return carpeta


def formatear_tabla(
    ruta_imagen: Path, limpia: np.ndarray, tipo_ruido: str, valor_ruido: float,
    tipo_filtro: str, clip_limit: float, semilla: int, metricas: dict[str, dict[str, float]],
) -> str:
    """Devuelve el encabezado y la tabla de métricas con 4 decimales."""
    alto, ancho = limpia.shape[:2]
    lineas = [
        f"Imagen: {ruta_imagen.name} ({alto}×{ancho}) | ruido: {tipo_ruido} {format(valor_ruido, 'g')}"
        f" | filtro: {tipo_filtro} | CLAHE: {format(clip_limit, 'g')} | semilla: {semilla}",
        "",
        "Métrica".ljust(ANCHO_COLUMNA_METRICA)
        + "".join(t.ljust(a) for t, a in zip(TITULOS_COLUMNAS, ANCHOS_COLUMNAS)).rstrip(),
    ]
    for nombre in METRICAS:
        valores = [f"{metricas[col][nombre]:.{DECIMALES_TABLA}f}" for col in COLUMNAS]
        lineas.append(nombre.ljust(ANCHO_COLUMNA_METRICA)
                      + "".join(v.ljust(a) for v, a in zip(valores, ANCHOS_COLUMNAS)).rstrip())
    return "\n".join(lineas)
