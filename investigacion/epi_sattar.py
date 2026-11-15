"""Prueba aparte: EPI de Sattar et al. (1997) sobre las 50 filas de EPI del paper.

    python investigacion/epi_sattar.py      (escribe resultados/epi_sattar.csv)

[LITERATURA] Sattar, Floreby, Salomonsson y Lövström, "Image enhancement based on a
nonlinear multiscale method", IEEE Trans. Image Processing 6(6), 1997:
EPI = Σ(Δs − media)(Δŝ − media) / sqrt(Σ(Δs − media)² · Σ(Δŝ − media)²), donde Δs y Δŝ
son la imagen original y la procesada pasadas por una aproximación 3×3 del laplaciano.
No pude consultar el artículo original; la fórmula sale de fuentes secundarias que lo citan:
https://www.mathworks.com/matlabcentral/fileexchange/65245-edge-preservation
https://arxiv.org/pdf/2507.15306

Candidato fuera de la grilla del prompt 01; se probó porque el usuario lo pidió. Usa la
configuración de `resultados/configuracion_final.csv`, semilla 0, y no cambia esa configuración.
"""

import ast
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from investigacion import busqueda, filtros_candidatos, metricas_candidatas, ruido_candidatos  # noqa: E402

# Aproximaciones 3×3 del laplaciano: 4 vecinos, 8 vecinos y fspecial('laplacian') (alpha = 0.2,
# valor por defecto según https://www.mathworks.com/help/images/ref/fspecial.html; no verificado).
NUCLEOS: dict[str, np.ndarray] = {
    "4 vecinos": np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float64),
    "8 vecinos": np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float64),
    "fspecial 0.2": np.array([[1, 4, 1], [4, -20, 4], [1, 4, 1]], dtype=np.float64) / 6,
}


def epi_sattar(x: list[np.ndarray], y: list[np.ndarray], nucleo: np.ndarray) -> float:
    """Correlación entre los laplacianos de la imagen y de la referencia, promediada sobre los planos."""
    valores = []
    for a, b in zip(x, y):
        da = cv2.filter2D(a, cv2.CV_64F, nucleo, borderType=cv2.BORDER_CONSTANT)
        db = cv2.filter2D(b, cv2.CV_64F, nucleo, borderType=cv2.BORDER_CONSTANT)
        da, db = da - da.mean(), db - db.mean()
        valores.append(float((da * db).sum() / np.sqrt((da * da).sum() * (db * db).sum())))
    return float(np.mean(valores))


def leer_configuracion() -> dict:
    """Configuración final escrita por `busqueda.py`."""
    with (busqueda.RESULTADOS / "configuracion_final.csv").open(encoding="utf-8", newline="") as archivo:
        config = {f["variable"]: f["valor"] for f in csv.DictReader(archivo)}
    for clave, valor in config.items():
        if valor.startswith("("):
            config[clave] = ast.literal_eval(valor)
    return config


def main() -> None:
    config = leer_configuracion()
    tablas = busqueda.cargar_tablas()
    desvios = busqueda.desvios_del_paper(tablas)
    filas = [f for f in tablas if f["metrica"] == "EPI" and f["filtro"] != "HM-LCE"]
    limpia = busqueda.leer_limpia(busqueda.IMAGEN_PAPER)
    referencia = metricas_candidatas.planos(limpia)
    salida, errores = [], {}
    for fila in filas:
        _, ruido, nivel, llamada, semilla, filtro, parametros, clahe, clip = busqueda.crear_tarea(
            config, fila["filtro"], fila["ruido"], fila["nivel"])
        ruidosa = ruido_candidatos.agregar_ruido_candidato(limpia, ruido, nivel, semilla, llamada or "varianza")
        filtrada = filtros_candidatos.aplicar_filtro(ruidosa, filtro, parametros)
        casos = {"con CLAHE": metricas_candidatas.planos(busqueda._aplicar_clahe(filtrada, clahe, clip)),
                 "sin CLAHE": metricas_candidatas.planos(filtrada)}
        for (nombre, nucleo), (caso, imagen), dominio in ((n, c, d) for n in NUCLEOS.items()
                                                         for c in casos.items()
                                                         for d in metricas_candidatas.DOMINIOS):
            valor = epi_sattar(imagen[dominio], referencia[dominio], nucleo)
            error = valor - fila["valor_paper"]
            grupo = "calibración" if fila["tabla"] in busqueda.TABLAS_CALIBRACION else "validación"
            errores.setdefault((nombre, caso, dominio, grupo), []).append(
                (abs(error), abs(error) / desvios[fila["tabla"], "EPI"]))
            salida.append([nombre, caso, dominio, fila["tabla"], fila["filtro"], fila["ruido"],
                           f"{fila['nivel']:g}", f"{fila['valor_paper']:.6f}", f"{valor:.6f}", f"{error:.6f}"])
    resumenes = [[nombre, caso, dominio, f"RESUMEN {grupo}", "mediana |error| y mediana normalizada", "", "",
                  "", f"{np.median([e[0] for e in lista]):.6f}", f"{np.median([e[1] for e in lista]):.6f}"]
                 for (nombre, caso, dominio, grupo), lista in errores.items()]
    busqueda.escribir_csv("epi_sattar.csv", ["laplaciano", "caso", "dominio", "tabla", "filtro", "ruido",
                                             "nivel", "valor_paper", "nuestro", "error"], resumenes + salida)
    for fila in resumenes:
        print(*fila[:4], fila[-2], fila[-1])


if __name__ == "__main__":
    main()
