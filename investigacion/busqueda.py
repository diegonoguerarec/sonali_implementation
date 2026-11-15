"""Búsqueda de la configuración que mejor reproduce las Tablas 3 a 8 de Sonali et al. (2019).

Corre las etapas de la sección 5 del prompt 01 y escribe `investigacion/resultados/`:

    python investigacion/busqueda.py

Una "evaluación del proceso completo" es una corrida ruido → filtro → CLAHE →
métricas sobre una imagen. Las métricas de cada corrida quedan en caché (en
memoria) con todas sus variantes, así que comparar definiciones de métricas no
cuesta corridas nuevas. Todo es determinista: las mismas entradas dan los
mismos CSV.
"""

import csv
import itertools
import multiprocessing
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from investigacion import clahe_matlab  # noqa: E402
from investigacion import filtros_candidatos  # noqa: E402
from investigacion import metricas_candidatas  # noqa: E402
from investigacion import ruido_candidatos  # noqa: E402

CARPETA = Path(__file__).resolve().parent
RESULTADOS = CARPETA / "resultados"
CSV_PAPER = RAIZ / "prompts" / "01_tablas_paper.csv"
PDF_PAPER = RAIZ / "Sonali.pdf"
CARPETA_STARE = RAIZ / "STARE"

# [VERIFICADO] Sección 1.1 del prompt 01 (se vuelve a medir en la etapa 1).
IMAGEN_PAPER = "im0001"
IMAGENES_STARE = ("im0001", "im0002", "im0003", "im0004", "im0005", "im0006")
# [PAPER] Fig. 5.
CLIP_LIMIT = 0.0005
# Sección 4.3 del prompt 01: sensibilidad del clip limit.
CLIPS_SENSIBILIDAD = (0.0005, 0.001, 0.005, 0.01)
# Sección 5 del prompt 01: semillas de la evaluación final.
SEMILLAS_FINALES = tuple(range(10))
# Regla 5 del prompt 01: tope de evaluaciones del proceso completo.
MAX_EVALUACIONES = 5000
# Sección 5 del prompt 01: criterio "cerca".
TOLERANCIA_CERCA = {"PSNR": 0.5, "SSIM": 0.01, "CoC": 0.01, "EPI": 0.01}
MAX_VUELTAS = 3
# [DECISIÓN] Un candidato solo desplaza al valor por defecto si baja la mediana del error
# normalizado en más de esta tolerancia; por debajo, la diferencia no se considera evidencia.
TOLERANCIA_EMPATE = 0.01
PROCESOS = max(1, min(6, (multiprocessing.cpu_count() or 2) - 1))

FILTROS = ("mediana", "wiener", "media", "gaussiano", "mediana_ponderada")
METRICAS = ("PSNR", "SSIM", "CoC", "EPI")
# [PAPER] Tablas 3–6 (calibración) y Tablas 7–8 (validación).
CONDICIONES_CALIBRACION = [(r, n) for r in ("sal_pimienta", "gaussiano") for n in (0.001, 0.01, 0.1, 0.2)]
CONDICIONES_VALIDACION = [("sal_pimienta", 0.005), ("sal_pimienta", 0.5)]
TABLAS_CALIBRACION = ("Tabla 3", "Tabla 4", "Tabla 5", "Tabla 6")

VARIANTES_CLAHE = list(itertools.product(clahe_matlab.VARIANTES_TABLA, clahe_matlab.VARIANTES_FINAL,
                                         clahe_matlab.VARIANTES_RELLENO))

# Punto de partida: valores por defecto de MATLAB, σ = 1 del paper, métricas del proyecto.
CONFIG_INICIAL: dict = {
    "ruido_gauss": "varianza",
    "clahe": VARIANTES_CLAHE[0],
    **{f: filtros_candidatos.CANDIDATOS[f][0] for f in FILTROS},
    "PSNR": "rgb|255",
    "SSIM": "plano|rgb",
    "CoC": "rgb|limpia",
    "EPI": "rgb",
}


# --- Proceso completo (se ejecuta en los procesos de trabajo) -------------------

class _CacheLRU(OrderedDict):
    """Caché de imágenes con tamaño máximo (descarta la menos usada)."""

    def __init__(self, maximo: int) -> None:
        super().__init__()
        self.maximo = maximo

    def obtener(self, clave, crear):
        if clave in self:
            self.move_to_end(clave)
            return self[clave]
        valor = crear()
        self[clave] = valor
        if len(self) > self.maximo:
            self.popitem(last=False)
        return valor


_LIMPIAS: dict[str, np.ndarray] = {}
_RUIDOSAS = _CacheLRU(8)
_FILTRADAS = _CacheLRU(16)


def leer_limpia(imagen: str) -> np.ndarray:
    """Imagen limpia de STARE en RGB uint8."""
    if imagen not in _LIMPIAS:
        bgr = cv2.imread(str(CARPETA_STARE / f"{imagen}.ppm"), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"No se pudo leer STARE/{imagen}.ppm")
        _LIMPIAS[imagen] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return _LIMPIAS[imagen]


def _aplicar_clahe(imagen: np.ndarray, clahe, clip: float) -> np.ndarray:
    """CLAHE por canal: variante de `adapthisteq`, "opencv" (la original del proyecto) o None (sin CLAHE)."""
    if clahe is None:
        return imagen
    if clahe == "opencv":
        return np.stack([clahe_matlab.clahe_opencv(imagen[:, :, c], clip) for c in range(3)], axis=2)
    tabla, final, relleno = clahe
    canales = [clahe_matlab.adapthisteq(imagen[:, :, c], clip, tabla, final, relleno) for c in range(3)]
    return np.stack(canales, axis=2)


def evaluar_tarea(tarea: tuple) -> tuple[tuple, dict]:
    """Corre el proceso completo de una tarea y devuelve todas las variantes de las métricas.

    tarea = (imagen, ruido, nivel, llamada_gauss, semilla, filtro, parámetros, clahe, clip).
    filtro "ninguno" evalúa la imagen ruidosa; clahe None, la imagen solo filtrada.
    """
    imagen, ruido, nivel, llamada, semilla, filtro, parametros, clahe, clip = tarea
    limpia = leer_limpia(imagen)
    clave_ruido = (imagen, ruido, nivel, llamada, semilla)
    if ruido == "ninguno":
        ruidosa = limpia
    else:
        ruidosa = _RUIDOSAS.obtener(clave_ruido, lambda: ruido_candidatos.agregar_ruido_candidato(
            limpia, ruido, nivel, semilla, llamada or "varianza"))
    if filtro == "ninguno":
        filtrada = ruidosa
    else:
        filtrada = _FILTRADAS.obtener(clave_ruido + (filtro, parametros), lambda: (
            filtros_candidatos.aplicar_filtro(ruidosa, filtro, parametros)))
    final = _aplicar_clahe(filtrada, clahe, clip)
    valores = metricas_candidatas.calcular_todas(final, limpia, ruidosa)
    valores["media", "rgb"] = float(final.mean())
    valores["media", "B"] = float(final[:, :, 2].mean())
    return tarea, valores


def crear_tarea(config: dict, filtro: str, ruido: str, nivel: float, semilla: int = 0,
                imagen: str = IMAGEN_PAPER, clip: float = CLIP_LIMIT, **cambios) -> tuple:
    """Tarea del proceso completo para una fila; `cambios` reemplaza campos (clahe, parametros)."""
    llamada = config["ruido_gauss"] if ruido == "gaussiano" else ""
    parametros = cambios.get("parametros", config.get(filtro, ()))
    clahe = cambios.get("clahe", config["clahe"])
    return (imagen, ruido, nivel, llamada, semilla, filtro, parametros, clahe, clip)


class Evaluador:
    """Corre las tareas que falten (en paralelo) y guarda sus métricas."""

    def __init__(self) -> None:
        self.metricas: dict[tuple, dict] = {}
        self.evaluaciones = 0
        self._pool = multiprocessing.Pool(PROCESOS) if PROCESOS > 1 else None

    def evaluar(self, tareas) -> None:
        pendientes = sorted({t for t in tareas if t not in self.metricas}, key=repr)
        if self.evaluaciones + len(pendientes) > MAX_EVALUACIONES:
            raise RuntimeError(f"Se superaría el tope de {MAX_EVALUACIONES} evaluaciones (regla 5).")
        self.evaluaciones += len(pendientes)
        if not pendientes:
            return
        mapear = self._pool.imap_unordered if self._pool else map
        argumentos = (pendientes, 4) if self._pool else (pendientes,)
        for tarea, valores in mapear(evaluar_tarea, *argumentos):
            self.metricas[tarea] = valores

    def valor(self, tarea: tuple, metrica: str, variante: str) -> float:
        self.evaluar([tarea])
        return self.metricas[tarea][metrica, variante]

    def cerrar(self) -> None:
        if self._pool:
            self._pool.close()
            self._pool.join()


# --- Tablas del paper -----------------------------------------------------------

def cargar_tablas() -> list[dict]:
    """Filas de `prompts/01_tablas_paper.csv`, con nivel y valor como números."""
    with CSV_PAPER.open(encoding="utf-8", newline="") as archivo:
        filas = list(csv.DictReader(archivo))
    for fila in filas:
        fila["nivel"] = float(fila["nivel"])
        fila["valor_paper"] = float(fila["valor_paper"])
    return filas


def desvios_del_paper(filas: list[dict]) -> dict[tuple[str, str], float]:
    """Desvío estándar (poblacional) de los valores del paper por tabla y métrica, sin HM-LCE."""
    grupos: dict[tuple[str, str], list[float]] = {}
    for fila in filas:
        if fila["filtro"] != "HM-LCE":
            grupos.setdefault((fila["tabla"], fila["metrica"]), []).append(fila["valor_paper"])
    return {clave: float(np.std(valores)) for clave, valores in grupos.items()}


def leer_tablas_del_pdf() -> dict[tuple, float] | None:
    """Tablas 3 a 8 leídas del PDF con `pdftotext -layout`; None si no está la herramienta.

    Tablas 3–6: 20 renglones "nivel  s&p  gaussiano" por tabla, en el orden mediana,
    Wiener, media, gaussiano y mediana ponderada. Tablas 7–8: renglones "PSNR (in dB)",
    "SSIM", "CoC" y "EPI" con 6 columnas (orden verificado en la página renderizada).
    """
    if shutil.which("pdftotext") is None or not PDF_PAPER.is_file():
        return None
    texto = subprocess.run(["pdftotext", "-layout", str(PDF_PAPER), "-"], capture_output=True,
                           text=True, check=True).stdout
    leidas: dict[tuple, float] = {}
    patron = re.compile(r"(?<![\d.])(0\.001|0\.01|0\.1|0\.2)\s+(\d+\.\d{6})\s+(\d+\.\d{6})(?!\d)")
    pares = [(float(a), float(b), float(c)) for a, b, c in patron.findall(texto)]
    orden = ("mediana", "wiener", "media", "gaussiano", "mediana_ponderada")
    for i, (nivel, sal, gauss) in enumerate(pares[:80]):
        tabla, metrica = f"Tabla {3 + i // 20}", METRICAS[i // 20]
        filtro = orden[(i % 20) // 4]
        leidas[tabla, metrica, filtro, "sal_pimienta", nivel] = sal
        leidas[tabla, metrica, filtro, "gaussiano", nivel] = gauss
    columnas = ("HM-LCE", "mediana", "gaussiano", "mediana_ponderada", "wiener", "media")
    patron_78 = re.compile(r"^\s*(PSNR \(in dB\)|SSIM|CoC|EPI)\s+((?:\d+\.\d{6}\s*){6})$", re.MULTILINE)
    for i, (nombre, numeros) in enumerate(patron_78.findall(texto)[:8]):
        tabla, nivel = ("Tabla 7", 0.005) if i < 4 else ("Tabla 8", 0.5)
        for filtro, numero in zip(columnas, numeros.split()):
            leidas[tabla, nombre.split()[0], filtro, "sal_pimienta", nivel] = float(numero)
    return leidas


def diferencias_csv_pdf(filas: list[dict]) -> int | None:
    """Cantidad de filas del CSV que no coinciden en valor y posición con el PDF (None: sin pdftotext)."""
    leidas = leer_tablas_del_pdf()
    if leidas is None:
        return None
    distintas = sum(
        leidas.get((f["tabla"], f["metrica"], f["filtro"], f["ruido"], f["nivel"])) != f["valor_paper"]
        for f in filas)
    return distintas + abs(len(leidas) - len(filas))


# --- Error de una configuración ----------------------------------------------------

def valor_de_fila(ev: Evaluador, config: dict, fila: dict, **opciones) -> float:
    """Nuestro valor para una fila del paper con una configuración."""
    tarea = crear_tarea(config, fila["filtro"], fila["ruido"], fila["nivel"], **opciones)
    return ev.valor(tarea, fila["metrica"], config[fila["metrica"]])


def errores_normalizados(ev: Evaluador, config: dict, filas: list[dict], desvios: dict) -> np.ndarray:
    """|nuestro − paper| / desvío del paper (misma métrica y tabla), fila por fila."""
    ev.evaluar(crear_tarea(config, f["filtro"], f["ruido"], f["nivel"]) for f in filas)
    return np.array([abs(valor_de_fila(ev, config, f) - f["valor_paper"]) / desvios[f["tabla"], f["metrica"]]
                     for f in filas])


def criterio(errores: np.ndarray) -> tuple[float, float]:
    """(mediana, promedio) de los errores absolutos normalizados: criterio principal y secundario."""
    return float(np.median(errores)), float(np.mean(errores))


def _f(x: float) -> str:
    """Número con 6 decimales para los CSV."""
    return f"{x:.6f}"


def escribir_csv(nombre: str, encabezado: list[str], filas: list[list]) -> None:
    RESULTADOS.mkdir(exist_ok=True)
    with (RESULTADOS / nombre).open("w", encoding="utf-8", newline="") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(encabezado)
        escritor.writerows(filas)


# --- Etapa 1: base ----------------------------------------------------------------

def _correlacion_con_figura(figura: np.ndarray, imagen: str) -> float:
    """Correlación entre una figura del PDF (gris) y una imagen de STARE redimensionada y suavizada."""
    gris = cv2.cvtColor(leer_limpia(imagen), cv2.COLOR_RGB2GRAY)
    chica = cv2.resize(gris, (figura.shape[1], figura.shape[0]), interpolation=cv2.INTER_AREA)
    a = cv2.GaussianBlur(figura.astype(np.float64), (0, 0), 3)
    b = cv2.GaussianBlur(chica.astype(np.float64), (0, 0), 3)
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def identificar_imagen() -> list[list]:
    """Sección 1.1: correlación de las Figs. 1 y 7 del PDF con cada imagen de STARE."""
    if shutil.which("pdfimages") is None or not PDF_PAPER.is_file():
        return [["1.1", "pdfimages no disponible: no se pudo repetir la identificación", "", ""]]
    filas = []
    with tempfile.TemporaryDirectory(dir=CARPETA) as temporal:
        subprocess.run(["pdfimages", "-j", str(PDF_PAPER), str(Path(temporal) / "img")], check=True)
        # Objetos del PDF: img-002 es la Fig. 1 (limpia) e img-006, la ruidosa de la Fig. 7.
        for figura, archivo in (("Fig. 1", "img-002.jpg"), ("Fig. 7 (ruidosa)", "img-006.jpg")):
            gris = cv2.imread(str(Path(temporal) / archivo), cv2.IMREAD_GRAYSCALE)
            for imagen in IMAGENES_STARE:
                filas.append(["1.1", f"correlación {figura} con {imagen}",
                              _f(_correlacion_con_figura(gris, imagen)), ""])
    return filas


def etapa_base(ev: Evaluador, filas_paper: list[dict]) -> None:
    """Reproduce los números de la sección 1 del prompt 01 (valor nuestro y valor del prompt)."""
    base = dict(CONFIG_INICIAL)
    filas: list[list] = []

    def psnr(filtro, ruido, nivel, **opciones) -> float:
        return ev.valor(crear_tarea(base, filtro, ruido, nivel, **opciones), "PSNR", "rgb|255")

    def ssim(filtro, ruido, nivel, **opciones) -> float:
        return ev.valor(crear_tarea(base, filtro, ruido, nivel, **opciones), "SSIM", "plano|rgb")

    distintas = diferencias_csv_pdf(filas_paper)
    filas.append(["3", "filas del CSV que difieren del PDF (valor o posición)",
                  "sin pdftotext" if distintas is None else distintas, 0])
    filas += identificar_imagen()

    for clahe, etiqueta, esperado in (("opencv", "OpenCV", (25.51, 0.864)),
                                      (base["clahe"], "adapthisteq reconstruido", (36.92, 0.975))):
        filas.append(["1.2", f"CLAHE sobre la limpia, {etiqueta}: PSNR",
                      _f(psnr("ninguno", "ninguno", 0.0, clahe=clahe)), esperado[0]])
        filas.append(["1.2", f"CLAHE sobre la limpia, {etiqueta}: SSIM",
                      _f(ssim("ninguno", "ninguno", 0.0, clahe=clahe)), esperado[1]])

    tabla_13 = {("sal_pimienta", 0.001): (34.50, 50.64, 26.09, 37.15), ("sal_pimienta", 0.005): (27.44, 50.07, 26.08, 37.11),
                ("sal_pimienta", 0.01): (24.42, 49.62, 26.07, 37.08), ("sal_pimienta", 0.1): (14.35, 31.40, 24.63, 30.20),
                ("sal_pimienta", 0.2): (11.36, 22.52, 20.37, 22.22), ("sal_pimienta", 0.5): (7.38, 11.33, 10.98, 11.27),
                ("gaussiano", 0.001): (30.41, 35.21, 23.22, 31.88), ("gaussiano", 0.01): (20.91, 25.78, 18.56, 24.11),
                ("gaussiano", 0.1): (12.01, 16.53, 12.64, 15.27), ("gaussiano", 0.2): (9.83, 13.87, 11.04, 12.76)}
    for (ruido, nivel), esperado in tabla_13.items():
        nuestros = (psnr("ninguno", ruido, nivel, clahe=None), psnr("mediana_ponderada", ruido, nivel, clahe=None),
                    psnr("mediana_ponderada", ruido, nivel, clahe="opencv"), psnr("mediana_ponderada", ruido, nivel))
        for nombre, nuestro, prompt in zip(("ruidosa", "solo filtro", "+ CLAHE OpenCV", "+ CLAHE MATLAB"),
                                           nuestros, esperado):
            filas.append(["1.3", f"PSNR mediana ponderada, {ruido} {nivel:g}, {nombre}", _f(nuestro), prompt])
    filas.append(["1.3", "SSIM mediana ponderada + CLAHE MATLAB, sal_pimienta 0.001",
                  _f(ssim("mediana_ponderada", "sal_pimienta", 0.001)), 0.971])
    filas.append(["1.3", "SSIM mediana ponderada + CLAHE MATLAB, gaussiano 0.001",
                  _f(ssim("mediana_ponderada", "gaussiano", 0.001)), 0.709])

    for nivel, esperado in ((0.2, (11.36, 31.47, 30.45)), (0.5, (7.38, 14.47, 14.46))):
        nuestros = (psnr("ninguno", "sal_pimienta", nivel, clahe=None),
                    psnr("mediana", "sal_pimienta", nivel, clahe=None), psnr("mediana", "sal_pimienta", nivel))
        for nombre, nuestro, prompt in zip(("ruidosa", "solo mediana", "+ CLAHE"), nuestros, esperado):
            filas.append(["1.4", f"PSNR mediana 3×3, sal_pimienta {nivel:g}, {nombre}", _f(nuestro), prompt])

    literal = {"parametros": ("W/12",)}
    tarea = crear_tarea(base, "mediana_ponderada_literal", "sal_pimienta", 0.001, clahe=None, **literal)
    filas.append(["1.5", "media de intensidad de la limpia",
                  _f(ev.valor(crear_tarea(base, "ninguno", "ninguno", 0.0, clahe=None), "media", "rgb")), 72.9])
    filas.append(["1.5", "media de intensidad tras la mediana literal W/12", _f(ev.valor(tarea, "media", "rgb")), 6.1])
    filas.append(["1.5", "PSNR mediana literal W/12 sin CLAHE, sal_pimienta 0.001",
                  _f(ev.valor(tarea, "PSNR", "rgb|255")), 8.95])
    filas.append(["1.5", "PSNR mediana literal W/12 con CLAHE, sal_pimienta 0.001",
                  _f(psnr("mediana_ponderada_literal", "sal_pimienta", 0.001, **literal)), 9.07])
    filas.append(["4.2", "PSNR mediana literal W sin dividir (saturada) con CLAHE, sal_pimienta 0.001",
                  _f(psnr("mediana_ponderada_literal", "sal_pimienta", 0.001, parametros=("W_saturada",))), ""])

    semillas = [psnr("mediana_ponderada", "gaussiano", 0.001, semilla=s) for s in range(5)]
    filas.append(["1.6", "PSNR gaussiano 0.001, mínimo de 5 semillas", _f(min(semillas)), 31.88])
    filas.append(["1.6", "PSNR gaussiano 0.001, máximo de 5 semillas", _f(max(semillas)), 31.90])
    imagenes = [psnr("mediana_ponderada", "sal_pimienta", 0.001, imagen=i) for i in IMAGENES_STARE]
    filas.append(["1.6", "PSNR sal_pimienta 0.001, mínimo de las 6 imágenes", _f(min(imagenes)), 36.4])
    filas.append(["1.6", "PSNR sal_pimienta 0.001, máximo de las 6 imágenes", _f(max(imagenes)), 38.8])

    media = dict(base, ruido_gauss="media")
    tabla_17 = {"mediana_ponderada": ((24.07, 0.256), (23.64, 0.247), (17.78, 0.207), (13.46, 0.200)),
                "mediana": ((26.18, 0.389), (25.58, 0.375), (18.28, 0.311), (13.61, 0.290))}
    for filtro, esperados in tabla_17.items():
        for nivel, (e_psnr, e_ssim) in zip((0.001, 0.01, 0.1, 0.2), esperados):
            tarea = crear_tarea(media, filtro, "gaussiano", nivel)
            filas.append(["1.7", f"PSNR {filtro}, gaussiano media = {nivel:g} y varianza 0.01",
                          _f(ev.valor(tarea, "PSNR", "rgb|255")), e_psnr])
            filas.append(["1.7", f"SSIM {filtro}, gaussiano media = {nivel:g} y varianza 0.01",
                          _f(ev.valor(tarea, "SSIM", "plano|rgb")), e_ssim])
    escribir_csv("base_seccion1.csv", ["seccion", "descripcion", "valor_nuestro", "valor_prompt"], filas)


# --- Etapa 2: control de CLAHE ----------------------------------------------------

def etapa_control_clahe(ev: Evaluador) -> None:
    """Techo (CLAHE sobre la limpia) y media del canal B (Fig. 8) para cada implementación."""
    base = dict(CONFIG_INICIAL)
    filas = []
    for clahe in VARIANTES_CLAHE + ["opencv"]:
        techo = crear_tarea(base, "ninguno", "ninguno", 0.0, clahe=clahe)
        fig8 = crear_tarea(base, "mediana", "sal_pimienta", 0.01, clahe=clahe)
        nombre = clahe if clahe == "opencv" else "/".join(clahe)
        filas.append([nombre, _f(ev.valor(techo, "PSNR", "rgb|255")), _f(ev.valor(techo, "SSIM", "plano|rgb")),
                      _f(ev.valor(techo, "EPI", "rgb")), _f(ev.valor(techo, "media", "B")),
                      _f(ev.valor(fig8, "media", "B"))])
    ruidosa = crear_tarea(base, "ninguno", "sal_pimienta", 0.01, clahe=None)
    filas.append(["sin CLAHE (ruidosa s&p 0.01)", "", "", "", "", _f(ev.valor(ruidosa, "media", "B"))])
    escribir_csv("control_clahe.csv", ["clahe (tabla/final/relleno)", "techo_psnr", "techo_ssim", "techo_epi",
                                       "media_B_limpia_realzada", "media_B_fig8_mediana_sp001"], filas)


# --- Etapas 3 y 4: búsqueda -------------------------------------------------------

def etapa_compartidas(ev: Evaluador, config: dict, filas_cal: list[dict], desvios: dict,
                      vuelta: int, registro: list[list]) -> dict:
    """Etapa 3: CLAHE (detalles inciertos), llamada del ruido gaussiano y definición de las métricas.

    Recorre el producto completo con los filtros fijos y toma el mínimo de
    (mediana, promedio). Después vuelve cada variable a su valor por defecto si
    eso no empeora la mediana en más de TOLERANCIA_EMPATE (`simplificar`).
    """
    mejor, mejor_criterio = None, None
    combinaciones = list(itertools.product(*(metricas_candidatas.VARIANTES[m] for m in METRICAS)))
    for clahe, llamada in itertools.product(VARIANTES_CLAHE, ruido_candidatos.LLAMADAS_GAUSS):
        candidato = dict(config, clahe=clahe, ruido_gauss=llamada)
        ev.evaluar(crear_tarea(candidato, f["filtro"], f["ruido"], f["nivel"]) for f in filas_cal)
        errores = {}
        for metrica in METRICAS:
            propias = [f for f in filas_cal if f["metrica"] == metrica]
            for variante in metricas_candidatas.VARIANTES[metrica]:
                errores[metrica, variante] = errores_normalizados(
                    ev, dict(candidato, **{metrica: variante}), propias, desvios)
        for variantes in combinaciones:
            juntos = np.concatenate([errores[m, v] for m, v in zip(METRICAS, variantes)])
            actual = criterio(juntos)
            registro.append([vuelta, "/".join(clahe), llamada, *variantes, _f(actual[0]), _f(actual[1])])
            if mejor_criterio is None or actual < mejor_criterio:
                mejor_criterio = actual
                mejor = dict(candidato, **dict(zip(METRICAS, variantes)))
    return simplificar(ev, mejor, mejor_criterio[0], filas_cal, desvios, vuelta)


SIMPLIFICACIONES: list[list] = []


def simplificar(ev: Evaluador, config: dict, mejor_mediana: float, filas_cal: list[dict],
                desvios: dict, vuelta: int) -> dict:
    """Vuelve al valor por defecto las variables compartidas cuya mejora es menor que la tolerancia."""
    reversiones = [("ruido_gauss", CONFIG_INICIAL["ruido_gauss"])]
    reversiones += [(("clahe", i), CONFIG_INICIAL["clahe"][i]) for i in range(3)]
    reversiones += [(m, CONFIG_INICIAL[m]) for m in METRICAS]
    for variable, defecto in reversiones:
        if isinstance(variable, tuple):
            clahe = tuple(defecto if i == variable[1] else v for i, v in enumerate(config["clahe"]))
            prueba, nombre, actual = dict(config, clahe=clahe), f"clahe[{variable[1]}]", config["clahe"][variable[1]]
        else:
            prueba, nombre, actual = dict(config, **{variable: defecto}), variable, config[variable]
        if prueba == config:
            continue
        mediana = criterio(errores_normalizados(ev, prueba, filas_cal, desvios))[0]
        aceptada = mediana <= mejor_mediana + TOLERANCIA_EMPATE
        SIMPLIFICACIONES.append([vuelta, nombre, actual, defecto, _f(mejor_mediana), _f(mediana),
                                 "sí" if aceptada else "no"])
        if aceptada:
            config = prueba
    return config


def etapa_filtros(ev: Evaluador, config: dict, filas_cal: list[dict], desvios: dict,
                  vuelta: int, registro: list[list]) -> dict:
    """Etapa 4: la grilla de cada filtro por separado, sobre las filas de ese filtro.

    Entre los candidatos que quedan a menos de TOLERANCIA_EMPATE del mejor, se
    elige el que está primero en la lista (el valor por defecto de MATLAB).
    """
    nueva = dict(config)
    for filtro in FILTROS:
        propias = [f for f in filas_cal if f["filtro"] == filtro]
        evaluados = []
        for parametros in filtros_candidatos.CANDIDATOS[filtro]:
            actual = criterio(errores_normalizados(ev, dict(config, **{filtro: parametros}), propias, desvios))
            registro.append([vuelta, filtro, repr(parametros), _f(actual[0]), _f(actual[1])])
            evaluados.append((parametros, actual))
        mejor_mediana = min(actual for _, actual in evaluados)[0]
        # El primero de la lista (el valor por defecto va primero) que queda dentro de la tolerancia.
        nueva[filtro] = next(par for par, actual in evaluados if actual[0] <= mejor_mediana + TOLERANCIA_EMPATE)
    return nueva


def buscar(ev: Evaluador, filas_cal: list[dict], desvios: dict) -> dict:
    """Etapas 3 a 5: alterna compartidas y filtros hasta que la configuración no cambie."""
    config = dict(CONFIG_INICIAL)
    registro_3, registro_4, historial = [], [], []
    inicial = criterio(errores_normalizados(ev, config, filas_cal, desvios))
    historial.append([0, "inicial", repr(config), _f(inicial[0]), _f(inicial[1])])
    for vuelta in range(1, MAX_VUELTAS + 1):
        anterior = dict(config)
        config = etapa_compartidas(ev, config, filas_cal, desvios, vuelta, registro_3)
        actual = criterio(errores_normalizados(ev, config, filas_cal, desvios))
        historial.append([vuelta, "etapa 3", repr(config), _f(actual[0]), _f(actual[1])])
        config = etapa_filtros(ev, config, filas_cal, desvios, vuelta, registro_4)
        actual = criterio(errores_normalizados(ev, config, filas_cal, desvios))
        historial.append([vuelta, "etapa 4", repr(config), _f(actual[0]), _f(actual[1])])
        print(f"  vuelta {vuelta}: mediana {actual[0]:.4f}, promedio {actual[1]:.4f} "
              f"({ev.evaluaciones} evaluaciones)", flush=True)
        if config == anterior:
            break
    escribir_csv("etapa3_compartidas.csv", ["vuelta", "clahe", "ruido_gauss", *METRICAS,
                                            "mediana_error", "promedio_error"], registro_3)
    escribir_csv("etapa4_filtros.csv", ["vuelta", "filtro", "parametros", "mediana_error", "promedio_error"],
                 registro_4)
    escribir_csv("simplificaciones.csv", ["vuelta", "variable", "valor_optimo", "valor_por_defecto",
                                          "mediana_optimo", "mediana_con_defecto", "vuelve_al_defecto"],
                 SIMPLIFICACIONES)
    escribir_csv("historial.csv", ["vuelta", "etapa", "configuracion", "mediana_error", "promedio_error"],
                 historial)
    return config


def evidencia_por_variable(ev: Evaluador, config: dict, filas_cal: list[dict], desvios: dict) -> None:
    """Error de cada candidato cambiando una sola variable de la configuración final."""
    filas = []

    def agregar(variable: str, candidato, cambios: dict, propias: list[dict] | None = None) -> None:
        prueba = dict(config, **cambios)
        todo = criterio(errores_normalizados(ev, prueba, filas_cal, desvios))
        parcial = criterio(errores_normalizados(ev, prueba, propias, desvios)) if propias else ("", "")
        filas.append([variable, candidato, "sí" if prueba == config else "",
                      _f(todo[0]), _f(todo[1]), *(x if x == "" else _f(x) for x in parcial)])

    for llamada in ruido_candidatos.LLAMADAS_GAUSS:
        agregar("ruido_gauss", llamada, {"ruido_gauss": llamada},
                [f for f in filas_cal if f["ruido"] == "gaussiano"])
    for posicion, (nombre, variantes) in enumerate((("clahe_tabla", clahe_matlab.VARIANTES_TABLA),
                                                    ("clahe_final", clahe_matlab.VARIANTES_FINAL),
                                                    ("clahe_relleno", clahe_matlab.VARIANTES_RELLENO))):
        for variante in variantes:
            clahe = tuple(variante if i == posicion else v for i, v in enumerate(config["clahe"]))
            agregar(nombre, variante, {"clahe": clahe})
    agregar("clahe_implementacion", "opencv (solo referencia)", {"clahe": "opencv"})
    for metrica in METRICAS:
        for variante in metricas_candidatas.VARIANTES[metrica]:
            agregar(metrica, variante, {metrica: variante}, [f for f in filas_cal if f["metrica"] == metrica])
    for filtro in FILTROS:
        for parametros in filtros_candidatos.CANDIDATOS[filtro]:
            agregar(filtro, repr(parametros), {filtro: parametros},
                    [f for f in filas_cal if f["filtro"] == filtro])
    escribir_csv("evidencia_variables.csv",
                 ["variable", "candidato", "elegido", "mediana_error_calibracion", "promedio_error_calibracion",
                  "mediana_error_filas_propias", "promedio_error_filas_propias"], filas)


# --- Etapa 6: semillas, comparación final y sensibilidad -------------------------

def categoria(metrica: str, error: float, desvio_semillas: float) -> str:
    """Criterio de coincidencia de la sección 5 del prompt 01."""
    if abs(error) <= 2 * desvio_semillas:
        return "coincide"
    if abs(error) <= TOLERANCIA_CERCA[metrica]:
        return "cerca"
    return "no coincide"


def comparacion_final(ev: Evaluador, config: dict, filas_paper: list[dict], desvios: dict) -> list[dict]:
    """Las 208 filas: paper, nuestro valor (media ± desvío con 10 semillas), error y categoría."""
    replicadas = [f for f in filas_paper if f["filtro"] != "HM-LCE"]
    ev.evaluar(crear_tarea(config, f["filtro"], f["ruido"], f["nivel"], semilla=s)
               for f in replicadas for s in SEMILLAS_FINALES)
    salida, detalle = [], []
    for fila in filas_paper:
        base = [fila["tabla"], fila["metrica"], fila["filtro"], fila["ruido"], f"{fila['nivel']:g}",
                _f(fila["valor_paper"])]
        if fila["filtro"] == "HM-LCE":
            salida.append(base + ["", "", "", "", "no se replica (otro método)"])
            continue
        valores = np.array([valor_de_fila(ev, config, fila, semilla=s) for s in SEMILLAS_FINALES])
        media, desvio = float(valores.mean()), float(valores.std(ddof=1))
        error = media - fila["valor_paper"]
        normalizado = abs(error) / desvios[fila["tabla"], fila["metrica"]]
        clase = categoria(fila["metrica"], error, desvio)
        salida.append(base + [_f(media), _f(desvio), _f(error), _f(normalizado), clase])
        detalle.append(dict(fila, media=media, desvio=desvio, error=error, normalizado=normalizado, categoria=clase))
    escribir_csv("comparacion_final.csv", ["tabla", "metrica", "filtro", "ruido", "nivel", "valor_paper",
                                           "nuestro_media", "nuestro_desvio", "error", "error_normalizado",
                                           "categoria"], salida)
    return detalle


def tabla_markdown() -> None:
    """Pasa `comparacion_final.csv` a una tabla Markdown (la del punto 2 del informe)."""
    with (RESULTADOS / "comparacion_final.csv").open(encoding="utf-8", newline="") as archivo:
        filas = list(csv.DictReader(archivo))
    lineas = ["| Tabla | Métrica | Filtro | Ruido | Nivel | Paper | Nuestro (media ± desvío) | Error | Categoría |",
              "|---|---|---|---|---|---|---|---|---|"]
    for f in filas:
        nuestro = f"{f['nuestro_media']} ± {f['nuestro_desvio']}" if f["nuestro_media"] else "—"
        lineas.append(f"| {f['tabla'][-1]} | {f['metrica']} | {f['filtro']} | {f['ruido']} | {f['nivel']} | "
                      f"{f['valor_paper']} | {nuestro} | {f['error'] or '—'} | {f['categoria']} |")
    (RESULTADOS / "comparacion_final.md").write_text("\n".join(lineas) + "\n", encoding="utf-8")


def resumen(detalle: list[dict]) -> None:
    """Cantidad de filas por categoría, agrupadas de varias maneras."""
    grupos = {
        "total": lambda f: "todas",
        "calibración/validación": lambda f: "calibración" if f["tabla"] in TABLAS_CALIBRACION else "validación",
        "tabla": lambda f: f["tabla"],
        "métrica": lambda f: f["metrica"],
        "filtro": lambda f: f["filtro"],
        "ruido": lambda f: f["ruido"],
        "filtro y ruido": lambda f: f"{f['filtro']} / {f['ruido']}",
        "ruido y nivel": lambda f: f"{f['ruido']} {f['nivel']:g}",
        "métrica y ruido": lambda f: f"{f['metrica']} / {f['ruido']}",
    }
    filas = []
    for nombre, clave in grupos.items():
        conteos: dict[str, list[dict]] = {}
        for fila in detalle:
            conteos.setdefault(clave(fila), []).append(fila)
        for grupo, propias in conteos.items():
            clases = [f["categoria"] for f in propias]
            normalizados = [f["normalizado"] for f in propias]
            filas.append([nombre, grupo, len(propias), clases.count("coincide"), clases.count("cerca"),
                          clases.count("no coincide"), _f(float(np.median(normalizados))),
                          _f(float(np.mean(normalizados)))])
    escribir_csv("resumen.csv", ["agrupado_por", "grupo", "filas", "coincide", "cerca", "no_coincide",
                                 "mediana_error_normalizado", "promedio_error_normalizado"], filas)


def sensibilidad(ev: Evaluador, config: dict, filas_paper: list[dict], desvios: dict) -> None:
    """Sensibilidad al clip limit y a la imagen, con la configuración final y semilla 0."""
    replicadas = [f for f in filas_paper if f["filtro"] != "HM-LCE"]
    for nombre, opcion, valores in (("sensibilidad_clip.csv", "clip", CLIPS_SENSIBILIDAD),
                                    ("sensibilidad_imagen.csv", "imagen", IMAGENES_STARE)):
        filas, resumenes = [], []
        for valor in valores:
            ev.evaluar(crear_tarea(config, f["filtro"], f["ruido"], f["nivel"], **{opcion: valor})
                       for f in replicadas)
            errores = {"calibración": [], "validación": []}
            for fila in replicadas:
                nuestro = valor_de_fila(ev, config, fila, **{opcion: valor})
                error = nuestro - fila["valor_paper"]
                grupo = "calibración" if fila["tabla"] in TABLAS_CALIBRACION else "validación"
                errores[grupo].append(abs(error) / desvios[fila["tabla"], fila["metrica"]])
                filas.append([valor, fila["tabla"], fila["metrica"], fila["filtro"], fila["ruido"],
                              f"{fila['nivel']:g}", _f(fila["valor_paper"]), _f(nuestro), _f(error)])
            resumenes.append([valor, "RESUMEN", "mediana y promedio del error normalizado", "", "", "",
                              "calibración", _f(float(np.median(errores["calibración"]))),
                              _f(float(np.mean(errores["calibración"])))])
            resumenes.append([valor, "RESUMEN", "mediana y promedio del error normalizado", "", "", "",
                              "validación", _f(float(np.median(errores["validación"]))),
                              _f(float(np.mean(errores["validación"])))])
        escribir_csv(nombre, [opcion, "tabla", "metrica", "filtro", "ruido", "nivel", "valor_paper",
                              "nuestro", "error"], resumenes + filas)


def diagnosticos(ev: Evaluador, config: dict, filas_paper: list[dict]) -> None:
    """Valores de referencia para el informe: sin CLAHE, CLAHE de OpenCV y medianas literales."""
    replicadas = [f for f in filas_paper if f["filtro"] != "HM-LCE"]
    casos = [("solo filtro (sin CLAHE)", {"clahe": None}, None),
             ("CLAHE de OpenCV", {"clahe": "opencv"}, None),
             ("mediana ponderada literal W/12", {"parametros": ("W/12",)}, "mediana_ponderada"),
             ("mediana ponderada literal W saturada", {"parametros": ("W_saturada",)}, "mediana_ponderada"),
             ("mediana ponderada c = 4 (File Exchange)", {"parametros": (4,)}, "mediana_ponderada"),
             ("mediana 3×3 (por defecto)", {"parametros": (3, "ceros")}, "mediana"),
             ("mediana 7×7 symmetric", {"parametros": (7, "symmetric")}, "mediana")]
    filas = []
    for nombre, cambios, solo_filtro in casos:
        for fila in replicadas:
            if solo_filtro and fila["filtro"] != solo_filtro:
                continue
            filtro = "mediana_ponderada_literal" if "literal" in nombre else fila["filtro"]
            tarea = crear_tarea(config, filtro, fila["ruido"], fila["nivel"], **cambios)
            nuestro = ev.valor(tarea, fila["metrica"], config[fila["metrica"]])
            filas.append([nombre, fila["tabla"], fila["metrica"], fila["filtro"], fila["ruido"],
                          f"{fila['nivel']:g}", _f(fila["valor_paper"]), _f(nuestro),
                          _f(nuestro - fila["valor_paper"])])
    escribir_csv("diagnosticos.csv", ["diagnostico", "tabla", "metrica", "filtro", "ruido", "nivel",
                                      "valor_paper", "nuestro", "error"], filas)


def main() -> None:
    inicio = time.time()
    filas_paper = cargar_tablas()
    desvios = desvios_del_paper(filas_paper)
    filas_cal = [f for f in filas_paper if f["tabla"] in TABLAS_CALIBRACION]
    ev = Evaluador()
    try:
        print("Etapa 1: base (sección 1 del prompt)", flush=True)
        etapa_base(ev, filas_paper)
        print("Etapa 2: control de CLAHE", flush=True)
        etapa_control_clahe(ev)
        print("Etapas 3 a 5: búsqueda", flush=True)
        config = buscar(ev, filas_cal, desvios)
        escribir_csv("configuracion_final.csv", ["variable", "valor"],
                     [[k, v if isinstance(v, str) else repr(v)] for k, v in config.items()])
        evidencia_por_variable(ev, config, filas_cal, desvios)
        print("Etapa 6: semillas, comparación final y sensibilidad", flush=True)
        detalle = comparacion_final(ev, config, filas_paper, desvios)
        tabla_markdown()
        resumen(detalle)
        sensibilidad(ev, config, filas_paper, desvios)
        diagnosticos(ev, config, filas_paper)
    finally:
        ev.cerrar()
    clases = [f["categoria"] for f in detalle]
    print("Configuración final:")
    for variable, valor in config.items():
        print(f"  {variable}: {valor}")
    print(f"Filas: {clases.count('coincide')} coinciden, {clases.count('cerca')} cerca, "
          f"{clases.count('no coincide')} no coinciden (de {len(clases)}).")
    print(f"Evaluaciones del proceso completo: {ev.evaluaciones} (tope {MAX_EVALUACIONES}). "
          f"Tiempo: {time.time() - inicio:.0f} s.")


if __name__ == "__main__":
    main()
