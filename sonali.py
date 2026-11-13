"""Punto de entrada por consola del método de Sonali et al. (2019).

Uso:
    python sonali.py IMAGEN TIPO_RUIDO VARIANZA TIPO_FILTRO [--semilla N] [--clip-limit C]
"""

import argparse
import shlex
import sys
from pathlib import Path

import cv2
import numpy as np

from procesamiento import parametros as p
from procesamiento import procesar_imagen
from procesamiento.filtros import FILTROS
from procesamiento.ruido import RUIDOS
from procesamiento.salida import datos_parametros, formatear_tabla, guardar_resultados, nombre_carpeta

CARPETA_IMAGENES = Path(__file__).resolve().parent / "STARE"
CARPETA_RESULTADO = Path("resultado")


def crear_parser() -> argparse.ArgumentParser:
    """Define los argumentos de la consola; los filtros salen del registro FILTROS."""
    parser = argparse.ArgumentParser(
        description="Eliminación de ruido y realce de contraste de imágenes de fondo de ojo "
                    "(Sonali et al., 2019): ruido → separar RGB → filtrar → CLAHE → unir.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("imagen", metavar="IMAGEN",
                        help="ruta a una imagen RGB de 8 bits; si no existe, se busca dentro de STARE/")
    parser.add_argument("tipo_ruido", metavar="TIPO_RUIDO", choices=list(RUIDOS),
                        help=f"tipo de ruido: {', '.join(RUIDOS)}")
    parser.add_argument("varianza", metavar="VARIANZA", type=float,
                        help="en 'gaussiano': varianza del ruido, con intensidades en [0, 1] (> 0). "
                             "En 'sal_pimienta': densidad d de imnoise, la fracción de valores "
                             "afectados, en (0, 1]")
    parser.add_argument("tipo_filtro", metavar="TIPO_FILTRO", choices=list(FILTROS),
                        help=f"filtro de eliminación de ruido: {', '.join(FILTROS)}")
    parser.add_argument("--semilla", type=int, default=p.SEMILLA_POR_DEFECTO,
                        help=f"semilla del generador de ruido (por defecto, {p.SEMILLA_POR_DEFECTO})")
    parser.add_argument("--clip-limit", type=float, default=p.CLIP_LIMIT_POR_DEFECTO,
                        help="clip limit normalizado de CLAHE, en (0, 1] "
                             f"(por defecto, {format(p.CLIP_LIMIT_POR_DEFECTO, 'g')})")
    return parser


def resolver_ruta_imagen(imagen: str) -> Path:
    """Usa la ruta tal cual si existe; si no, la busca dentro de STARE/."""
    ruta = Path(imagen)
    if ruta.is_file():
        return ruta
    en_stare = CARPETA_IMAGENES / imagen
    if en_stare.is_file():
        return en_stare
    raise ValueError(f"La imagen '{imagen}' no existe (tampoco en {CARPETA_IMAGENES}/).")


def leer_imagen_rgb(ruta: Path) -> np.ndarray:
    """Lee la imagen con OpenCV, verifica que sea RGB de 8 bits y la pasa de BGR a RGB."""
    imagen = cv2.imread(str(ruta), cv2.IMREAD_UNCHANGED)
    if imagen is None:
        raise ValueError(f"No se pudo leer '{ruta}' como imagen.")
    if imagen.dtype != np.uint8 or imagen.ndim != 3 or imagen.shape[2] != p.CANALES_RGB:
        raise ValueError(f"'{ruta}' no es una imagen RGB de 8 bits "
                         f"(tipo {imagen.dtype}, forma {imagen.shape}).")
    return cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)


def validar_argumentos(args: argparse.Namespace) -> None:
    """Valida la varianza o densidad y el clip limit."""
    if args.varianza <= 0:
        raise ValueError(f"La varianza tiene que ser > 0 (se recibió {args.varianza}).")
    if args.tipo_ruido == "sal_pimienta" and args.varianza > p.INTENSIDAD_MAX_NORMALIZADA:
        raise ValueError(f"La densidad de sal y pimienta tiene que estar en (0, 1] "
                         f"(se recibió {args.varianza}).")
    if not p.CLIP_LIMIT_MIN_EXCLUSIVO < args.clip_limit <= p.CLIP_LIMIT_MAX:
        raise ValueError(f"El clip limit tiene que estar en (0, 1] (se recibió {args.clip_limit}).")


def main(argv: list[str] | None = None, dir_resultado: Path = CARPETA_RESULTADO) -> Path:
    """Corre el método sobre una imagen, muestra la tabla y guarda los resultados."""
    parser = crear_parser()
    argumentos = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(argumentos)
    try:
        validar_argumentos(args)
        ruta = resolver_ruta_imagen(args.imagen)
        limpia = leer_imagen_rgb(ruta)
    except ValueError as error:
        parser.error(str(error))

    resultado = procesar_imagen(limpia, args.tipo_ruido, args.varianza, args.tipo_filtro,
                                args.clip_limit, args.semilla)
    print(formatear_tabla(ruta, limpia, args.tipo_ruido, args.varianza, args.tipo_filtro,
                          args.clip_limit, args.semilla, resultado["metricas"]))

    comando = shlex.join(["python", "sonali.py", *argumentos])
    datos = datos_parametros(comando, ruta, limpia, args.tipo_ruido, args.varianza,
                              args.tipo_filtro, args.clip_limit, args.semilla, resultado["clahe"])
    nombre = nombre_carpeta(ruta, args.tipo_ruido, args.varianza, args.tipo_filtro,
                            args.clip_limit, args.semilla)
    carpeta = guardar_resultados(dir_resultado, nombre, resultado, datos)
    print(f"\nResultados guardados en: {carpeta}/")
    return carpeta


if __name__ == "__main__":
    main()
