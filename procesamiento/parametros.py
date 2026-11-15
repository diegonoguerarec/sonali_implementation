"""Constantes del método de Sonali et al. (2019), cada una con su procedencia.

Etiquetas:
    [PAPER]          está en el paper.
    [MATLAB]         valor por defecto o comportamiento documentado de MATLAB.
    [FILE EXCHANGE]  código de MATLAB File Exchange #29386.
    [LITERATURA]     definición tomada de la bibliografía.
    [OPENCV]         comportamiento de OpenCV.
    [DECISIÓN]       decisión del proyecto.
    [CONVENCIÓN]     convención de implementación.
    [VERIFICADO]     deducido con un experimento reproducible.
    [AJUSTE]         candidato con fuente que mejor reproduce las Tablas 3–6 del
                     paper; la evidencia está en investigacion/INFORME.md.
"""

import numpy as np

# --- Imagen -----------------------------------------------------------------

# [MATLAB] Máximo de una imagen uint8; se usa para pasar a [0, 1] y volver.
MAX_INTENSIDAD: int = 255

# [MATLAB] uint8(round(x)) redondea los empates hacia arriba: floor(x + 0.5).
DESPLAZAMIENTO_REDONDEO: float = 0.5

# [PAPER] Imagen de fondo de ojo RGB: 3 canales.
CANALES_RGB: int = 3

# --- Ruido (imnoise) --------------------------------------------------------

# [AJUSTE] El "nivel" gaussiano del paper se usa como imnoise(I,'gaussian',nivel):
# el nivel es la MEDIA del ruido y la varianza queda en su valor por defecto.
# [MATLAB] imnoise: varianza por defecto del ruido gaussiano.
VARIANZA_RUIDO_GAUSSIANO: float = 0.01

# [MATLAB] imnoise 'gaussian': la varianza se expresa con intensidades en [0, 1].
INTENSIDAD_MIN_NORMALIZADA: float = 0.0
INTENSIDAD_MAX_NORMALIZADA: float = 1.0

# [MATLAB] imnoise 'salt & pepper': u < d/2 -> 0 ; d/2 <= u < d -> 255.
FRACCION_PIMIENTA: float = 0.5
VALOR_PIMIENTA: int = 0
VALOR_SAL: int = MAX_INTENSIDAD

# [CONVENCIÓN] Semilla inicial de MATLAB (rng por defecto), con otro generador.
SEMILLA_POR_DEFECTO: int = 0

# --- Mediana ponderada ------------------------------------------------------

# [FILE EXCHANGE] #29386: ventana 3×3 y W = [1 1 1; 1 4 1; 1 1 1]/12.
# Se usan los pesos sin dividir por 12 ([LITERATURA]: Brownrigg 1984; Yin et al. 1996).
PESOS_MEDIANA_PONDERADA: np.ndarray = np.array(
    [[1, 1, 1],
     [1, 4, 1],
     [1, 1, 1]],
    dtype=np.int64,
)
TAMANO_VENTANA_MEDIANA: int = 3

# [MATLAB] medfilt2: relleno por defecto 'zeros'.
VALOR_RELLENO_MEDIANA: int = 0

# [MATLAB] medfilt2 con cantidad par de valores: promedio de los 2 centrales,
# en división entera (se descarta la fracción).
DIVISOR_EMPATE_MEDIANA: int = 2

# --- Mediana, media, gaussiano y Wiener ---------------------------------------

# [AJUSTE] medfilt2 con ventana 5×5 (por defecto es 3×3; evidencia débil).
TAMANO_MEDIANA: int = 5

# [AJUSTE] fspecial('average', 5) + imfilter (por defecto es 3×3).
TAMANO_MEDIA: int = 5

# [MATLAB] imfilter: relleno por defecto 0, correlación, salida uint8 redondeada.
VALOR_RELLENO_IMFILTER: int = 0

# [AJUSTE] + [MATLAB] fspecial('gaussian', 3, 0.5): sigma 0.5 es el valor por
# defecto de fspecial. La Tabla 1 del paper dice "default 1", pero 0.5 reproduce
# mucho mejor sus tablas.
TAMANO_GAUSSIANO: int = 3
SIGMA_GAUSSIANO: float = 0.5

# [AJUSTE] wiener2 con ventana 5×5 (por defecto es 3×3). [MATLAB] El ruido se
# estima como el promedio de las varianzas locales.
TAMANO_WIENER: int = 5

# --- CLAHE (adapthisteq) ----------------------------------------------------

# [MATLAB] adapthisteq: NumTiles por defecto [8 8].
TESELAS_CLAHE: tuple[int, int] = (8, 8)

# [MATLAB] adapthisteq: NBins por defecto 256.
BINS_CLAHE: int = 256

# [MATLAB] adapthisteq: Distribution por defecto 'uniform'.
DISTRIBUCION_CLAHE: str = "uniforme"

# [MATLAB] Ejemplo HDL de MathWorks (declara coincidir con adapthisteq): el lado
# de cada tesela tiene que ser par; si no, la imagen se rellena simétricamente.
PARIDAD_TESELA: int = 2

# [PAPER] Fig. 5: clip limit 0.0005 (normalizado, como en MATLAB).
CLIP_LIMIT_POR_DEFECTO: float = 0.0005
CLIP_LIMIT_MIN_EXCLUSIVO: float = 0.0
CLIP_LIMIT_MAX: float = 1.0

# --- Métricas ----------------------------------------------------------------

# [MATLAB] rgb2gray: 0.298936021293775·R + 0.587043074451121·G + 0.114020904255103·B.
# [AJUSTE] PSNR y SSIM se calculan sobre la imagen en gris.
COEFICIENTES_GRIS: tuple[float, float, float] = (0.298936021293775, 0.587043074451121, 0.114020904255103)

# [AJUSTE] + [PAPER] El pico del PSNR es "the maximum difference in the input
# image" (texto de la ecuación 2): máximo − mínimo de la limpia en gris.
# [MATLAB] 255 es el PeakValue por defecto de psnr para uint8; ya no se usa como
# pico, pero se conserva como referencia.
PICO_PSNR: float = 255.0
FACTOR_DECIBELES: float = 10.0

# [MATLAB] ssim: ventana gaussiana σ = 1.5 (documentación); tamaño 11×11
# (radio ceil(3σ) = 5) y bordes replicados (fuente secundaria:
# https://www.cnblogs.com/hyddd/p/7132187.html).
SIGMA_SSIM: float = 1.5
TAMANO_VENTANA_SSIM: int = 11

# [MATLAB] + [PAPER] ssim: K = [0.01 0.03], C1 = (K1·L)², C2 = (K2·L)², C3 = C2/2,
# exponentes 1 (ecuaciones 4–7).
K1_SSIM: float = 0.01
K2_SSIM: float = 0.03
RANGO_DINAMICO_SSIM: float = 255.0
EXPONENTE_SSIM: float = 1.0
DIVISOR_C3_SSIM: float = 2.0
FACTOR_NUMERADOR_SSIM: float = 2.0  # el "2" de 2μxμy y 2σxy (ecuaciones 5 y 7)

# --- Tabla de procedencia -----------------------------------------------------

TABLA_PROCEDENCIA: list[dict[str, str]] = [
    {"parametro": 'Orden del proceso', "valor": 'ruido → separar → filtrar → CLAHE → unir',
     "etiqueta": '[PAPER]', "fuente": 'Sección 3, pasos 1–5'},
    {"parametro": 'Imagen de las tablas del paper', "valor": 'STARE/im0001.ppm',
     "etiqueta": '[VERIFICADO]', "fuente": 'Correlación con las Figs. 1 y 7 del PDF'},
    {"parametro": 'Ruido gaussiano de nivel v', "valor": "media v y varianza 0.01: imnoise(I,'gaussian',v)",
     "etiqueta": '[AJUSTE]', "fuente": 'imnoise; investigacion/INFORME.md'},
    {"parametro": 'Escala del ruido gaussiano', "valor": 'intensidades en [0, 1]',
     "etiqueta": '[MATLAB]', "fuente": 'imnoise'},
    {"parametro": 'Recorte y vuelta a uint8', "valor": 'recortar a [0, 1], ×255, redondear',
     "etiqueta": '[MATLAB]', "fuente": 'imnoise, uint8(round(·))'},
    {"parametro": '"Varianza" en sal y pimienta', "valor": 'densidad d',
     "etiqueta": '[DECISIÓN]', "fuente": 'El paper la llama varianza; imnoise usa densidad'},
    {"parametro": 'Regla de sal y pimienta', "valor": 'u < d/2 → 0; d/2 ≤ u < d → 255',
     "etiqueta": '[MATLAB]', "fuente": 'imnoise'},
    {"parametro": 'Ruido en RGB', "valor": 'independiente por valor, antes de separar canales',
     "etiqueta": '[MATLAB] + [PAPER]', "fuente": 'imnoise; paso 1'},
    {"parametro": 'Semilla', "valor": '0',
     "etiqueta": '[CONVENCIÓN]', "fuente": 'Semilla inicial de MATLAB (con otro generador)'},
    {"parametro": 'Mediana', "valor": '5×5, relleno con ceros',
     "etiqueta": '[AJUSTE]', "fuente": 'medfilt2 (evidencia débil)'},
    {"parametro": 'Media', "valor": "fspecial('average', 5) + imfilter, relleno con ceros",
     "etiqueta": '[AJUSTE]', "fuente": 'fspecial; imfilter'},
    {"parametro": 'Gaussiano', "valor": "fspecial('gaussian', 3, 0.5) + imfilter, relleno con ceros",
     "etiqueta": '[AJUSTE] + [MATLAB]', "fuente": 'fspecial (sigma por defecto); el paper dice "default 1"'},
    {"parametro": 'Wiener', "valor": 'wiener2 5×5, ruido = promedio de las varianzas locales',
     "etiqueta": '[AJUSTE] + [MATLAB]', "fuente": 'wiener2'},
    {"parametro": 'Ventana de la mediana ponderada', "valor": '3×3',
     "etiqueta": '[FILE EXCHANGE]', "fuente": '#29386'},
    {"parametro": 'Pesos', "valor": '[[1,1,1],[1,4,1],[1,1,1]]',
     "etiqueta": '[FILE EXCHANGE]', "fuente": '#29386, sin dividir por 12'},
    {"parametro": 'Definición de mediana ponderada', "valor": 'repetir cada valor según su peso y tomar la mediana',
     "etiqueta": '[LITERATURA]', "fuente": 'Brownrigg 1984; Yin et al. 1996'},
    {"parametro": 'Empate (suma par)', "valor": 'promedio de los 2 centrales, sin fracción',
     "etiqueta": '[MATLAB]', "fuente": 'medfilt2'},
    {"parametro": 'Bordes de las medianas', "valor": 'relleno con ceros',
     "etiqueta": '[MATLAB]', "fuente": 'medfilt2'},
    {"parametro": 'Teselas de CLAHE', "valor": '8×8, de lado par, con relleno simétrico',
     "etiqueta": '[MATLAB]', "fuente": 'adapthisteq; ejemplo HDL de MathWorks'},
    {"parametro": 'Bins de CLAHE', "valor": '256',
     "etiqueta": '[MATLAB]', "fuente": 'adapthisteq'},
    {"parametro": 'Distribución de CLAHE', "valor": 'uniforme',
     "etiqueta": '[MATLAB]', "fuente": 'adapthisteq'},
    {"parametro": 'Clip limit', "valor": '0.0005',
     "etiqueta": '[PAPER]', "fuente": 'Fig. 5'},
    {"parametro": 'Clip limit en cuentas', "valor": 'ceil(P/256) + round(c·(P − ceil(P/256)))',
     "etiqueta": '[MATLAB]', "fuente": 'Ejemplo HDL de MathWorks'},
    {"parametro": 'Reparto del exceso recortado', "valor": 'ningún bin supera el clip',
     "etiqueta": '[MATLAB]', "fuente": 'Ejemplo HDL de MathWorks (Distribute/Redistribute)'},
    {"parametro": 'Tabla de cada tesela', "valor": 'round(255·cdf/P)',
     "etiqueta": '[MATLAB]', "fuente": 'Ejemplo HDL de MathWorks (grayxform)'},
    {"parametro": 'Unión de teselas', "valor": 'interpolación bilineal',
     "etiqueta": '[MATLAB] + [LITERATURA]', "fuente": 'adapthisteq; Zuiderveld 1994 (pesos no verificados)'},
    {"parametro": 'Dominio de PSNR y SSIM', "valor": 'rgb2gray',
     "etiqueta": '[AJUSTE] + [MATLAB]', "fuente": 'rgb2gray; investigacion/INFORME.md'},
    {"parametro": 'Pico del PSNR', "valor": 'máximo − mínimo de la limpia en gris',
     "etiqueta": '[AJUSTE] + [PAPER]', "fuente": 'Texto de la ecuación 2'},
    {"parametro": 'Ventana de SSIM', "valor": 'gaussiana 11×11, σ = 1.5, bordes replicados',
     "etiqueta": '[MATLAB]', "fuente": 'ssim (σ: documentación; tamaño y bordes: fuente secundaria)'},
    {"parametro": 'Constantes de SSIM', "valor": 'C1, C2, C3 = C2/2, exponentes 1',
     "etiqueta": '[MATLAB] + [PAPER]', "fuente": 'ssim; ecuaciones 4–7'},
    {"parametro": 'CoC', "valor": 'Pearson sobre todos los valores RGB',
     "etiqueta": '[PAPER] + [DECISIÓN]', "fuente": 'Ecuación 8'},
    {"parametro": 'EPI', "valor": 'ecuación 9, sumando los 3 canales',
     "etiqueta": '[PAPER] + [DECISIÓN]', "fuente": 'Ecuación 9 (no reproduce la Tabla 6 del paper)'},
    {"parametro": 'Métricas intermedias', "valor": 'columna "solo filtro"',
     "etiqueta": '[DECISIÓN]', "fuente": '—'},
]
