# Sonali et al. (2019) en Python

Implementación del método de *"An approach for de-noising and contrast enhancement of retinal fundus image using CLAHE"* (Sonali, Sahu, Singh, Ghrera y Elhoseny, *Optics and Laser Technology* 110, 2019, pp. 87–98).

## Qué hace

Para una imagen de fondo de ojo RGB de 8 bits:

1. agrega ruido (gaussiano o sal y pimienta) a la imagen RGB completa, como `imnoise` de MATLAB;
2. separa los canales R, G y B;
3. filtra cada canal con uno de los cinco filtros del paper: **mediana**, **media**, **gaussiano**, **Wiener** o **mediana ponderada**;
4. aplica CLAHE a cada canal filtrado, al estilo de `adapthisteq` de MATLAB;
5. une los canales y calcula PSNR, SSIM, CoC y EPI contra la imagen limpia, para la ruidosa, la "solo filtro" (antes de CLAHE) y la final.

Los parámetros que el paper no publica se eligieron buscando los que mejor reproducen sus Tablas 3 a 6. La búsqueda, la evidencia de cada elección y lo que no se pudo reproducir están en [`investigacion/INFORME.md`](investigacion/INFORME.md).

Para agregar otro filtro alcanza con escribir su función (canal `uint8` → canal `uint8`) y sumarla al diccionario `FILTROS` de `procesamiento/filtros.py`; la consola lo toma sola.

## Instalación

```bash
pyenv install -s 3.14.6
pyenv local 3.14.6
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Uso

```
python sonali.py IMAGEN TIPO_RUIDO VARIANZA TIPO_FILTRO [--semilla N] [--clip-limit C]
```

```bash
python sonali.py im0001.ppm gaussiano 0.001 mediana
python sonali.py im0001.ppm sal_pimienta 0.01 mediana_ponderada --semilla 3
```

- `IMAGEN`: ruta a la imagen; si no existe, se busca en `STARE/`.
- `TIPO_RUIDO`: `gaussiano` o `sal_pimienta`.
- `VARIANZA`: el nivel de ruido, que el paper llama "noise variance". Tiene que ser mayor que 0.
  - En `gaussiano` es la **media** del ruido (un corrimiento de brillo, con intensidades en [0, 1]); la varianza queda fija en 0.01. Equivale a `imnoise(I,'gaussian',nivel)`.
  - En `sal_pimienta` es la densidad *d* de `imnoise`: la fracción de valores afectados, en (0, 1].
- `TIPO_FILTRO`: `mediana`, `media`, `gaussiano`, `wiener` o `mediana_ponderada`.
- `--semilla`: semilla del ruido (por defecto 0).
- `--clip-limit`: clip limit normalizado de CLAHE, en (0, 1] (por defecto 0.0005).

Pruebas: `pytest -q`.

## Qué guarda y dónde

Muestra una tabla en consola y guarda en
`resultado/<imagen>__ruido-<tipo>__var-<valor>__filtro-<filtro>__clahe-<clip>__semilla-<n>/`
(se sobrescribe si ya existe):

- `ruidosa.png`, `solo_filtro.png`, `final.png` (PNG sin pérdida);
- `metricas.csv` (`metrica,ruidosa,solo_filtro,final`, precisión completa);
- `parametros.json` (comando, fecha, imagen, parámetros de ruido, filtro y CLAHE, definiciones de las métricas, versiones y procedencia).

## Procedencia de los parámetros

Etiquetas: `[PAPER]` paper; `[MATLAB]` valor por defecto o comportamiento documentado de MATLAB; `[FILE EXCHANGE]` MATLAB File Exchange #29386; `[LITERATURA]` bibliografía; `[DECISIÓN]` decisión del proyecto; `[CONVENCIÓN]` convención de implementación; `[VERIFICADO]` deducido con un experimento reproducible; `[AJUSTE]` candidato con fuente que mejor reproduce las tablas del paper (evidencia en `investigacion/INFORME.md`).

| Parámetro | Valor | Etiqueta | Fuente |
|---|---|---|---|
| Orden del proceso | ruido → separar → filtrar → CLAHE → unir | [PAPER] | Sección 3, pasos 1–5 |
| Imagen de las tablas del paper | STARE/im0001.ppm | [VERIFICADO] | Correlación con las Figs. 1 y 7 del PDF |
| Ruido gaussiano de nivel v | media v y varianza 0.01: imnoise(I,'gaussian',v) | [AJUSTE] | imnoise; investigacion/INFORME.md |
| Escala del ruido gaussiano | intensidades en [0, 1] | [MATLAB] | imnoise |
| Recorte y vuelta a uint8 | recortar a [0, 1], ×255, redondear | [MATLAB] | imnoise, uint8(round(·)) |
| "Varianza" en sal y pimienta | densidad d | [DECISIÓN] | El paper la llama varianza; imnoise usa densidad |
| Regla de sal y pimienta | u < d/2 → 0; d/2 ≤ u < d → 255 | [MATLAB] | imnoise |
| Ruido en RGB | independiente por valor, antes de separar canales | [MATLAB] + [PAPER] | imnoise; paso 1 |
| Semilla | 0 | [CONVENCIÓN] | Semilla inicial de MATLAB (con otro generador) |
| Mediana | 5×5, relleno con ceros | [AJUSTE] | medfilt2 (evidencia débil) |
| Media | fspecial('average', 5) + imfilter, relleno con ceros | [AJUSTE] | fspecial; imfilter |
| Gaussiano | fspecial('gaussian', 3, 0.5) + imfilter, relleno con ceros | [AJUSTE] + [MATLAB] | fspecial (sigma por defecto); el paper dice "default 1" |
| Wiener | wiener2 5×5, ruido = promedio de las varianzas locales | [AJUSTE] + [MATLAB] | wiener2 |
| Ventana de la mediana ponderada | 3×3 | [FILE EXCHANGE] | #29386 |
| Pesos | [[1,1,1],[1,4,1],[1,1,1]] | [FILE EXCHANGE] | #29386, sin dividir por 12 |
| Definición de mediana ponderada | repetir cada valor según su peso y tomar la mediana | [LITERATURA] | Brownrigg 1984; Yin et al. 1996 |
| Empate (suma par) | promedio de los 2 centrales, sin fracción | [MATLAB] | medfilt2 |
| Bordes de las medianas | relleno con ceros | [MATLAB] | medfilt2 |
| Teselas de CLAHE | 8×8, de lado par, con relleno simétrico | [MATLAB] | adapthisteq; ejemplo HDL de MathWorks |
| Bins de CLAHE | 256 | [MATLAB] | adapthisteq |
| Distribución de CLAHE | uniforme | [MATLAB] | adapthisteq |
| Clip limit | 0.0005 | [PAPER] | Fig. 5 |
| Clip limit en cuentas | ceil(P/256) + round(c·(P − ceil(P/256))) | [MATLAB] | Ejemplo HDL de MathWorks |
| Reparto del exceso recortado | ningún bin supera el clip | [MATLAB] | Ejemplo HDL de MathWorks (Distribute/Redistribute) |
| Tabla de cada tesela | round(255·cdf/P) | [MATLAB] | Ejemplo HDL de MathWorks (grayxform) |
| Unión de teselas | interpolación bilineal | [MATLAB] + [LITERATURA] | adapthisteq; Zuiderveld 1994 (pesos no verificados) |
| Dominio de PSNR y SSIM | rgb2gray | [AJUSTE] + [MATLAB] | rgb2gray; investigacion/INFORME.md |
| Pico del PSNR | máximo − mínimo de la limpia en gris | [AJUSTE] + [PAPER] | Texto de la ecuación 2 |
| Ventana de SSIM | gaussiana 11×11, σ = 1.5, bordes replicados | [MATLAB] | ssim (σ: documentación; tamaño y bordes: fuente secundaria) |
| Constantes de SSIM | C1, C2, C3 = C2/2, exponentes 1 | [MATLAB] + [PAPER] | ssim; ecuaciones 4–7 |
| CoC | Pearson sobre todos los valores RGB | [PAPER] + [DECISIÓN] | Ecuación 8 |
| EPI | ecuación 9, sumando los 3 canales | [PAPER] + [DECISIÓN] | Ecuación 9 (no reproduce la Tabla 6 del paper) |
| Métricas intermedias | columna "solo filtro" | [DECISIÓN] | — |

Detalles:

- **Ruido gaussiano.** `imnoise(I,'gaussian',m)` con un solo número usa *m* como media y deja la varianza en 0.01. Con esa lectura, las columnas gaussianas del paper se reproducen mucho mejor que tomando el nivel como varianza.
- **Filtros.** Mediana 5×5 (`medfilt2`), media 5×5 (`fspecial('average')` + `imfilter`), gaussiano 3×3 con σ = 0.5 (`fspecial('gaussian')` + `imfilter`) y Wiener 5×5 (`wiener2`), todos con relleno de ceros. La Tabla 1 del paper dice σ = 1 para el gaussiano, pero σ = 0.5 (el valor por defecto de `fspecial`) reproduce mucho mejor sus tablas.
- **Mediana ponderada.** La Tabla 1 del paper describe casi textualmente el código de File Exchange #29386, que multiplica la ventana por `W/12` y toma la mediana de los productos (eso divide la intensidad por ~12). Por eso se usan los mismos pesos sin dividir, con la definición clásica: cada valor se repite según su peso (12 valores) y, como 12 es par, la salida es el promedio entero de los valores 6.º y 7.º, como `medfilt2`.
- **CLAHE.** Se reconstruyó `adapthisteq` con NumPy a partir de su documentación y del ejemplo HDL de MathWorks: teselas de lado par con relleno simétrico, `min_clip = ceil(P/256)`, `cuentas = min_clip + round(c·(P − min_clip))`, reparto del exceso sin que ningún bin supere el clip, tabla `round(255·cdf/P)` e interpolación bilineal. Para STARE (605×700): teselas de 76×88, P = 6688 y, con c = 0.0005, 30 cuentas. El CLAHE de OpenCV no se usa porque reparte el exceso sin tope y realza mucho más.
- **PSNR y SSIM.** Se calculan sobre la imagen en gris (`rgb2gray`). El pico del PSNR es el máximo menos el mínimo de la limpia en gris (texto de la ecuación 2 del paper). En el SSIM, el tamaño 11×11 y los bordes replicados salen de una fuente secundaria sobre `ssim.m` (https://www.cnblogs.com/hyddd/p/7132187.html); σ = 1.5 sale de la documentación de MATLAB.
- **EPI.** Usa solo diferencias horizontales (ecuación 9) sumadas en los 3 canales. Puede pasar de 1, por ejemplo cuando queda ruido o CLAHE agrega bordes.

## Limitaciones

- **El ruido no es idéntico al de MATLAB.** La semilla por defecto es 0, como la de MATLAB, pero el generador es otro (`numpy.random.default_rng`), así que los valores sorteados son distintos.
- **CLAHE no está verificado contra `adapthisteq`.** Es una reconstrucción a partir de la documentación; el lado del relleno, el redondeo de la salida y los pesos de la interpolación no están documentados.
- **Las tablas del paper se reproducen solo en parte.** Con `im0001.ppm`, PSNR, SSIM y CoC quedan cerca en la mayoría de las filas de las Tablas 3 a 5 (error típico: 0.7 dB, 0.02 y 0.007). No se reproducen:
  - el EPI (Tabla 6), con ninguna definición probada;
  - la mediana ponderada con ruido gaussiano (2 a 3 dB por debajo);
  - la Tabla 8 (sal y pimienta 0.5).

  El detalle fila por fila está en `investigacion/INFORME.md`. Los valores de la mediana ponderada de ese informe corresponden a un peso central de 1; el programa conserva el peso 4 de File Exchange.
