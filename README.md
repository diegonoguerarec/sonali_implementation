# Sonali et al. (2019) en Python

Implementación del método de *"An approach for de-noising and contrast enhancement of retinal fundus image using CLAHE"* (Sonali, Sahu, Singh, Ghrera y Elhoseny, *Optics and Laser Technology* 110, 2019, pp. 87–98).

## Qué hace

Para una imagen de fondo de ojo RGB de 8 bits:

1. agrega ruido (gaussiano o sal y pimienta) a la imagen RGB completa, como `imnoise` de MATLAB;
2. separa los canales R, G y B;
3. filtra cada canal con la **mediana ponderada** 3×3;
4. aplica CLAHE a cada canal filtrado (valores por defecto de `adapthisteq`);
5. une los canales y calcula PSNR, SSIM, CoC y EPI contra la imagen limpia, para la ruidosa, la "solo filtro" (antes de CLAHE) y la final.

Por ahora solo está la mediana ponderada. Para agregar otro filtro alcanza con escribir su función (canal `uint8` → canal `uint8`) y sumarla al diccionario `FILTROS` de `procesamiento/filtros.py`; la consola lo toma sola.

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
python sonali.py im0001.ppm gaussiano 0.001 mediana_ponderada
python sonali.py im0001.ppm sal_pimienta 0.01 mediana_ponderada --semilla 3
```

- `IMAGEN`: ruta a la imagen; si no existe, se busca en `STARE/`.
- `TIPO_RUIDO`: `gaussiano` o `sal_pimienta`.
- `VARIANZA`: en `gaussiano`, la varianza del ruido con intensidades en [0, 1]; en `sal_pimienta`, la densidad *d* de `imnoise` (fracción de valores afectados, en (0, 1]).
- `TIPO_FILTRO`: `mediana_ponderada`.
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

Etiquetas: `[PAPER]` paper; `[MATLAB]` valor por defecto o comportamiento documentado de MATLAB; `[FILE EXCHANGE]` MATLAB File Exchange #29386; `[LITERATURA]` bibliografía; `[OPENCV]` comportamiento de OpenCV; `[DECISIÓN]` decisión del proyecto; `[CONVENCIÓN]` convención de implementación.

| Parámetro | Valor | Etiqueta | Fuente |
|---|---|---|---|
| Orden del proceso | ruido → separar → filtrar → CLAHE → unir | [PAPER] | Sección 3, pasos 1–5 |
| Media del ruido gaussiano | 0 | [MATLAB] | imnoise |
| Escala de la varianza gaussiana | intensidades en [0, 1] | [MATLAB] | imnoise |
| Recorte y vuelta a uint8 | recortar a [0, 1], ×255, redondear | [MATLAB] | imnoise, uint8(round(·)) |
| "Varianza" en sal y pimienta | densidad d | [DECISIÓN] | El paper la llama varianza; imnoise usa densidad |
| Regla de sal y pimienta | u < d/2 → 0; d/2 ≤ u < d → 255 | [MATLAB] | imnoise |
| Ruido en RGB | independiente por valor, antes de separar canales | [MATLAB] + [PAPER] | imnoise; paso 1 |
| Semilla | 0 | [CONVENCIÓN] | Semilla inicial de MATLAB (con otro generador) |
| Ventana de la mediana ponderada | 3×3 | [FILE EXCHANGE] | #29386 |
| Pesos | [[1,1,1],[1,4,1],[1,1,1]] | [FILE EXCHANGE] | #29386, sin dividir por 12 |
| Definición de mediana ponderada | repetir cada valor según su peso y tomar la mediana | [LITERATURA] | Brownrigg 1984; Yin et al. 1996 |
| Empate (suma par) | promedio de los 2 centrales, sin fracción | [MATLAB] | medfilt2 |
| Bordes del filtro | relleno con ceros | [MATLAB] | medfilt2 |
| Teselas de CLAHE | 8×8 | [MATLAB] | adapthisteq |
| Bins de CLAHE | 256 | [MATLAB] | adapthisteq |
| Distribución de CLAHE | uniforme | [MATLAB] | adapthisteq |
| Clip limit | 0.0005 | [PAPER] | Fig. 5 |
| Conversión del clip limit | fórmula de la sección 7 | [MATLAB] + [OPENCV] | Ejemplo HDL de MathWorks; clahe.cpp |
| Pico del PSNR | 255 | [MATLAB] | psnr |
| Ventana de SSIM | gaussiana 11×11, σ = 1.5, bordes replicados | [MATLAB] | ssim (σ: documentación; tamaño y bordes: fuente secundaria) |
| Constantes de SSIM | C1, C2, C3 = C2/2, exponentes 1 | [MATLAB] + [PAPER] | ssim; ecuaciones 4–7 |
| SSIM en RGB | ventana por plano, promedio de todo el mapa | [DECISIÓN] | Equivale a "SSC" + promedio |
| CoC | Pearson sobre todos los valores RGB | [PAPER] + [DECISIÓN] | Ecuación 8 |
| EPI | ecuación 9, sumando los 3 canales | [PAPER] + [DECISIÓN] | Ecuación 9 |
| Métricas intermedias | columna "solo filtro" | [DECISIÓN] | — |

Detalles:

- **Mediana ponderada.** La Tabla 1 del paper describe casi textualmente el código de File Exchange #29386, que multiplica la ventana por `W/12` y toma la mediana de los productos (eso divide la intensidad por ~12). Por eso se usan los mismos pesos sin dividir, con la definición clásica: cada valor se repite según su peso (12 valores) y, como 12 es par, la salida es el promedio entero de los valores 6.º y 7.º, como `medfilt2`.
- **Clip limit.** MATLAB lo expresa normalizado (0 a 1) y OpenCV en otra escala. Con P = píxeles por tesela (como OpenCV): `min_clip = ceil(P/256)`, `cuentas = min_clip + round(c·(P − min_clip))` y `clipLimit = (cuentas + 0.5)·256/P`. Para STARE (605×700): P = 6688; con c = 0.0005 son 30 cuentas (clipLimit ≈ 1.167464).
- **SSIM.** El tamaño 11×11 y los bordes replicados salen de una fuente secundaria sobre `ssim.m` (https://www.cnblogs.com/hyddd/p/7132187.html); σ = 1.5 sale de la documentación de MATLAB.
- **EPI.** Usa solo diferencias horizontales (ecuación 9) sumadas en los 3 canales. Puede pasar de 1, por ejemplo cuando el ruido o CLAHE agregan bordes.

## Limitaciones

- **El ruido no es idéntico al de MATLAB.** La semilla por defecto es 0, como la de MATLAB, pero el generador es otro (`numpy.random.default_rng`), así que los valores sorteados son distintos.
- **CLAHE no da lo mismo bit a bit que `adapthisteq`.** MATLAB y OpenCV rellenan los bordes y redistribuyen el exceso del histograma de forma algo distinta.
- **Los resultados no se pueden comparar con las tablas del paper.** El paper no publica casi ninguno de sus parámetros (pesos, tamaños de ventana, definición exacta de las métricas, etc.); los valores de acá son reconstrucciones documentadas, no los de los autores.
