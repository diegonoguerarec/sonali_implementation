# Tarea: implementar en Python el método de Sonali et al. (2019)

Respondeme siempre en español.

## 0. Contexto y regla principal

En la raíz de esta carpeta está el PDF del paper "An approach for de-noising and contrast enhancement of retinal fundus image using CLAHE" (Sonali, Sahu, Singh, Ghrera y Elhoseny, *Optics and Laser Technology* 110, 2019, pp. 87–98). Los autores lo implementaron en MATLAB R2016a. Por eso, todo lo que el paper no define se completa con el comportamiento por defecto de MATLAB, siempre que exista uno documentado.

El método (sección 3 del paper) es:
1. Leer la imagen de fondo de ojo con ruido.
2. Separarla en los canales R, G y B.
3. Filtrar cada canal para quitar el ruido.
4. Aplicar CLAHE a cada canal filtrado.
5. Unir los tres canales.
6. Repetir con otros filtros, ruidos y niveles. Esto se hace corriendo el programa varias veces; no hay que automatizarlo.

**Regla principal: no inventes ningún valor.** Cada parámetro del código tiene que salir de este documento, con la etiqueta de procedencia de la tabla de la sección 9. Si necesitás un valor que no está acá, o si el PDF contradice este documento, frená y preguntame. Leé el PDF como referencia, pero las decisiones de este documento mandan: el paper no publica casi ningún parámetro, y cada valor que falta ya está resuelto acá con su fuente.

Etiquetas de procedencia:
- `[PAPER]`: está en el paper.
- `[MATLAB]`: valor por defecto o comportamiento documentado de MATLAB.
- `[FILE EXCHANGE]`: código de MATLAB File Exchange #29386 (sección 6).
- `[LITERATURA]`: definición tomada de la bibliografía.
- `[OPENCV]`: comportamiento de OpenCV.
- `[DECISIÓN]`: decisión del proyecto, que tomé yo.
- `[CONVENCIÓN]`: convención de implementación.

Alcance de esta primera versión:
- Se procesa una sola imagen por ejecución.
- De los filtros solo se implementa la **mediana ponderada**. El código y la consola tienen que quedar preparados para agregar más adelante los otros filtros del paper (media, mediana, Wiener y gaussiano). No los implementes ahora.

Forma de trabajo: primero leé el PDF y este documento y mostrame un plan breve con los archivos y las funciones. Esperá mi confirmación antes de escribir código.

## 1. Entorno

- Ubuntu, pyenv 2.8.6, Python 3.14.6, con un entorno virtual dentro del proyecto:
  ```bash
  pyenv install -s 3.14.6
  pyenv local 3.14.6
  python -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```
- `requirements.txt` con estas versiones exactas. Verifiqué que tienen wheels para Python 3.14 en Linux x86_64:
  ```
  numpy==2.5.4
  opencv-python-headless==5.0.0.93
  pytest==9.2.0
  ```
- No agregues otras dependencias (ni SciPy ni scikit-image). Si alguna instalación falla, avisame; no cambies de librería por tu cuenta.

## 2. Estructura

```
./
├── <paper>.pdf                 (ya está)
├── STARE/                      (ya está: imágenes .ppm RGB de 8 bits, 605×700)
├── sonali.py                   punto de entrada por consola
├── procesamiento/
│   ├── __init__.py
│   ├── parametros.py           todas las constantes, cada una con su procedencia
│   ├── ruido.py
│   ├── filtros.py              mediana ponderada y registro de filtros
│   ├── realce.py               CLAHE
│   ├── metricas.py
│   └── salida.py               carpeta de resultados, archivos y tabla en consola
├── tests/
├── requirements.txt
└── README.md
```

## 3. Uso por consola

```
python sonali.py IMAGEN TIPO_RUIDO VARIANZA TIPO_FILTRO [--semilla N] [--clip-limit C]
```

Ejemplos:
```
python sonali.py im0001.ppm gaussiano 0.001 mediana_ponderada
python sonali.py im0001.ppm sal_pimienta 0.01 mediana_ponderada --semilla 3
```

- `IMAGEN`: si es una ruta que existe, se usa tal cual; si no, se busca dentro de `STARE/`. Tiene que ser una imagen RGB de 8 bits.
- `TIPO_RUIDO`: `gaussiano` o `sal_pimienta`.
- `VARIANZA`: un número.
  - En `gaussiano` es la varianza del ruido, con las intensidades en [0, 1].
  - En `sal_pimienta` se interpreta como la densidad *d* de `imnoise`, es decir, la fracción de valores afectados.
  - La ayuda de la consola tiene que explicar las dos interpretaciones.
- `TIPO_FILTRO`: por ahora solo `mediana_ponderada`. Las opciones válidas se leen del registro de filtros (`FILTROS` en `filtros.py`), así que para agregar un filtro no hay que tocar la consola.
- `--semilla`: entero; por defecto, 0.
- `--clip-limit`: clip limit normalizado de CLAHE, en el intervalo (0, 1]; por defecto, 0.0005.
- Validaciones, con mensajes claros:
  - la imagen no existe o no es RGB de 8 bits;
  - la varianza es ≤ 0;
  - la densidad está fuera de (0, 1];
  - el clip limit está fuera de (0, 1].

## 4. Proceso

Las imágenes se manejan en `uint8`, como haría MATLAB con una imagen de 8 bits.

1. Leer la imagen limpia con OpenCV y pasarla de BGR a RGB. Esta imagen es la referencia de todas las métricas.
2. Agregar el ruido a la imagen RGB completa, antes de separar canales (sección 5).
3. Separar los canales R, G y B.
4. Filtrar cada canal con la mediana ponderada (sección 6).
5. Unir los canales filtrados. El resultado es la imagen "solo filtro".
6. Aplicar CLAHE a cada canal filtrado (sección 7) y unir los canales. El resultado es la imagen "final".
7. Calcular las métricas de la ruidosa, de la "solo filtro" y de la final, siempre contra la limpia (sección 8).
8. Mostrar la tabla en consola y guardar los resultados (sección 9).

## 5. Ruido (como lo hace `imnoise` de MATLAB)

Fuente: https://www.mathworks.com/help/images/ref/imnoise.html

**Gaussiano**
1. Pasar la imagen a [0, 1] dividiendo por 255.
2. Sumar ruido normal con media 0 (el valor por defecto de imnoise) y la varianza pedida.
3. Recortar el resultado a [0, 1].
4. Volver a `uint8` con `floor(x * 255 + 0.5)`. Así redondea MATLAB (`uint8(round(x*255))`). No uses `round` de Python ni `np.round`, porque redondean los empates al número par.

**Sal y pimienta** (densidad *d*)

Para cada valor de la imagen (alto × ancho × 3), sortear *u* uniforme en (0, 1):
- si *u* < *d*/2, el valor pasa a 0;
- si *d*/2 ≤ *u* < *d*, pasa a 255;
- si no, queda igual.

**Para los dos tipos**
- El ruido se sortea por separado para cada valor de los 3 canales, porque imnoise trata el arreglo elemento a elemento.
- Generador: `numpy.random.default_rng(semilla)`.
- La semilla por defecto es 0 porque MATLAB también arranca con semilla 0. Pero el generador es otro, así que el ruido no va a ser idéntico al de MATLAB. Aclaralo en el README.

## 6. Mediana ponderada

Se aplica a cada canal; entra y sale en `uint8`.

- Ventana de 3×3.
- Pesos enteros `W = [[1, 1, 1], [1, 4, 1], [1, 1, 1]]`.
- Definición clásica: cada valor de la ventana se repite tantas veces como su peso (12 valores en total) y la salida es la mediana de esos 12 valores.
- Como 12 es par, la mediana es el promedio de los valores 6.º y 7.º ordenados, descartando la parte fraccionaria (división entera). Así lo hace `medfilt2` de MATLAB con imágenes enteras cuando la cantidad de valores es par. Cuidado con el desborde de `uint8` al sumar.
- Bordes: relleno con ceros, que es el relleno por defecto de `medfilt2`.
- La función recibe los pesos como parámetro, con `W` por defecto, y tiene que funcionar con cualquier matriz 3×3 de pesos enteros positivos, sea la suma par o impar.

De dónde salen estos valores (ponelo en el docstring):
- MATLAB no trae un filtro de mediana ponderada, así que no hay valores por defecto que copiar.
- La descripción del filtro en la Tabla 1 del paper es casi textual la del código de MATLAB File Exchange #29386 (M. K. Sharma, 2010): https://www.mathworks.com/matlabcentral/fileexchange/29386-weighted-median-filter
- Ese código usa una ventana de 3×3 y `W = [1 1 1; 1 4 1; 1 1 1]/12`. Multiplica la ventana por W y toma la mediana de los 9 productos, lo que divide la intensidad por unos 12.
- Por eso no se usa esa versión multiplicativa. Se toman los pesos sin dividir y se aplican con la definición clásica:
  - Brownrigg, "The weighted median filter", *Communications of the ACM* 27(8), 1984.
  - Yin, Yang, Gabbouj y Neuvo, "Weighted median filters: a tutorial", *IEEE TCAS-II* 43(3), 1996, doi:10.1109/82.486465.
- Empate y bordes: https://www.mathworks.com/help/images/ref/medfilt2.html

Implementación:
- Versión principal vectorizada con NumPy: `sliding_window_view` sobre la imagen rellenada, `np.repeat` con los pesos y `np.sort`. Un bucle de Python píxel por píxel tardaría minutos en 605×700.
- Además, una versión de referencia con bucles explícitos, corta y fácil de leer. Solo se usa en las pruebas, para verificar la vectorizada.
- Registro: un diccionario `FILTROS` que asocia cada nombre a una función que recibe un canal `uint8` y devuelve un canal `uint8`. Por ahora solo tiene `mediana_ponderada`. Para agregar un filtro alcanza con su función y una entrada nueva en el registro.

## 7. CLAHE

Se usan los valores por defecto de `adapthisteq` de MATLAB: 8×8 teselas, 256 bins y distribución uniforme. Fuente: https://www.mathworks.com/help/images/ref/adapthisteq.html

El clip limit es normalizado, como en MATLAB: va de 0 a 1. Por defecto vale 0.0005, el único valor que da el paper (Fig. 5), y se puede cambiar con `--clip-limit`.

Se implementa con `cv2.createCLAHE(clipLimit=..., tileGridSize=(8, 8))`, aplicado a cada canal filtrado. OpenCV usa otra escala para el clip limit, así que hay que convertirlo con una función `clip_limit_matlab_a_opencv(clip_normalizado, alto, ancho)`:

1. **Tamaño de tesela, igual que OpenCV.** Si el alto y el ancho son divisibles por 8, la tesela mide (alto/8) × (ancho/8). Si no, OpenCV rellena hasta alto + (8 − alto % 8) y ancho + (8 − ancho % 8) y divide por 8. *P* es la cantidad de píxeles de una tesela. (Lo verifiqué en `modules/imgproc/src/clahe.cpp` de la rama 5.x de OpenCV.)
2. **Límite en cuentas, como lo calcula MATLAB.** La fórmula sale del ejemplo de MathWorks que toma adapthisteq como referencia (https://www.mathworks.com/help/visionhdl/ug/contrast-adaptive-histogram-equalization.html):
   - `min_clip = ceil(P / 256)`
   - `clip_cuentas = min_clip + round_matlab(c * (P - min_clip))`, donde `round_matlab(x) = floor(x + 0.5)` para x ≥ 0.
3. **Valor para OpenCV.** OpenCV calcula `int(clipLimit * P / 256)`, con un mínimo de 1. Para que dé exactamente `clip_cuentas`, usar `clipLimit = (clip_cuentas + 0.5) * 256 / P`.

Valores esperados para STARE (605×700), que también van en las pruebas:
- La tesela es de 76×88, así que P = 6688.
- Con c = 0.0005: 30 cuentas, y el clipLimit de OpenCV ≈ 1.167464.
- Con c = 0.01: 94 cuentas, y el clipLimit de OpenCV ≈ 3.617225.

Limitación, para el README: el resultado no es idéntico bit a bit al de `adapthisteq`, porque MATLAB y OpenCV rellenan los bordes y redistribuyen el exceso del histograma de forma algo distinta.

## 8. Métricas

- Todas se calculan sobre la imagen RGB completa (alto × ancho × 3).
- Hay un único valor por métrica; no se muestran valores por canal.
- La referencia es siempre la imagen limpia.
- Los cálculos se hacen en float64.

**PSNR**
- MSE = promedio de (A − ref)² sobre todos los valores RGB.
- PSNR = 10·log10(255² / MSE).
- El pico es 255 porque es el valor por defecto de `psnr` de MATLAB para `uint8`. Así se interpreta el "Peakvalue" de la ecuación 2 del paper.
- Si MSE = 0, el resultado es infinito.
- Fuente: https://www.mathworks.com/help/images/ref/psnr.html

**SSIM** (valores por defecto de `ssim` de MATLAB y ecuaciones 4 a 7 del paper)
- Ventana gaussiana de 11×11 con σ = 1.5 y bordes replicados: `cv2.GaussianBlur(x, (11, 11), 1.5, sigmaY=1.5, borderType=cv2.BORDER_REPLICATE)`.
  - σ = 1.5 sale de la documentación.
  - El tamaño (radio ceil(3σ) = 5) y los bordes replicados salen de un análisis publicado del código de ssim.m, que es una fuente secundaria y hay que etiquetarla así: https://www.cnblogs.com/hyddd/p/7132187.html
- Constantes: C1 = (0.01·255)², C2 = (0.03·255)², C3 = C2/2, exponentes 1. Con esos valores la fórmula se simplifica a ((2μxμy + C1)(2σxy + C2)) / ((μx² + μy² + C1)(σx² + σy² + C2)).
- En RGB, la ventana recorre cada plano de color sin mezclarlos. Eso da un mapa SSIM de alto × ancho × 3, y el resultado es el promedio de todo ese mapa.
  - Equivale a `ssim` de MATLAB con `DataFormat="SSC"`, promediado sobre los canales, que es lo que recomienda su documentación para imágenes RGB.
  - No uses una ventana 3D que mezcle los canales.
- Fuente: https://www.mathworks.com/help/images/ref/ssim.html

**CoC** (ecuación 8 del paper)
- Coeficiente de correlación de Pearson entre todos los valores RGB de la imagen evaluada y los de la limpia.

**EPI** (ecuación 9 del paper)
- EPI = (suma de |X'(m, n+1) − X'(m, n)| en la imagen evaluada X') / (la misma suma en la imagen limpia X).
- Cada suma recorre todas las filas, las columnas n = 1…N−1 y los 3 canales.
- Son solo diferencias horizontales, como en la ecuación 9.
- El valor puede pasar de 1, por ejemplo cuando el ruido agrega bordes. Aclaralo en el README.

Si algún denominador da cero (imagen constante), devolvé NaN y mostrá un aviso.

## 9. Salida

**Tabla en consola**, con 4 decimales:

```
Imagen: im0001.ppm (605×700) | ruido: gaussiano 0.001 | filtro: mediana_ponderada | CLAHE: 0.0005 | semilla: 0

Métrica      Ruidosa    Solo filtro    Final
PSNR (dB)    ...        ...            ...
SSIM         ...        ...            ...
CoC          ...        ...            ...
EPI          ...        ...            ...
```

- "Ruidosa" es el antes: la imagen ruidosa contra la limpia.
- "Solo filtro" es el paso intermedio, antes de CLAHE.
- "Final" es el después: filtro y CLAHE, contra la limpia.

**Carpeta de resultados**: `resultado/<imagen>__ruido-<tipo>__var-<valor>__filtro-<filtro>__clahe-<clip>__semilla-<n>/`
- Ejemplo: `resultado/im0001__ruido-gaussiano__var-0.001__filtro-mediana_ponderada__clahe-0.0005__semilla-0/`
- `<imagen>` es el nombre del archivo sin extensión, y los números se escriben con `format(x, "g")`.
- `__` separa los campos; dentro de cada campo, el primer `-` separa el nombre del valor.
- Si la carpeta ya existe, se sobrescribe: los mismos parámetros dan el mismo resultado.

**Archivos dentro de la carpeta**
- `ruidosa.png`, `solo_filtro.png` y `final.png`: PNG sin pérdida. Antes de `cv2.imwrite`, pasar de RGB a BGR.
- `metricas.csv`: columnas `metrica,ruidosa,solo_filtro,final`, con precisión completa.
- `parametros.json`, con:
  - el comando ejecutado, la fecha y hora, y la ruta y el tamaño de la imagen;
  - los parámetros del ruido;
  - los del filtro: pesos, regla de empate y bordes;
  - los de CLAHE: clip normalizado, P, cuentas y clipLimit de OpenCV;
  - las definiciones de las métricas;
  - las versiones de Python, NumPy y OpenCV;
  - la procedencia de cada parámetro.

**Tabla de procedencia.** Va en `parametros.py` (como comentario junto a cada constante), en el README y en `parametros.json`.

| Parámetro | Valor | Etiqueta | Fuente |
|---|---|---|---|
| Orden del proceso | ruido → separar → filtrar → CLAHE → unir | [PAPER] | Sección 3, pasos 1–5 |
| Media del ruido gaussiano | 0 | [MATLAB] | imnoise |
| Escala de la varianza gaussiana | intensidades en [0, 1] | [MATLAB] | imnoise |
| Recorte y vuelta a uint8 | recortar a [0, 1], ×255, redondear | [MATLAB] | imnoise, `uint8(round(·))` |
| "Varianza" en sal y pimienta | densidad *d* | [DECISIÓN] | El paper la llama varianza; imnoise usa densidad |
| Regla de sal y pimienta | *u* < *d*/2 → 0; *d*/2 ≤ *u* < *d* → 255 | [MATLAB] | imnoise |
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

## 10. Estilo de código

- Nombres de archivos, funciones y variables en español (`agregar_ruido`, `mediana_ponderada`, `calcular_metricas`).
- Nada de `from x import *`.
- Funciones cortas, con *type hints* y docstrings en español.
- Toda constante numérica del método vive en `parametros.py`, con un comentario que indique su etiqueta y su fuente. No dejes números sueltos en el resto del código.
- Usá bucles explícitos donde ayuden a leer y vectorizá donde haga falta por velocidad (sección 6).
- Separá la lógica de la entrada/salida: una función `procesar_imagen(...)` que devuelva las imágenes y las métricas, para poder reutilizarla más adelante con muchas imágenes.
- No agregues opciones ni funciones que no estén en este documento.

## 11. Pruebas (pytest, en `tests/`)

**Mediana ponderada**
- Con imágenes aleatorias chicas (por ejemplo, 7×9), la versión vectorizada y la de bucles dan exactamente lo mismo.
- Caso a mano: con la ventana `[[0,0,0],[0,101,0],[0,101,101]]` y los pesos `W`, el píxel central vale **50**. Los 12 valores son seis 0 y seis 101; el promedio de los dos centrales es 50.5, y sin la fracción queda 50.
- Con todos los pesos en 1, el resultado es igual al de una mediana común de 3×3 con relleno de ceros.

**Ruido** (sobre una imagen gris de valor 128, de 200×200×3)
- Sal y pimienta con *d* = 0.1: la fracción de ceros y la de 255 dan cada una ≈ 0.05 (±0.005).
- Gaussiano con varianza 0.01: la varianza de (salida/255 − 128/255) da ≈ 0.01 (±5 %).
- Con la misma semilla el ruido es idéntico, y con otra semilla cambia.

**CLAHE**
- Para 605×700 y 8×8 teselas, P = 6688.
- Con c = 0.0005 salen 30 cuentas, y con c = 0.01, 94.
- En los dos casos, `int(clipLimit * P / 256)` devuelve esas mismas cuentas.

**Métricas**
- Con dos imágenes idénticas: PSNR = inf, SSIM = 1, CoC = 1 y EPI = 1.
- Con la referencia toda en 100 y la otra toda en 110: PSNR = 28.1308 dB.
- EPI a mano: con la referencia en filas `[0, 10, 20]` en los 3 canales y la otra en `[0, 20, 40]`, EPI = 2.0.

**Prueba completa**
- Con una imagen sintética chica y cada tipo de ruido, el resultado se escribe en una carpeta temporal (`tmp_path`), no en el `resultado/` real.
- Se crean los 5 archivos, con las formas y tipos correctos, y la carpeta tiene el nombre esperado.

## 12. Verificación final (antes de darme el trabajo por terminado)

1. `pytest -q` pasa sin fallas.
2. Correr `python sonali.py im0001.ppm gaussiano 0.001 mediana_ponderada` y `python sonali.py im0001.ppm sal_pimienta 0.01 mediana_ponderada`. Si no existe `STARE/im0001.ppm`, usá el primer `.ppm` de `STARE/` y decime cuál usaste.
3. Mostrame la salida de consola de las dos corridas y el contenido de las carpetas que se crearon.
4. Abrí el `final.png` de una corrida y confirmá que los colores son correctos (no quedaron en BGR) y que la imagen no salió oscura ni saturada.
5. No ajustes parámetros para mejorar las métricas.
6. No compares los resultados con las tablas del paper: no son comparables, porque el paper no publica sus parámetros. Poné esa advertencia en el README.

## 13. README.md

Breve, en español, con:
- qué hace el programa;
- cómo instalar el entorno;
- cómo correrlo, con ejemplos;
- qué guarda y dónde;
- la tabla de procedencia;
- las limitaciones:
  - el ruido no es idéntico al de MATLAB, porque el generador es otro;
  - CLAHE no da lo mismo bit a bit que adapthisteq;
  - los resultados no se pueden comparar con las tablas del paper.
