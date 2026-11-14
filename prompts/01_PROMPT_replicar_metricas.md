# Tarea 01: encontrar la configuración que replica las métricas del paper de Sonali et al. (2019)

Respondeme siempre en español.

## 0. Contexto, objetivo y reglas

Este proyecto implementa en Python el método de Sonali et al. (2019), "An approach for de-noising and contrast enhancement of retinal fundus image using CLAHE" (*Optics and Laser Technology* 110, pp. 87–98). El PDF está en la raíz (`Sonali.pdf`). La primera versión se hizo con `prompts/00_PROMPT_sonali_claude_code.md`; leelo entero, junto con el código actual (`sonali.py`, `procesamiento/`, `tests/`, `README.md`).

**Objetivo.** Encontrar los valores de las variables que **el paper no define** con los que nuestras métricas (PSNR, SSIM, CoC y EPI) coincidan con las Tablas 3 a 8 del paper, o queden lo más cerca posible. Lo que sí define el paper no se toca.

**Entregable: investigación y propuesta.** No modifiques el código del proyecto (`sonali.py`, `procesamiento/`, `tests/`, `README.md`, `requirements.txt`). Todo lo nuevo va en una carpeta `investigacion/` (sección 7). Al final me proponés los cambios concretos y yo decido si se aplican.

**Reglas**
1. **No alucines.** Cada candidato que pruebes tiene que salir de una fuente citable: el paper, la documentación de MATLAB, el código de File Exchange #29386, la literatura o este documento. Separá siempre lo verificado (con su fuente o con un experimento reproducible) de las hipótesis. Si una afirmación sobre MATLAB no la pudiste verificar en la documentación, decilo así; no la presentes como un hecho.
2. **Una sola configuración global.** Los parámetros compartidos (ruido, CLAHE, métricas) son los mismos para todos los filtros y todas las filas. Los parámetros de cada filtro son los mismos para los dos tipos de ruido y todos los niveles. No se permite elegir parámetros distintos por fila ni por nivel de ruido.
3. **Filas que no cierran.** Si con la mejor configuración global una fila sigue lejos del paper, no fuerces el ajuste: documentá el error y las hipótesis que probaste y descartaste.
4. **Solo variables no definidas.** Lo de la sección 2 queda fijo. Si creés que hace falta cambiar algo de esa lista, frená y preguntame.
5. **Grillas cerradas.** Probá solo los candidatos de la sección 4. Si encontrás un candidato nuevo con fuente citable, o si la búsqueda supera unas 5000 evaluaciones del proceso completo o una hora de cómputo, frená y preguntame.
6. **Entorno.** Las mismas dependencias que el proyecto (`numpy`, `opencv-python-headless`, `pytest`, con las versiones de `requirements.txt`). No agregues SciPy ni scikit-image; lo que haga falta (por ejemplo, `wiener2`) se implementa con NumPy u OpenCV. Si algo no se puede hacer así, avisame.
7. **Forma de trabajo.** Primero leé el PDF, este documento, el prompt 00 y el código. Después mostrame un plan breve (archivos, etapas y tamaño estimado de las grillas) y esperá mi confirmación antes de escribir código.

Etiquetas de procedencia: las del prompt 00 (`[PAPER]`, `[MATLAB]`, `[FILE EXCHANGE]`, `[LITERATURA]`, `[OPENCV]`, `[DECISIÓN]`, `[CONVENCIÓN]`) y dos nuevas:
- `[VERIFICADO]`: se dedujo con un experimento reproducible (por ejemplo, qué imagen de STARE usó el paper).
- `[AJUSTE]`: se eligió porque es el candidato con fuente que mejor reproduce las tablas. Siempre va con la evidencia: el error con ese valor y con los otros candidatos.

## 1. Lo que ya se sabe (sesión anterior)

Todo esto se midió con `im0001`, semilla 0 y la mediana ponderada clásica con W = [[1,1,1],[1,4,1],[1,1,1]], salvo que se indique otra cosa. Reproducilo antes de usarlo.

**1.1. La imagen del paper es `STARE/im0001.ppm`** `[VERIFICADO]`. Comparé las figuras extraídas del PDF (`pdfimages -j Sonali.pdf`) con las 6 imágenes de `STARE/`, redimensionadas y suavizadas:
- Fig. 1: correlación 0.978 con `im0001`; la siguiente es `im0003` con 0.939.
- Fig. 7 (ruidosa): 0.993 con `im0001`; la siguiente es 0.934.

**1.2. El CLAHE de OpenCV no se comporta como `adapthisteq` con clip 0.0005.** Las dos implementaciones calculan el mismo límite (30 cuentas), pero reparten distinto el exceso recortado:
- **MATLAB:** ningún bin pasa el límite. Fuente: https://www.mathworks.com/help/visionhdl/ug/contrast-adaptive-histogram-equalization.html (subsistemas "Distribute" y "Redistribute").
- **OpenCV:** suma `redistBatch` a todos los bins sin tope. Fuente: `modules/imgproc/src/clahe.cpp`, rama 5.x.

En una tesela interior del canal G, el histograma recortado queda en 19–30 con MATLAB y en 18–48 con OpenCV.

Aplicando CLAHE 0.0005 a la imagen limpia y comparando con la misma limpia:

| Implementación | PSNR | SSIM |
|---|---|---|
| OpenCV (la del proyecto) | 25.51 dB | 0.864 |
| Reconstrucción de MATLAB (`prompts/01_clahe_matlab_borrador.py`) | 36.92 dB | 0.975 |

**Esa reconstrucción es un borrador sin verificar.** Sigue la descripción del ejemplo HDL y el algoritmo de Zuiderveld. Detalles inciertos:
- el redondeo de `grayxform`;
- si la tabla de valores está en [0, 1] o en [0, 255];
- el redondeo final;
- el relleno `symmetric` hasta teselas de tamaño par.

**1.3. Con el CLAHE estilo MATLAB, las filas de poco ruido quedan cerca** (mediana ponderada + CLAHE, PSNR en dB):

| Ruido | Ruidosa | Solo filtro | + CLAHE OpenCV | + CLAHE MATLAB | Paper |
|---|---|---|---|---|---|
| s&p 0.001 | 34.50 | 50.64 | 26.09 | 37.15 | 36.72 |
| s&p 0.005 | 27.44 | 50.07 | 26.08 | 37.11 | 36.19 |
| s&p 0.01 | 24.42 | 49.62 | 26.07 | 37.08 | 35.82 |
| s&p 0.1 | 14.35 | 31.40 | 24.63 | 30.20 | 35.43 |
| s&p 0.2 | 11.36 | 22.52 | 20.37 | 22.22 | 35.17 |
| s&p 0.5 | 7.38 | 11.33 | 10.98 | 11.27 | 34.82 |
| gauss 0.001 | 30.41 | 35.21 | 23.22 | 31.88 | 29.72 |
| gauss 0.01 | 20.91 | 25.78 | 18.56 | 24.11 | 29.33 |
| gauss 0.1 | 12.01 | 16.53 | 12.64 | 15.27 | 19.78 |
| gauss 0.2 | 9.83 | 13.87 | 11.04 | 12.76 | 14.41 |

SSIM con CLAHE MATLAB: 0.971 en s&p 0.001 (paper 0.975) y 0.709 en gauss 0.001 (paper 0.728).

**1.4. Filas de sal y pimienta con mucho ruido.** La mediana común 3×3 (relleno de ceros) con CLAHE MATLAB da:

| d | Ruidosa | Solo mediana | + CLAHE | Paper |
|---|---|---|---|---|
| 0.2 | 11.36 | 31.47 | 30.45 | 34.64 |
| 0.5 | 7.38 | 14.47 | 14.46 | 33.78 |

Con d = 0.5, el 22.5 % de los valores filtrados sigue en 0 o 255; en la limpia es el 13.8 %. En las filas de poco ruido, el paper queda cerca del techo de 36.9 dB que pone CLAHE sobre la limpia. Eso sugiere que la referencia de las métricas es la imagen limpia.

**1.5. La versión literal de la Tabla 1 queda descartada.** La mediana ponderada multiplicativa de File Exchange #29386 (mediana de ventana·W/12) baja la intensidad media de 72.9 a 6.1 y da 8.95 dB sin CLAHE y 9.07 dB con CLAHE, contra 36.72 en el paper.

**1.6. Otras mediciones**
- La semilla casi no influye: con 5 semillas, gauss 0.001 da entre 31.88 y 31.90 dB.
- La imagen sí influye: con las 6 imágenes, s&p 0.001 da entre 36.4 y 38.8 dB.

**1.7. Hipótesis sin confirmar: el "varianza" gaussiano puede ser la media.** `imnoise(I,'gaussian',m)` con un solo número toma *m* como media y deja la varianza en 0.01 (verificalo en la documentación). Con media = v y varianza 0.01, mediana ponderada + CLAHE MATLAB:

| Filtro | 0.001 | 0.01 | 0.1 | 0.2 |
|---|---|---|---|---|
| WM | 24.07 / 0.256 (paper 29.72 / 0.728) | 23.64 / 0.247 (paper 29.33 / 0.722) | 17.78 / 0.207 (paper 19.78 / 0.664) | 13.46 / 0.200 (paper 14.41 / 0.607) |
| Mediana 3×3 | 26.18 / 0.389 (paper 29.21 / 0.772) | 25.58 / 0.375 (paper 27.89 / 0.767) | 18.28 / 0.311 (paper 18.54 / 0.692) | 13.61 / 0.290 (paper 13.69 / 0.633) |

Cada celda es PSNR / SSIM.
- **A favor:** reproduce la *forma* de las columnas gaussianas del paper. Son casi planas entre 0.001 y 0.01 y caen después, y el SSIM baja poco. Con varianza = v, en cambio, el SSIM cae de 0.71 a 0.03 mientras el paper baja de 0.73 a 0.61.
- **En contra:** los *niveles* no coinciden. Con 0.001 y 0.01 el PSNR queda unos 3 dB abajo y el SSIM unos 0.4 abajo.

Puede que se combine con otra variable sin definir (ventanas más grandes, SSIM sobre gris o en 3D, etc.).

## 2. Lo que define el paper (queda fijo)

| Qué | Valor | Dónde |
|---|---|---|
| Imagen | `STARE/im0001.ppm` (RGB, 605×700) | Sección 4 + 1.1 `[VERIFICADO]` |
| Orden del proceso | ruido en la RGB → separar R, G y B → filtrar cada canal → CLAHE a cada canal → unir | Sección 3, pasos 1–5; Fig. 6 |
| Tipos de ruido | gaussiano y sal y pimienta | Tabla 2; Tablas 3–8 |
| Niveles | 0.001, 0.01, 0.1, 0.2 (Tablas 3–6); 0.005 (Tabla 7); 0.5 (Tabla 8) | Tablas |
| Filtros | media, mediana, Wiener, gaussiano y mediana ponderada | Tabla 1 |
| Herramienta | MATLAB R2016a (la referencia de cualquier comportamiento por defecto) | Sección 4 |
| Realce | CLAHE | Secciones 2.2 y 3 |
| Clip limit | 0.0005 | Fig. 5. **Ojo:** el paper lo da para el histograma de la Fig. 1 y no dice que los experimentos usen ese valor. Mantenelo fijo; en el informe agregá un análisis de sensibilidad (sección 4.3), sin proponer cambiarlo. |
| PSNR | Ecs. 2 y 3 | Sección 4 |
| SSIM | Ecs. 4 a 7, con c3 = c2/2 | Sección 4 |
| CoC | Ec. 8 | Sección 4 |
| EPI | Ec. 9 | Sección 4 |
| Filtro gaussiano | "σ² is the variance (default 1)" | Tabla 1. Es ambiguo, pero está en el paper: σ = 1 es el primer candidato (sección 4.2). |

## 3. Materiales

- `prompts/01_tablas_paper.csv`: las 208 cifras de las Tablas 3 a 8, con columnas `tabla,metrica,filtro,ruido,nivel,valor_paper`.
  - Cada valor está textualmente en el PDF; lo comprobé con `pdftotext`. Lo que no comprobé es la *posición* (fila y columna) de cada cifra: verificala contra el PDF renderizado (`pdftoppm -r 150`) antes de usar el CSV.
  - En las Tablas 7 y 8 aparece además la fila `HM-LCE`, que es otro método. No se replica.
- `prompts/01_clahe_matlab_borrador.py`: el borrador de `adapthisteq` (sección 1.2). Es lento: tiene bucles de Python en el recorte. Verificalo y, si hace falta, vectorizalo.
- Figuras del PDF (extraelas con `pdfimages -j Sonali.pdf <carpeta>/img`):
  - **Fig. 1:** la imagen limpia.
  - **Fig. 5:** histograma antes y después de CLAHE 0.0005.
  - **Fig. 7:** el proceso con sal y pimienta 0.01 y mediana + CLAHE.
  - **Fig. 8:** histogramas por canal de la ruidosa y de la realzada en ese mismo caso. En la Fig. 8, el canal B realzado sigue entre 0 y unos 40, igual que el ruidoso; con OpenCV la media de B sube de 13.6 a 22.1, y con el borrador de MATLAB, a 14.8.
  - Usá las Figs. 5 y 8 como control cualitativo de CLAHE. No las uses como objetivo numérico: son capturas en JPEG.
- Comprobación de la transcripción: el texto del paper calcula las mejoras de la mediana ponderada sobre la mediana como (WM − M)/WM. Por ejemplo, (36.719707 − 35.376870)/36.719707 = 3.66 %, y el texto dice 3.65 %. Los porcentajes del resumen salen de la Tabla 7 contra HM-LCE: 7.85 % de PSNR.

## 4. Variables sin definir y candidatos

Antes de usar cualquier valor por defecto de MATLAB, verificalo con WebFetch en la documentación oficial y citá la URL. Si existe la documentación archivada de R2016a, preferila; si no la encontrás, decilo.

### 4.1. Ruido

| Variable | Candidatos | Fuente |
|---|---|---|
| Llamada al ruido gaussiano | (a) `imnoise(I,'gaussian',0,v)`: varianza = v<br>(b) `imnoise(I,'gaussian',v)`: media = v y varianza 0.01 (sección 1.7) | Documentación de imnoise |
| Sal y pimienta | `imnoise(I,'salt & pepper',d)` con d = nivel; regla del prompt 00 | Documentación de imnoise |
| Generador y semilla | `default_rng`. Casi no influye (1.6). La configuración final se evalúa con 10 semillas (sección 5). | — |

### 4.2. Filtros (por canal, en uint8)

| Filtro | Variable | Candidatos | Fuente |
|---|---|---|---|
| Mediana | ventana | 3×3 (por defecto en `medfilt2`), 5×5, 7×7 | medfilt2 |
| | bordes | ceros (por defecto), `symmetric` | medfilt2 |
| Media | núcleo y aplicación | `fspecial('average',h)` + `imfilter` (correlación, relleno de ceros, salida uint8 redondeada) | fspecial, imfilter |
| | h | 3 (por defecto en `fspecial`), 5, 7 | fspecial |
| | bordes | ceros (por defecto en `imfilter`), `replicate`, `symmetric` | imfilter |
| Gaussiano | σ | 1 (Tabla 1 del paper), 0.5 (por defecto en `fspecial` e `imgaussfilt`) | paper; fspecial; imgaussfilt |
| | tamaño y función | `fspecial('gaussian',h,σ)` + `imfilter` (ceros) con h = 3 (por defecto) o 5; `imgaussfilt(I,σ)`, con tamaño 2·ceil(2σ)+1 y relleno `replicate` (por defecto, verificar) | fspecial; imgaussfilt |
| Wiener | ventana | 3×3 (por defecto en `wiener2`), 5×5, 7×7 | wiener2 |
| | estimación del ruido | la de `wiener2`: promedio de las varianzas locales | wiener2 |
| | bordes | los de `wiener2` (verificar en la documentación cómo calcula la media y la varianza locales) | wiener2 |
| Mediana ponderada | pesos | `[[1,1,1],[1,c,1],[1,1,1]]` con c ∈ {1, 2, 3, 4, 5}; c = 4 es el de File Exchange #29386 | File Exchange #29386 |
| | definición | la clásica (repetir según el peso), con el empate y los bordes de `medfilt2`, como en el proyecto | Brownrigg 1984; Yin et al. 1996 |

Variantes literales de la Tabla 1 (multiplicar la ventana por la máscara y tomar la mediana):
- con W/12: ya descartada (sección 1.5); solo confirmala;
- con W sin dividir, saturando en uint8 como `immultiply`: probala una vez y documentá el resultado.

### 4.3. CLAHE

| Variable | Candidatos | Fuente |
|---|---|---|
| Implementación | `adapthisteq` reconstruido (verificado contra la documentación y el ejemplo HDL); OpenCV (la actual), solo como referencia | 1.2 |
| Teselas, bins, distribución, rango | 8×8, 256, `uniform`, `full` | Documentación de adapthisteq; fijos salvo evidencia fuerte |
| Detalles inciertos del borrador | redondeo de la tabla de valores, redondeo final, relleno hasta teselas pares | Probar las variantes que permita la documentación y elegir la que mejor reproduzca las tablas, como `[AJUSTE]` |
| Clip limit | 0.0005 fijo | Fig. 5 |

Para la sensibilidad del clip limit, evaluá además 0.001, 0.005 y 0.01 con la configuración final y reportalo aparte.

### 4.4. Métricas

| Variable | Candidatos | Fuente |
|---|---|---|
| Referencia | la imagen limpia; para el CoC, además, la ruidosa (el texto de la Ec. 8 dice "original degraded image") | paper; 1.4 |
| Dominio de los datos | (a) RGB, todos los valores juntos (la del proyecto)<br>(b) por canal y promediado<br>(c) `rgb2gray` (coeficientes y redondeo de MATLAB, verificar) | psnr, ssim, corr2, rgb2gray |
| Pico del PSNR | 255 (por defecto en `psnr` para uint8); "maximum difference in the input image" (texto de la Ec. 2): max(ref) − min(ref) | psnr; paper |
| SSIM | (a) la actual: gaussiana 11×11, σ = 1.5, bordes replicados, por plano<br>(b) qué hace `ssim` de R2016a con un arreglo 3-D (¿volumen con gaussiana 3D?); **verificalo en la documentación de esa versión antes de implementarlo**<br>(c) `ssim_index.m` de Wang et al. (2004): gaussiana 11×11, σ = 1.5, filtrado `valid`, K = [0.01 0.03], L = 255 | ssim; Wang et al. 2004 |
| CoC | Pearson según el dominio de arriba. `corr2` acepta solo 2-D (verificar), así que con RGB los autores tuvieron que elegir canal, gris o `A(:)` | corr2; Ec. 8 |
| EPI | Ec. 9 según el dominio de arriba. En el texto extraído, las barras de valor absoluto del numerador no se ven: miralo en la página renderizada. Si la ecuación no las tiene, probá con y sin. | Ec. 9 |

## 5. Procedimiento

Estas son decisiones mías y podés proponerme otras en el plan.

**Objetivo de la búsqueda**
- Para cada fila del CSV: error = nuestro valor − valor del paper.
- Para comparar métricas distintas, cada error se divide por el desvío estándar de los valores del paper de esa misma métrica en esa tabla.
- El criterio principal es la **mediana** de los errores absolutos normalizados en todas las filas de calibración. Así unas pocas filas imposibles no arrastran la configuración. El criterio secundario es el promedio.

**Calibración y validación**
- Se calibra con las Tablas 3 a 6: 5 filtros × 2 ruidos × 4 niveles × 4 métricas.
- Se valida con las Tablas 7 (d = 0.005) y 8 (d = 0.5). Esos niveles no se usan para elegir nada; solo se reportan.

**Etapas**
1. **Base.** Reproducí los números de la sección 1 y verificá la posición de cada cifra del CSV.
2. **CLAHE.** Verificá y corregí el borrador contra la documentación. Controlalo con las Figs. 5 y 8 y con el "techo": CLAHE sobre la limpia.
3. **Ruido y métricas** (variables compartidas). Con los filtros en sus valores por defecto de MATLAB, recorré la grilla de 4.1 y 4.4.
4. **Filtros.** Con lo compartido fijo, recorré la grilla de cada filtro (4.2) por separado.
5. **Iteración.** Repetí las etapas 3 y 4 hasta que la configuración no cambie (como máximo 3 vueltas).
6. **Semillas.** Evaluá la configuración final con las semillas 0 a 9 y reportá la media y el desvío.

Guardá en caché las imágenes ruidosas, las salidas de cada filtro y las de CLAHE, para no recalcularlas.

**Criterio de coincidencia** (decisión mía)
- **"Coincide":** |error| ≤ 2 desvíos de las semillas.
- **"Cerca":** |error| ≤ 0.5 dB en PSNR o ≤ 0.01 en SSIM, CoC o EPI.
- **"No coincide":** el resto.

## 6. Qué tiene que responder el informe

1. La configuración propuesta: cada variable con su valor, su etiqueta, su fuente y su evidencia (el error con ese valor y con los otros candidatos).
2. La tabla completa con las 208 filas: paper, nuestro valor (media ± desvío con 10 semillas), error y categoría (coincide, cerca o no coincide).
3. Un resumen por tabla, filtro y ruido de qué filas coinciden y cuáles no.
4. Para cada grupo de filas que no coincide: las hipótesis que se probaron, con sus números, y por qué se descartaron. Si ninguna hipótesis con fuente lo explica, decilo explícitamente.
5. La sensibilidad al clip limit (sección 4.3) y a la imagen (las 6 de STARE con la configuración final).
6. Los cambios concretos que propondrías al proyecto (archivo por archivo y la tabla de procedencia nueva), sin aplicarlos. Incluí:
   - cómo cambia la interpretación de `VARIANZA` en la consola si gana la hipótesis 1.7;
   - qué pruebas habría que agregar.
7. Las preguntas que me queden por decidir.

## 7. Estructura de `investigacion/`

```
investigacion/
├── INFORME.md                 informe de la sección 6
├── clahe_matlab.py            adapthisteq reconstruido y verificado
├── ruido_candidatos.py
├── filtros_candidatos.py      media, mediana, Wiener, gaussiano y variantes de mediana ponderada
├── metricas_candidatas.py
├── busqueda.py                corre las etapas de la sección 5 y escribe resultados/
├── resultados/                CSV con todas las configuraciones evaluadas y la comparación final
└── tests/                     pruebas de las piezas nuevas (por ejemplo, que clahe_matlab cumpla el tope del clip y cuentas conocidas)
```

- Reutilizá las funciones del proyecto cuando sirvan (por ejemplo, `procesamiento.ruido.a_uint8_matlab`), pero no las modifiques.
- `pytest -q` desde la raíz tiene que seguir pasando: las pruebas del proyecto y las de `investigacion/tests/`.
- Estilo: el mismo del prompt 00 (nombres en español, type hints, docstrings y procedencia de cada constante).

## 8. Verificación final, antes de darme el trabajo por terminado

1. `pytest -q` pasa.
2. `python investigacion/busqueda.py` reproduce los CSV de `resultados/` y la tabla del informe.
3. En el informe, cada afirmación sobre MATLAB lleva su URL o la marca "no verificado".
4. No modificaste nada fuera de `investigacion/` (mostrame `git status`).
