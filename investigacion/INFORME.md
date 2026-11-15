# Informe: configuración que mejor reproduce las Tablas 3 a 8 de Sonali et al. (2019)

Todo lo que sigue sale de `python investigacion/busqueda.py` (CSV en `investigacion/resultados/`). La investigación se hizo sin modificar nada fuera de `investigacion/`; la propuesta se aplicó al proyecto después (ver el recuadro de la sección 6).

## Resumen en simple

- **Qué se buscó.** El paper no dice con qué ajustes exactos corrió sus experimentos. Probé las combinaciones de ajustes posibles (cada una con una fuente) y medí cuál da números más parecidos a las tablas del paper.
- **Qué salió bien.** Con la configuración propuesta, PSNR, SSIM y CoC quedan cerca del paper en la mayoría de las filas de las Tablas 3 a 5: el error típico es 0.7 dB en PSNR, 0.02 en SSIM y 0.007 en CoC. Con la configuración actual del proyecto esas filas estaban a varios dB.
- **Los tres cambios que más acercan.**
  1. Usar un CLAHE que imita al de MATLAB en lugar del de OpenCV.
  2. Interpretar el "ruido gaussiano 0.001…0.2" del paper como `imnoise(I,'gaussian',v)`: eso suma un corrimiento de brillo *v* y deja la varianza fija en 0.01.
  3. Calcular PSNR y SSIM sobre la imagen en gris.
- **Qué no se pudo reproducir con ningún candidato.**
  - El EPI: las 50 filas quedan lejos, tanto con la fórmula del paper como con la definición alternativa de Sattar et al. (1997).
  - La mediana ponderada con ruido gaussiano: 2 a 3 dB por debajo del paper.
  - La Tabla 8 (sal y pimienta 0.5) y la fila del filtro gaussiano con sal y pimienta 0.2.
- **Cuánto coincide, con el criterio estricto del prompt.** De 200 filas: 3 "coinciden", 56 quedan "cerca" y 141 "no coinciden". El criterio "coincide" exige un error menor que 2 desvíos entre semillas (unos 0.04 dB), así que casi ninguna fila puede cumplirlo.
- **Advertencia.** Varias elecciones se apoyan en diferencias chicas y están marcadas como evidencia débil. La configuración es "la que menos se aleja", no una reconstrucción confirmada de lo que hicieron los autores.

## 0. Cómo se hizo

- **Imagen:** `STARE/im0001.ppm`. Repetí la identificación de la sección 1.1: la Fig. 1 correlaciona 0.956 con `im0001` (la siguiente, `im0003`, 0.876) y la ruidosa de la Fig. 7, 0.993 (la siguiente, 0.887). Los valores difieren algo de los de la sesión anterior porque el suavizado es otro; el orden es el mismo. `[VERIFICADO]`
- **CSV del paper:** las 208 cifras coinciden en valor y posición con el PDF (comprobación automática contra `pdftotext -layout`, más la lectura de las páginas renderizadas para el orden de columnas de las Tablas 7 y 8). `[VERIFICADO]`
- **Sección 1 del prompt:** todos los números se reproducen (`resultados/base_seccion1.csv`); por ejemplo, techo de CLAHE 36.92 dB / 0.975 con el estilo MATLAB y 25.51 dB / 0.864 con OpenCV.
- **Error de una fila:** (nuestro − paper) dividido por el desvío estándar de los valores del paper de esa métrica en esa tabla. Criterio principal: mediana de los errores absolutos normalizados de las 160 filas de calibración (Tablas 3 a 6); secundario: el promedio.
- **Búsqueda:** etapa 3 (CLAHE, llamada del ruido gaussiano y métricas, producto completo: 8640 combinaciones por vuelta) y etapa 4 (cada filtro por separado), alternadas 3 vueltas.
- **Regla agregada `[DECISIÓN]`:** un candidato solo reemplaza al valor por defecto si baja la mediana en más de 0.01. Sin esa regla, la primera corrida elegía variantes por diferencias de 0.001, que no son evidencia. Las reversiones quedaron en `resultados/simplificaciones.csv`.
- **Parada:** la búsqueda llegó al máximo de 3 vueltas. En la tercera solo cambiaron dos variantes de métricas casi equivalentes (SSIM "wang" → "plano" y CoC gris → RGB); los filtros, el ruido y CLAHE ya no cambiaron desde la vuelta 2.
- **Costo:** 2151 evaluaciones del proceso completo (tope 5000) y 13 minutos.
- **Ec. 9 (EPI):** en la página renderizada tiene barras de valor absoluto en el numerador y en el denominador, así que no hubo variante "sin barras".

### Documentación de MATLAB consultada

La documentación archivada de R2016a no está en línea (`https://www.mathworks.com/help/releases/R2016a/...` responde "Documentation for the requested release is not available online"). Todo lo que sigue es de la documentación actual.

| Función | Verificado | URL |
|---|---|---|
| `imnoise` | `imnoise(I,'gaussian',m)` usa media *m* y varianza 0.01; media y varianza se expresan para imágenes en [0, 1]; regla de sal y pimienta con *d*/2 | https://www.mathworks.com/help/images/ref/imnoise.html |
| `medfilt2` | ventana 3×3 y relleno `zeros` por defecto; opción `symmetric` | https://www.mathworks.com/help/images/ref/medfilt2.html |
| `fspecial` | `average` 3×3 por defecto; `gaussian` con σ = 0.5 por defecto y núcleo normalizado por su suma | https://www.mathworks.com/help/images/ref/fspecial.html |
| `imfilter` | correlación, relleno 0, salida del mismo tamaño y tipo; en enteros redondea y recorta | https://www.mathworks.com/help/images/ref/imfilter.html |
| `imgaussfilt` | σ = 0.5, tamaño 2·ceil(2σ)+1, relleno `replicate`; existe desde R2015a | https://www.mathworks.com/help/images/ref/imgaussfilt.html |
| `wiener2` | 3×3 por defecto; ruido = promedio de las varianzas locales; fórmula de salida | https://www.mathworks.com/help/images/ref/wiener2.html |
| `adapthisteq` | 8×8 teselas, 256 bins, `uniform`, `full`; interpolación bilineal entre teselas | https://www.mathworks.com/help/images/ref/adapthisteq.html |
| Ejemplo HDL de CLAHE | teselas pares con relleno simétrico; fórmula del clip limit; "Distribute"/"Redistribute"; `grayxform = round(255·cdf/P)`; declara coincidir con `adapthisteq` salvo "unos pocos píxeles" | https://www.mathworks.com/help/visionhdl/ug/contrast-adaptive-histogram-equalization.html |
| `psnr` | pico 255 para `uint8`; acepta arreglos de cualquier dimensión | https://www.mathworks.com/help/images/ref/psnr.html |
| `ssim` | σ = 1.5; C1, C2, C3 = C2/2; exponentes 1; acepta imagen 2-D o volumen 3-D; `DataFormat` llegó en R2021b | https://www.mathworks.com/help/images/ref/ssim.html |
| `corr2` | fórmula de Pearson con `mean2` | https://www.mathworks.com/help/images/ref/corr2.html |
| `rgb2gray` | 0.298936021293775·R + 0.587043074451121·G + 0.114020904255103·B | https://www.mathworks.com/help/matlab/ref/rgb2gray.html |

**No verificado** (no figura en la documentación consultada):

- el tamaño por defecto de `fspecial('gaussian')`: la página actual no lo dice;
- que `corr2` rechace arreglos 3-D: la página no lo dice;
- cómo trata `wiener2` los bordes y la clase `uint8` (asumí relleno de ceros sobre la imagen en `double`);
- que `ssim(A, ref)` con un arreglo M×N×3 use una gaussiana 3-D: una respuesta de MATLAB Answers (no es personal de MathWorks) dice que equivale a `DataFormat="SSS"` (https://www.mathworks.com/matlabcentral/answers/2142916-ssim-calculation-for-rgb-images); el tamaño y los bordes de esa ventana los asumí;
- en `adapthisteq`: de qué lado va el relleno, el redondeo de la salida y los pesos exactos de la interpolación. No encontré una copia pública citable de `adapthisteq.m`;
- el código de File Exchange #29386: la página no muestra el archivo; uso lo que dice el prompt 00.

## 1. Configuración propuesta

"Mediana global" es el criterio principal sobre las 160 filas de calibración; "filas propias" es la mediana sobre las filas que esa variable afecta. Con la configuración propuesta la mediana global es **0.144** (con la configuración inicial del proyecto, más los cuatro filtros nuevos por defecto, era 0.724).

### Variables compartidas

| Variable | Valor propuesto | Etiqueta | Fuente | Evidencia (error con ese valor y con los otros) |
|---|---|---|---|---|
| Implementación de CLAHE | `adapthisteq` reconstruido (`clahe_matlab.py`) | `[MATLAB]` | adapthisteq; ejemplo HDL | Global 0.144; con OpenCV, 0.701. **Fuerte.** |
| CLAHE: tabla de cada tesela | redondeada a enteros | `[MATLAB]` | ejemplo HDL (`grayxform`) | 0.1442; sin redondear, 0.1444. Indistinguible; queda la documentada. |
| CLAHE: redondeo de la salida | redondear | no verificado | — | 0.1442; truncando, 0.1463. Indistinguible. |
| CLAHE: lado del relleno | centrado | no verificado | — | 0.1442; todo al final, 0.1432. Indistinguible. |
| CLAHE: teselas, bins, distribución, clip | 8×8, 256, uniforme, 0.0005 | `[MATLAB]` + `[PAPER]` | adapthisteq; Fig. 5 | Fijos. |
| Ruido gaussiano de nivel *v* | `imnoise(I,'gaussian',v)`: media *v*, varianza 0.01 | `[AJUSTE]` | imnoise | Global 0.144 contra 0.440 con varianza = *v*. Filas gaussianas: 0.262 contra 0.949. **Fuerte.** |
| Sal y pimienta de nivel *d* | densidad *d* | `[MATLAB]` + `[DECISIÓN]` | imnoise | No se buscó (único candidato). |
| PSNR | sobre `rgb2gray`, pico = máx − mín de la limpia en gris (222) | `[AJUSTE]` | rgb2gray; texto de la Ec. 2 | Filas propias: 0.096. Gris con pico 255: 0.166. RGB con pico 255: 0.161. Por canal con pico por rango: 0.511. **Moderada.** |
| SSIM | sobre `rgb2gray`, gaussiana 11×11, σ = 1.5, bordes replicados | `[AJUSTE]` | rgb2gray; ssim | Filas propias: 0.081. Wang sobre gris: 0.109 (indistinguible). RGB por plano: 0.709. Volumen 3-D: 0.531. **Fuerte a favor del gris.** |
| CoC | Pearson sobre todos los valores RGB, contra la limpia | `[PAPER]` + `[DECISIÓN]` | Ec. 8 | Filas propias: 0.217. Gris contra la limpia: 0.142 (mejor, pero en el criterio global la diferencia es 0.001 y quedó el valor por defecto). Contra la ruidosa: 1.37 o más. Por canal: 1.67. |
| EPI | Ec. 9 sobre los 3 canales | `[PAPER]` + `[DECISIÓN]` | Ec. 9 | **Ningún candidato reproduce la Tabla 6:** filas propias 8.77 (RGB), 9.43 (por canal), 6.78 (gris). Queda el valor por defecto. |

El pico por rango en RGB da exactamente lo mismo que 255, porque la limpia tiene mínimo 0 y máximo 255.

### Filtros

| Filtro | Valor propuesto | Etiqueta | Fuente | Evidencia (mediana de las filas propias) |
|---|---|---|---|---|
| Mediana | 5×5, relleno de ceros | `[AJUSTE]` | medfilt2 | 5×5: 0.121. 3×3: 0.142. 7×7: 0.255. `symmetric` es indistinguible de `zeros`. **Débil** (ver 4.3: la 7×7 reproduce mejor la Tabla 8). |
| Wiener | 5×5 | `[AJUSTE]` | wiener2 | 5×5: 0.100. 3×3: 0.548. 7×7: 0.343. **Fuerte.** |
| Media | 5×5, relleno de ceros | `[AJUSTE]` | fspecial; imfilter | 5×5: 0.200. 3×3: 0.393. 7×7: 0.310. Los bordes no se distinguen. **Moderada.** |
| Gaussiano | `fspecial('gaussian', 3, 0.5)` + `imfilter` | `[AJUSTE]` + `[MATLAB]` | fspecial | σ = 0.5: 0.133. σ = 1: 1.14 a 1.43. **Fuerte a favor de σ = 0.5.** Con σ = 0.5, el tamaño (3 o 5) y la función (`fspecial` o `imgaussfilt`) no se distinguen. |
| Mediana ponderada | pesos `[[1,1,1],[1,c,1],[1,1,1]]` con c = 1, definición clásica | `[AJUSTE]` | File Exchange #29386; Brownrigg 1984 | c = 1: 0.232. c = 2: 0.317. c = 3: 0.509. c = 4: 0.610. c = 5: 0.734. **Moderada, y con un problema:** con c = 1 el filtro es idéntico a la mediana común 3×3 (ver 4.2). |

Variantes literales de la Tabla 1, probadas una vez cada una y descartadas (`diagnosticos.csv`):

- **W/12:** 9.07 dB en sal y pimienta 0.001 (paper: 36.72). Confirma la sección 1.5.
- **W sin dividir, saturando en `uint8`:** 35.37 dB en sal y pimienta 0.001 y 25.72 dB en gaussiano 0.001 (paper: 36.72 y 29.72). Peor que la clásica en todas las filas.

## 2. Tabla completa (208 filas)

Media ± desvío con las semillas 0 a 9. Categorías: "coincide" si |error| ≤ 2 desvíos de las semillas; "cerca" si |error| ≤ 0.5 dB (PSNR) o ≤ 0.01 (SSIM, CoC, EPI); "no coincide" el resto. La misma tabla está en `resultados/comparacion_final.csv`.

| Tabla | Métrica | Filtro | Ruido | Nivel | Paper | Nuestro (media ± desvío) | Error | Categoría |
|---|---|---|---|---|---|---|---|---|
| 3 | PSNR | mediana | sal_pimienta | 0.001 | 35.376870 | 35.730364 ± 0.009340 | 0.353494 | cerca |
| 3 | PSNR | mediana | gaussiano | 0.001 | 29.205802 | 29.694329 ± 0.036432 | 0.488527 | cerca |
| 3 | PSNR | mediana | sal_pimienta | 0.01 | 35.342524 | 35.709816 ± 0.013422 | 0.367292 | cerca |
| 3 | PSNR | mediana | gaussiano | 0.01 | 27.890991 | 28.284398 ± 0.037732 | 0.393407 | cerca |
| 3 | PSNR | mediana | sal_pimienta | 0.1 | 35.011961 | 35.443109 ± 0.026920 | 0.431148 | cerca |
| 3 | PSNR | mediana | gaussiano | 0.1 | 18.544124 | 17.739471 ± 0.007627 | -0.804653 | no coincide |
| 3 | PSNR | mediana | sal_pimienta | 0.2 | 34.644201 | 35.140192 ± 0.018431 | 0.495991 | cerca |
| 3 | PSNR | mediana | gaussiano | 0.2 | 13.685337 | 12.614430 ± 0.004998 | -1.070907 | no coincide |
| 3 | PSNR | wiener | sal_pimienta | 0.001 | 32.783764 | 33.382389 ± 0.075311 | 0.598625 | no coincide |
| 3 | PSNR | wiener | gaussiano | 0.001 | 28.612019 | 29.217652 ± 0.020153 | 0.605633 | no coincide |
| 3 | PSNR | wiener | sal_pimienta | 0.01 | 28.545231 | 27.813417 ± 0.049225 | -0.731814 | no coincide |
| 3 | PSNR | wiener | gaussiano | 0.01 | 27.424313 | 27.933843 ± 0.020641 | 0.509530 | no coincide |
| 3 | PSNR | wiener | sal_pimienta | 0.1 | 23.233115 | 22.573034 ± 0.020346 | -0.660081 | no coincide |
| 3 | PSNR | wiener | gaussiano | 0.1 | 18.685778 | 17.955940 ± 0.005376 | -0.729838 | no coincide |
| 3 | PSNR | wiener | sal_pimienta | 0.2 | 20.894587 | 20.190858 ± 0.026945 | -0.703729 | no coincide |
| 3 | PSNR | wiener | gaussiano | 0.2 | 13.755823 | 12.709250 ± 0.006476 | -1.046573 | no coincide |
| 3 | PSNR | media | sal_pimienta | 0.001 | 32.075036 | 33.422421 ± 0.025007 | 1.347385 | no coincide |
| 3 | PSNR | media | gaussiano | 0.001 | 28.866618 | 29.947904 ± 0.027448 | 1.081286 | no coincide |
| 3 | PSNR | media | sal_pimienta | 0.01 | 31.059567 | 32.085429 ± 0.034345 | 1.025862 | no coincide |
| 3 | PSNR | media | gaussiano | 0.01 | 27.656338 | 28.424036 ± 0.025104 | 0.767698 | no coincide |
| 3 | PSNR | media | sal_pimienta | 0.1 | 25.302295 | 25.340165 ± 0.034008 | 0.037870 | coincide |
| 3 | PSNR | media | gaussiano | 0.1 | 18.690148 | 17.983355 ± 0.004786 | -0.706793 | no coincide |
| 3 | PSNR | media | sal_pimienta | 0.2 | 21.903061 | 21.661754 ± 0.029683 | -0.241307 | cerca |
| 3 | PSNR | media | gaussiano | 0.2 | 13.740520 | 12.702410 ± 0.002582 | -1.038110 | no coincide |
| 3 | PSNR | gaussiano | sal_pimienta | 0.001 | 33.921009 | 34.391284 ± 0.045215 | 0.470275 | cerca |
| 3 | PSNR | gaussiano | gaussiano | 0.001 | 24.371021 | 24.279067 ± 0.010731 | -0.091954 | cerca |
| 3 | PSNR | gaussiano | sal_pimienta | 0.01 | 29.783372 | 29.363481 ± 0.051647 | -0.419891 | cerca |
| 3 | PSNR | gaussiano | gaussiano | 0.01 | 23.836875 | 23.721126 ± 0.010919 | -0.115749 | cerca |
| 3 | PSNR | gaussiano | sal_pimienta | 0.1 | 21.037163 | 20.124496 ± 0.021129 | -0.912667 | no coincide |
| 3 | PSNR | gaussiano | gaussiano | 0.1 | 17.814673 | 17.191406 ± 0.004899 | -0.623267 | no coincide |
| 3 | PSNR | gaussiano | sal_pimienta | 0.2 | 21.005991 | 16.898109 ± 0.022491 | -4.107882 | no coincide |
| 3 | PSNR | gaussiano | gaussiano | 0.2 | 13.585131 | 12.579599 ± 0.003392 | -1.005532 | no coincide |
| 3 | PSNR | mediana_ponderada | sal_pimienta | 0.001 | 36.719707 | 36.012325 ± 0.005633 | -0.707382 | no coincide |
| 3 | PSNR | mediana_ponderada | gaussiano | 0.001 | 29.718484 | 27.333385 ± 0.027248 | -2.385099 | no coincide |
| 3 | PSNR | mediana_ponderada | sal_pimienta | 0.01 | 35.819953 | 35.991833 ± 0.009580 | 0.171880 | cerca |
| 3 | PSNR | mediana_ponderada | gaussiano | 0.01 | 29.329734 | 26.402291 ± 0.024094 | -2.927443 | no coincide |
| 3 | PSNR | mediana_ponderada | sal_pimienta | 0.1 | 35.425546 | 35.486581 ± 0.060911 | 0.061035 | coincide |
| 3 | PSNR | mediana_ponderada | gaussiano | 0.1 | 19.783958 | 17.551648 ± 0.007710 | -2.232310 | no coincide |
| 3 | PSNR | mediana_ponderada | sal_pimienta | 0.2 | 35.171530 | 32.081994 ± 0.111334 | -3.089536 | no coincide |
| 3 | PSNR | mediana_ponderada | gaussiano | 0.2 | 14.407455 | 12.627481 ± 0.005490 | -1.779974 | no coincide |
| 4 | SSIM | mediana | sal_pimienta | 0.001 | 0.962327 | 0.973616 ± 0.000008 | 0.011289 | no coincide |
| 4 | SSIM | mediana | gaussiano | 0.001 | 0.772254 | 0.800028 ± 0.001175 | 0.027774 | no coincide |
| 4 | SSIM | mediana | sal_pimienta | 0.01 | 0.962192 | 0.973492 ± 0.000026 | 0.011300 | no coincide |
| 4 | SSIM | mediana | gaussiano | 0.01 | 0.767369 | 0.792960 ± 0.001215 | 0.025591 | no coincide |
| 4 | SSIM | mediana | sal_pimienta | 0.1 | 0.958761 | 0.971413 ± 0.000094 | 0.012652 | no coincide |
| 4 | SSIM | mediana | gaussiano | 0.1 | 0.691514 | 0.719842 ± 0.000961 | 0.028328 | no coincide |
| 4 | SSIM | mediana | sal_pimienta | 0.2 | 0.951407 | 0.967630 ± 0.000155 | 0.016223 | no coincide |
| 4 | SSIM | mediana | gaussiano | 0.2 | 0.632898 | 0.658698 ± 0.000813 | 0.025800 | no coincide |
| 4 | SSIM | wiener | sal_pimienta | 0.001 | 0.916593 | 0.936781 ± 0.001162 | 0.020188 | no coincide |
| 4 | SSIM | wiener | gaussiano | 0.001 | 0.755744 | 0.763589 ± 0.000601 | 0.007845 | cerca |
| 4 | SSIM | wiener | sal_pimienta | 0.01 | 0.736117 | 0.730336 ± 0.001942 | -0.005781 | cerca |
| 4 | SSIM | wiener | gaussiano | 0.01 | 0.750255 | 0.760317 ± 0.000566 | 0.010062 | no coincide |
| 4 | SSIM | wiener | sal_pimienta | 0.1 | 0.459403 | 0.451302 ± 0.001147 | -0.008101 | cerca |
| 4 | SSIM | wiener | gaussiano | 0.1 | 0.692935 | 0.704115 ± 0.000557 | 0.011180 | no coincide |
| 4 | SSIM | wiener | sal_pimienta | 0.2 | 0.365518 | 0.364757 ± 0.000653 | -0.000761 | coincide |
| 4 | SSIM | wiener | gaussiano | 0.2 | 0.627231 | 0.638955 ± 0.000472 | 0.011724 | no coincide |
| 4 | SSIM | media | sal_pimienta | 0.001 | 0.926023 | 0.956396 ± 0.000187 | 0.030373 | no coincide |
| 4 | SSIM | media | gaussiano | 0.001 | 0.816556 | 0.847931 ± 0.000384 | 0.031375 | no coincide |
| 4 | SSIM | media | sal_pimienta | 0.01 | 0.895237 | 0.923676 ± 0.000453 | 0.028439 | no coincide |
| 4 | SSIM | media | gaussiano | 0.01 | 0.810642 | 0.842388 ± 0.000368 | 0.031746 | no coincide |
| 4 | SSIM | media | sal_pimienta | 0.1 | 0.666808 | 0.689959 ± 0.000937 | 0.023151 | no coincide |
| 4 | SSIM | media | gaussiano | 0.1 | 0.739105 | 0.772865 ± 0.000425 | 0.033760 | no coincide |
| 4 | SSIM | media | sal_pimienta | 0.2 | 0.516100 | 0.541336 ± 0.001541 | 0.025236 | no coincide |
| 4 | SSIM | media | gaussiano | 0.2 | 0.671041 | 0.702234 ± 0.000324 | 0.031193 | no coincide |
| 4 | SSIM | gaussiano | sal_pimienta | 0.001 | 0.962614 | 0.967438 ± 0.000858 | 0.004824 | cerca |
| 4 | SSIM | gaussiano | gaussiano | 0.001 | 0.378019 | 0.366200 ± 0.000683 | -0.011819 | no coincide |
| 4 | SSIM | gaussiano | sal_pimienta | 0.01 | 0.791894 | 0.780728 ± 0.002219 | -0.011166 | no coincide |
| 4 | SSIM | gaussiano | gaussiano | 0.01 | 0.374464 | 0.361711 ± 0.000698 | -0.012753 | no coincide |
| 4 | SSIM | gaussiano | sal_pimienta | 0.1 | 0.237781 | 0.211666 ± 0.001002 | -0.026115 | no coincide |
| 4 | SSIM | gaussiano | gaussiano | 0.1 | 0.335231 | 0.325761 ± 0.000479 | -0.009470 | cerca |
| 4 | SSIM | gaussiano | sal_pimienta | 0.2 | 0.237073 | 0.107553 ± 0.000395 | -0.129520 | no coincide |
| 4 | SSIM | gaussiano | gaussiano | 0.2 | 0.311754 | 0.300870 ± 0.000450 | -0.010884 | no coincide |
| 4 | SSIM | mediana_ponderada | sal_pimienta | 0.001 | 0.975304 | 0.982922 ± 0.000004 | 0.007618 | cerca |
| 4 | SSIM | mediana_ponderada | gaussiano | 0.001 | 0.727873 | 0.595561 ± 0.001241 | -0.132312 | no coincide |
| 4 | SSIM | mediana_ponderada | sal_pimienta | 0.01 | 0.965923 | 0.982741 ± 0.000022 | 0.016818 | no coincide |
| 4 | SSIM | mediana_ponderada | gaussiano | 0.01 | 0.721606 | 0.589332 ± 0.001243 | -0.132274 | no coincide |
| 4 | SSIM | mediana_ponderada | sal_pimienta | 0.1 | 0.961078 | 0.977487 ± 0.000399 | 0.016409 | no coincide |
| 4 | SSIM | mediana_ponderada | gaussiano | 0.1 | 0.663598 | 0.534538 ± 0.001112 | -0.129060 | no coincide |
| 4 | SSIM | mediana_ponderada | sal_pimienta | 0.2 | 0.959336 | 0.922158 ± 0.001675 | -0.037178 | no coincide |
| 4 | SSIM | mediana_ponderada | gaussiano | 0.2 | 0.607094 | 0.493024 ± 0.000955 | -0.114070 | no coincide |
| 5 | CoC | mediana | sal_pimienta | 0.001 | 0.995936 | 0.998845 ± 0.000002 | 0.002909 | cerca |
| 5 | CoC | mediana | gaussiano | 0.001 | 0.984200 | 0.992591 ± 0.000042 | 0.008391 | cerca |
| 5 | CoC | mediana | sal_pimienta | 0.01 | 0.995904 | 0.998839 ± 0.000003 | 0.002935 | cerca |
| 5 | CoC | mediana | gaussiano | 0.01 | 0.983187 | 0.992348 ± 0.000048 | 0.009161 | cerca |
| 5 | CoC | mediana | sal_pimienta | 0.1 | 0.995609 | 0.998750 ± 0.000007 | 0.003141 | cerca |
| 5 | CoC | mediana | gaussiano | 0.1 | 0.980631 | 0.991237 ± 0.000048 | 0.010606 | no coincide |
| 5 | CoC | mediana | sal_pimienta | 0.2 | 0.995154 | 0.998597 ± 0.000005 | 0.003443 | cerca |
| 5 | CoC | mediana | gaussiano | 0.2 | 0.980155 | 0.990017 ± 0.000049 | 0.009862 | cerca |
| 5 | CoC | wiener | sal_pimienta | 0.001 | 0.992425 | 0.996376 ± 0.000076 | 0.003951 | cerca |
| 5 | CoC | wiener | gaussiano | 0.001 | 0.985162 | 0.991603 ± 0.000028 | 0.006441 | cerca |
| 5 | CoC | wiener | sal_pimienta | 0.01 | 0.978108 | 0.979553 ± 0.000205 | 0.001445 | cerca |
| 5 | CoC | wiener | gaussiano | 0.01 | 0.984797 | 0.991703 ± 0.000026 | 0.006906 | cerca |
| 5 | CoC | wiener | sal_pimienta | 0.1 | 0.933659 | 0.936790 ± 0.000136 | 0.003131 | cerca |
| 5 | CoC | wiener | gaussiano | 0.1 | 0.982089 | 0.991352 ± 0.000037 | 0.009263 | cerca |
| 5 | CoC | wiener | sal_pimienta | 0.2 | 0.894428 | 0.903501 ± 0.000418 | 0.009073 | cerca |
| 5 | CoC | wiener | gaussiano | 0.2 | 0.979562 | 0.988735 ± 0.000037 | 0.009173 | cerca |
| 5 | CoC | media | sal_pimienta | 0.001 | 0.991667 | 0.998051 ± 0.000009 | 0.006384 | cerca |
| 5 | CoC | media | gaussiano | 0.001 | 0.987125 | 0.994882 ± 0.000029 | 0.007757 | cerca |
| 5 | CoC | media | sal_pimienta | 0.01 | 0.989854 | 0.996732 ± 0.000021 | 0.006878 | cerca |
| 5 | CoC | media | gaussiano | 0.01 | 0.986607 | 0.994860 ± 0.000030 | 0.008253 | cerca |
| 5 | CoC | media | sal_pimienta | 0.1 | 0.971225 | 0.981368 ± 0.000078 | 0.010143 | no coincide |
| 5 | CoC | media | gaussiano | 0.1 | 0.983454 | 0.993909 ± 0.000038 | 0.010455 | no coincide |
| 5 | CoC | media | sal_pimienta | 0.2 | 0.943358 | 0.958041 ± 0.000281 | 0.014683 | no coincide |
| 5 | CoC | media | gaussiano | 0.2 | 0.977634 | 0.991489 ± 0.000035 | 0.013855 | no coincide |
| 5 | CoC | gaussiano | sal_pimienta | 0.001 | 0.994516 | 0.997850 ± 0.000035 | 0.003334 | cerca |
| 5 | CoC | gaussiano | gaussiano | 0.001 | 0.947788 | 0.965605 ± 0.000056 | 0.017817 | no coincide |
| 5 | CoC | gaussiano | sal_pimienta | 0.01 | 0.984102 | 0.987264 ± 0.000144 | 0.003162 | cerca |
| 5 | CoC | gaussiano | gaussiano | 0.01 | 0.947453 | 0.965149 ± 0.000056 | 0.017696 | no coincide |
| 5 | CoC | gaussiano | sal_pimienta | 0.1 | 0.878259 | 0.880344 ± 0.000367 | 0.002085 | cerca |
| 5 | CoC | gaussiano | gaussiano | 0.1 | 0.942873 | 0.961113 ± 0.000054 | 0.018240 | no coincide |
| 5 | CoC | gaussiano | sal_pimienta | 0.2 | 0.876521 | 0.762688 ± 0.000573 | -0.113833 | no coincide |
| 5 | CoC | gaussiano | gaussiano | 0.2 | 0.942859 | 0.957923 ± 0.000054 | 0.015064 | no coincide |
| 5 | CoC | mediana_ponderada | sal_pimienta | 0.001 | 0.998919 | 0.998967 ± 0.000001 | 0.000048 | cerca |
| 5 | CoC | mediana_ponderada | gaussiano | 0.001 | 0.981838 | 0.984022 ± 0.000059 | 0.002184 | cerca |
| 5 | CoC | mediana_ponderada | sal_pimienta | 0.01 | 0.998559 | 0.998960 ± 0.000003 | 0.000401 | cerca |
| 5 | CoC | mediana_ponderada | gaussiano | 0.01 | 0.981312 | 0.983658 ± 0.000060 | 0.002346 | cerca |
| 5 | CoC | mediana_ponderada | sal_pimienta | 0.1 | 0.998174 | 0.998680 ± 0.000040 | 0.000506 | cerca |
| 5 | CoC | mediana_ponderada | gaussiano | 0.1 | 0.980056 | 0.981625 ± 0.000066 | 0.001569 | cerca |
| 5 | CoC | mediana_ponderada | sal_pimienta | 0.2 | 0.997298 | 0.994096 ± 0.000114 | -0.003202 | cerca |
| 5 | CoC | mediana_ponderada | gaussiano | 0.2 | 0.979335 | 0.980727 ± 0.000057 | 0.001392 | cerca |
| 6 | EPI | mediana | sal_pimienta | 0.001 | 0.975770 | 0.475240 ± 0.000062 | -0.500530 | no coincide |
| 6 | EPI | mediana | gaussiano | 0.001 | 0.919333 | 2.184531 ± 0.006411 | 1.265198 | no coincide |
| 6 | EPI | mediana | sal_pimienta | 0.01 | 0.967323 | 0.476254 ± 0.000231 | -0.491069 | no coincide |
| 6 | EPI | mediana | gaussiano | 0.01 | 0.908845 | 2.235580 ± 0.006156 | 1.326735 | no coincide |
| 6 | EPI | mediana | sal_pimienta | 0.1 | 0.932775 | 0.491987 ± 0.000666 | -0.440788 | no coincide |
| 6 | EPI | mediana | gaussiano | 0.1 | 0.874766 | 2.337238 ± 0.005938 | 1.462472 | no coincide |
| 6 | EPI | mediana | sal_pimienta | 0.2 | 0.901980 | 0.521292 ± 0.000966 | -0.380688 | no coincide |
| 6 | EPI | mediana | gaussiano | 0.2 | 0.802964 | 2.248235 ± 0.005729 | 1.445271 | no coincide |
| 6 | EPI | wiener | sal_pimienta | 0.001 | 0.934824 | 0.634476 ± 0.004210 | -0.300348 | no coincide |
| 6 | EPI | wiener | gaussiano | 0.001 | 0.814890 | 3.226239 ± 0.004206 | 2.411349 | no coincide |
| 6 | EPI | wiener | sal_pimienta | 0.01 | 0.839759 | 1.776720 ± 0.011540 | 0.936961 | no coincide |
| 6 | EPI | wiener | gaussiano | 0.01 | 0.806004 | 3.212752 ± 0.003891 | 2.406748 | no coincide |
| 6 | EPI | wiener | sal_pimienta | 0.1 | 0.663500 | 6.112181 ± 0.015418 | 5.448681 | no coincide |
| 6 | EPI | wiener | gaussiano | 0.1 | 0.792641 | 3.174927 ± 0.003023 | 2.382286 | no coincide |
| 6 | EPI | wiener | sal_pimienta | 0.2 | 0.582712 | 8.417395 ± 0.015285 | 7.834683 | no coincide |
| 6 | EPI | wiener | gaussiano | 0.2 | 0.788118 | 3.345950 ± 0.002907 | 2.557832 | no coincide |
| 6 | EPI | media | sal_pimienta | 0.001 | 0.967379 | 0.501181 ± 0.000936 | -0.466198 | no coincide |
| 6 | EPI | media | gaussiano | 0.001 | 0.859345 | 1.717537 ± 0.001622 | 0.858192 | no coincide |
| 6 | EPI | media | sal_pimienta | 0.01 | 0.905243 | 0.777296 ± 0.003060 | -0.127947 | no coincide |
| 6 | EPI | media | gaussiano | 0.01 | 0.851678 | 1.739553 ± 0.001827 | 0.887875 | no coincide |
| 6 | EPI | media | sal_pimienta | 0.1 | 0.833512 | 2.890225 ± 0.004428 | 2.056713 | no coincide |
| 6 | EPI | media | gaussiano | 0.1 | 0.816744 | 1.877021 ± 0.001792 | 1.060277 | no coincide |
| 6 | EPI | media | sal_pimienta | 0.2 | 0.761389 | 4.403633 ± 0.007727 | 3.642244 | no coincide |
| 6 | EPI | media | gaussiano | 0.2 | 0.799861 | 1.884647 ± 0.001581 | 1.084786 | no coincide |
| 6 | EPI | gaussiano | sal_pimienta | 0.001 | 0.971780 | 1.008754 ± 0.003224 | 0.036974 | no coincide |
| 6 | EPI | gaussiano | gaussiano | 0.001 | 0.493119 | 9.756640 ± 0.009581 | 9.263521 | no coincide |
| 6 | EPI | gaussiano | sal_pimienta | 0.01 | 0.824640 | 2.078556 ± 0.012342 | 1.253916 | no coincide |
| 6 | EPI | gaussiano | gaussiano | 0.01 | 0.482138 | 9.899583 ± 0.010282 | 9.417445 | no coincide |
| 6 | EPI | gaussiano | sal_pimienta | 0.1 | 0.567712 | 11.510364 ± 0.029706 | 10.942652 | no coincide |
| 6 | EPI | gaussiano | gaussiano | 0.1 | 0.455450 | 10.755171 ± 0.008825 | 10.299721 | no coincide |
| 6 | EPI | gaussiano | sal_pimienta | 0.2 | 0.559466 | 19.841316 ± 0.036761 | 19.281850 | no coincide |
| 6 | EPI | gaussiano | gaussiano | 0.2 | 0.413555 | 10.687165 ± 0.008380 | 10.273610 | no coincide |
| 6 | EPI | mediana_ponderada | sal_pimienta | 0.001 | 0.984367 | 0.619903 ± 0.000024 | -0.364464 | no coincide |
| 6 | EPI | mediana_ponderada | gaussiano | 0.001 | 0.889385 | 4.326483 ± 0.008682 | 3.437098 | no coincide |
| 6 | EPI | mediana_ponderada | sal_pimienta | 0.01 | 0.980937 | 0.621846 ± 0.000421 | -0.359091 | no coincide |
| 6 | EPI | mediana_ponderada | gaussiano | 0.01 | 0.832670 | 4.409053 ± 0.008637 | 3.576383 | no coincide |
| 6 | EPI | mediana_ponderada | sal_pimienta | 0.1 | 0.966822 | 0.659863 ± 0.001689 | -0.306959 | no coincide |
| 6 | EPI | mediana_ponderada | gaussiano | 0.1 | 0.776450 | 4.678537 ± 0.007943 | 3.902087 | no coincide |
| 6 | EPI | mediana_ponderada | sal_pimienta | 0.2 | 0.961093 | 0.920325 ± 0.004497 | -0.040768 | no coincide |
| 6 | EPI | mediana_ponderada | gaussiano | 0.2 | 0.705152 | 4.508540 ± 0.007289 | 3.803388 | no coincide |
| 7 | PSNR | HM-LCE | sal_pimienta | 0.005 | 33.347485 | — | — | no se replica (otro método) |
| 7 | PSNR | mediana | sal_pimienta | 0.005 | 35.350071 | 35.723565 ± 0.019300 | 0.373494 | cerca |
| 7 | PSNR | gaussiano | sal_pimienta | 0.005 | 32.156432 | 31.398621 ± 0.044032 | -0.757811 | no coincide |
| 7 | PSNR | mediana_ponderada | sal_pimienta | 0.005 | 36.189543 | 36.003767 ± 0.007846 | -0.185776 | cerca |
| 7 | PSNR | wiener | sal_pimienta | 0.005 | 31.684723 | 29.625563 ± 0.039301 | -2.059160 | no coincide |
| 7 | PSNR | media | sal_pimienta | 0.005 | 31.977563 | 32.781389 ± 0.012631 | 0.803826 | no coincide |
| 7 | SSIM | HM-LCE | sal_pimienta | 0.005 | 0.960207 | — | — | no se replica (otro método) |
| 7 | SSIM | mediana | sal_pimienta | 0.005 | 0.962201 | 0.973562 ± 0.000011 | 0.011361 | no coincide |
| 7 | SSIM | gaussiano | sal_pimienta | 0.005 | 0.958962 | 0.876095 ± 0.001631 | -0.082867 | no coincide |
| 7 | SSIM | mediana_ponderada | sal_pimienta | 0.005 | 0.971790 | 0.982843 ± 0.000014 | 0.011053 | no coincide |
| 7 | SSIM | wiener | sal_pimienta | 0.005 | 0.901265 | 0.820782 ± 0.001750 | -0.080483 | no coincide |
| 7 | SSIM | media | sal_pimienta | 0.005 | 0.913341 | 0.941234 ± 0.000268 | 0.027893 | no coincide |
| 7 | CoC | HM-LCE | sal_pimienta | 0.005 | 0.994712 | — | — | no se replica (otro método) |
| 7 | CoC | mediana | sal_pimienta | 0.005 | 0.994919 | 0.998842 ± 0.000004 | 0.003923 | cerca |
| 7 | CoC | gaussiano | sal_pimienta | 0.005 | 0.994516 | 0.993064 ± 0.000098 | -0.001452 | cerca |
| 7 | CoC | mediana_ponderada | sal_pimienta | 0.005 | 0.995936 | 0.998964 ± 0.000002 | 0.003028 | cerca |
| 7 | CoC | wiener | sal_pimienta | 0.005 | 0.992425 | 0.987574 ± 0.000172 | -0.004851 | cerca |
| 7 | CoC | media | sal_pimienta | 0.005 | 0.991667 | 0.997464 ± 0.000014 | 0.005797 | cerca |
| 7 | EPI | HM-LCE | sal_pimienta | 0.005 | 0.959674 | — | — | no se replica (otro método) |
| 7 | EPI | mediana | sal_pimienta | 0.005 | 0.969445 | 0.475575 ± 0.000244 | -0.493870 | no coincide |
| 7 | EPI | gaussiano | sal_pimienta | 0.005 | 0.954491 | 1.492767 ± 0.008552 | 0.538276 | no coincide |
| 7 | EPI | mediana_ponderada | sal_pimienta | 0.005 | 0.972134 | 0.620836 ± 0.000227 | -0.351298 | no coincide |
| 7 | EPI | wiener | sal_pimienta | 0.005 | 0.911346 | 1.196017 ± 0.009511 | 0.284671 | no coincide |
| 7 | EPI | media | sal_pimienta | 0.005 | 0.955223 | 0.627495 ± 0.002068 | -0.327728 | no coincide |
| 8 | PSNR | HM-LCE | sal_pimienta | 0.5 | 32.649113 | — | — | no se replica (otro método) |
| 8 | PSNR | mediana | sal_pimienta | 0.5 | 33.775695 | 27.842817 ± 0.132338 | -5.932878 | no coincide |
| 8 | PSNR | gaussiano | sal_pimienta | 0.5 | 20.614237 | 12.295346 ± 0.007529 | -8.318891 | no coincide |
| 8 | PSNR | mediana_ponderada | sal_pimienta | 0.5 | 34.824575 | 17.246760 ± 0.020154 | -17.577815 | no coincide |
| 8 | PSNR | wiener | sal_pimienta | 0.5 | 20.500871 | 15.645338 ± 0.013849 | -4.855533 | no coincide |
| 8 | PSNR | media | sal_pimienta | 0.5 | 20.964781 | 15.884465 ± 0.016734 | -5.080316 | no coincide |
| 8 | SSIM | HM-LCE | sal_pimienta | 0.5 | 0.912563 | — | — | no se replica (otro método) |
| 8 | SSIM | mediana | sal_pimienta | 0.5 | 0.944113 | 0.815606 ± 0.002747 | -0.128507 | no coincide |
| 8 | SSIM | gaussiano | sal_pimienta | 0.5 | 0.209523 | 0.039698 ± 0.000289 | -0.169825 | no coincide |
| 8 | SSIM | mediana_ponderada | sal_pimienta | 0.5 | 0.949867 | 0.175417 ± 0.001193 | -0.774450 | no coincide |
| 8 | SSIM | wiener | sal_pimienta | 0.5 | 0.316455 | 0.258831 ± 0.001054 | -0.057624 | no coincide |
| 8 | SSIM | media | sal_pimienta | 0.5 | 0.489056 | 0.326194 ± 0.001183 | -0.162862 | no coincide |
| 8 | CoC | HM-LCE | sal_pimienta | 0.5 | 0.977153 | — | — | no se replica (otro método) |
| 8 | CoC | mediana | sal_pimienta | 0.5 | 0.984528 | 0.980218 ± 0.000586 | -0.004310 | cerca |
| 8 | CoC | gaussiano | sal_pimienta | 0.5 | 0.857529 | 0.433188 ± 0.000797 | -0.424341 | no coincide |
| 8 | CoC | mediana_ponderada | sal_pimienta | 0.5 | 0.991619 | 0.784925 ± 0.000711 | -0.206694 | no coincide |
| 8 | CoC | wiener | sal_pimienta | 0.5 | 0.856218 | 0.758306 ± 0.000795 | -0.097912 | no coincide |
| 8 | CoC | media | sal_pimienta | 0.5 | 0.915224 | 0.813501 ± 0.000727 | -0.101723 | no coincide |
| 8 | EPI | HM-LCE | sal_pimienta | 0.5 | 0.837165 | — | — | no se replica (otro método) |
| 8 | EPI | mediana | sal_pimienta | 0.5 | 0.854892 | 1.350686 ± 0.018691 | 0.495794 | no coincide |
| 8 | EPI | gaussiano | sal_pimienta | 0.5 | 0.511638 | 36.521096 ± 0.025343 | 36.009458 | no coincide |
| 8 | EPI | mediana_ponderada | sal_pimienta | 0.5 | 0.927736 | 10.049410 ± 0.022421 | 9.121674 | no coincide |
| 8 | EPI | wiener | sal_pimienta | 0.5 | 0.494759 | 10.691060 ± 0.012497 | 10.196301 | no coincide |
| 8 | EPI | media | sal_pimienta | 0.5 | 0.618762 | 6.961159 ± 0.006747 | 6.342397 | no coincide |

## 3. Resumen de coincidencias

Totales: **3 coinciden, 56 cerca, 141 no coinciden** (de 200; las 8 filas de HM-LCE no se replican). El desvío entre semillas del PSNR es de unos 0.02 dB, así que "coincide" exige acertar a 0.04 dB.

| Grupo | Filas | Coincide | Cerca | No coincide | Error absoluto típico (mediana) |
|---|---|---|---|---|---|
| Tabla 3 (PSNR) | 40 | 2 | 12 | 26 | 0.71 dB |
| Tabla 4 (SSIM) | 40 | 1 | 6 | 33 | 0.019 |
| Tabla 5 (CoC) | 40 | 0 | 30 | 10 | 0.007 |
| Tabla 6 (EPI) | 40 | 0 | 0 | 40 | 1.39 |
| Tabla 7 (s&p 0.005, validación) | 20 | 0 | 7 | 13 | — |
| Tabla 8 (s&p 0.5, validación) | 20 | 0 | 1 | 19 | — |

Por filtro (las 40 filas de cada uno, las 4 métricas):

| Filtro | Coincide | Cerca | No coincide |
|---|---|---|---|
| Mediana | 0 | 16 | 24 |
| Wiener | 1 | 12 | 27 |
| Media | 1 | 6 | 33 |
| Gaussiano | 0 | 10 | 30 |
| Mediana ponderada | 1 | 12 | 27 |

Por ruido: sal y pimienta, 3 / 37 / 80 (de 120); gaussiano, 0 / 19 / 61 (de 80). El detalle por filtro y ruido, por nivel y por métrica está en `resultados/resumen.csv`.

Lectura:

- **PSNR, SSIM y CoC en calibración** quedan cerca aunque no entren en la categoría: tres cuartos de las filas de PSNR tienen un error menor que 1.04 dB, y tres cuartos de las de SSIM, menor que 0.031.
- **Hay sesgos sistemáticos**, que indican que algo sigue sin identificar: el SSIM de la mediana y de la media queda entre 0.01 y 0.03 por encima del paper en todas las filas; el PSNR con ruido gaussiano 0.1 y 0.2 queda entre 0.6 y 1.0 dB por debajo en cuatro de los cinco filtros (y más en la mediana ponderada).
- **La validación sale peor que la calibración** (mediana normalizada 1.81 contra 0.144). La Tabla 8 es la causa principal; ver 4.3.

## 4. Filas que no coinciden: hipótesis probadas

### 4.1. EPI (Tablas 6, 7 y 8: 50 filas)

- **Qué pasa.** El paper da EPI entre 0.41 y 0.98, siempre menor que 1. Nuestros valores van de 0.48 a 36.
- **Por qué.** La imagen limpia es muy suave, así que cualquier resto de ruido multiplica las diferencias horizontales (con ruido gaussiano: 1.7 a 10.8), y un filtro que suaviza las baja demasiado (mediana 5×5 con sal y pimienta 0.001: 0.48, contra 0.976 del paper).
- **Hipótesis probadas** (todas las de la grilla):
  - dominio RGB, por canal o gris: mediana de las filas propias 8.77, 9.43 y 6.78;
  - sin CLAHE (solo filtro): la mediana da 0.42 con sal y pimienta 0.001 y 1.92 con gaussiano 0.001;
  - mediana 3×3 en lugar de 5×5: 0.62 y 4.33;
  - llamada gaussiana con varianza = *v*: tampoco (2.82 con la mediana ponderada, medido en la etapa 1);
  - solo CLAHE sobre la limpia: 1.13.
- **Conclusión.** Ninguna hipótesis con fuente de la grilla explica la Tabla 6. La Ec. 9 tal como está escrita no puede dar valores siempre menores que 1 con estas imágenes.

**Prueba adicional, fuera de la grilla (pedida después de la primera versión de este informe): EPI de Sattar et al. (1997).**

- **Definición.** Correlación entre la imagen limpia y la procesada después de pasarlas por un laplaciano 3×3; por construcción no supera 1. No pude consultar el artículo original: la fórmula sale de fuentes secundarias que lo citan (https://www.mathworks.com/matlabcentral/fileexchange/65245-edge-preservation, https://arxiv.org/pdf/2507.15306).
- **Qué se probó** (`investigacion/epi_sattar.py`, `resultados/epi_sattar.csv`): 3 aproximaciones del laplaciano (4 vecinos, 8 vecinos y la de `fspecial('laplacian')`), 3 dominios (RGB, por canal, gris), con y sin CLAHE; 18 variantes sobre las 50 filas, con la configuración propuesta y semilla 0.
- **Resultado.** Tampoco reproduce la Tabla 6. La mejor variante (8 vecinos, gris) tiene un error absoluto mediano de 0.66 (0.64 sin CLAHE) y una mediana normalizada de 4.15, contra 1.39 y 8.77 de la Ec. 9. En esa variante los valores van de 0.02 a 0.72, mientras que el paper da de 0.41 a 0.98. Ejemplos: mediana con sal y pimienta 0.001, 0.66 (paper 0.976); mediana con gaussiano 0.001, 0.06 (paper 0.919); Wiener con sal y pimienta 0.2, 0.03 (paper 0.583).
- **Lectura.** Queda siempre por debajo de 1, como en el paper, pero muy por debajo: apenas queda algo de ruido, la correlación entre laplacianos cae a casi 0. El paper, en cambio, informa EPI de 0.8 a 0.9 con ruido gaussiano. **Se descarta**; la configuración propuesta no cambia.

### 4.2. Mediana ponderada con ruido gaussiano (8 filas de PSNR y SSIM)

- **Qué pasa.** PSNR 27.3 / 26.4 / 17.6 / 12.6 contra 29.7 / 29.3 / 19.8 / 14.4 del paper; SSIM alrededor de 0.13 por debajo.
- **Hipótesis probadas:** c = 1 a 5 (el mejor es c = 1, y sigue 2.4 dB abajo; con c = 4 da 25.4 dB), las dos variantes literales (25.7 dB la saturada) y las dos llamadas de ruido.
- **Problema de fondo.** En el paper la mediana ponderada supera a la mediana en todas las columnas, también con ruido gaussiano. Con una ventana 3×3 eso no es posible: más peso en el centro siempre filtra menos. Haría falta una mediana ponderada con una ventana más grande, que no tiene fuente (File Exchange #29386 es 3×3).
- **Conclusión.** Ninguna hipótesis con fuente lo explica. El c = 1 elegido es solo "el menos malo" y convierte el filtro en una mediana 3×3 común.

### 4.3. Tabla 8 (sal y pimienta 0.5) y mediana con mucho ruido

- **Qué pasa.** Con la configuración propuesta, la mediana 5×5 da 27.8 dB (paper: 33.8) y la mediana ponderada, 17.2 dB (paper: 34.8). Wiener, media y gaussiano quedan entre 4.8 y 8.3 dB abajo.
- **Hallazgo.** La **mediana 7×7** (`symmetric`) da **33.82 dB** con d = 0.5 (paper: 33.78), 35.13 dB con d = 0.2 (paper: 34.64) y 35.57 dB con d = 0.001 (paper: 35.38). Es decir, reproduce toda la columna de sal y pimienta de la mediana, incluida la Tabla 8. La calibración no la eligió porque con ruido gaussiano se pasa (30.8 dB contra 29.2 del paper) y su mediana de filas propias es 0.255 contra 0.121 de la 5×5.
- **No lo usé para elegir**, porque la Tabla 8 es de validación. Lo dejo como pregunta (sección 7).
- **Sin explicación:** la mediana ponderada a 34.8 dB con d = 0.5 (en 3×3 da 17.2 dB con c = 1 y 13.9 dB con c = 4) y los 20.5 a 21.0 dB de Wiener, media y gaussiano.

### 4.4. Filtro gaussiano con sal y pimienta 0.2 (3 filas)

- **Qué pasa.** PSNR 16.9 contra 21.0; SSIM 0.108 contra 0.237; CoC 0.763 contra 0.877.
- **Observación.** En el paper, esta fila es casi idéntica a la de d = 0.1 (21.006 contra 21.037; 0.2371 contra 0.2378; 0.8765 contra 0.8783), mientras que en los demás filtros lineales el PSNR baja entre 2.3 y 3.4 dB al pasar de 0.1 a 0.2. La fila de d = 0.1 queda a 0.9 dB (20.1 contra 21.0).
- **Conclusión.** Ningún candidato lo explica. Es compatible con un error en la tabla del paper, pero no lo puedo verificar.

### 4.5. Fila CoC de la Tabla 7

Tres valores de la Tabla 7 son idénticos a los de la Tabla 5 con d = 0.001 (gaussiano 0.994516, Wiener 0.992425, media 0.991667), y el de la mediana ponderada (0.995936) es el de la *mediana* con d = 0.001. Parecen copiados dentro del paper. Aun así, las cinco filas quedan "cerca" (error ≤ 0.006).

### 4.6. Sesgos chicos que quedan

- **SSIM de mediana y media:** +0.011 a +0.034 en todas las filas. Probé las cinco variantes de SSIM; la elegida es la mejor.
- **PSNR con gaussiano 0.1 y 0.2:** −0.6 a −1.0 dB en mediana, Wiener, media y gaussiano. Con la media en 0.2, el PSNR queda dominado por el corrimiento de brillo, así que ningún filtro lo cambia. Con pico 255 en lugar de 222 esas filas subirían 1.2 dB y quedarían cerca, pero empeorarían las de poco ruido; ningún candidato cierra las dos a la vez.

## 5. Sensibilidad

Mediana y promedio del error normalizado con la configuración propuesta y semilla 0 (`sensibilidad_clip.csv`, `sensibilidad_imagen.csv`).

**Clip limit** (análisis, sin proponer cambiarlo):

| Clip limit | Calibración: mediana | Calibración: promedio | Validación: mediana |
|---|---|---|---|
| 0.0005 (Fig. 5) | 0.144 | 5.23 | 1.82 |
| 0.001 | 0.235 | 5.79 | 2.19 |
| 0.005 | 1.082 | 9.58 | 4.95 |
| 0.01 (por defecto en MATLAB) | 1.796 | 13.88 | 6.64 |

El error crece con el clip limit: 0.0005 es el que mejor reproduce las tablas, lo que apoya que los experimentos usaron el valor de la Fig. 5.

**Imagen:**

| Imagen | Calibración: mediana | Calibración: promedio | Validación: mediana |
|---|---|---|---|
| im0001 | 0.144 | 5.23 | 1.82 |
| im0002 | 0.253 | 4.41 | 1.62 |
| im0003 | 0.273 | 3.36 | 1.21 |
| im0004 | 0.337 | 5.17 | 1.90 |
| im0005 | 0.177 | 3.54 | 1.46 |
| im0006 | 0.268 | 5.63 | 2.16 |

`im0001` da la menor mediana de calibración, coherente con la sección 1.1. La configuración se calibró con esa imagen, así que esto no es una prueba independiente.

**Control cualitativo de CLAHE** (`control_clahe.csv`): con sal y pimienta 0.01 y mediana, la media del canal B pasa de 13.6 (ruidosa) a 14.8 con el CLAHE reconstruido y a 22.1 con OpenCV. En la Fig. 8 el canal B realzado sigue entre 0 y unos 40, como el ruidoso, y en la Fig. 5 el histograma casi no cambia: las dos figuras son compatibles con el reconstruido y no con OpenCV.

## 6. Cambios que propondría al proyecto

> **Estado: aplicados** (después de escribir este informe y con aprobación). Se aplicaron los cambios de CLAHE, ruido gaussiano, los cuatro filtros nuevos (mediana 5×5, media 5×5, gaussiano 3×3 con σ = 0.5 y Wiener 5×5) y PSNR y SSIM en gris con pico por rango. No se cambió la mediana ponderada (sigue con peso central 4), ni el CoC, ni el EPI. Por eso, los valores de la mediana ponderada que da el programa no son los de la tabla de la sección 2, que usa peso central 1; los de peso 4 están en `resultados/diagnosticos.csv`. El argumento de consola sigue llamándose `VARIANZA` y la carpeta de resultados sigue usando `var-<valor>`. Para que la búsqueda siga dando lo mismo, `investigacion/` ya no usa el ruido ni el CLAHE del proyecto: tiene su propia copia de las dos llamadas de ruido y del CLAHE de OpenCV.

Separo los cambios por la fuerza de la evidencia. Recomiendo aplicar los del primer grupo y decidir los otros con las preguntas de la sección 7.

**Con evidencia fuerte**

| Archivo | Cambio |
|---|---|
| `procesamiento/realce.py` | Reemplazar `cv2.createCLAHE` por el `adapthisteq` de `investigacion/clahe_matlab.py`. Desaparecen `clip_limit_matlab_a_opencv` y la conversión a la escala de OpenCV; quedan `pixeles_por_tesela` (76 × 88 = 6688) y el clip en cuentas (30). |
| `procesamiento/ruido.py` | Ruido gaussiano como `imnoise(I,'gaussian',v)`: media *v*, varianza 0.01. |
| `procesamiento/filtros.py` | Agregar al registro `FILTROS`: `mediana`, `media`, `gaussiano` y `wiener`, con el código de `investigacion/filtros_candidatos.py`. Gaussiano con σ = 0.5; Wiener 5×5. |
| `procesamiento/metricas.py` | SSIM sobre `rgb2gray`. |

**Con evidencia moderada o débil**

| Archivo | Cambio |
|---|---|
| `procesamiento/metricas.py` | PSNR sobre `rgb2gray` con pico = máx − mín de la limpia. |
| `procesamiento/filtros.py` | Mediana y media 5×5; mediana ponderada con c = 1 (yo dejaría c = 4: ver pregunta 2). |

**En todos los casos**

- `procesamiento/parametros.py`: constantes nuevas con su etiqueta (varianza 0.01 del ruido gaussiano, coeficientes de `rgb2gray`, σ y tamaños de los filtros) y la tabla de procedencia nueva.
- `procesamiento/salida.py`: `parametros.json` deja de escribir siempre los datos de la mediana ponderada y escribe los del filtro usado; cambia el bloque de CLAHE.
- `sonali.py`: no cambia la estructura; cambia la ayuda de `VARIANZA`.
- `README.md`: tabla de procedencia nueva; reemplazar la limitación "los resultados no se pueden comparar con las tablas del paper" por un enlace a este informe y la lista de lo que no se reproduce.
- `requirements.txt`: sin cambios.

**Interpretación de `VARIANZA` en la consola (gana la hipótesis 1.7)**

- En `gaussiano`, el número deja de ser la varianza: pasa a ser la **media** del ruido (un corrimiento de brillo, con las intensidades en [0, 1]); la varianza queda fija en 0.01.
- En `sal_pimienta` sigue siendo la densidad *d*.
- La validación `VARIANZA > 0` se mantiene. Convendría renombrar el argumento a `NIVEL` y explicar en la ayuda las dos interpretaciones, aclarando que el paper lo llama "noise variance".
- El nombre de la carpeta de resultados usa `var-<valor>`; habría que decidir si se cambia a `nivel-<valor>`.

**Tabla de procedencia nueva** (filas que cambian o se agregan)

| Parámetro | Valor | Etiqueta | Fuente |
|---|---|---|---|
| Imagen de las tablas | `STARE/im0001.ppm` | `[VERIFICADO]` | Correlación con las Figs. 1 y 7 |
| Ruido gaussiano de nivel *v* | media *v*, varianza 0.01 | `[AJUSTE]` | imnoise; informe, sección 1 |
| CLAHE | `adapthisteq` reconstruido | `[MATLAB]` | adapthisteq; ejemplo HDL |
| Reparto del exceso recortado | ningún bin supera el clip | `[MATLAB]` | ejemplo HDL |
| Relleno de CLAHE | simétrico, hasta teselas pares | `[MATLAB]` | ejemplo HDL (lado del relleno: no verificado) |
| Mediana | 5×5, ceros | `[AJUSTE]` | medfilt2 |
| Media | `fspecial('average', 5)` + `imfilter` | `[AJUSTE]` | fspecial; imfilter |
| Gaussiano | `fspecial('gaussian', 3, 0.5)` + `imfilter` | `[AJUSTE]` + `[MATLAB]` | fspecial |
| Wiener | `wiener2` 5×5, ruido estimado | `[AJUSTE]` | wiener2 |
| Peso central de la mediana ponderada | 1 (o 4, según la pregunta 2) | `[AJUSTE]` o `[FILE EXCHANGE]` | #29386 |
| Dominio de PSNR y SSIM | `rgb2gray` | `[AJUSTE]` | rgb2gray |
| Pico del PSNR | máx − mín de la limpia | `[AJUSTE]` | Texto de la Ec. 2 |
| CoC | sin cambios | `[PAPER]` + `[DECISIÓN]` | Ec. 8 |
| EPI | sin cambios; no reproduce la Tabla 6 | `[PAPER]` + `[DECISIÓN]` | Ec. 9 |

Desaparecen las filas "Conversión del clip limit" y `[OPENCV]` de CLAHE.

**Pruebas que habría que agregar**

- CLAHE: ningún bin supera el clip y se conservan las cuentas; para 605×700, tesela 76×88, P = 6688 y 30 cuentas; techo sobre `im0001` (36.92 dB) como prueba de regresión si la imagen está disponible.
- Ruido: con `gaussiano 0.2` sobre gris 100, la media sube 0.2 y la varianza da 0.01.
- Filtros nuevos: cada uno contra una versión con bucles; media con relleno de ceros a mano; núcleo `fspecial('gaussian', 3, 0.5)` (centro 0.6193); Wiener sobre una imagen constante.
- Métricas: `rgb2gray` a mano; PSNR con pico por rango; métricas idénticas a las actuales cuando los tres canales son iguales.
- Las de `investigacion/tests/` cubren casi todo esto y se pueden mover.

## 7. Preguntas que quedan por decidir

1. **σ del filtro gaussiano.** La Tabla 1 del paper dice "default 1" y el prompt lo fija como primer candidato, pero σ = 0.5 (el valor por defecto de `fspecial`) reproduce las filas del filtro gaussiano mucho mejor (0.133 contra 1.14). ¿Se acepta 0.5 aunque contradiga el texto del paper?
2. **Mediana ponderada.** La búsqueda eligió c = 1, que la vuelve idéntica a la mediana 3×3, y aun así no reproduce las filas gaussianas. ¿Se cambia a c = 1 o se mantiene c = 4 de File Exchange y se documenta que esas filas no cierran? Yo mantendría c = 4.
3. **Ventana de la mediana.** La calibración eligió 5×5 con evidencia débil, pero la 7×7 reproduce la Tabla 8 (33.82 contra 33.78 dB). ¿Se permite usar la validación para decidir este caso?
4. **CoC en gris.** Sobre sus filas propias el gris es algo mejor (0.142 contra 0.217) y sería coherente con PSNR y SSIM en gris (`corr2` es una función 2-D). ¿Se unifica todo en gris?
5. **EPI.** Probé también la definición de Sattar et al. (1997) y tampoco reproduce la Tabla 6 (sección 4.1). No me queda ningún candidato con fuente. ¿Se deja el EPI de la Ec. 9 y se documenta que la Tabla 6 no es reproducible?
6. **SSIM de Wang con submuestreo: candidato nuevo.** La versión posterior de `ssim.m` de Wang submuestrea la imagen antes de calcular. No está en la grilla y no la probé. Podría explicar el sesgo de +0.01 a +0.03 en el SSIM. ¿La agrego?
7. **Mediana ponderada con ventana mayor que 3×3.** Es lo único que podría explicar que supere a la mediana con ruido gaussiano y con d = 0.5, pero no tiene fuente citable. ¿Se deja documentado como no reproducible?
8. **Código fuente de `adapthisteq.m`.** No encontré una copia pública. Si tenés acceso a MATLAB, alcanzaría con comparar `adapthisteq(I,'ClipLimit',0.0005)` de un canal de `im0001` contra `clahe_matlab.py` para cerrar los cuatro detalles marcados como no verificados.
