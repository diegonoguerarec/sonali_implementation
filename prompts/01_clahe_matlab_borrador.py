"""BORRADOR NO VERIFICADO: reconstrucción de adapthisteq (MATLAB R2016a), solo para investigación.

Sigue el algoritmo de adapthisteq.m (Zuiderveld), tal como lo describe el ejemplo HDL de MathWorks:
padding 'symmetric' hasta teselas pares, clip = ceil(P/256) + round(c*(P - ceil(P/256))),
reparto del exceso SIN superar el clip limit, LUT = round(255*cdf/P), interpolación bilineal
entre teselas con regiones de media tesela en los bordes.
"""
import math
import numpy as np


def _clip_histograma(h, clip, nbins=256):
    h = h.astype(np.int64).copy()
    exceso = int(np.maximum(h - clip, 0).sum())
    inc = exceso // nbins
    limite_sup = clip - inc
    for k in range(nbins):
        if h[k] > clip:
            h[k] = clip
        elif h[k] > limite_sup:
            exceso -= clip - h[k]
            h[k] = clip
        else:
            exceso -= inc
            h[k] += inc
    k = 0
    while exceso > 0:
        paso = max(nbins // exceso, 1)
        for m in range(k, nbins, paso):
            if h[m] < clip:
                h[m] += 1
                exceso -= 1
                if exceso == 0:
                    break
        k = (k + 1) % nbins
    return h


def adapthisteq(canal, c=0.0005, tiles=(8, 8), nbins=256):
    alto, ancho = canal.shape
    dims, pads = [], []
    for n, t in zip((alto, ancho), tiles):
        dim = n // t if n % t == 0 else n // t + 1
        pad = dim * t - n
        if dim % 2:
            pad += t
            dim += 1
        dims.append(dim); pads.append(pad)
    pre = [p_ // 2 for p_ in pads]; post = [math.ceil(p_ / 2) for p_ in pads]
    img = np.pad(canal, ((pre[0], post[0]), (pre[1], post[1])), mode="symmetric")
    P = dims[0] * dims[1]
    min_clip = math.ceil(P / nbins)
    clip = min_clip + math.floor(c * (P - min_clip) + 0.5)
    luts = np.empty(tiles + (nbins,), dtype=np.float64)
    for i in range(tiles[0]):
        for j in range(tiles[1]):
            t = img[i*dims[0]:(i+1)*dims[0], j*dims[1]:(j+1)*dims[1]]
            h = _clip_histograma(np.bincount(t.ravel(), minlength=nbins), clip, nbins)
            luts[i, j] = np.floor(255 * np.minimum(np.cumsum(h) / P, 1) + 0.5)
    salida = np.zeros(img.shape, dtype=np.float64)
    def regiones(n_t, dim):
        r, pos = [], 0
        for k in range(n_t + 1):
            if k == 0: n, m = dim // 2, (0, 0)
            elif k == n_t: n, m = dim // 2, (n_t - 1, n_t - 1)
            else: n, m = dim, (k - 1, k)
            r.append((pos, n, m)); pos += n
        return r
    for (f0, nf, mf) in regiones(tiles[0], dims[0]):
        for (c0, nc, mc) in regiones(tiles[1], dims[1]):
            sub = img[f0:f0+nf, c0:c0+nc]
            rw = np.arange(nf)[:, None]; rrw = np.arange(nf, 0, -1)[:, None]
            cw = np.arange(nc)[None, :]; crw = np.arange(nc, 0, -1)[None, :]
            ul, ur = luts[mf[0], mc[0]][sub], luts[mf[0], mc[1]][sub]
            bl, br = luts[mf[1], mc[0]][sub], luts[mf[1], mc[1]][sub]
            salida[f0:f0+nf, c0:c0+nc] = (rrw*(crw*ul + cw*ur) + rw*(crw*bl + cw*br)) / (nf*nc)
    salida = np.floor(salida + 0.5).astype(np.uint8)  # asignación a uint8 de MATLAB (redondeo)
    return salida[pre[0]:pre[0]+alto, pre[1]:pre[1]+ancho], clip, P
