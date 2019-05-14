from cuviz.colors import build_shades

from PIL.Image import fromarray
from numba import cuda
import numpy as np
from math import log1p

@cuda.jit('void(float64[:], float64)')
def add_k(inp, k):
    t = cuda.threadIdx.x
    b = cuda.blockIdx.x
    bw = cuda.blockDim.x
    i = b * bw + t

    if i < inp.size:
        inp[i] += k


@cuda.jit('void(float64[:])')
def log_k(inp):
    t = cuda.threadIdx.x
    b = cuda.blockIdx.x
    bw = cuda.blockDim.x
    i = b * bw + t

    if i < inp.size:
        val = inp[i]
        if val > 0.0:
            inp[i] = log1p(val)


def log(agg):
    n_points = agg.shape[0]
    bpg = int(n_points / 512) + 1
    log_k[bpg, 512](agg)
    return agg


@cuda.jit('void(float64[:])')
def cbrt_k(inp):
    t = cuda.threadIdx.x
    b = cuda.blockIdx.x
    bw = cuda.blockDim.x
    i = b * bw + t

    cbrt = 1.0 / 3.0
    if i < inp.size:
        val = inp[i]
        if val > 0.0:
            inp[i] = val ** cbrt


def cbrt(agg):
    n_points = agg.shape[0]
    bpg = int(n_points / 512) + 1
    cbrt_k[bpg, 512](agg)
    return agg


def linear(agg):
    return agg


@cuda.jit('void(float64[:], int32[:])')
def bincount_k(inp, hist):
    t = cuda.threadIdx.x
    b = cuda.blockIdx.x
    bw = cuda.blockDim.x
    i = b * bw + t

    if i < inp.size:
        val = inp[i]
        if val > 0.0:
            cuda.atomic.add(hist, int(val), 1)


@cuda.jit('void(float64[:], float64[:])')
def eq_hist_k(inp, cdf):
    t = cuda.threadIdx.x
    b = cuda.blockIdx.x
    bw = cuda.blockDim.x
    i = b * bw + t

    if i < inp.size:
        val = inp[i]
        if val > 0.0:
            s = int(inp[i]) # start
            e = int(inp[i]) + 1 # end
            v1 = cdf[s]
            v2 = cdf[e]
            diffv = v2 - v1
            prop = float(val - s) / 1.0
            inp[i] = v1 + prop * diffv


def eq_hist(agg):
    n_points = agg.shape[0]
    bpg = int(n_points / 512) + 1

    h_agg = agg.copy_to_host()
    max = int(h_agg.max())

    hist = cuda.device_array(max, dtype=np.int32)
    bincount_k[bpg, 512](agg, hist)
    h_hist = hist.copy_to_host()
    h_cdf = h_hist.cumsum()
    h_cdf = h_cdf / float(h_cdf[-1])
    cdf = cuda.to_device(h_cdf)
    eq_hist_k[bpg, 512](agg, cdf)

    return agg


@cuda.jit('void(float64[:], float64[:], uint8[:], uint8[:])')
def interpolate_k(inp, span, rgb, out):
    t = cuda.threadIdx.x
    b = cuda.blockIdx.x
    bw = cuda.blockDim.x
    i = b * bw + t

    if i < inp.size:
        val = inp[i]
        out_idx = i * 4

        if val < 0.0:
            out[out_idx + 3] = 0
            return

        s = span.size - 2 # start
        e = span.size - 1 # end
        for j in range(1, span.size - 1):
            if val <= span[j]:
                s = j-1
                e = j
                break
        
        r1 = rgb[s*3]
        g1 = rgb[s*3 + 1]
        b1 = rgb[s*3 + 2]
        r2 = rgb[e*3]
        g2 = rgb[e*3 + 1]
        b2 = rgb[e*3 + 2]

        prop = float(val - span[s]) / float(span[e] - span[s])

        diffr = r2 - r1
        diffg = g2 - g1
        diffb = b2 - b1

        out[out_idx] = r1 + prop * diffr
        out[out_idx + 1] = g1 + prop * diffg
        out[out_idx + 2] = b1 + prop * diffb
        out[out_idx + 3] = 255


_how_lookup = {'log': log, 'cbrt': cbrt, 'linear': linear, 'eq_hist': eq_hist}


def shade(agg, cmap, how):
    n_points = agg.agg.shape[0]
    bpg = int(n_points / 512) + 1

    h_agg = agg.agg.copy_to_host()
    h_agg = h_agg[h_agg>0.0]
    min = h_agg.min()
    
    add_k[bpg, 512](agg.agg, -min)

    how_func = _how_lookup[how]
    how_func(agg.agg)

    h_agg = agg.agg.copy_to_host()
    h_agg = h_agg[h_agg>0.0]
    max = h_agg.max()

    span = cuda.to_device(np.linspace(0.0, max, len(cmap)))
    shades = build_shades(cmap)
    img = cuda.device_array(agg.width * agg.height * 4, dtype=np.uint8)

    interpolate_k[bpg, 512](agg.agg, span, shades, img)

    img = img.reshape((agg.height, agg.width, 4), order='C').copy_to_host()
    return Image(img)


class Image():
    border=1

    def __init__(self, img):
        self.img = img
    
    def to_pil(self):
        return fromarray(self.img, 'RGBA')

    def to_bytesio(self, format='png'):
        fp = BytesIO()
        self.to_pil().save(fp, format)
        fp.seek(0)
        return fp

    def _repr_png_(self):
        """Supports rich PNG display in a Jupyter notebook"""
        return self.to_pil()._repr_png_()

    def _repr_html_(self):
        """Supports rich HTML display in a Jupyter notebook"""
        # imported here to avoid depending on these packages unless actually used
        from io import BytesIO
        from base64 import b64encode

        b = BytesIO()
        self.to_pil().save(b, format='png')

        h = """<img style="margin: auto; border:""" + str(self.border) + """px solid" """ + \
            """src='data:image/png;base64,{0}'/>""".\
                format(b64encode(b.getvalue()).decode('utf-8'))
        return h