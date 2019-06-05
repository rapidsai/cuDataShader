from cuviz.colors import build_shades

from PIL.Image import fromarray
from numba import cuda
import numpy as np
from math import log1p


maxThreadsPerBlock = cuda.get_current_device().MAX_THREADS_PER_BLOCK


@cuda.jit('void(float64[:], float64)')
def substract_k(input, to_add):
    i = cuda.grid(1)
    if i < input.size:
        input[i] -= to_add


@cuda.jit('void(float64[:])')
def log_k(input):
    i = cuda.grid(1)
    if i < input.size:
        val = input[i]
        if val > 0.0:
            input[i] = log1p(val)

def log(input, bpg, tpb):
    log_k[bpg, tpb](input)


@cuda.jit('void(float64[:])')
def cbrt_k(input):
    i = cuda.grid(1)
    if i < input.size:
        val = input[i]
        if val > 0.0:
            input[i] = val ** (1.0 / 3.0)

def cbrt(input, bpg, tpb):
    log_k[bpg, tpb](input)


def linear(input, bpg, tpb):
    pass


@cuda.jit('void(float64[:], int32[:])')
def bincount_k(input, hist):
    i = cuda.grid(1)
    if i < input.size:
        val = input[i]
        if val > 0.0:
            cuda.atomic.add(hist, int(val), 1)


@cuda.jit('void(float64[:], float64[:])')
def eq_hist_k(input, cdf):
    i = cuda.grid(1)
    if i < input.size:
        val = input[i]
        if val > 0.0:
            s = int(input[i]) # start
            e = int(input[i]) + 1 # end
            v1 = cdf[s]
            v2 = cdf[e]
            diffv = v2 - v1
            prop = float(val - s) / 1.0
            input[i] = v1 + prop * diffv

def eq_hist(input, bpg, tpb):
    min, max = get_min_max(input)
    hist = cuda.device_array(int(max), dtype=np.int32)
    bincount_k[bpg, tpb](input, hist)
    h_hist = hist.copy_to_host()
    h_cdf = np.nancumsum(h_hist) # overloaded by Numba
    h_cdf = h_cdf / float(h_cdf[-1])
    cdf = cuda.to_device(h_cdf)
    eq_hist_k[bpg, tpb](input, cdf)


@cuda.jit('void(float64[:,:], float64[:], uint8[:,:], uint8[:,:,:])')
def interpolate_k(input, span, rgb, out):
    x, y = cuda.grid(2)
    N, M = input.shape
    if x >= 0 and x < N and y >= 0 and y < M:
        val = input[x, y]

        if val < 0.0:
            out[x, y, 3] = 0
            return

        s = span.size - 2 # start
        e = span.size - 1 # end
        for j in range(1, span.size - 1):
            if val <= span[j]:
                s = j-1
                e = j
                break
        
        r1 = float(rgb[s, 0])
        g1 = float(rgb[s, 1])
        b1 = float(rgb[s, 2])
        r2 = float(rgb[e, 0])
        g2 = float(rgb[e, 1])
        b2 = float(rgb[e, 2])

        prop = (val - span[s]) / (span[e] - span[s])

        diffr = r2 - r1
        diffg = g2 - g1
        diffb = b2 - b1

        out[x, y, 0] = r1 + prop * diffr
        out[x, y, 1] = g1 + prop * diffg
        out[x, y, 2] = b1 + prop * diffb
        out[x, y, 3] = 255


_how_lookup = {'log': log, 'cbrt': cbrt, 'linear': linear, 'eq_hist': eq_hist}


@cuda.jit('void(float64[:], float64[:])')
def min_max(val_array, min_max_array):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    local_min = min_max_array[0]
    local_max = min_max_array[1]

    for i in range(start, val_array.size, stride):
        element = val_array[i]
        if element > 0.0:
            local_min = min(element, local_min)
        local_max = max(element, local_max)

    cuda.atomic.min(min_max_array, 0, local_min)
    cuda.atomic.max(min_max_array, 1, local_max)


def get_min_max(val_array):
    min_max_array_gpu = cuda.to_device(np.array([np.finfo(np.float64).max, np.finfo(np.float64).min], dtype=np.float64))
    min_max_bpg = int((val_array.size / maxThreadsPerBlock) / 16.0) + 1
    min_max[min_max_bpg, maxThreadsPerBlock](val_array, min_max_array_gpu)
    min, max = min_max_array_gpu.copy_to_host()
    return min, max


def shade(agg, cmap=["lightblue", "darkblue"], how='eq_hist'):
    agg_copy = agg.ravel(order='C')

    bpg = int(agg_copy.shape[0] / maxThreadsPerBlock) + 1

    min, max = get_min_max(agg_copy)
    
    if min != max:
        substract_k[bpg, maxThreadsPerBlock](agg_copy, min)
    else:
        substract_k[bpg, maxThreadsPerBlock](agg_copy, min - 0.0001) # yes, that's a hack ^^

    how_func = _how_lookup[how]
    how_func(agg_copy, bpg, maxThreadsPerBlock)

    min, max = get_min_max(agg_copy)

    agg_copy = agg_copy.reshape(agg.shape, order='C')

    span = cuda.to_device(np.linspace(0.0, max, len(cmap)))
    shades = build_shades(cmap)
    
    blockDim = int(np.sqrt(maxThreadsPerBlock))
    tpb = (blockDim, blockDim)
    height, width = agg_copy.shape
    bpg = (int(height / blockDim) + 1, int(width / blockDim) + 1)
    img = cuda.device_array((height, width, 4), dtype=np.uint8)
    interpolate_k[bpg, tpb](agg_copy, span, shades, img)
    
    return Image(img.copy_to_host())


@cuda.jit('void(uint8[:,:,:])')
def memset_k(input):
    x, y = cuda.grid(2)
    N, M, channels = input.shape
    if x >= 0 and x < N and y >= 0 and y < M:
        input[x, y, 0] = 0
        input[x, y, 1] = 0
        input[x, y, 2] = 0
        input[x, y, 3] = 0


def memset(input):
    blockDim = int(np.sqrt(maxThreadsPerBlock))
    tpb = (blockDim, blockDim)
    bpg = (int(input.shape[0] / blockDim) + 1, int(input.shape[1] / blockDim) + 1)
    memset_k[bpg, tpb](input)


def _square_mask(px):
    """Produce a square mask with sides of length ``2 * px + 1``"""
    px = int(px)
    w = 2 * px + 1
    x_bool = np.ones((w, w), dtype='bool')
    return cuda.to_device(x_bool)


def _circle_mask(r):
    """Produce a circular mask with a diameter of ``2 * r + 1``"""
    x = np.arange(-r, r + 1, dtype='i4')
    x_bool = np.where(np.sqrt(x**2 + x[:, None]**2) <= r+0.5, True, False)
    return cuda.to_device(x_bool)


_mask_lookup = {'square': _square_mask, 'circle': _circle_mask}


@cuda.jit('void(float64, float64, float64, float64, float64, float64, float64, float64, uint8[:])', device=True, inline=True)
def over_k(sr, sg, sb, sa, dr, dg, db, da, dst):
    factor = 1.0 - sa
    a = sa + da * factor
    if a == 0.0:
        dst[3] = 0
    else:
        dst[0] = ((sr * sa + dr * da * factor) / a) * 255.0
        dst[1] = ((sg * sa + dg * da * factor) / a) * 255.0
        dst[2] = ((sb * sa + db * da * factor) / a) * 255.0
        dst[3] = a * 255.0


@cuda.jit('void(float64, float64, float64, float64, float64, float64, float64, float64, uint8[:])', device=True, inline=True)
def add_k(sr, sg, sb, sa, dr, dg, db, da, dst):
    a = min(1.0, sa + da)
    if a == 0.0:
        dst[3] = 0
    else:
        dst[0] = ((sr * sa + dr * da) / a) * 255.0
        dst[1] = ((sg * sa + dg * da) / a) * 255.0
        dst[2] = ((sb * sa + db * da) / a) * 255.0
        dst[3] = a * 255.0


@cuda.jit('void(float64, float64, float64, float64, float64, float64, float64, float64, uint8[:])', device=True, inline=True)
def saturate_k(sr, sg, sb, sa, dr, dg, db, da, dst):
    a = min(1.0, sa + da)
    if a == 0.0:
        dst[3] = 0
    else:
        factor = min(sa, 1.0 - da)
        dst[0] = ((factor * sr + dr * da) / a) * 255.0
        dst[1] = ((factor * sg + dg * da) / a) * 255.0
        dst[2] = ((factor * sb + db * da) / a) * 255.0
        dst[3] = a * 255.0


@cuda.jit('void(float64, float64, float64, float64, uint8[:])', device=True, inline=True)
def source_k(sr, sg, sb, sa, dst):
    if sa > 0:
        dst[0] = sr * 255.0
        dst[1] = sg * 255.0
        dst[2] = sb * 255.0
        dst[3] = sa * 255.0


@cuda.jit('void(float64, float64, float64, float64, uint8[:], uint8)', device=True, inline=True)
def composite_k(sr, sg, sb, sa, dst, composite_type):
    if composite_type == 3:
        source_k(sr, sg, sb, sa, dst)
    else:
        dr = float(dst[0]) / 255.0
        dg = float(dst[1]) / 255.0
        db = float(dst[2]) / 255.0
        da = float(dst[3]) / 255.0
        if composite_type == 0:
            over_k(sr, sg, sb, sa, dr, dg, db, da, dst)
        elif composite_type == 1:
            add_k(sr, sg, sb, sa, dr, dg, db, da, dst)
        else:
            saturate_k(sr, sg, sb, sa, dr, dg, db, da, dst)


@cuda.jit('void(uint8[:,:,:], bool_[:,:], uint8[:,:,:], uint8)')
def spreading_k(img, mask, dest, composite_type):
    x, y = cuda.grid(2)
    N, M, channels = img.shape
    if x >= 0 and x < N and y >= 0 and y < M:
        a = float(img[x, y, 3]) / 255.0
        if a > 0.0:
            r = float(img[x, y, 0]) / 255.0
            g = float(img[x, y, 1]) / 255.0
            b = float(img[x, y, 2]) / 255.0

            h, w = mask.shape
            x -= int((h-1) / 2)
            y -= int((w-1) / 2)
            for mask_x in range(h):
                for mask_y in range(w):
                    if mask[mask_x, mask_y]:
                        nx = int(x + mask_x)
                        ny = int(y + mask_y)
                        if nx >= 0 and nx < N and ny >= 0 and ny < M:
                            composite_k(r, g, b, a, dest[nx, ny], composite_type)


_composite_op_lookup = { 'over': 0, 'add': 1, 'saturate': 2, 'source': 3 }


def spread(img, px=1, shape='circle', how='over'):
    if not isinstance(img, Image):
        raise TypeError("Expected `Image`, got: `{0}`".format(type(img)))

    blockDim = int(np.sqrt(maxThreadsPerBlock) / 3.0) # not using maxThreadsPerBlock to have enough memory
    tpb = (blockDim, blockDim)
    bpg = (int(img.data.shape[0] / blockDim) + 1, int(img.data.shape[1] / blockDim) + 1)

    mask = _mask_lookup[shape](px)
    dest = cuda.device_array(img.data.shape, dtype=np.uint8)
    memset(dest)
    composite_kernel = _composite_op_lookup[how]
    spreading_k[bpg, tpb](img.data, mask, dest, composite_kernel)

    return Image(dest.copy_to_host())


@cuda.jit('void(uint8[:,:,:], float64[:])')
def density_k(src, counters):
    x, y = cuda.grid(2)
    N, M, channels = src.shape
    if x >= 0 and x < N and y >= 0 and y < M:
        if src[x, y, 3] > 0:
            cuda.atomic.add(counters, 0, 1.0)
            for i in range(x - 1, x + 2):
                for j in range(y - 1, y + 2):
                    if i >= 0 and i < N and j >= 0 and j < M:
                        if src[i, j, 3] > 0:
                            cuda.atomic.add(counters, 1, 1.0)


def density(src):
    blockDim = int(np.sqrt(maxThreadsPerBlock))
    tpb = (blockDim, blockDim)
    bpg = (int(src.shape[0] / blockDim) + 1, int(src.shape[1] / blockDim) + 1)
    counters = cuda.to_device(np.zeros(2, dtype=np.float64))
    density_k[bpg, tpb](src, counters)
    cnt, total = counters.copy_to_host()
    return (total - cnt) / (cnt * 8.0) if cnt else np.inf


def dynspread(img, threshold=0.5, max_px=3, shape='circle', how='over'):
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    if not isinstance(max_px, int) or max_px < 0:
        raise ValueError("max_px must be >= 0")
    for px in range(max_px + 1):
        out = spread(img, px, shape=shape, how=how)
        if density(out.data) >= threshold:
            break
    return out


@cuda.jit('void(uint8[:,:,:], uint8[:,:,:], uint8)')
def stack_k(img, dest, composite_type):
    x, y = cuda.grid(2)
    N, M, channels = img.shape
    if x >= 0 and x < N and y >= 0 and y < M:
        r = float(img[x, y, 0]) / 255.0
        g = float(img[x, y, 1]) / 255.0
        b = float(img[x, y, 2]) / 255.0
        a = float(img[x, y, 3]) / 255.0
        composite_k(r, g, b, a, dest[x, y], composite_type)


def stack(img1, img2, how="over"):
    if img1.data.shape[0] != img2.data.shape[0] or img1.data.shape[1] != img2.data.shape[1]:
        raise ValueError("Images must have same shapes")

    dest = cuda.to_device(img1.data)
    composite_kernel = _composite_op_lookup[how]

    blockDim = int(np.sqrt(maxThreadsPerBlock))
    tpb = (blockDim, blockDim)
    bpg = (int(dest.shape[0] / blockDim) + 1, int(dest.shape[1] / blockDim) + 1)
    stack_k[bpg, tpb](img2.data, dest, composite_kernel)

    return Image(dest.copy_to_host())


class Image():
    border=1

    def __init__(self, img):
        self.data = img
    
    def to_pil(self):
        return fromarray(np.flip(self.data, axis=0), 'RGBA')

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