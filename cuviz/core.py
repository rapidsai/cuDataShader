from numba import cuda
import numpy as np

@cuda.jit('void(float64[:], float64, float64)')
def map_onto_pixel(inp, scale, transform):
    t = cuda.threadIdx.x
    b = cuda.blockIdx.x
    bw = cuda.blockDim.x
    i = b * bw + t
    if i < inp.size:
        inp[i] = int((inp[i] * scale) + transform)

class Axis:
    def compute_scale_and_translate(self, range, n):
        start, end = map(self.mapper, range)
        s = n/(end - start)
        t = -start * s
        return s, t

class LinearAxis(Axis):
    def mapper(self, x):
        return x

class LogAxis(Axis):
    def mapper(self, x):
        return np.log(x)

_axis_lookup = {'linear': LinearAxis(), 'log': LogAxis()}

class Canvas:
    """An abstract canvas representing the space in which to bin.
    Parameters
    ----------
    plot_width, plot_height : int, optional
        Width and height of the output aggregate in pixels.
    x_range, y_range : tuple, optional
        A tuple representing the bounds inclusive space ``[min, max]`` along
        the axis.
    x_axis_type, y_axis_type : str, optional
        The type of the axis. Valid options are ``'linear'`` [default], and
        ``'log'``.
    """
    def __init__(self, plot_width=600, plot_height=600,
                 x_range=None, y_range=None,
                 x_axis_type='linear', y_axis_type='linear'):
        self.plot_width = plot_width
        self.plot_height = plot_height
        self.x_range = x_range
        self.y_range = y_range
        self.x_axis = _axis_lookup[x_axis_type]
        self.y_axis = _axis_lookup[y_axis_type]
        self.x_st = self.x_axis.compute_scale_and_translate(x_range, plot_width)
        self.y_st = self.y_axis.compute_scale_and_translate(y_range, plot_height)

    def points(self, source, x, y, agg=None):
        """Compute a reduction by pixel, mapping data to pixels as points.
        Parameters
        ----------
        source : cudf.DataFrame
            The input datasource.
        x, y : str
            Column names for the x and y coordinates of each point.
        agg : Reduction, optional
            Reduction to compute. Default is ``count()``.
        """
        agg.validate(source)

        threadsperblock = 512
        blockspergrid = int(len(source) / threadsperblock) + 1

        sx, tx = self.x_st
        x_coords = source.as_gpu_matrix([x]).ravel(order='F')
        map_onto_pixel[blockspergrid, threadsperblock](x_coords, sx, tx)

        sy, ty = self.y_st
        y_coords = source.as_gpu_matrix([y]).ravel(order='F')
        map_onto_pixel[blockspergrid, threadsperblock](y_coords, sy, ty)

        agg_data = source.as_gpu_matrix([agg.column]).ravel(order='F')
        agg.set_scheme(self.plot_width, self.plot_height, len(agg_data))
        agg.set_data(x_coords, y_coords, agg_data)
        return agg.reduct()
