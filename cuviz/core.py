from numba import cuda
import numpy as np
from cuviz.reductions import count, any
import cudf
from numba.cuda.cudadrv.devicearray import DeviceNDArrayBase


maxThreadsPerBlock = cuda.get_current_device().MAX_THREADS_PER_BLOCK


@cuda.jit('void(float64[:], float64, float64)')
def map_onto_pixel(input, scale, transform):
    i = cuda.grid(1)
    if i < input.shape[0]:
        val = input[i]
        if val > -1.79e+308:
            input[i] = int((val * scale) + transform)
        else:
            input[i] = np.nan


@cuda.jit('void(float64[:], float64[:,:], int32)')
def copy_k(dst, src, idx):
    i = cuda.grid(1)
    if i < dst.shape[0]:
        dst[i] = src[i, idx]


def device_to_device_copy(src, idx):
    n_points = src.shape[0]
    dst = cuda.device_array(n_points, dtype=np.float64)
    blockspergrid = int(n_points / maxThreadsPerBlock) + 1
    copy_k[blockspergrid, maxThreadsPerBlock](dst, src, idx)
    return dst


class Axis:
    def compute_scale_and_translate(self, range, n):
        start, end = map(self.mapper, range)
        s = n / (end - start)
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
        source : cudf.DataFrame or Numba device ndarray
            The input datasource.
        x, y : str
            Column names for the x and y coordinates of each point.
        agg : Reduction, optional
            Reduction to compute. Default is ``count()``.
        """
        if agg is None:
            if isinstance(source, cudf.DataFrame):
                agg = count(source.columns[-1])
            elif isinstance(data, DeviceNDArrayBase):
                agg = count(2)
            else:
                raise ValueError("input should be cudf DataFrame or Numba device ndarray")

        agg.validate(source)

        sx, tx = self.x_st
        sy, ty = self.y_st

        if isinstance(source, cudf.DataFrame):
            x_coords = source.as_gpu_matrix([x]).ravel(order='F')
            y_coords = source.as_gpu_matrix([y]).ravel(order='F')
            agg_data = source.as_gpu_matrix([agg.column]).ravel(order='F')
        else:
            x_coords = device_to_device_copy(source, x)
            y_coords = device_to_device_copy(source, y)
            agg_data = device_to_device_copy(source, agg.column)

        blockspergrid = int(len(agg_data) / maxThreadsPerBlock) + 1
        map_onto_pixel[blockspergrid, maxThreadsPerBlock](x_coords, sx, tx)
        map_onto_pixel[blockspergrid, maxThreadsPerBlock](y_coords, sy, ty)
        
        agg.set_scheme("points", self.plot_width, self.plot_height, len(agg_data))
        agg.set_data(x_coords, y_coords, agg_data)
        return agg.reduct()


    def line(self, source, x, y, agg=None, axis=0):
        """Compute a reduction by pixel, mapping data to pixels as one or more lines.
        Parameters
        ----------
        source : cudf.DataFrame
            The input datasource.
        x, y : str
            Column names for the x and y coordinates of each point.
        agg : Reduction, optional
            Reduction to compute. Default is ``any()``.
        axis : 0 or 1, default 0
            Axis in source to draw lines along
            * 0: Draw lines using data from the specified columns across
                 all rows in source
            * 1: Draw one line per row in source using data from the
                 specified columns
        """

        if agg is None:
            if isinstance(source, cudf.DataFrame):
                agg = any(source.columns[-1])
            elif isinstance(data, DeviceNDArrayBase):
                agg = any(2)
            else:
                raise ValueError("input should be cudf DataFrame or Numba device ndarray")
        
        agg.validate(source)

        sx, tx = self.x_st
        sy, ty = self.y_st

        if isinstance(source, cudf.DataFrame):
            mini = np.finfo(np.float64).min
            source.fillna({x: mini, y: mini, agg.column: mini}, inplace=True)
            x_coords = source.as_gpu_matrix([x]).ravel(order='F')
            y_coords = source.as_gpu_matrix([y]).ravel(order='F')
            agg_data = source.as_gpu_matrix([agg.column]).ravel(order='F')
        else:
            x_coords = device_to_device_copy(source, x)
            y_coords = device_to_device_copy(source, y)
            agg_data = device_to_device_copy(source, agg.column)

        blockspergrid = int(len(agg_data) / maxThreadsPerBlock) + 1
        map_onto_pixel[blockspergrid, maxThreadsPerBlock](x_coords, sx, tx)
        map_onto_pixel[blockspergrid, maxThreadsPerBlock](y_coords, sy, ty)
        
        agg.set_scheme("lines", self.plot_width, self.plot_height, len(agg_data))
        agg.set_data(x_coords, y_coords, agg_data)
        return agg.reduct()