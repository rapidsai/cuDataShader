from numba import cuda
import numpy as np
from cudatashader.reductions import count, any
import cudf
from numba.cuda.cudadrv.devicearray import DeviceNDArrayBase
from math import ceil, log


maxThreadsPerBlock = cuda.get_current_device().MAX_THREADS_PER_BLOCK

# Mask value (-1.79e+308 minimal float)
mask_value = np.finfo(np.float64).min


@cuda.jit('void(float64[:], float64, float64)')
def map_onto_pixel(input, scale, transform):
    i = cuda.grid(1)
    if i < input.shape[0]:
        val = input[i]
        if val > mask_value: # Only apply if value is not mask value (-1.79e+308 minimal float)
            input[i] = int((val * scale) + transform) # Apply transformation
        else:
            input[i] = np.nan # Set to nan when encountering mask value


@cuda.jit('void(float64[:], float64[:,:], int32)')
def copy_k(dst, src, idx):
    i = cuda.grid(1)
    if i < dst.shape[0]:
        dst[i] = src[i, idx]


def device_to_device_copy(src, idx):
    n_points = src.shape[0]
    dst = cuda.device_array(n_points, dtype=np.float64)
    blockspergrid = int(ceil(n_points / maxThreadsPerBlock))
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

    def data_mapper(self, x):
        pass


class LogAxis(Axis):
    def mapper(self, x):
        return np.log(x)

    @cuda.jit('void(float64[:])')
    def log_k(input):
        i = cuda.grid(1)
        if i < input.size:
            val = input[i]
            if val > 0.0: # Only valid for positive values
                input[i] = log(val)

    def data_mapper(self, x):
        blockspergrid = int(ceil(x.shape[0] / maxThreadsPerBlock))
        self.log_k[blockspergrid, maxThreadsPerBlock](x)


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

        # Compute scale and transformation factors
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
        # Look for default aggregation column
        if agg is None:
            if isinstance(source, cudf.DataFrame):
                agg = count(y)
            elif isinstance(data, DeviceNDArrayBase):
                agg = count(2)
            else:
                raise ValueError("input should be cudf DataFrame or Numba device ndarray")

        agg.validate(source) # Check source validity

        sx, tx = self.x_st
        sy, ty = self.y_st

        # Copying columns in Numba device ndarray
        if isinstance(source, cudf.DataFrame):
            x_coords = source.as_gpu_matrix([x]).ravel(order='F')
            y_coords = source.as_gpu_matrix([y]).ravel(order='F')
            agg_data = source.as_gpu_matrix([agg.column]).ravel(order='F')
        else:
            x_coords = device_to_device_copy(source, x)
            y_coords = device_to_device_copy(source, y)
            agg_data = device_to_device_copy(source, agg.column)

        # Apply axis transformations
        self.x_axis.data_mapper(x_coords)
        self.y_axis.data_mapper(y_coords)

        # Apply scale and transform transformations
        blockspergrid = int(ceil(len(agg_data) / maxThreadsPerBlock))
        map_onto_pixel[blockspergrid, maxThreadsPerBlock](x_coords, sx, tx)
        map_onto_pixel[blockspergrid, maxThreadsPerBlock](y_coords, sy, ty)
        
        # Sets up aggregation
        agg.set_scheme("points", self.plot_width, self.plot_height, len(agg_data))

        # Sets data for aggregation
        agg.set_data(x_coords, y_coords, agg_data)

        # Apply reduction
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
        # Look for default aggregation column
        if agg is None:
            if isinstance(source, cudf.DataFrame):
                agg = any(y)
            elif isinstance(data, DeviceNDArrayBase):
                agg = any(2)
            else:
                raise ValueError("input should be cudf DataFrame or Numba device ndarray")
        
        agg.validate(source) # Check source validity

        sx, tx = self.x_st
        sy, ty = self.y_st

        # Copying columns in Numba device ndarray
        if isinstance(source, cudf.DataFrame):
            # Replace nan values with mask value (-1.79e+308 minimal float)
            source.fillna({x: mask_value, y: mask_value, agg.column: mask_value}, inplace=True)
            x_coords = source.as_gpu_matrix([x]).ravel(order='F')
            y_coords = source.as_gpu_matrix([y]).ravel(order='F')
            agg_data = source.as_gpu_matrix([agg.column]).ravel(order='F')
        else:
            x_coords = device_to_device_copy(source, x)
            y_coords = device_to_device_copy(source, y)
            agg_data = device_to_device_copy(source, agg.column)

        # Apply axis transformations
        self.x_axis.data_mapper(x_coords)
        self.y_axis.data_mapper(y_coords)

        # Apply scale and transform transformations
        blockspergrid = int(ceil(len(agg_data) / maxThreadsPerBlock))
        map_onto_pixel[blockspergrid, maxThreadsPerBlock](x_coords, sx, tx)
        map_onto_pixel[blockspergrid, maxThreadsPerBlock](y_coords, sy, ty)
        
        # Sets data for aggregation
        agg.set_scheme("lines", self.plot_width, self.plot_height, len(agg_data))

        # Apply reduction
        agg.set_data(x_coords, y_coords, agg_data)

        # Apply reduction
        return agg.reduct()