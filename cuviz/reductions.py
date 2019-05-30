from numba import cuda
import numpy as np
import cudf
from numba.cuda.cudadrv.devicearray import DeviceNDArrayBase


maxThreadsPerBlock = cuda.get_current_device().MAX_THREADS_PER_BLOCK


@cuda.jit('void(float64[:,:])')
def memset_k(input):
    x, y = cuda.grid(2)
    N, M = input.shape
    if x >= 0 and x < N and y >= 0 and y < M:
        input[x, y] = 0.0


@cuda.jit('void(float64[:], float64[:], float64[:,:])')
def count_k(x_coords, y_coords, agg):
    i = cuda.grid(1)
    N, M = agg.shape
    nx = x_coords[i]
    ny = y_coords[i]
    if nx >= 0 and nx < M and ny >= 0 and ny < N:
        cuda.atomic.add(agg, (ny, nx), 1.0)


@cuda.jit('void(float64[:], float64[:], float64[:], float64[:,:])')
def sum_k(x_coords, y_coords, agg_data, agg):
    i = cuda.grid(1)
    N, M = agg.shape
    nx = x_coords[i]
    ny = y_coords[i]
    if nx >= 0 and nx < M and ny >= 0 and ny < N:
        cuda.atomic.add(agg, (ny, nx), agg_data[i])


@cuda.jit('void(float64[:], float64[:], float64[:], float64[:,:])')
def max_k(x_coords, y_coords, agg_data, agg):
    i = cuda.grid(1)
    N, M = agg.shape
    nx = x_coords[i]
    ny = y_coords[i]
    if nx >= 0 and nx < M and ny >= 0 and ny < N:
        cuda.atomic.max(agg, (ny, nx), agg_data[i])


class Reduction:
    def __init__(self, column):
        self.column = column

    def set_scheme(self, width, height, n_points):
        self.agg = cuda.device_array((height, width), dtype=np.float64)
        self.set_agg(width, height)
        self.tpb = maxThreadsPerBlock
        self.bpg = int(n_points / maxThreadsPerBlock) + 1

    def set_data(self, x_coords, y_coords, agg_data):
        self.x_coords = x_coords
        self.y_coords = y_coords
        self.agg_data = agg_data

    def set_agg(self, width, height):
        blockDim = int(np.sqrt(maxThreadsPerBlock))
        memset_tpb = (blockDim, blockDim)
        memset_bpg = (int(height / blockDim) + 1, int(width / blockDim) + 1)
        memset_k[memset_bpg, memset_tpb](self.agg)

    def validate(self, data):
        if isinstance(data, cudf.DataFrame):
            if self.column not in data:
                raise ValueError("specified column not found")
            if not np.issubdtype(data[self.column].dtype, np.number):
                raise ValueError("input must be numeric")
            if not np.issubdtype(data[self.column].dtype, np.float64):
                raise ValueError("must use float64")
        elif isinstance(data, DeviceNDArrayBase):
            if len(data.shape) != 2 or data.shape[1] < 3:
                raise ValueError("input should have at least 3 columns for x, y and aggregation value")
            if data.dtype != np.float64:
                raise ValueError("must use float64")
        else:
            raise ValueError("input must be cudf DataFrame or Numba device ndarray")


class count(Reduction):
    def reduct(self):
        count_k[self.bpg, self.tpb](self.x_coords, self.y_coords, self.agg)
        return self.agg


class sum(Reduction):
    def reduct(self):
        sum_k[self.bpg, self.tpb](self.x_coords, self.y_coords, self.agg_data, self.agg)
        return self.agg


class max(Reduction):
    def reduct(self):
        max_k[self.bpg, self.tpb](self.x_coords, self.y_coords, self.agg_data, self.agg)
        return self.agg