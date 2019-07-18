from numba import cuda
import numpy as np
import cudf
from numba.cuda.cudadrv.devicearray import DeviceNDArrayBase
from math import ceil, isnan


maxThreadsPerBlock = cuda.get_current_device().MAX_THREADS_PER_BLOCK


@cuda.jit('void(float64[:,:])')
def memset_k(input): # Memset aggregation array to 0
    x, y = cuda.grid(2)
    N, M = input.shape
    if x >= 0 and x < N and y >= 0 and y < M:
        input[x, y] = 0.0


@cuda.jit('void(float64[:], float64[:], float64[:,:])')
def count_k(x_coords, y_coords, agg): # Count reduction for points
    i = cuda.grid(1)
    n_points = x_coords.size

    if i < n_points:
        N, M = agg.shape
        nx = int(x_coords[i])
        ny = int(y_coords[i])
        if nx >= 0 and nx < M and ny >= 0 and ny < N:
            cuda.atomic.add(agg, (ny, nx), 1.0)


@cuda.jit('void(float64[:], float64[:], float64[:,:])')
def any_k(x_coords, y_coords, agg): # Any reduction for points
    i = cuda.grid(1)
    n_points = x_coords.shape[0]

    if i < n_points:
        N, M = agg.shape
        nx = int(x_coords[i])
        ny = int(y_coords[i])
        if nx >= 0 and nx < M and ny >= 0 and ny < N:
            agg[ny, nx] = 1.0


@cuda.jit('void(float64[:], float64[:], float64[:,:])')
def any_lines_k(x_coords, y_coords, agg): # Any reduction for lines
    i = cuda.grid(1)
    n_points = x_coords.shape[0]

    if i < n_points - 1:
        n1_x = x_coords[i]
        n1_y = y_coords[i]
        if isnan(n1_x) or isnan(n1_y):
            return # no segment
        n2_x = x_coords[i+1]
        n2_y = y_coords[i+1]
        if isnan(n2_x) or isnan(n2_x):
            return # last point of edge, no segment to display

        M, N = agg.shape

        d_x = abs(n2_x - n1_x)
        d_y = abs(n2_y - n1_y)

        x, y = int(n1_x), int(n1_y)
        s_x = -1 if n1_x > n2_x else 1
        s_y = -1 if n1_y > n2_y else 1

        if d_x > d_y:
            err = d_x / 2.0
            while x != n2_x:
                if x >= 0 and x < N and y >= 0 and y < M:
                    agg[y, x] = 1.0
                err -= d_y
                if err < 0:
                    y += s_y
                    err += d_x
                x += s_x
        else:
            err = d_y / 2.0
            while y != n2_y:
                if x >= 0 and x < N and y >= 0 and y < M:
                    agg[y, x] = 1.0
                err -= d_x
                if err < 0:
                    x += s_x
                    err += d_y
                y += s_y
        if x >= 0 and x < N and y >= 0 and y < M:     
            agg[y, x] = 1.0 # Draw segment by setting all pixels to 1


@cuda.jit('void(float64[:], float64[:], float64[:], float64[:,:])')
def sum_k(x_coords, y_coords, agg_data, agg): # Sum reduction for points
    i = cuda.grid(1)
    n_points = x_coords.shape[0]

    if i < n_points:
        N, M = agg.shape
        nx = int(x_coords[i])
        ny = int(y_coords[i])
        if nx >= 0 and nx < M and ny >= 0 and ny < N:
            cuda.atomic.add(agg, (ny, nx), agg_data[i])


@cuda.jit('void(float64[:], float64[:], float64[:], float64[:,:])')
def max_k(x_coords, y_coords, agg_data, agg): # Max reduction for points
    i = cuda.grid(1)
    n_points = x_coords.shape[0]

    if i < n_points:
        N, M = agg.shape
        nx = int(x_coords[i])
        ny = int(y_coords[i])
        if nx >= 0 and nx < M and ny >= 0 and ny < N:
            cuda.atomic.max(agg, (ny, nx), agg_data[i])


class Reduction:
    def __init__(self, column):
        self.column = column

    def set_scheme(self, glyph_type, width, height, n_points):
        self.agg = cuda.device_array((height, width), dtype=np.float64) # Allocate aggregation array on GPU
        self.set_agg(width, height) # Memset aggregation array to 0
        self.glyph_type = glyph_type # Lines or points
        self.tpb = maxThreadsPerBlock # Compute CUDA thread per block
        self.bpg = int(ceil(n_points / maxThreadsPerBlock)) # Compute CUDA block per grid

    def set_data(self, x_coords, y_coords, agg_data):
        self.x_coords = x_coords
        self.y_coords = y_coords
        self.agg_data = agg_data

    def set_agg(self, width, height):
        blockDim = int(np.sqrt(maxThreadsPerBlock))
        memset_tpb = (blockDim, blockDim)
        memset_bpg = (int(ceil(height / blockDim)), int(ceil(width / blockDim)))
        memset_k[memset_bpg, memset_tpb](self.agg) # Memset aggregation array to 0

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
        if self.glyph_type == "points":
            count_k[self.bpg, self.tpb](self.x_coords, self.y_coords, self.agg)
        return self.agg


class any(Reduction):
    def reduct(self):
        if self.glyph_type == "points":
            any_k[self.bpg, self.tpb](self.x_coords, self.y_coords, self.agg)
        if self.glyph_type == "lines":
            any_lines_k[self.bpg, self.tpb](self.x_coords, self.y_coords, self.agg)
        return self.agg


class sum(Reduction):
    def reduct(self):
        if self.glyph_type == "points":
            sum_k[self.bpg, self.tpb](self.x_coords, self.y_coords, self.agg_data, self.agg)
        return self.agg


class max(Reduction):
    def reduct(self):
        if self.glyph_type == "points":
            max_k[self.bpg, self.tpb](self.x_coords, self.y_coords, self.agg_data, self.agg)
        return self.agg