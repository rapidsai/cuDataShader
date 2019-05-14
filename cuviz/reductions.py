from numba import cuda
import numpy as np


@cuda.jit('void(float64[:])')
def memset_k(inp):
    t = cuda.threadIdx.x
    b = cuda.blockIdx.x
    bw = cuda.blockDim.x
    i = b * bw + t
    if i < inp.size:
        inp[i] = 0.0


@cuda.jit('void(float64[:], float64[:], float64[:], int32, int32)')
def count_k(xd, yd, dest_img, xdim, ydim):
    t = cuda.threadIdx.x
    b = cuda.blockIdx.x
    bw = cuda.blockDim.x
    i = b * bw + t

    if i < xd.size:
        x = xd[i]
        y = yd[i]
        img_idx = int(((ydim-y) * xdim) + x)
        #img_idx = int((y * xdim) + x)

        if x >= 0 and x < xdim and y >= 0 and y < ydim:
            cuda.atomic.add(dest_img, img_idx, 1.0)


@cuda.jit('void(float64[:], float64[:], float64[:], float64[:], int32, int32)')
def sum_k(xd, yd, cat, dest_img, xdim, ydim):
    t = cuda.threadIdx.x
    b = cuda.blockIdx.x
    bw = cuda.blockDim.x
    i = b * bw + t

    if i < xd.size:
        x = xd[i]
        y = yd[i]
        img_idx = int(((ydim-y) * xdim) + x)
        #img_idx = int((y * xdim) + x)

        if x >= 0 and x < xdim and y >= 0 and y < ydim:
            cuda.atomic.add(dest_img, img_idx, cat[i])


class Aggregation():
    def __init__(self, agg, width, height):
        self.agg = agg
        self.width = width
        self.height = height


class Reduction:
    def __init__(self, column):
        self.column = column
        self.tpb = 512

    def set_scheme(self, width, height, n_points):
        self.width = width
        self.height = height
        self.bpg = int(n_points / self.tpb) + 1
        self.agg = cuda.device_array(self.width * self.height, dtype=np.float64)

    def set_data(self, x_coords, y_coords, agg_data):
        self.x_coords = x_coords
        self.y_coords = y_coords
        self.agg_data = agg_data

    def validate(self, data):
        if self.column not in data:
            raise ValueError("specified column not found")
        if not np.issubdtype(data[self.column].dtype, np.number):
            raise ValueError("input must be numeric")
        if not np.issubdtype(data[self.column].dtype, np.float64):
            raise ValueError("must use float64")


class count(Reduction):
    def reduct(self):
        memset_k[self.bpg, self.tpb](self.agg)
        count_k[self.bpg, self.tpb](self.x_coords, self.y_coords,
                                self.agg, self.width, self.height)
        return Aggregation(self.agg, self.width, self.height)


class sum(Reduction):
    def reduct(self):
        memset_k[self.bpg, self.tpb](self.agg)
        sum_k[self.bpg, self.tpb](self.x_coords, self.y_coords,
                self.agg_data, self.agg, self.width, self.height)
        return Aggregation(self.agg, self.width, self.height)
