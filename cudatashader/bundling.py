from numba import cuda
import numpy as np
import cudf
from math import ceil, sqrt

maxThreadsPerBlock = cuda.get_current_device().MAX_THREADS_PER_BLOCK


@cuda.jit('void(float64[:,:], int64[:,:], float64[:,:])')
def connect_edges_k(nodes, edges, connected_nodes): # Connect edges kernel
    i = cuda.grid(1)

    if i < edges.shape[0]:
        v1 = edges[i, 0]
        v2 = edges[i, 1]

        i *= 3

        connected_nodes[i, 0] = nodes[v1, 0]
        connected_nodes[i, 1] = nodes[v1, 1]
        connected_nodes[i+1, 0] = nodes[v2, 0]
        connected_nodes[i+1, 1] = nodes[v2, 1]
        connected_nodes[i+2, 0] = np.nan
        connected_nodes[i+2, 1] = np.nan


def connect_edges(nodes, edges):
    nodes = nodes.as_gpu_matrix()
    edges = edges.as_gpu_matrix()
    n_edges = edges.shape[0]
    # Allocate for storing lines
    connected_nodes = cuda.device_array((n_edges * 3, 2), dtype=np.float64)

    bpg = int(ceil(n_edges / maxThreadsPerBlock))
    # Apply connect edges kernel
    connect_edges_k[bpg, maxThreadsPerBlock](nodes, edges, connected_nodes)
    # Convert to cuDF DataFrame
    gdf = cudf.DataFrame.from_gpu_matrix(connected_nodes)
    gdf.rename({0: 'x', 1: 'y'}, copy=False, inplace=True)
    return gdf


@cuda.jit('void(float64[:,:], int64[:,:], float64[:,:,:])')
def connect_nodes_k(nodes_original, edges, nodes): # Connect nodes kernel
    i = cuda.grid(1)

    if i < edges.shape[0]:
        v1 = edges[i, 0]
        v2 = edges[i, 1]

        nodes[i, 0, 0] = nodes_original[v1, 0] # copy v1's x
        nodes[i, 0, 1] = nodes_original[v1, 1] # copy v1's y
        nodes[i, 1, 0] = nodes_original[v2, 0] # copy v2's x
        nodes[i, 1, 1] = nodes_original[v2, 1] # copy v2's y


def connect_nodes(nodes_original, edges, nodes): # Used to initialise GPU FDEB Edge Bundling
    n_edges = edges.shape[0]
    bpg = int(ceil(n_edges / maxThreadsPerBlock))
    # Apply connect nodes kernel
    connect_nodes_k[bpg, maxThreadsPerBlock](nodes_original, edges, nodes)


@cuda.jit('void(float64[:,:,:], float64, float64[:])')
def compute_stiffness_k(nodes, K, stiffness): # Stiffness kernel : compute edge-wise stiffness factor
    i = cuda.grid(1)

    if i < nodes.shape[0]:
        v1_x = nodes[i, 0, 0]
        v1_y = nodes[i, 0, 1]
        v2_x = nodes[i, 1, 0]
        v2_y = nodes[i, 1, 1]

        P_length = sqrt(((v1_x - v2_x) ** 2.0) + ((v1_y - v2_y) ** 2.0))
        stiffness[i] = K / P_length


def compute_stiffness(nodes, K, stiffness):
    n_edges = nodes.shape[0]
    bpg = int(ceil(n_edges / maxThreadsPerBlock))
    compute_stiffness_k[bpg, maxThreadsPerBlock](nodes, K, stiffness) # Apply stiffness kernel


@cuda.jit('void(float64[:,:,:], float64[:,:])')
def compatibility_k(nodes, c_matrix): # Compatibility kernel : compute pairwise edge compatibility
    x, y = cuda.grid(2)
    N, M = c_matrix.shape

    if y < x and x < N:
        v1_x = nodes[x, 0, 0]
        v1_y = nodes[x, 0, 1]
        v2_x = nodes[x, 1, 0]
        v2_y = nodes[x, 1, 1]
        v3_x = nodes[y, 0, 0]
        v3_y = nodes[y, 0, 1]
        v4_x = nodes[y, 1, 0]
        v4_y = nodes[y, 1, 1]

        d1_x = v1_x - v2_x
        d1_y = v1_y - v2_y
        d2_x = v3_x - v4_x
        d2_y = v3_y - v4_y

        P_length = sqrt((d1_x ** 2.0) + (d1_y ** 2.0))
        Q_length = sqrt((d2_x ** 2.0) + (d2_y ** 2.0))
        avg_length = (P_length + Q_length) / 2.0

        total_compatibility = 1.0

        # angle compatibility
        dot_prod = d1_x * d2_x + d1_y * d2_y
        total_compatibility *= abs(dot_prod / (P_length * Q_length))

        # scale compatibility
        total_compatibility *= 2.0 / ((avg_length * min(P_length, Q_length)) + (max(P_length, Q_length) / avg_length))

        # position compatibility
        pm_x = (v1_x + v2_x) / 2.0
        pm_y = (v1_y + v2_y) / 2.0
        qm_x = (v3_x + v4_x) / 2.0
        qm_y = (v3_y + v4_y) / 2.0
        dist = sqrt(((pm_x - qm_x) ** 2.0) + ((pm_y - qm_y) ** 2.0))
        total_compatibility *= avg_length / (avg_length + dist)

        # visibility compatibility TODO
        total_compatibility *= 1.0

        c_matrix[x, y] = total_compatibility


def compute_compatibility_matrix(nodes):
    n_edges = int(nodes.shape[0])
    blockDim = int(sqrt(maxThreadsPerBlock) / 2) # to get enough ressources per block
    tpb = (blockDim, blockDim)
    gridDim = int(ceil(n_edges / blockDim))
    bpg = (gridDim, gridDim)
    compatibility_matrix = cuda.device_array((n_edges, n_edges), dtype=np.float64)
    compatibility_k[bpg, tpb](nodes, compatibility_matrix) # Apply compatibility kernel

    return compatibility_matrix


@cuda.jit('void(float64[:,:,:], float64[:,:,:])')
def copy_nodes_k(nodes, nodes_copy): # Nodes copy kernel
    iEdge, iSubdiv = cuda.grid(2)
    if iEdge < nodes.shape[0]:
        nodes_copy[iEdge, iSubdiv, 0] = nodes[iEdge, iSubdiv, 0]
        nodes_copy[iEdge, iSubdiv, 1] = nodes[iEdge, iSubdiv, 1]


@cuda.jit('void(float64[:,:,:], float64[:,:,:], int32, float64[:,:], float64[:,:], float64[:,:], float64[:])')
def regenerate_subdivisions_k(nodes, old_nodes, P, d_xs, d_ys, segment_lengths, edge_lengths):
    iEdge, iSubdiv = cuda.grid(2)
    nEdges = nodes.shape[0]

    if iEdge < nEdges:
        if iSubdiv == 0:
            edge_lengths[iEdge] = 0.0
        cuda.syncthreads()

        v1_x = nodes[iEdge, iSubdiv, 0]
        v1_y = nodes[iEdge, iSubdiv, 1]
        v2_x = nodes[iEdge, iSubdiv + 1, 0]
        v2_y = nodes[iEdge, iSubdiv + 1, 1]

        d_x = v2_x - v1_x
        d_xs[iEdge, iSubdiv] = d_x

        d_y = v2_y - v1_y
        d_ys[iEdge, iSubdiv] = d_y

        segment_length = sqrt((d_x ** 2.0) + (d_y ** 2.0))
        segment_lengths[iEdge, iSubdiv] = segment_length

        cuda.atomic.add(edge_lengths, iEdge, segment_length)

        if iSubdiv == 0:
            cuda.syncthreads()

            nSegments = float(P + 1)                                    # number of segments in regeneration
            segment_length = edge_lengths[iEdge] / nSegments            # length of new segments
            new_cumsum = segment_length                                 # position of new segment to be inserted
            old_length = segment_lengths[iEdge, 0]                      # length of old segment that we work with
            old_cumsum = 0.0                                            # start position of old segment that we work with
            old_idx = 0                                                 # index of old segment that we work with
            
            for i in range(1, nSegments + 1):
                while new_cumsum > (old_cumsum + old_length) + 0.0001:
                    if (old_idx + 1) < d_xs.shape[1]:
                        old_cumsum += old_length
                        old_idx += 1
                        old_length = segment_lengths[iEdge, old_idx]
                        d_x = d_xs[iEdge, old_idx]
                        d_y = d_ys[iEdge, old_idx]
                    else:
                        nodes[iEdge, i, 0] = old_nodes[iEdge, old_idx, 0] + d_x
                        nodes[iEdge, i, 1] = old_nodes[iEdge, old_idx, 1] + d_y
                        return

                prop = (new_cumsum - old_cumsum) / old_length
                nodes[iEdge, i, 0] = old_nodes[iEdge, old_idx, 0] + (d_x * prop)
                nodes[iEdge, i, 1] = old_nodes[iEdge, old_idx, 1] + (d_y * prop)

                new_cumsum += segment_length


def regenerate_subdivisions(nodes, P):
    n_edges = nodes.shape[0]
    nMeasures = int(P / 2) + 1
    tpb = (int((maxThreadsPerBlock / nMeasures) / 2), nMeasures)
    bpg = (int(ceil(n_edges / tpb[0])), 1)

    nodes_copy = cuda.device_array((n_edges, nMeasures, 2), dtype=np.float64)
    copy_nodes_k[bpg, tpb](nodes, nodes_copy) # Copy nodes before subdivision

    d_xs = cuda.device_array((n_edges, nMeasures), dtype=np.float64)
    d_ys = cuda.device_array((n_edges, nMeasures), dtype=np.float64)
    segment_lengths = cuda.device_array((n_edges, nMeasures), dtype=np.float64)
    edge_lengths = cuda.device_array(n_edges, dtype=np.float64)

    # Apply subdivision of each edges
    regenerate_subdivisions_k[bpg, tpb](nodes, nodes_copy, P, d_xs, d_ys, segment_lengths, edge_lengths)


@cuda.jit('void(float64[:,:,:], float64[:], float64[:,:,:])')
def compute_spring_force_k(nodes, stiffness, forces):
    edge_idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    subdiv_in = cuda.blockIdx.y + 1
    subdiv_out = cuda.blockIdx.y

    if edge_idx < nodes.shape[0]:
        p1_x = nodes[edge_idx, subdiv_in - 1,  0]
        p1_y = nodes[edge_idx, subdiv_in - 1,  1]
        p2_x = nodes[edge_idx, subdiv_in,      0]
        p2_y = nodes[edge_idx, subdiv_in,      1]
        p3_x = nodes[edge_idx, subdiv_in + 1,  0]
        p3_y = nodes[edge_idx, subdiv_in + 1,  1]

        d1_x = p1_x - p2_x
        d1_y = p1_y - p2_y
        d2_x = p3_x - p2_x
        d2_y = p3_y - p2_y

        kp = stiffness[edge_idx]
        forces[edge_idx, subdiv_out, 0] += kp * (d1_x + d2_x)
        forces[edge_idx, subdiv_out, 1] += kp * (d1_y + d2_y)


def compute_spring_force(nodes, P, stiffness, forces):
    n_edges = nodes.shape[0]
    tpb = maxThreadsPerBlock
    bpg = (int(ceil(n_edges / maxThreadsPerBlock)), P)

    compute_spring_force_k[bpg, tpb](nodes, stiffness, forces)


@cuda.jit('void(float64[:,:,:], float64[:,:], float64[:,:,:])')
def compute_electrostatic_force_k(nodes, compatibility_matrix, forces):
    subdiv_in = cuda.blockIdx.x + 1
    subdiv_out = cuda.blockIdx.x
    x = cuda.blockIdx.y * cuda.blockDim.x + cuda.threadIdx.x
    y = cuda.blockIdx.z * cuda.blockDim.y + cuda.threadIdx.y

    N, M, xy = forces.shape
    if y < x and x < N:
        p_x = nodes[x, subdiv_in, 0]
        p_y = nodes[x, subdiv_in, 1]
        q_x = nodes[y, subdiv_in, 0]
        q_y = nodes[y, subdiv_in, 1]


        d_x = q_x - p_x
        d_y = q_y - p_y

        d = sqrt((d_x ** 2.0) + (d_y ** 2.0))

        d_x /= d
        d_y /= d
        
        if abs(d) > 0.000001:
            compatibility = compatibility_matrix[x, y]

            f_x = (d_x / d) * compatibility
            cuda.atomic.add(forces, (x, subdiv_out, 0), f_x)
            cuda.atomic.add(forces, (y, subdiv_out, 0), -f_x)

            f_y = (d_y / d) * compatibility
            cuda.atomic.add(forces, (x, subdiv_out, 1), f_y)
            cuda.atomic.add(forces, (y, subdiv_out, 1), -f_y)


def compute_electrostatic_force(nodes, P, compatibility_matrix, forces):
    n_edges = nodes.shape[0]
    blockDim = int(sqrt(maxThreadsPerBlock))
    tpb = (blockDim, blockDim)
    gridDim = int(ceil(n_edges / blockDim))
    bpg = (P, gridDim, gridDim)

    compute_electrostatic_force_k[bpg, tpb](nodes, compatibility_matrix, forces)


@cuda.jit('void(float64[:,:,:], float64[:,:,:], float64)')
def update_positions_k(nodes, forces, S):
    edge_idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    if edge_idx < nodes.shape[0]:
        subdiv_in = cuda.blockIdx.y
        subdiv_out = cuda.blockIdx.y + 1

        f_x = forces[edge_idx, subdiv_in, 0]
        f_y = forces[edge_idx, subdiv_in, 1]
        norm = sqrt((f_x ** 2.0) + (f_y ** 2.0))
        d_x = (f_x / norm) * S
        d_y = (f_y / norm) * S

        nodes[edge_idx, subdiv_out, 0] += d_x
        forces[edge_idx, subdiv_in, 0] = 0.0 # reseting for future use

        nodes[edge_idx, subdiv_out, 1] += d_y
        forces[edge_idx, subdiv_in, 1] = 0.0 # reseting for future use


def update_positions(nodes, P, forces, S):
    n_edges = nodes.shape[0]
    bpg = (int(ceil(n_edges / maxThreadsPerBlock)), P)

    update_positions_k[bpg, maxThreadsPerBlock](nodes, forces, S)


@cuda.jit('void(float64[:,:,:], float64[:,:])')
def as_edges_k(nodes, edges):
    edge_idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    subdiv_idx = cuda.blockIdx.y
    
    nEdges = nodes.shape[0]
    if edge_idx < nEdges:
        new_edge_idx = edge_idx * (nodes.shape[1] + 1) + subdiv_idx

        if subdiv_idx < nodes.shape[1]:
            edges[new_edge_idx, 0] = nodes[edge_idx, subdiv_idx, 0]
            edges[new_edge_idx, 1] = nodes[edge_idx, subdiv_idx, 1]
        else:
            edges[new_edge_idx, 0] = np.nan
            edges[new_edge_idx, 1] = np.nan


def as_edges(nodes):
    n_edges, P, xy = nodes.shape
    edges = cuda.device_array((n_edges * (P + 1), 2), dtype=np.float64)

    bpg = (int(ceil(n_edges / maxThreadsPerBlock)), P + 1)
    as_edges_k[bpg, maxThreadsPerBlock](nodes, edges)
    
    gdf = cudf.DataFrame.from_gpu_matrix(edges)
    gdf.rename({0: 'x', 1: 'y'}, copy=False, inplace=True)
    return gdf


def fdeb_bundle(nodes, edges, params=None):
    """
    Run GPU FDEB Edge Bundling algorithm.

    References :
        - https://ieeexplore.ieee.org/document/6385238
        - http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.212.7989&rep=rep1&type=pdf
    """

    if params is None:
        params = {
            'C': 5,
            'I': 50,
            'S': 0.001,
            'K': 0.03
        }

    # initialize parameters
    C = params['C']     # number of cycles
    I = params['I']     # number of iteration steps in a cycle
    P = 1               # number of subdivision points
    S = params['S']     # step size
    K = params['K']     # stiffness factor

    max_P = 2 ** (C-1)

    # Convert cuDF DataFrames to Numba Cuda ndarray
    nodes_original = nodes.as_gpu_matrix()
    edges = edges.as_gpu_matrix()

    n_edges = edges.shape[0]
    nodes = cuda.device_array((n_edges, max_P + 2, 2), dtype=np.float64)

    # setting up nodes buffer on GPU
    connect_nodes(nodes_original, edges, nodes)
    del nodes_original  # no more use
    del edges           # no more use

    stiffness = cuda.device_array(n_edges, dtype=np.float64)
    forces = cuda.device_array((n_edges, max_P, 2), dtype=np.float64)

    # calculate edge's stiffness
    compute_stiffness(nodes, K, stiffness)

    # calculate edge compatibility pairwise
    compatibility_matrix = compute_compatibility_matrix(nodes)

    # bundling
    for c in range(C):
        # regenerate subdivision points
        regenerate_subdivisions(nodes, P)

        # adjust the positions of subdivision points
        for i in range(I):
            # calculate spring force
            compute_spring_force(nodes, P, stiffness, forces)

            # calculate electrostatic force
            compute_electrostatic_force(nodes, P, compatibility_matrix, forces)

            # update position accordingly
            update_positions(nodes, P, forces, S)

        # update parameters
        P *= 2
        I = int(I * (2.0 / 3.0))
        S /= 2.0
    
    # Generate cuDF DataFrame to display segments with a lines canvas
    return as_edges(nodes)