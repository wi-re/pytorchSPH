import inspect
import re
def debugPrint(x):
    frame = inspect.currentframe().f_back
    s = inspect.getframeinfo(frame).code_context[0]
    r = re.search(r"\((.*)\)", s).group(1)
    print("{} [{}] = {}".format(r,type(x).__name__, x))
    
    
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch_geometric.nn import radius
from torch_geometric.nn import SplineConv, fps, global_mean_pool, radius_graph, radius
from torch_scatter import scatter
from torch.profiler import profile, record_function, ProfilerActivity

# Wendland 2 Kernel function and its derivative
@torch.jit.script
def wendland(q, h):
    C = 7 / np.pi
    b1 = torch.pow(1. - q, 4)
    b2 = 1.0 + 4.0 * q
    return b1 * b2 * C / h**2    
@torch.jit.script
def wendlandGrad(q,r,h):
    C = 7 / np.pi    
    return - r * C / h**3 * (20. * q * (1. -q)**3)[:,None]
# Spiky kernel function used mainly in DFSPH to avoid particle clustering
@torch.jit.script
def spikyGrad(q,r,h):
    return -r * 30 / np.pi / h**3 * ((1 - q)**2)[:,None]
# Cohesion kernel is used for the akinci surface tension module
@torch.jit.script
def cohesionKernel(q, h):
    res = q.new_zeros(q.shape)
    Cd = -1 / (2 * np.pi) * 1 / h**2
    k1 = 128 * (q-1)**3 * q**3 + 1
    k2 = 64 * (q-1)**3 * q**3
    
    res[q <= 0.5] = k1[q<=0.5]
    res[q > 0.5] = k2[q>0.5]
    
    return -res
# Convenient alias functions for easier usage
@torch.jit.script
def kernel(q, h):
    return wendland(q,h)
@torch.jit.script
def kernelGradient(q,r,h):
    return wendlandGrad(q,r,h)

# This function was inteded to be used to swap to different kernel functions
# However, pytorch SPH makes this overly cumbersome so this is not implemented
# TODO: Someday this should be possible in torch script.
def getKernelFunctions(kernel):
    if kernel == 'wendland2':
        return wendland, wendlandGrad
