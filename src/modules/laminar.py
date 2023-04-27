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

from ..kernels import kernel, spikyGrad, kernelGradient
from ..module import Module
from ..parameter import Parameter
from ..util import *

import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator
import matplotlib.ticker as mticker

@torch.jit.script
def computeLaminarViscosity(i, j, ri, rj, Vi, Vj, distances, radialDistances, support, numParticles : int, eps : float, rhoi, rhoj, ui, uj, alpha : float, c0 : float, restDensity : float):
    gradW = kernelGradient(radialDistances, distances, support)

    uij = ui[i] - uj[j]
    rij = ri[j] - rj[i]
    rij2 = torch.linalg.norm(rij, dim=1)**2 + eps

    mui = rhoi[i] * alpha
    muj = rhoj[i] * alpha
    mj = rhoj[j] * Vj[j] 

    nominator = 4 * mj * (mui + muj) * torch.einsum('nu, nu -> n', rij, gradW) 
    denominator = (rhoi[i] + rhoj[j])**2 * (rij2)
    term = nominator / denominator

    return -scatter_sum(term[:,None] * uij, i, dim=0, dim_size = numParticles)


class laminarViscosityModule(Module):
    def __init__(self):
        super().__init__('densityInterpolation', 'Evaluates density at the current timestep')
    
    def initialize(self, simulationConfig, simulationState):
        self.support = simulationConfig['particle']['support']    
        self.backgroundPressure = simulationConfig['fluid']['backgroundPressure']
        self.restDensity = simulationConfig['fluid']['restDensity']
        self.gamma = simulationConfig['pressure']['gamma']
        
        self.boundaryScheme = simulationConfig['simulation']['boundaryScheme']
        self.boundaryCounter = len(simulationConfig['solidBC']) if 'solidBC' in simulationConfig else 0

        self.computeBodyForces = simulationConfig['simulation']['bodyForces'] 
        
        self.dtype = simulationConfig['compute']['precision']
        self.device = simulationConfig['compute']['device'] 
        
        self.alpha = simulationConfig['diffusion']['alpha']
        self.boundaryDiffusion = simulationConfig['diffusion']['boundaryDiffusion']
        self.delta = simulationConfig['diffusion']['delta'] 
        dx = simulationConfig['particle']['support'] * simulationConfig['particle']['packing']
        c0 = 10.0 * np.sqrt(2.0*9.81*0.3)
        h0 = simulationConfig['particle']['support']
        dt = 0.25 * h0 / (1.1 * c0)
        if simulationConfig['fluid']['c0'] < 0:
            simulationConfig['fluid']['c0'] = c0
        
        self.c0 = simulationConfig['fluid']['c0']
        self.eps = self.support **2 * 0.1
        self.kinematic = simulationConfig['diffusion']['kinematic']
    def resetState(self, simulationState):
        self.laminarViscosity = None

    def computeLaminarViscosity(self, simulationState, simulation):
        with record_function('diffusion[laminar] - compute laminar diffusion'):
            self.laminarViscosity = computeLaminarViscosity(simulationState['fluidNeighbors'][0], simulationState['fluidNeighbors'][1], \
                                                                                                  simulationState['fluidPosition'], simulationState['fluidPosition'], simulationState['fluidVolume'], simulationState['fluidVolume'],\
                                                                                                  simulationState['fluidDistances'], simulationState['fluidRadialDistances'],\
                                                                                                  self.support, simulationState['fluidDensity'].shape[0], self.eps,\
                                                                                                  simulationState['fluidDensity'] * self.restDensity, simulationState['fluidDensity'] * self.restDensity,\
                                                                                                  simulationState['fluidVelocity'],simulationState['fluidVelocity'],
                                                                                                  self.kinematic, self.c0, self.restDensity)
            # if self.boundaryDiffusion:
                # self.laminarViscosity += simulation.boundaryModule.computeLaminarViscosity(simulationState, simulation)
            simulationState['fluidAcceleration'] += self.laminarViscosity
            simulation.sync(simulationState['fluidAcceleration'])
            # return self.laminarViscosity
        