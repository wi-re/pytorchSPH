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
def computeVelocityDiffusion(i, j, ri, rj, Vi, Vj, distances, radialDistances, support, numParticles : int, eps : float, rhoi, rhoj, ui, uj, alpha : float, c0 : float, restDensity : float):
    gradW = kernelGradient(radialDistances, distances, support)

    uji = uj[j] - ui[i]
    rji = rj[j] - ri[i]
    rji2 = torch.linalg.norm(rji, dim=1)**2 + eps

    pi_ij = torch.einsum('nu, nu -> n', uji, rji) 
    pi_ij[pi_ij > 0] = 0

    pi_ij = pi_ij / rji2
    term = (pi_ij * Vj[j] * rhoj[j]  / (rhoi[i] + rhoj[j]))[:,None] * gradW

    return (support * alpha * c0) * scatter_sum(term, i, dim=0, dim_size = numParticles)


class diffusionModule(Module):
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
        
    def resetState(self, simulationState):
        self.velocityDiffusion = None

    def evaluate(self, simulationState, simulation):
        with record_function('velocity[diffusion] - compute velocity diffusion'):
            self.velocityDiffusion = computeVelocityDiffusion(simulationState['fluidNeighbors'][0], simulationState['fluidNeighbors'][1], \
                                                                                                  simulationState['fluidPosition'], simulationState['fluidPosition'], simulationState['fluidVolume'], simulationState['fluidVolume'],\
                                                                                                  simulationState['fluidDistances'], simulationState['fluidRadialDistances'],\
                                                                                                  self.support, simulationState['fluidDensity'].shape[0], self.eps,\
                                                                                                  simulationState['fluidDensity'] * self.restDensity, simulationState['fluidDensity'] * self.restDensity,\
                                                                                                  simulationState['fluidVelocity'],simulationState['fluidVelocity'],
                                                                                                  self.alpha, self.c0, self.restDensity)
            if self.boundaryDiffusion:
                self.velocityDiffusion += simulation.boundaryModule.computeVelocityDiffusion(simulationState, simulation)
            simulationState['fluidAcceleration'] += self.velocityDiffusion
            simulation.sync(simulationState['fluidAcceleration'])
            # return self.velocityDiffusion