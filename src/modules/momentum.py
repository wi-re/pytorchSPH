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

from ..kernels import kernel, kernelGradient
from ..module import Module, BoundaryModule
from ..parameter import Parameter
from ..util import *

from ..ghostParticles import *

from .deltaSPH import *
from .diffusion import computeVelocityDiffusion
from .densityDiffusion import *


@torch.jit.script
def computeDivergenceTerm(i, j, ri, rj, Vi, Vj, distances, radialDistances, support, numParticles : int, eps : float, rhoi, rhoj, ui, uj):
    gradW = kernelGradient(radialDistances, distances, support)

    uji = uj[j] - ui[i]
    prod = torch.einsum('nu,nu -> n', uji, gradW) 

    return - scatter_sum(prod * Vj[j] * rhoj[j], i, dim=0, dim_size = numParticles)


class momentumModule(Module):    
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
        self.divergenceTerm = None
        self.dpdt = None

        simulationState.pop('dpdt', None)

    def computeDpDt(self, simulationState, simulation):
        with record_function('density[continuity] - compute drho/dt'):
            self.divergenceTerm = computeDivergenceTerm(simulationState['fluidNeighbors'][0], simulationState['fluidNeighbors'][1], \
                                                                                                  simulationState['fluidPosition'], simulationState['fluidPosition'], simulationState['fluidVolume'], simulationState['fluidVolume'],\
                                                                                                  simulationState['fluidDistances'], simulationState['fluidRadialDistances'],\
                                                                                                  self.support, simulationState['fluidDensity'].shape[0], self.eps,\
                                                                                                  simulationState['fluidDensity'] * self.restDensity, simulationState['fluidDensity'] * self.restDensity,\
                                                                                                  simulationState['fluidVelocity'], simulationState['fluidVelocity'])
            self.divergenceTerm += simulation.boundaryModule.computeDpDt(simulationState, simulation)
            
            self.dpdt = self.divergenceTerm #+ self.densityDiffusion
            simulationState['dpdt'] = self.dpdt
            simulation.sync(simulationState['dpdt'])
            # return self.dpdt

