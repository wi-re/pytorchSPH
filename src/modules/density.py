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
from ..module import Module
from ..parameter import Parameter
from ..util import *


# @torch.jit.script
@torch.jit.script
def computeDensity(radialDistances, areas, neighbors, support):
    with record_function("sph - density 2"): 
        rho =  scatter_sum(kernel(radialDistances, support) * areas[neighbors[1]], neighbors[0], dim=0, dim_size=areas.shape[0])
        return rho


class densityModule(Module):
    def __init__(self):
        super().__init__('densityInterpolation', 'Evaluates density at the current timestep')
    
    def initialize(self, simulationConfig, simulationState):
        self.support = simulationConfig['particle']['support']
        self.scheme = simulationConfig['simulation']['densityScheme']

    def resetState(self, simulationState):
        simulationState.pop('fluidVolume', None)

    def evaluate(self, simulationState, simulation):
        if self.scheme == 'summation':
            with record_function('density[summation] - evaluate'):
                fluidRadialDistances = simulationState['fluidRadialDistances']
                fluidArea = simulationState['fluidArea']
                fluidNeighbors = simulationState['fluidNeighbors']
                particleSupport = self.support
                simulationState['fluidDensity'] = computeDensity(fluidRadialDistances, fluidArea, fluidNeighbors, particleSupport)
        
        simulationState['fluidVolume'] = simulationState['fluidArea'] / simulationState['fluidDensity']
        # simulation.sync(simulationState['fluidDensity'])
        # simulation.sync(simulationState['fluidVolume'])

def testFunctionality(sphSimulation):
    density = densityModule()
    density.initialize(sphSimulation.config, sphSimulation)
            
    sphSimulation.sphDensity.evaluate(sphSimulation.simulationState, sphSimulation)
    density.evaluate(sphSimulation.simulationState, sphSimulation)