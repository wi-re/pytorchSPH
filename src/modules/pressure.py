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
def computePressureAccel(i, j, ri, rj, Vi, Vj, distances, radialDistances, support, numParticles : int, eps : float, rhoi, rhoj, pi, pj):
    gradW = kernelGradient(radialDistances, distances, support)

    pij = pi[i] + pj[j]        
    term = (pij * Vj[j])[:,None] * gradW

    return - 1 / rhoi[:,None] * scatter_sum(term, i, dim=0, dim_size = numParticles)

@torch.jit.script
def computePressureAccelDeltaPlus(i, j, ri, rj, Vi, Vj, distances, radialDistances, support, numParticles : int, eps : float, rhoi, rhoj, pi, pj, surfaceMask):
    gradW = kernelGradient(radialDistances, distances, support)

    pij = pj[j] - pi[i]    
    switch = torch.logical_or(pi[i] >= 0, surfaceMask[i] < 1.5)
    pij[switch] = pi[i][switch] + pj[j][switch]


    term = (pij * Vj[j])[:,None] * gradW

    return - 1 / rhoi[:,None] * scatter_sum(term, i, dim=0, dim_size = numParticles)


class pressureModule(Module):
    def getParameters(self):
        return [
            # Parameter('deltaSPH', 'alpha', 'float', 0.01, required = False, export = True, hint = ''),
            # Parameter('deltaSPH', 'delta', 'float', 0.1, required = False, export = True, hint = ''),
            # Parameter('deltaSPH', 'c0', 'float', -1.0, required = False, export = True, hint = ''),
            # Parameter('deltaSPH', 'gamma', 'float', 7.0, required = False, export = True, hint = ''),
            # Parameter('deltaSPH', 'beta', 'float', -1.0, required = False, export = True, hint = ''),
            Parameter('pressure', 'HughesGrahamCorrection', 'bool', True, required = False, export = True, hint = '')
        ]
        
    def __init__(self):
        super().__init__('densityInterpolation', 'Evaluates density at the current timestep')
    
    def initialize(self, simulationConfig, simulationState):
        self.support = simulationConfig['particle']['support']    
        self.backgroundPressure = simulationConfig['fluid']['backgroundPressure']
        self.restDensity = simulationConfig['fluid']['restDensity']
        self.gamma = simulationConfig['pressure']['gamma']
        self.kappa = simulationConfig['pressure']['kappa']
        
        self.simulationScheme = simulationConfig['simulation']['scheme']

        self.boundaryScheme = simulationConfig['simulation']['boundaryScheme']
        self.boundaryCounter = len(simulationConfig['solidBC']) if 'solidBC' in simulationConfig else 0

        self.fluidPressureScheme = simulationConfig['pressure']['fluidPressureTerm'] 
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


    def computePressure(self, simulationState, simulation):
        with record_function('pressure[EOS] - compute pressure'):
            if self.fluidPressureScheme == "TaitEOS":
                self.pressure = self.restDensity * self.c0**2 /self.gamma * (torch.pow(simulationState['fluidDensity'], self.gamma) - 1)
            if self.fluidPressureScheme == "deltaSPH":
                self.pressure = self.c0**2 * (simulationState['fluidDensity'] * self.restDensity  - self.restDensity )
            if self.fluidPressureScheme == "compressible":
                self.pressure = self.kappa * (simulationState['fluidDensity'] * self.restDensity  - self.restDensity )
            simulation.boundaryModule.computePressure(simulationState, simulation)

            simulationState['fluidPressure'] = self.pressure
            simulation.sync(simulationState['fluidPressure'])
            return self.pressure

    def resetState(self, simulationState):
        self.pressure = None
        self.pressureAccel = None
        
    def computePressureAcceleration(self, simulationState, simulation):
        with record_function('pressure[EOS] - compute pressure acceleration'):
            if self.simulationScheme != 'deltaPlus':
                self.pressureAccel = computePressureAccel(simulationState['fluidNeighbors'][0], simulationState['fluidNeighbors'][1], \
                                                                                                    simulationState['fluidPosition'], simulationState['fluidPosition'], simulationState['fluidVolume'], simulationState['fluidVolume'],\
                                                                                                    simulationState['fluidDistances'], simulationState['fluidRadialDistances'],\
                                                                                                    self.support, simulationState['fluidDensity'].shape[0], self.eps,\
                                                                                                    simulationState['fluidDensity'] * self.restDensity, simulationState['fluidDensity'] * self.restDensity, \
                                                                                                    self.pressure, self.pressure)
            else:            
                self.pressureAccel = computePressureAccelDeltaPlus(simulationState['fluidNeighbors'][0], simulationState['fluidNeighbors'][1], \
                                                                                                  simulationState['fluidPosition'], simulationState['fluidPosition'], simulationState['fluidVolume'], simulationState['fluidVolume'],\
                                                                                                  simulationState['fluidDistances'], simulationState['fluidRadialDistances'],\
                                                                                                  self.support, simulationState['fluidDensity'].shape[0], self.eps,\
                                                                                                  simulationState['fluidDensity'] * self.restDensity, simulationState['fluidDensity'] * self.restDensity, \
                                                                                                  self.pressure, self.pressure, simulationState['fluidSurfaceMask'])
            self.pressureAccel += simulation.boundaryModule.computePressureAcceleration(simulationState, simulation)

            simulationState['fluidAcceleration'] += self.pressureAccel
            simulation.sync(simulationState['fluidAcceleration'])
        
