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
def computeNormalizationMatrix(i, j, ri, rj, Vi, Vj, distances, radialDistances, support, numParticles : int, eps : float):
    gradW = kernelGradient(radialDistances, distances, support)

    r_ba = rj[j] - ri[i]
    fac = Vj[j]

    term = torch.einsum('nu,nv -> nuv', r_ba, gradW)
    term[:,0,0] = term[:,0,0] * fac
    term[:,0,1] = term[:,0,1] * fac
    term[:,1,0] = term[:,1,0] * fac
    term[:,1,1] = term[:,1,1] * fac

    fluidNormalizationMatrix = scatter_sum(term, i, dim=0, dim_size=numParticles)
    return fluidNormalizationMatrix
@torch.jit.script 
def computeRenormalizedDensityGradient(i, j, ri, rj, Vi, Vj, distances, radialDistances, support, numParticles : int, eps : float, Li, Lj, rhoi, rhoj):
    gradW = kernelGradient(radialDistances, distances, support)
    
    rho_ba = rhoj[j] - rhoi[i] 
    grad = torch.matmul(Li[i], gradW.unsqueeze(2))[:,:,0]

    gradMagnitude = torch.linalg.norm(grad, dim=1)
    kernelMagnitude = torch.linalg.norm(gradW, dim=1)        
    change =  torch.abs(gradMagnitude - kernelMagnitude) / (kernelMagnitude + eps)
    # grad[change > 0.1,:] = gradW[change > 0.1, :]
    # grad = gradW

    renormalizedGrad = grad
    renormalizedDensityGradient = -scatter_sum((rho_ba * Vj[j] * 2)[:,None] * grad, i, dim = 0, dim_size=numParticles)
    
    return renormalizedGrad, renormalizedDensityGradient

@torch.jit.script
def computeDensityDiffusionDeltaSPH(i, j, ri, rj, Vi, Vj, distances, radialDistances, support, numParticles : int, eps : float, gradRhoi, gradRhoj, rhoi, rhoj, delta : float, c0 : float):
    gradW = kernelGradient(radialDistances, distances, support)
    rji = rj[j] - ri[i]
    rji2 = torch.linalg.norm(rji, dim=1)**2 + eps

    psi_ij = (2 * (rhoj[j] - rhoi[i]) / rji2)[:,None] * rji - (gradRhoi[i] + gradRhoj[j])
    prod = torch.einsum('nu,nu -> n', psi_ij, gradW) 
    return support * delta * c0 * scatter_sum(prod * Vj[j], i, dim=0, dim_size = numParticles)


@torch.jit.script
def computeDensityDiffusionMOG(i, j, ri, rj, Vi, Vj, distances, radialDistances, support, numParticles : int, eps : float, rhoi, rhoj, delta : float, c0 : float):
    gradW = kernelGradient(radialDistances, distances, support)
    rji = rj[j] - ri[i]
    rji2 = torch.linalg.norm(rji, dim=1)**2 + eps

    psi_ij = (2 * (rhoj[j] - rhoi[i]) / rji2)[:,None] * rji
    prod = torch.einsum('nu,nu -> n', psi_ij, gradW) 
    return support * delta * c0 * scatter_sum(prod * Vj[j], i, dim=0, dim_size = numParticles)



class densityDiffusionModule(Module):
    def getParameters(self):
        return [
            # Parameter('deltaSPH', 'alpha', 'float', 0.01, required = False, export = True, hint = ''),
            # Parameter('deltaSPH', 'delta', 'float', 0.1, required = False, export = True, hint = ''),
            # Parameter('deltaSPH', 'c0', 'float', -1.0, required = False, export = True, hint = ''),
            # Parameter('deltaSPH', 'gamma', 'float', 7.0, required = False, export = True, hint = ''),
            # Parameter('deltaSPH', 'beta', 'float', -1.0, required = False, export = True, hint = ''),
            Parameter('deltaSPH', 'HughesGrahamCorrection', 'bool', True, required = False, export = True, hint = '')
        ]
        
    def __init__(self):
        super().__init__('densityInterpolation', 'Evaluates density at the current timestep')
    
    def initialize(self, simulationConfig, simulationState):
        self.support = simulationConfig['particle']['support']    
        self.backgroundPressure = simulationConfig['fluid']['backgroundPressure']
        self.restDensity = simulationConfig['fluid']['restDensity']
        self.gamma = simulationConfig['pressure']['gamma']
        
        self.boundaryScheme = simulationConfig['simulation']['boundaryScheme']
        self.boundaryCounter = len(simulationConfig['solidBC']) if 'solidBC' in simulationConfig else 0

        self.pressureScheme = simulationConfig['pressure']['fluidPressureTerm'] 
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
        self.scheme = simulationConfig['diffusion']['densityScheme']

    def resetState(self, simulationState):
        self.normalizationMatrix = None
        self.fluidL = None
        self.eigVals = None
        self.densityDiffusion = None

    def computeNormalizationMatrices(self, simulationState, simulation):
        with record_function('density[diffusion] - compute normalization matrices'):
            self.normalizationMatrix = computeNormalizationMatrix(simulationState['fluidNeighbors'][0], simulationState['fluidNeighbors'][1], \
                                                                                                  simulationState['fluidPosition'], simulationState['fluidPosition'], simulationState['fluidVolume'], simulationState['fluidVolume'],\
                                                                                                  simulationState['fluidDistances'], simulationState['fluidRadialDistances'],\
                                                                                                  self.support, simulationState['fluidDensity'].shape[0], self.eps)     
            self.normalizationMatrix += simulation.boundaryModule.computeNormalizationMatrices(simulationState, simulation)
            self.fluidL, self.eigVals = pinv2x2(self.normalizationMatrix)
    def computeRenormalizedDensityGradient(self, simulationState, simulation):
        with record_function('density[diffusion] - compute renormalized density gradient'):
            self.renormalizedGrad, self.renormalizedDensityGradient = computeRenormalizedDensityGradient(simulationState['fluidNeighbors'][0], simulationState['fluidNeighbors'][1], \
                                                                                                  simulationState['fluidPosition'], simulationState['fluidPosition'], simulationState['fluidVolume'], simulationState['fluidVolume'],\
                                                                                                  simulationState['fluidDistances'], simulationState['fluidRadialDistances'],\
                                                                                                  self.support, simulationState['fluidDensity'].shape[0], self.eps,\
                                                                                                  self.fluidL, self.fluidL, simulationState['fluidDensity'] * self.restDensity,\
                                                                                                  simulationState['fluidDensity'] * self.restDensity)     
            self.renormalizedDensityGradient  += simulation.boundaryModule.computeRenormalizedDensityGradient(simulationState, simulation)
  
    def computeDensityDiffusion(self, simulationState, simulation):
        with record_function('density[diffusion] - compute density diffusion'):
            if self.scheme == 'deltaSPH':
                if 'fluidL' in simulationState:
                    self.normalizationMatrix = simulationState['normalizationMatrix']
                    self.fluidL = simulationState['fluidL']
                    self.eigVals = simulationState['eigVals']
                else:
                    self.computeNormalizationMatrices(simulationState, simulation)
                self.computeRenormalizedDensityGradient(simulationState, simulation)
                self.densityDiffusion = computeDensityDiffusionDeltaSPH(simulationState['fluidNeighbors'][0], simulationState['fluidNeighbors'][1], \
                                                                                                  simulationState['fluidPosition'], simulationState['fluidPosition'], simulationState['fluidVolume'], simulationState['fluidVolume'],\
                                                                                                  simulationState['fluidDistances'], simulationState['fluidRadialDistances'],\
                                                                                                  self.support, simulationState['fluidDensity'].shape[0], self.eps,\
                                                                                                  self.renormalizedDensityGradient, self.renormalizedDensityGradient, \
                                                                                                  simulationState['fluidDensity'] * self.restDensity,simulationState['fluidDensity'] * self.restDensity,\
                                                                                                  self.delta, self.c0)
                simulationState['dpdt'] += self.densityDiffusion
            elif self.scheme == 'MOG':
                self.densityDiffusion = computeDensityDiffusionMOG(simulationState['fluidNeighbors'][0], simulationState['fluidNeighbors'][1], \
                                                                                                  simulationState['fluidPosition'], simulationState['fluidPosition'], simulationState['fluidVolume'], simulationState['fluidVolume'],\
                                                                                                  simulationState['fluidDistances'], simulationState['fluidRadialDistances'],\
                                                                                                  self.support, simulationState['fluidDensity'].shape[0], self.eps,\
                                                                                                  simulationState['fluidDensity'] * self.restDensity,simulationState['fluidDensity'] * self.restDensity,\
                                                                                                  self.delta, self.c0)
                simulationState['dpdt'] += self.densityDiffusion
            simulation.sync(simulationState['dpdt'])

            # self.densityDiffusion += simulation.boundaryModule.computeDensityDiffusion(simulationState, simulation)