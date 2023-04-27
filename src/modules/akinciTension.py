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

@torch.jit.script
def cohesionKernel(q, h):
    res = q.new_zeros(q.shape)
    Cd = -1 / (2 * np.pi) * 1 / h**2
    k1 = 128 * (q-1)**3 * q**3 + 1
    k2 = 64 * (q-1)**3 * q**3
    
    res[q <= 0.5] = k1[q<=0.5]
    res[q > 0.5] = k2[q>0.5]
    
    return -res

@torch.jit.script
def computeNormalFunction(neighbors, fluidAreas, fluidDensities, fluidRadialDistances, fluidDistances, support):
    i = neighbors[0]
    j = neighbors[1]            

    fac = fluidAreas[j] / fluidDensities[j]

    grad = kernelGradient(fluidRadialDistances, fluidDistances, support)

    term = (fac)[:,None] * grad
    normals = support * scatter_sum(term, i, dim=0, dim_size=fluidAreas.shape[0])
    
    return normals
    
@torch.jit.script
def computeCurvatureForce(neighbors, normals, gamma):
    i = neighbors[0]
    j = neighbors[1]      
    fac = normals[i] - normals[j] 
    curvatureForce = -gamma * scatter_sum(fac, i, dim=0, dim_size=normals.shape[0])     
    return curvatureForce

@torch.jit.script
def computeCohesionForce(neighbors, fluidAreas, fluidRestDensity, gamma, fluidRadialDistances, fluidDistances, support):
    i = neighbors[0]
    j = neighbors[1]      
    
    fac = fluidAreas[j] * fluidRestDensity[j]

    kernel = cohesionKernel(fluidRadialDistances, support)

    term = (fac * kernel)[:,None] * fluidDistances

    normals = -gamma * scatter_sum(term, i, dim=0, dim_size=fluidAreas.shape[0])
    return normals


class akinciTensionModule(Module):
    def __init__(self):
        super().__init__('densityInterpolation', 'Evaluates density at the current timestep')
    
    def getParameters(self):
        return [
            Parameter('surfaceTension', 'gamma', 'float', 1, required = False, export = True, hint = '')
        ]
    
    def resetState(self, simulationState):
        self.normals =  None
        self.curvatureForceVal = None
        self.cohesionForceVal = None

    def initialize(self, simulationConfig, simulationState):
        self.support = simulationConfig['particle']['support']
        self.dtype = simulationConfig['compute']['precision']
        self.device = simulationConfig['compute']['device']  
        self.gamma = simulationConfig['surfaceTension']['gamma']
    
    def computeNormals(self, simulationState, simulation):
        with record_function("surface Tension[akinci] - computeNormals"): 
            self.normals = computeNormalFunction(simulationState['fluidNeighbors'], simulationState['fluidArea'], simulationState['fluidDensity'], simulationState['fluidRadialDistances'], simulationState['fluidDistances'], self.support)
        
    def curvatureForce(self, simulationState, simulation):   
        with record_function("surface Tension[akinci] - curvatureForce"):      
            self.curvatureForceVal = computeCurvatureForce(simulationState['fluidNeighbors'], self.normals, self.gamma)
            simulationState['fluidAcceleration'] += self.curvatureForceVal
            simulation.sync(simulationState['fluidAcceleration'])
        
    def cohesionForce(self, simulationState, simulation):
        with record_function("surface Tension[akinci] - cohesionForce"): 
            self.cohesionForceVal = computeCohesionForce(simulationState['fluidNeighbors'], simulationState['fluidArea'], simulationState['fluidRestDensity'], self.gamma, simulationState['fluidRadialDistances'], simulationState['fluidDistances'], self.support)
            simulationState['fluidAcceleration'] += self.cohesionForceVal
            simulation.sync(simulationState['fluidAcceleration'])


    
# surfaceTension = akinciTensionModule()
# surfaceTension.initialize(sphSimulation.config, sphSimulation.simulationState)
# sphSimulation.simulationState['fluidNormals'] = surfaceTension.computeNormals(sphSimulation.simulationState, sphSimulation)
# sphSimulation.simulationState['curvatureFurce'] = surfaceTension.curvatureForce(sphSimulation.simulationState, sphSimulation)
# sphSimulation.simulationState['cohesionForce'] = surfaceTension.cohesionForce(sphSimulation.simulationState, sphSimulation)

# fig, axis = sphSimulation.createPlot(plotScale = 2)

# positions = sphSimulation.simulationState['shiftedPositions'].detach().cpu().numpy()

# # data = sphSimulation.simulationState['fluidPosition'] - initialPositions
# data = sphSimulation.simulationState['fluidUpdate']
# colors = torch.linalg.norm(data.detach(),axis=1).cpu().numpy()
# colors = sphSimulation.simulationState['fluidLambda'].detach().cpu().numpy()
# colors = sphSimulation.simulationState['fluidSurfaceMask'].detach().cpu().numpy()
# # colors = sphSimulation.simulationState['angleMax'].detach().cpu().numpy()


# sc = axis[0,0].scatter(positions[:,0], positions[:,1], c = colors, s = 32)
# axis[0,0].axis('equal')

# quiverData = sphSimulation.simulationState['cohesionForce'].detach().cpu().numpy()
# qv = axis[0,0].quiver(positions[:,0], positions[:,1], quiverData[:,0], quiverData[:,1], \
#                       scale_units='xy', scale = 10) #scale = 2/ sphSimulation.config['particle']['support'], alpha=0.5)
# #                       scale_units='xy', scale = 2/ sphSimulation.config['particle']['support'], alpha=0.5)

# ax1_divider = make_axes_locatable(axis[0,0])
# cax1 = ax1_divider.append_axes("right", size="4%", pad="1%")
# cbar = fig.colorbar(sc, cax=cax1,orientation='vertical')
# cbar.ax.tick_params(labelsize=8) 

# fig.tight_layout()


# fig, axis = plt.subplots(1, 1, figsize=(9,6), sharex = False, sharey = False, squeeze = False)

# x = torch.linspace(0,1,127)
# axis[0,0].plot(x, cohesionKernel(x,1))
# # axis[0,0].axhline(0)
# axis[0,0].grid(True)
    