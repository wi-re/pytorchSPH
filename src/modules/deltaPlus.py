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
from .momentum import computeDivergenceTerm
from .pressure import computePressureAccel

from src.modules.densityDiffusion import *
from src.kernels import kernel, kernelGradient, spikyGrad, wendland, wendlandGrad, cohesionKernel, getKernelFunctions
class deltaPlusModule(Module):
#     def getParameters(self):
#         return [
#             # Parameter('deltaSPH', 'alpha', 'float', 0.01, required = False, export = True, hint = ''),
#             # Parameter('deltaSPH', 'delta', 'float', 0.1, required = False, export = True, hint = ''),
#             # Parameter('deltaSPH', 'c0', 'float', -1.0, required = False, export = True, hint = ''),
#             # Parameter('deltaSPH', 'gamma', 'float', 7.0, required = False, export = True, hint = ''),
#             # Parameter('deltaSPH', 'beta', 'float', -1.0, required = False, export = True, hint = ''),
#             Parameter('deltaSPH', 'HughesGrahamCorrection', 'bool', True, required = False, export = True, hint = '')
#         ]
        
    def exportState(self, simulationState, simulation, grp, mask):  
        if simulation.config['shifting']['enabled']:
            grp.create_dataset('fluidShifting', data = simulationState['fluidUpdate'].detach().cpu().numpy())


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
        
    def computeShiftAmount(self, simulationState, simulation):
        i,j = simulationState['fluidNeighbors']
        support = self.support
        eps = support **2 * 0.1

        CFL = 1.5
        supportTerm = 4 * support**2
        Ma = torch.linalg.norm(simulationState['fluidVelocity'], dim = 1) / simulation.config['fluid']['c0']
        Ma = torch.max(torch.linalg.norm(simulationState['fluidVelocity'], dim = 1)) / simulation.config['fluid']['c0']
        k0 = kernel(simulation.config['particle']['packing'], support)
        R = 0.2
        n = 4

        kernelTerm = 1 + R * torch.pow(kernel(simulationState['fluidRadialDistances'], support) / k0, n)
        gradientTerm = kernelGradient(simulationState['fluidRadialDistances'], simulationState['fluidDistances'], support)

        phi_ij = 1

        massTerm = simulationState['fluidArea'][j] / (simulationState['fluidDensity'][i] + simulationState['fluidDensity'][j])

        term = (kernelTerm * massTerm * phi_ij )[:,None] * gradientTerm

        simulationState['shiftAmount'] = - CFL * Ma * supportTerm * scatter_sum(term, i, dim=0, dim_size = simulationState['fluidDensity'].shape[0])

        bb, bf = simulation.boundaryModule.boundaryToFluidNeighbors

        kernelTerm = 1 + R * torch.pow(kernel(simulation.boundaryModule.boundaryToFluidNeighborRadialDistances, support) / k0, n)
        gradientTerm = kernelGradient(simulation.boundaryModule.boundaryToFluidNeighborRadialDistances, -simulation.boundaryModule.boundaryToFluidNeighborDistances, support)

        phi_ij = 1

        massTerm = simulation.boundaryModule.boundaryVolume[bb] / (simulationState['fluidDensity'][bf] + simulation.boundaryModule.boundaryDensity[bb])
        term = (kernelTerm * massTerm * phi_ij )[:,None] * gradientTerm

        simulationState['shiftAmount'] += - CFL * Ma * supportTerm * scatter_sum(term, bf, dim=0, dim_size = simulationState['fluidDensity'].shape[0])
    def computeNormalizationmatrix(self, simulationState, simulation):
        support = self.support
        eps = support **2 * 0.1

        if 'fluidL' in simulationState:
            self.normalizationMatrix = simulationState['normalizationMatrix']
            self.fluidL = simulationState['fluidL']
            self.eigVals = simulationState['eigVals']
        else:
            simulationState['normalizationMatrix'] = computeNormalizationMatrix(simulationState['fluidNeighbors'][0], simulationState['fluidNeighbors'][1], \
                                                                                          simulationState['fluidPosition'], simulationState['fluidPosition'], simulationState['fluidVolume'], simulationState['fluidVolume'],\
                                                                                          simulationState['fluidDistances'], simulationState['fluidRadialDistances'],\
                                                                                          support, simulationState['fluidDensity'].shape[0], eps)     
            simulationState['normalizationMatrix'] += simulation.boundaryModule.computeNormalizationMatrices(simulationState, simulation)
            simulationState['fluidL'], simulationState['eigVals'] = pinv2x2(simulationState['normalizationMatrix'])
        simulationState['fluidLambda'] = simulationState['eigVals'][:,1]
    def computeFluidNormal(self, simulationState, simulation):
        i,j = simulationState['fluidNeighbors']
        support = self.support
        eps = support **2 * 0.1

        
        volume = simulationState['fluidArea'][j] / simulationState['fluidDensity'][j]
        factor = simulationState['fluidLambda'][j] - simulationState['fluidLambda'][i]

        kernelGrad = simulation.kernelGrad(simulationState['fluidRadialDistances'], simulationState['fluidDistances'], support)

        correctedKernel = torch.bmm(simulationState['fluidL'][i], kernelGrad[:,:,None])
        # print(correctedKernel.shape)

        term = -(volume * factor)[:,None] * correctedKernel[:,:,0]

        simulationState['lambdaGrad'] = scatter(term, i, dim=0, dim_size=simulationState['numParticles'], reduce="add")

        bb, bf = simulation.boundaryModule.boundaryToFluidNeighbors
        volume = simulationState['fluidArea'][bb] / simulationState['fluidDensity'][bb]
        factor = simulation.boundaryModule.eigVals[:,1][bb] - simulationState['fluidLambda'][bf]

        kernelGrad = kernelGradient(simulation.boundaryModule.boundaryToFluidNeighborRadialDistances, -simulation.boundaryModule.boundaryToFluidNeighborDistances, support)

        correctedKernel = torch.bmm(simulationState['fluidL'][bf], kernelGrad[:,:,None])
        # print(correctedKernel.shape)

        term = -(volume * factor)[:,None] * correctedKernel[:,:,0]

        simulationState['lambdaGrad'] += scatter(term, bf, dim=0, dim_size=simulationState['numParticles'], reduce="add")

        simulationState['fluidNormal'] = simulationState['lambdaGrad'] / (torch.linalg.norm(simulationState['lambdaGrad'],dim=1) + eps)[:,None]
        
    def detectFluidSurface(self, simulationState, simulation):
        neighbors = simulationState['fluidNeighbors']
        i = neighbors[0, neighbors[0] != neighbors[1]]
        j = neighbors[1, neighbors[0] != neighbors[1]]
        support = self.support
        eps = support **2 * 0.1


        gradients = simulationState['fluidNormal'][i]
        distances = -simulationState['fluidDistances'][neighbors[0]!=neighbors[1]]

        dotProducts = torch.einsum('nd, nd -> n', distances, gradients)
        scattered = scatter(dotProducts, i, dim = 0, dim_size = simulationState['numParticles'], reduce='max')
        simulationState['angleMin'] = torch.arccos(scatter(dotProducts, i, dim = 0, dim_size = simulationState['numParticles'], reduce='min'))
        simulationState['angleMax'] = torch.arccos(scatter(dotProducts, i, dim = 0, dim_size = simulationState['numParticles'], reduce='max'))


        bb, bf = simulation.boundaryModule.boundaryToFluidNeighbors

        dotProducts = torch.einsum('nd, nd -> n', simulation.boundaryModule.boundaryToFluidNeighborDistances, simulationState['fluidNormal'][bf])
        scattered2 = scatter(dotProducts, bf, dim = 0, dim_size = simulationState['numParticles'], reduce='max')
        #             debugPrint(scattered.shape)
        #             debugPrint(scattered2.shape)
        scattered = torch.max(scattered, scattered2)

        scattered = torch.arccos(scattered)


        #         scattered = torch.arccos(scattered)
        mask = scattered.new_zeros(scattered.shape)
        mask[ torch.logical_and(scattered > np.pi/6, simulationState['fluidLambda'] < 0.6)] = 1
        # mask[ simulationState['fluidLambda'] < 0.6] = 1
        mask2 = scatter(mask[j],i, dim=0, dim_size = mask.shape[0], reduce='max')
        mask3 = scatter(mask2[j],i, dim=0, dim_size = mask.shape[0], reduce='max')
        finalMask = scattered.new_zeros(scattered.shape)
        finalMask[mask2 > 0] = 2/3
        finalMask[mask  > 0] = 1

        zeroRegion = finalMask > 0.7
        surfRegion = torch.logical_and(simulationState['fluidLambda'] >= 0.5, finalMask > 0.)
        bulkRegion = finalMask < 0.1
        simulationState['fluidSurfaceMask'] = torch.clone(finalMask)
        simulationState['fluidSurfaceMask'][surfRegion] = 1
        simulationState['fluidSurfaceMask'][zeroRegion] = 0
        simulationState['fluidSurfaceMask'][bulkRegion] = 2
        
    def adjustShiftingAmount(self, simulationState, simulation):
        normal = simulationState['fluidNormal']
        shiftAmount = simulationState['shiftAmount']
        shiftLength = torch.linalg.norm(shiftAmount, axis = 1)
        shiftAmount[shiftLength > self.support * 0.05] = \
            shiftAmount[shiftLength > self.support * 0.05] / shiftLength[shiftLength > self.support * 0.05,None] * self.support * 0.05
        
        surfaceMask = simulationState['fluidSurfaceMask']

        normalOuter = torch.einsum('nu, nv -> nuv', normal, normal)
        idMatrix = torch.tensor([[1,0],[0,1]], dtype = normal.dtype, device = normal.device)
        normalShift = torch.matmul(idMatrix - normalOuter, shiftAmount.unsqueeze(2))[:,:,0]
        # normalShift = torch.einsum('nuv, nu -> nu',idMatrix - normalOuter, shiftAmount)
        zeroRegion = surfaceMask < 0.5
        surfRegion = surfaceMask < 1.5
        bulkRegion = surfaceMask > 1.5

        shiftAmount[surfRegion] = normalShift[surfRegion]
        shiftAmount[bulkRegion] = shiftAmount[bulkRegion]
        shiftAmount[zeroRegion] = 0

        # shiftLength = torch.linalg.norm(shiftAmount, axis = 1)

        # shiftAmount[shiftLength > self.support * 0.1] = shiftAmount[shiftLength > self.support * 0.1] / shiftLength[shiftLength > self.support * 0.1,None] * self.support * 0.1

        simulationState['fluidUpdate'] = shiftAmount


        simulationState['fluidSurfaceMask'][surfRegion] = 1
        simulationState['fluidSurfaceMask'][zeroRegion] = 0
        simulationState['fluidSurfaceMask'][bulkRegion] = 2
    def detectSurface(self, simulationState, simulation):
        self.computeNormalizationmatrix(simulationState, simulation)
        self.computeFluidNormal(simulationState, simulation)
        self.detectFluidSurface(simulationState, simulation)
        
    def shift(self, simulationState, simulation):
        if 'fluidSurfaceMask' not in simulationState:
            self.detectSurface(simulationState, simulation)
        self.computeShiftAmount(simulationState, simulation)
        self.adjustShiftingAmount(simulationState, simulation)