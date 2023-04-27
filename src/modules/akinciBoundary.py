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


class akinciBoundaryModule(BoundaryModule):
    def getParameters(self):
        return [
            Parameter('akinciBoundary', 'recomputeBoundary', 'bool', False, required = False, export = True, hint = ''),
            Parameter('akinciBoundary', 'beta', 'float', 0.125, required = False, export = True, hint = ''),
            # Parameter('akinciBoundary', 'gamma', 'float', 0.7, required = False, export = True, hint = '')
        ]
        

    def exportState(self, simulationState, simulation, grp, mask):        
        if not simulation.config['export']['staticBoundary']:
            grp.create_dataset('boundaryPosition', data = self.boundaryPositions.detach().cpu().numpy())
            grp.create_dataset('boundaryVelocity', data = self.boundaryVelocity.detach().cpu().numpy())
            grp.create_dataset('boundarySupport', data = self.boundarySupport.detach().cpu().numpy())
            grp.create_dataset('boundaryRestDensity', data = self.boundaryRestDensity.detach().cpu().numpy())
            grp.create_dataset('boundaryArea', data = self.boundaryVolume.detach().cpu().numpy())
            grp.create_dataset('boundaryNormals', data = self.boundaryNormals.detach().cpu().numpy())
            grp.create_dataset('boundaryBodyAssociation', data = self.bodyAssociation.detach().cpu().numpy())

        grp.create_dataset('boundaryDensity', data = self.boundaryDensity.detach().cpu().numpy())
        grp.create_dataset('boundaryPressure', data = self.boundaryPressure.detach().cpu().numpy())


    def saveState(self, perennialState, copy):
        perennialState['boundaryPosition']    = self.boundaryPositions    if not copy else torch.clone(self.boundaryPositions)
        perennialState['boundaryVelocity']    = self.boundaryVelocity     if not copy else torch.clone(self.boundaryVelocity)
        perennialState['boundaryDensity']     = self.boundaryDensity      if not copy else torch.clone(self.boundaryDensity)
        perennialState['boundarySupport']     = self.boundarySupport      if not copy else torch.clone(self.boundarySupport)
        perennialState['boundaryRestDensity'] = self.boundaryRestDensity  if not copy else torch.clone(self.boundaryRestDensity)
        perennialState['boundaryArea']        = self.boundaryVolume       if not copy else torch.clone(self.boundaryVolume)
        perennialState['boundaryPressure']    = self.boundaryPressure             if not copy else torch.clone(self.boundaryPressure)
        perennialState['boundaryNormals']     = self.boundaryNormals             if not copy else torch.clone(self.boundaryNormals)
        perennialState['boundaryBodyAssociation'] =  self.bodyAssociation             if not copy else torch.clone(self.bodyAssociation)
        # perennialState['boundaryPressureForce']    = self.boundaryModule.boundaryPressureForce             if not copy else torch.clone(self.boundaryModule.boundaryPressureForce)
        # perennialState['boundaryDragForce']        = self.boundaryModule.boundaryDragForce             if not copy else torch.clone(self.boundaryModule.boundaryDragForce)
        
        perennialState['boundaryParticles'] = perennialState['boundaryPosition'].shape[0]


    def __init__(self):
        super().__init__('densityInterpolation', 'Evaluates density at the current timestep')
        
    def initialize(self, simulationConfig, simulationState):
        self.support = simulationConfig['particle']['support']
        self.active = True if 'solidBC' in simulationConfig else False
        self.maxNeighbors = simulationConfig['compute']['maxNeighbors']
        self.threshold = simulationConfig['neighborSearch']['gradientThreshold']
        self.beta = simulationConfig['akinciBoundary']['beta']
        self.gamma = simulationConfig['akinciBoundary']['gamma']

        self.dtype = simulationConfig['compute']['precision']
        self.device = simulationConfig['compute']['device']  
        self.recomputeBoundary = simulationConfig['akinciBoundary']['recomputeBoundary']  
        
        self.domainMin = torch.tensor(simulationConfig['domain']['min'], device = self.device)
        self.domainMax = torch.tensor(simulationConfig['domain']['max'], device = self.device)
        self.pressureScheme = simulationConfig['pressure']['boundaryPressureTerm'] 
        self.computeBodyForces = simulationConfig['simulation']['bodyForces'] 
        self.boundaryCounter = len(simulationConfig['solidBC']) if 'solidBC' in simulationConfig else 0
        self.relaxedJacobiOmega = simulationConfig['dfsph']['relaxedJacobiOmega'] if 'dfsph'in simulationConfig else 0.5
        self.backgroundPressure = simulationConfig['fluid']['backgroundPressure']

        if not self.active:
            return
        self.numBodies = len(simulationConfig['solidBC'])
        self.boundaryObjects = simulationConfig['solidBC']
        # simulationState['akinciBoundary'] = {}
        self.bodies = simulationConfig['solidBC']
        # self.bodies'] =  simulationConfig['solidBC']
        # self.kernel, _ = getKernelFunctions(simulationConfig['kernel']['defaultKernel'])
        
        bptcls = []
        gptcls = []
        bNormals = []
        bIndices = []
        bdyCounter = 0
        for b in self.bodies:
            bdy = self.bodies[b]
            packing = simulationConfig['particle']['packing'] * simulationConfig['particle']['support']
            offset = packing / 2 if bdy['inverted'] else -packing /2
            tempPtcls = []
            tempGPtcls = []
            for i in range(1):
                cptcls, cgptcls = samplePolygon(bdy['polygon'], packing, simulationConfig['particle']['support'], offset = 2 * i * offset + offset )#packing / 2 if bdy['inverted'] else -packing /2)    
                tempPtcls.append(cptcls)
                tempGPtcls.append(cgptcls)
            ptcls = np.vstack(tempPtcls)
            ghostPtcls = np.vstack(tempGPtcls)
            # debugPrint(ptcls)
            ptcls = torch.tensor(ptcls).type(self.dtype).to(self.device)
            ghostPtcls = torch.tensor(ghostPtcls).type(self.dtype).to(self.device)
            bptcls.append(ptcls)
            gptcls.append(ghostPtcls)
            dist, grad, _, _, _, _ = sdPolyDer(bdy['polygon'], ptcls)
            bNormals.append(grad)
            bIndices.append(torch.ones(ptcls.shape[0], dtype = torch.long, device = self.device) * bdyCounter)
        # debugPrint(bptcls)
        bptcls = torch.cat(bptcls)
        gptcls = torch.cat(gptcls)
        bNormals = torch.cat(bNormals)
        self.boundaryPositions = bptcls
        self.boundaryGhostPositions = gptcls
        self.boundaryNormals = bNormals
        bdyCounter = torch.hstack(bIndices)
        self.bodyAssociation = bdyCounter
        self.numPtcls = self.boundaryPositions.shape[0]

        # self.positions']
        # boundaryPositions = self.positions']

        bj, bi = radius(self.boundaryPositions, self.boundaryPositions, self.support)

        bbDistances = (self.boundaryPositions[bi] - self.boundaryPositions[bj])
        bbRadialDistances = torch.linalg.norm(bbDistances,axis=1)

        bbDistances[bbRadialDistances < self.threshold,:] = 0
        bbDistances[bbRadialDistances >= self.threshold,:] /= bbRadialDistances[bbRadialDistances >= self.threshold,None]
        bbRadialDistances /= self.support

        self.boundaryToBoundaryNeighbors = torch.vstack((bj, bi))
        self.boundaryToBoundaryNeighborDistances = bbDistances
        self.boundaryToBoundaryNeighborRadialDistances = bbRadialDistances

        boundaryKernelTerm = kernel(bbRadialDistances, self.support)

        # gamma = 0.7
        boundaryVolume = scatter(boundaryKernelTerm, bi, dim=0, dim_size = self.numPtcls, reduce='add')
        boundaryDensity = scatter(boundaryKernelTerm * self.gamma / boundaryVolume[bj], bi, dim=0, dim_size = self.numPtcls, reduce='add')

        self.boundaryDensityTerm = (boundaryDensity).type(self.dtype)
        self.boundaryVolume = self.gamma / boundaryVolume
        self.boundarySupport = torch.ones_like(boundaryVolume) * self.support
        self.boundaryPressure = torch.zeros_like(boundaryVolume)
        self.boundaryRestDensity = torch.ones_like(boundaryVolume) * simulationConfig['fluid']['restDensity'] 
        self.boundaryVelocity = torch.zeros_like(self.boundaryPositions) 
        self.boundaryAcceleration = torch.zeros_like(self.boundaryPositions) 
        self.boundaryDensity = torch.clone(boundaryDensity).type(self.dtype)


    def dfsphPrepareSolver(self, simulationState, simulation, density = True):
        if not(density) or self.boundaryToFluidNeighbors == None:
            return 
        self.boundaryPressure = torch.zeros_like(self.boundaryVolume)
        self.boundaryPressure2 = torch.zeros_like(self.boundaryVolume)
        self.boundaryActualArea = self.boundaryVolume / self.boundaryDensity
        self.fluidPredAccel = torch.zeros(self.boundaryPositions.shape, dtype = self.dtype, device = self.device)
        self.boundaryPredictedVelocity = self.boundaryVelocity + simulationState['dt'] * self.boundaryAcceleration
        
        if self.pressureScheme == "deltaMLS":
            self.pgPartial = precomputeMLS(self.boundaryPositions, simulationState['fluidPosition'], simulationState['fluidArea'], simulationState['fluidDensity'], self.support * 2, self.ghostToFluidNeighbors, self.ghostToFluidNeighborRadialDistances)
        if self.pressureScheme == "ghostMLS":
            self.pgPartial = precomputeMLS(self.boundaryGhostPositions, simulationState['fluidPosition'], simulationState['fluidArea'], simulationState['fluidDensity'], self.support * 2, self.ghostToFluidNeighbors, self.ghostToFluidNeighborRadialDistances)
        if self.pressureScheme == "MLSPressure":
            self.M1, self.vec, self.bbar = prepareMLSBoundaries(self.boundaryPositions, self.boundarySupport, self.ghostToFluidNeighbors, self.ghostToFluidNeighborRadialDistances, simulationState['fluidPosition'], simulationState['fluidActualArea'], self.support * 2)


    def dfsphBoundaryAccelTerm(self, simulationState, simulation, density):
        if not(density) or self.boundaryToFluidNeighbors == None:
            return torch.zeros_like(simulationState['fluidPosition'])
        bb,bf = self.boundaryToFluidNeighbors
        boundaryDistances = self.boundaryToFluidNeighborDistances
        boundaryRadialDistances = self.boundaryToFluidNeighborRadialDistances
        boundaryActualArea = self.boundaryActualArea
        boundaryArea = self.boundaryVolume
        boundaryRestDensity = self.boundaryRestDensity
        boundaryPredictedVelocity = self.boundaryPredictedVelocity
        
        grad = -spikyGrad(boundaryRadialDistances, boundaryDistances, self.support)

        fac = -(boundaryArea * boundaryRestDensity)[bb]
        
        pi = (simulationState['fluidPressure2'] / (simulationState['fluidDensity'] * simulationState['fluidRestDensity'])**2)[bf]

        if self.pressureScheme == "mirrored":
            self.boundaryPressure2 = 0
            pb = simulationState['fluidPressure2'][bf]

            
        if self.pressureScheme == "deltaMLS" or self.pressureScheme == "ghostMLS":
            neighbors2 = self.ghostToFluidNeighbors
            
            self.boundaryPressure2 = scatter(self.pgPartial * simulationState['fluidPressure2'][neighbors2[1]], neighbors2[0], dim=0, dim_size = boundaryArea.shape[0], reduce='add')

            self.boundaryGravity = torch.zeros_like(self.boundaryPositions)
            self.boundaryGravity[:,1] = -1

            # self.boundaryPressure2'] += 2 * 2 * boundaryRestDensity * self.boundaryDensity'] * torch.einsum('nd, nd -> n', self.boundaryNormals'], self.boundaryGravity'])
            self.boundaryPressure2[:] = torch.clamp(self.boundaryPressure2[:],min = 0)
            self.boundaryPressure[:] = self.boundaryPressure2[:]
            pb = self.boundaryPressure2[bb]
        
        if self.pressureScheme == "MLSPressure":
            pb = evalMLSBoundaries(self.M1, self.vec, self.bbar, self.ghostToFluidNeighbors, self.ghostToFluidNeighborRadialDistances, simulationState['fluidPosition'], simulationState['fluidActualArea'], simulationState['fluidPressure'], self.support * 2)
            pb = torch.clamp(pb,min = 0)
            # self.boundaryGravity'] = torch.zeros_like(self.positions'])
            # self.boundaryGravity'][:,1] = -1
            # pb += 2 * 2 * boundaryRestDensity * self.boundaryDensity'] * torch.einsum('nd, nd -> n', self.boundaryNormals'], self.boundaryGravity'])
            self.boundaryPressure[:] = self.boundaryPressure2[:] = pb

            pb = self.boundaryPressure[bb]


        if self.pressureScheme == "PBSPH":
            self.boundaryPressure2[:] = self.boundaryPressure[:]
            pb = self.boundaryPressure[bb]

            pb =  pb / ((self.boundaryDensity[bb] * self.boundaryRestDensity[bb])**2)
            
                            # pb =  pb / ((self.boundaryDensity'] * self.boundaryRestDensity'])**2)[bb]
            term = (fac * (pi + pb))[:,None] * grad

            # debugPrint(fac)
            # debugPrint(pi)
            # debugPrint(pb)
            # debugPrint(term)
            # debugPrint(grad)

            if self.computeBodyForces:
                force = -term * (simulationState['fluidArea'] * simulationState['fluidRestDensity'])[bf,None]
                # self.bodyAssociation']
                boundaryForces = scatter_sum(force, bb, dim=0, dim_size = self.boundaryDensity.shape[0])
                self.pressureForces = scatter_sum(boundaryForces, self.bodyAssociation, dim=0, dim_size = self.boundaryCounter)

            boundaryAccelTerm = scatter_sum(term, bf, dim=0, dim_size = simulationState['fluidArea'].shape[0])

            return boundaryAccelTerm
            
        else:
            area        = self.boundaryVolume
            restDensity = self.boundaryRestDensity
            density     = self.boundaryDensity
            actualArea  = area / 1
            
            fac = -(area * restDensity)[bb]
            pf = (simulationState['fluidPressure2'] / (simulationState['fluidDensity'] * simulationState['fluidRestDensity'])**2)[bf]
#                     pb = pb /  ((simulationState['fluidDensity'][bf] * restDensity[bb])**2)
            pb = pb /  ((1 * restDensity[bb])**2)
            
            term = (fac * (pf + pb))[:,None] * grad
            
            boundaryAccelTerm = scatter_sum(term, bf, dim=0, dim_size=simulationState['fluidArea'].shape[0])

            if self.computeBodyForces:
                force = -term * (simulationState['fluidArea'] * simulationState['fluidRestDensity'])[bf,None]
                # self.bodyAssociation']
                boundaryForces = scatter_sum(force, bb, dim=0, dim_size = self.boundaryDensity.shape[0])
                self.pressureForces = scatter_sum(boundaryForces, self.bodyAssociation, dim=0, dim_size = self.boundaryCounter)

            return boundaryAccelTerm
        
    def computeVelocityDiffusion(self, simulationState, simulation):
        return torch.zeros_like(simulationState['fluidAcceleration'])

    def dfsphBoundaryPressureSum(self, simulationState, simulation, density):
        if not(density) or self.boundaryToFluidNeighbors == None:
            return torch.zeros_like(simulationState['fluidDensity'])
        bb,bf = self.boundaryToFluidNeighbors
        boundaryDistances = self.boundaryToFluidNeighborDistances
        boundaryRadialDistances = self.boundaryToFluidNeighborRadialDistances
        boundaryActualArea = self.boundaryActualArea
        boundaryArea = self.boundaryVolume
        boundaryRestDensity = self.boundaryRestDensity
        boundaryPredictedVelocity = self.boundaryPredictedVelocity
        
        grad = spikyGrad(boundaryRadialDistances, boundaryDistances, self.support)


        facFluid = simulationState['dt']**2 * simulationState['fluidActualArea'][bf]
        facBoundary = simulationState['dt']**2 * boundaryActualArea[bb]
        aij = simulationState['fluidPredAccel'][bf]

        if self.pressureScheme == 'PBSPH':
            boundaryKernelSum = scatter_sum(torch.einsum('nd, nd -> n', facFluid[:,None] * aij, -grad), bb, dim=0, dim_size=boundaryArea.shape[0])

            self.boundaryResidual = boundaryKernelSum - self.boundarySource
            boundaryPressure = self.boundaryPressure - self.relaxedJacobiOmega * self.boundaryResidual / self.boundaryAlpha
            boundaryPressure = torch.clamp(boundaryPressure, min = 0.) if density else boundaryPressure
            if density and self.backgroundPressure:
                boundaryPressure = torch.clamp(boundaryPressure, min = (5**2) * self.boundaryRestDensity)
            self.boundaryPressure = boundaryPressure
            return scatter_sum(torch.einsum('nd, nd -> n', -facBoundary[:,None] * aij, grad), bf, dim=0, dim_size=simulationState['fluidActualArea'].shape[0])
        else:
            area        = self.boundaryVolume
            restDensity = self.boundaryRestDensity
            boundaryDensity     =  self.boundaryDensity
            actualArea  = area[bb] / 1 #simulationState['fluidDensity'][bf]
            
            fac = simulationState['dt']**2 * actualArea
            return scatter_sum(torch.einsum('nd, nd -> n', fac[:,None] * aij, -grad), bf, dim=0, dim_size=simulationState['fluidActualArea'].shape[0])
            
    def dfsphBoundaryAlphaTerm(self, simulationState, simulation, density):
        placeholder1 = torch.zeros(simulationState['fluidDensity'].shape, device=self.device, dtype= self.dtype)
        placeholder2 = torch.zeros(simulationState['fluidPosition'].shape, device=self.device, dtype= self.dtype)
        if not(density) or self.boundaryToFluidNeighbors == None:
            return placeholder2, placeholder1
        bb,bf = self.boundaryToFluidNeighbors
        boundaryDistances = self.boundaryToFluidNeighborDistances
        boundaryRadialDistances = self.boundaryToFluidNeighborRadialDistances
        boundaryActualArea = self.boundaryActualArea
        boundaryArea = self.boundaryVolume
        boundaryRestDensity = self.boundaryRestDensity

        grad = spikyGrad(boundaryRadialDistances, boundaryDistances, self.support)
        grad2 = torch.einsum('nd, nd -> n', grad, grad)

        fluidActualArea = simulationState['fluidActualArea']
        fluidArea = simulationState['fluidArea']

        termFluid = (boundaryActualArea**2 / (boundaryArea * boundaryRestDensity))[bb] * grad2
        termBoundary = (simulationState['fluidActualArea']**2 / (simulationState['fluidArea'] * simulationState['fluidRestDensity']))[bf] * grad2
        if self.pressureScheme == 'PBSPH':
            kSum1 = -scatter(boundaryActualArea[bb,None] * grad, bf, dim=0, dim_size=simulationState['numParticles'],reduce='add')
            kSum2 = scatter(termFluid, bf, dim=0, dim_size=simulationState['numParticles'],reduce='add')
            self.boundaryAlpha = -simulationState['dt']**2 * boundaryActualArea * scatter(termBoundary, bb, dim=0, dim_size=boundaryArea.shape[0],reduce='add')
        else:
            area        = self.boundaryVolume
            restDensity = self.boundaryRestDensity
            density     = self.boundaryDensity
            actualArea  = area /1# density
            
            term1 = actualArea[bb][:,None] * grad
            term2 = actualArea[bb]**2 / (area * restDensity)[bb] * grad2
            
            kSum1 = scatter(term1, bf, dim=0, dim_size=simulationState['numParticles'],reduce='add')
            kSum2 = scatter(term2, bf, dim=0, dim_size=simulationState['numParticles'],reduce='add')
        return kSum1, kSum2

    def dfsphBoundarySourceTerm(self, simulationState, simulation, density):
        if not(density) or self.boundaryToFluidNeighbors == None:
            return torch.zeros_like(simulationState['fluidDensity'])
        bb,bf = self.boundaryToFluidNeighbors
        boundaryDistances = self.boundaryToFluidNeighborDistances
        boundaryRadialDistances = self.boundaryToFluidNeighborRadialDistances
        boundaryActualArea = self.boundaryActualArea
        boundaryArea = self.boundaryVolume
        boundaryRestDensity = self.boundaryRestDensity
        boundaryPredictedVelocity = self.boundaryPredictedVelocity
        
        grad = spikyGrad(boundaryRadialDistances, boundaryDistances, self.support)
        velDifference = boundaryPredictedVelocity[bb] - simulationState['fluidPredictedVelocity'][bf]
        prod = torch.einsum('nd, nd -> n',  velDifference,  grad)
        # debugPrint(simulationState['fluidPredictedVelocity'][bf,0])
        # debugPrint(simulationState['fluidPredictedVelocity'][bf,1])
        # debugPrint(prod)
        # debugPrint(grad[:,0])
        # debugPrint(grad[:,1])

        if self.pressureScheme == 'PBSPH':
            boundarySource = - simulationState['dt'] * scatter(boundaryActualArea[bb] *prod, bf, dim = 0, dim_size = simulationState['numParticles'], reduce= 'add')
            boundarySourceTerm = - simulationState['dt'] * scatter(simulationState['fluidActualArea'][bf] *prod, bb, dim = 0, dim_size = boundaryArea.shape[0], reduce= 'add')
            self.boundarySource = 1. - self.boundaryDensity + boundarySourceTerm if density else boundarySourceTerm 

        else:
            area        = self.boundaryVolume
            restDensity = self.boundaryRestDensity
            boundaryDensity     = self.boundaryDensity
            actualArea  = area / 1#boundaryDensity
            
            fac = - simulationState['dt'] * actualArea[bb]
            boundarySource = scatter(fac * prod, bf, dim = 0, dim_size = simulationState['numParticles'], reduce= 'add')
            
#                     fluidActualArea = simulationState['fluidActualArea']
        return boundarySource
    def evalBoundaryPressure(self, simulationState, simulation):
        if self.boundaryToFluidNeighbors == None:
            return 


    def evalBoundaryDensity(self, simulationState, simulation):
        density = torch.zeros(simulationState['fluidDensity'].shape, device=simulation.device, dtype= simulation.dtype)
        if self.boundaryToFluidNeighbors == None:
            return density
        if self.recomputeBoundary :    
            bj, bi = radius(self.boundaryPositions, self.boundaryPositions, self.support)

            bbDistances = (self.boundaryPositions[bi] - self.boundaryPositions[bj])
            bbRadialDistances = torch.linalg.norm(bbDistances,axis=1)
            bbRadialDistances /= self.support

            boundaryKernelTerm = kernel(bbRadialDistances, self.support)

            boundaryVolume = scatter(boundaryKernelTerm, bi, dim=0, dim_size = self.numPtcls, reduce='add')
            boundaryDensity = scatter(boundaryKernelTerm * self.gamma / boundaryVolume[bj], bi, dim=0, dim_size = self.numPtcls, reduce='add')

            self.boundaryDensityTerm = (boundaryDensity).type(self.dtype)
            self.boundaryVolume = self.gamma / boundaryVolume


        bb,bf = self.boundaryToFluidNeighbors
        k = kernel(self.boundaryToFluidNeighborRadialDistances, self.support)

        self.boundaryDensityContribution = scatter(k * self.boundaryVolume[bb], bf, dim=0, dim_size = simulationState['numParticles'], reduce = 'add')
        self.boundaryDensity = self.boundaryDensityTerm + scatter(k * simulationState['fluidArea'][bf], bb, dim=0, dim_size = self.numPtcls, reduce = 'add') + self.beta
        simulationState['fluidDensity'] += self.boundaryDensityContribution

    def evalBoundaryFriction(self, simulationState, simulation):
        raise Exception('Operation boundaryFriction not implemented for ', self.identifier)


    def boundaryNeighborhoodSearch(self, simulationState, simulation):
        if not self.active:
            return
        self.boundaryToFluidNeighbors, self.boundaryToFluidNeighborDistances, self.boundaryToFluidNeighborRadialDistances = simulation.neighborSearch.searchExisting(self.boundaryPositions, self.boundarySupport, simulationState, simulation)
        if self.pressureScheme == 'ghostMLS':
            self.ghostToFluidNeighbors, self.ghostToFluidNeighborDistances, self.ghostToFluidNeighborRadialDistances = simulation.neighborSearch.searchExisting(self.boundaryGhostPositions, self.boundarySupport * 2, simulationState, simulation)
        if self.pressureScheme == 'deltaMLS' or self.pressureScheme == 'MLSPressure':
            self.ghostToFluidNeighbors, self.ghostToFluidNeighborDistances, self.ghostToFluidNeighborRadialDistances = simulation.neighborSearch.searchExisting(self.boundaryPositions, self.boundarySupport * 2, simulationState, simulation)
    
    def boundaryFilterNeighborhoods(self, simulationState, simulation):
        return # Default behavior here is do nothing so no exception needs to be thrown

    def getNormalizationMatrices(self, simulationState, simulation):
        if not self.active:
            return torch.ones((simulationState['fluidDensity'].shape[0],2,2), dtype = self.dtype, device = self.device)
        bb,bf = self.boundaryToFluidNeighbors
        # self.boundaryNormalizationMatrix = torch.zeros((self.numPtcls, 2,2), dtype = self.dtype, device = self.device)


        fluidVolume = simulationState['fluidArea'][bf]/simulationState['fluidDensity'][bf]
        boundaryVolume = self.boundaryVolume[bb]/self.boundaryDensity[bb]

        difference = simulationState['fluidPosition'][bf] - self.boundaryPositions[bb]
        kernel = simulation.kernelGrad(self.boundaryToFluidNeighborRadialDistances, self.boundaryToFluidNeighborDistances, self.support)

        fluidTerm = boundaryVolume[:,None,None] * torch.einsum('nu,nv -> nuv', -difference, -kernel)
        boundaryTerm = fluidVolume[:,None,None] * torch.einsum('nu,nv -> nuv',  difference,  kernel)

        normalizationMatrix = scatter(fluidTerm, bf, dim=0, dim_size=simulationState['numParticles'], reduce="add")
        self.boundaryNormalizationMatrix = scatter(boundaryTerm, bb, dim=0, dim_size=self.numPtcls, reduce="add")
        
        bi, bj = self.boundaryToBoundaryNeighbors
        volume = self.boundaryVolume[bj]/self.boundaryDensity[bj]
        difference = self.boundaryPositions[bj] - self.boundaryPositions[bi]
        kernel = simulation.kernelGrad(self.boundaryToBoundaryNeighborRadialDistances, self.boundaryToBoundaryNeighborDistances, self.support)
        term = volume[:,None,None] * torch.einsum('nu,nv -> nuv',  difference,  kernel)
        self.boundaryNormalizationMatrix += -scatter(term, bi, dim=0, dim_size=self.numPtcls, reduce="add")

        return normalizationMatrix
# solidBC = solidBCModule()
# solidBC.initialize(sphSimulation.config, sphSimulation.simulationState)
# solidBC.filterFluidNeighborhoods(sphSimulation.simulationState, sphSimulation)

# solidBC = solidBCModule()
# solidBC.initialize(sphSimulation.config, sphSimulation.simulationState)
# # solidBC.filterFluidNeighborhoods(sphSimulation.simulationState, sphSimulation)
# # sphSimulation.simulationState['fluidToGhostNeighbors'], sphSimulation.simulationState['ghostToFluidNeighbors'], sphSimulation.simulationState['ghostParticleBodyAssociation'], \
# #     sphSimulation.simulationState['ghostParticlePosition'], sphSimulation.simulationState['ghostParticleDistance'], sphSimulation.simulationState['ghostParticleGradient'], \
# #     sphSimulation.simulationState['ghostParticleKernelIntegral'], sphSimulation.simulationState['ghostParticleGradientIntegral'] = solidBC.search(sphSimulation.simulationState, sphSimulation)
# solidBC.density(sphSimulation.simulationState, sphSimulation)   
        
    
# with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],with_stack=True, profile_memory=True) as prof:    
#     for i in range(16):
#         with record_function("timing"): 
# #             solidBC.filterFluidNeighborhoods(sphSimulation.simulationState, sphSimulation)
#             solidBC.density(sphSimulation.simulationState, sphSimulation)   
        
# # print(prof.key_averages().table(sort_by='self_cpu_time_total'))
# print(prof.key_averages().table(sort_by='cpu_time_total'))


# prof.export_chrome_trace("trace.json")

# simulationState = sphSimulation.simulationState
# b = simulationState['solidBC']['domainBoundary']
# # polyDist, polyDer2 = domainDistanceAndDer(sphSimulation.simulationState['fluidPosition'], torch.Tensor(sphSimulation.config['domain']['min']), torch.Tensor(sphSimulation.config['domain']['max']))
# polyDist, polyDer, polyInt, polyGrad = sdPolyDerAndIntegral(b['polygon'], simulationState['fluidPosition'], solidBC.support, inverted = b['inverted'])
# polyDist2, polyDer2, polyInt2, polyGrad2 = domainDistanceDerAndIntegral(simulationState['fluidPosition'], b['polygon'], solidBC.support)
    

# fig, axis = plt.subplots(2,2, figsize=(9 *  1.09, 9), squeeze = False)
# for axx in axis:
#     for ax in axx:
#         ax.set_xlim(sphSimulation.config['domain']['virtualMin'][0], sphSimulation.config['domain']['virtualMax'][0])
#         ax.set_ylim(sphSimulation.config['domain']['virtualMin'][1], sphSimulation.config['domain']['virtualMax'][1])
#         ax.axis('equal')
#         ax.axvline(sphSimulation.config['domain']['min'][0], ls= '--', c = 'black')
#         ax.axvline(sphSimulation.config['domain']['max'][0], ls= '--', c = 'black')
#         ax.axhline(sphSimulation.config['domain']['min'][1], ls= '--', c = 'black')
#         ax.axhline(sphSimulation.config['domain']['max'][1], ls= '--', c = 'black')



# state = sphSimulation.simulationState

# positions = state['fluidPosition'].detach().cpu().numpy()
# # data = torch.linalg.norm(state['fluidUpdate'].detach(),axis=1).cpu().numpy()
# data = polyDer[:,0].detach().cpu().numpy()


# sc = axis[0,0].scatter(positions[:,0], positions[:,1], c = polyGrad[:,0].detach().cpu().numpy(), s = 4)
# ax1_divider = make_axes_locatable(axis[0,0])
# cax1 = ax1_divider.append_axes("right", size="4%", pad="1%")
# cbar = fig.colorbar(sc, cax=cax1,orientation='vertical')
# cbar.ax.tick_params(labelsize=8) 
# sc = axis[0,1].scatter(positions[:,0], positions[:,1], c = polyGrad[:,1].detach().cpu().numpy(), s = 4)
# ax1_divider = make_axes_locatable(axis[0,1])
# cax1 = ax1_divider.append_axes("right", size="4%", pad="1%")
# cbar = fig.colorbar(sc, cax=cax1,orientation='vertical')
# cbar.ax.tick_params(labelsize=8) 
# # axis[0,0].axis('equal')

# sc = axis[1,0].scatter(positions[:,0], positions[:,1], c = polyGrad2[:,0].detach().cpu().numpy(), s = 4)
# ax1_divider = make_axes_locatable(axis[1,0])
# cax1 = ax1_divider.append_axes("right", size="4%", pad="1%")
# cbar = fig.colorbar(sc, cax=cax1,orientation='vertical')
# cbar.ax.tick_params(labelsize=8) 
# sc = axis[1,1].scatter(positions[:,0], positions[:,1], c = polyGrad2[:,1].detach().cpu().numpy(), s = 4)
# ax1_divider = make_axes_locatable(axis[1,1])
# cax1 = ax1_divider.append_axes("right", size="4%", pad="1%")
# cbar = fig.colorbar(sc, cax=cax1,orientation='vertical')
# cbar.ax.tick_params(labelsize=8) 

# fig.tight_layout()

# simulationState = sphSimulation.simulationState

# b = simulationState['solidBC']['domainBoundary']
# polyDist, polyDer, _, _, _, _ = sdPolyDer(b['polygon'], simulationState['fluidPosition'], inverted = b['inverted'])

# with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],with_stack=True, profile_memory=True) as prof:    
#     for i in range(16):
#         with record_function("old"): 
#             sdPolyDerAndIntegral(b['polygon'], simulationState['fluidPosition'], solidBC.support, inverted = b['inverted'])
#     for i in range(16):
#         with record_function("new"): 
#             sdPolyDerAndIntegral2(b['polygon'], simulationState['fluidPosition'], solidBC.support, inverted = b['inverted'])
        
        
# # print(prof.key_averages().table(sort_by='self_cpu_time_total'))
# print(prof.key_averages().table(sort_by='cpu_time_total'))


# # prof.export_chrome_trace("trace.json")

# fig, axis = plt.subplots(1,1, figsize=(9 *  1.09, 3), squeeze = False)

# x = torch.linspace(-1,1,1023)
# xt = torch.linspace(-1,1,1023)

# # axis[0,0].plot(xt,boundaryKernelAnalytic(xt,xt))
# axis[0,0].plot(x,gradK(x), label = 'gradient')
# axis[0,0].plot(x,numGradK(x,1e-2), label = 'numerical Gradient')
# axis[0,0].plot(x,k(x), label = 'kernel')
# axis[0,0].grid(True)

# axis[0,0].legend()

# fig.tight_layout()

# x = torch.linspace(-1,1,1023 * 512)
# with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],with_stack=True, profile_memory=True) as prof:    
# #     for i in range(16):
# #         with record_function("timing old"): 
# #             _ = boundaryKernelAnalytic(xt,xt)
#     for i in range(16):
#         with record_function("timing kernel"): 
#             _ = k(x)
#     for i in range(16):
#         with record_function("timing gradient"): 
#             _ = gradK(x)
#     for i in range(16):
#         with record_function("timing num gradient"): 
#             _ = numGradK(x)
        
        
# # print(prof.key_averages().table(sort_by='self_cpu_time_total'))
# print(prof.key_averages().table(sort_by='cpu_time_total'))


# # prof.export_chrome_trace("trace.json")

# @torch.jit.script
# def sdPoly(poly, p):    
#     with record_function("sdPoly"): 
#         N = len(poly)

#         i = torch.arange(N, device = p.device, dtype = torch.int64)
#         i2 = (i + 1) % N
#         e = poly[i2] - poly[i]
#         v = p - poly[i][:,None]

#         ve = torch.einsum('npd, nd -> np', v, e)
#         ee = torch.einsum('nd, nd -> n', e, e)

#         pq = v - e[:,None] * torch.clamp(ve / ee[:,None], min = 0, max = 1)[:,:,None]

#         d = torch.einsum('npd, npd -> np', pq, pq)
#         d = torch.min(d, dim = 0).values

#         wn = torch.zeros((N, p.shape[0]), device = p.device, dtype = torch.int64)

#         cond1 = 0 <= v[i,:,1]
#         cond2 = 0 >  v[i2,:,1]
#         val3 = e[i,0,None] * v[i,:,1] - e[i,1,None] * v[i,:,0]

#         c1c2 = torch.logical_and(cond1, cond2)
#         nc1nc2 = torch.logical_and(torch.logical_not(cond1), torch.logical_not(cond2))

#         wn[torch.logical_and(c1c2, val3 > 0)] += 1
#         wn[torch.logical_and(nc1nc2, val3 < 0)] -= 1

#         wn = torch.sum(wn,dim=0)
#         s = torch.ones(p.shape[0], device = p.device, dtype = p.dtype)
#         s[wn != 0] = -1

#         return s * torch.sqrt(d)

# @torch.jit.script
# def boundaryKernelAnalytic(dr : torch.Tensor , q : torch.Tensor):
#     with record_function("boundaryKernelAnalytic"): 
#         d = dr + 0j
#         a = torch.zeros(d.shape, device = q.device, dtype=d.dtype)
#         b = torch.zeros(d.shape, device = q.device, dtype=d.dtype)

        
#         mask = torch.abs(d.real) > 1e-3
#         dno = d[mask]


#         a[mask] += ( 12 * dno**5 + 80 * dno**3) * torch.log(torch.sqrt(1 - dno**2) + 1)
#         a[mask] += (-12 * dno**5 - 80 * dno**3) * torch.log(1 - torch.sqrt(1 - dno**2))
#         a[mask] += (-12 * dno**5 - 80 * dno**3) * torch.log(torch.sqrt(1 - 4 * dno**2) + 1)
#         a[mask] += ( 12 * dno**5 + 80 * dno**3) * torch.log(1 - torch.sqrt(1 - 4 * dno**2))
#         a += -13 * torch.acos(2 * d)
#         a +=  16 * torch.acos(d)
#         a += torch.sqrt(1 - 4 * d**2) * (74 * d**3 + 49 * d)
#         a += torch.sqrt(1 - d **2) * (-136 * d**3 - 64 * d)


#         b += -36 * d**5 * torch.log(torch.sqrt(1-4 * d**2) + 1)
#         b[mask] += 36 * dno**5 * torch.log(1-torch.sqrt(1-4*dno**2))
#         b += 11 * torch.acos(2 * d)
# #         b += -36 * np.log(-1 + 0j) * d**5
#         b += -160j * d**4
#         b += torch.sqrt(1 -4 *d**2)*(62 *d**3 - 33*d)
#         b += 80j * d**2
#         res = (a + b) / (14 * np.pi)

#         gammaScale = 2.0
#         gamma = 1
# #         gamma = 1 + (1 - q / 2) ** gammaScale
#         # gamma = torch.log( 1 + torch.exp(gammaScale * q)) - np.log(1 + np.exp(-gammaScale) / np.log(2))

#         return res.real * gamma

# @torch.jit.script
# def sdPolyDer(poly, p, dh :float = 1e-4, inverted :bool = False):
#     with record_function("sdPolyDer"): 
#         dh = 1e-4
#         dpx = torch.zeros_like(p)
#         dnx = torch.zeros_like(p)
#         dpy = torch.zeros_like(p)
#         dny = torch.zeros_like(p)

#         dpx[:,0] += dh
#         dnx[:,0] -= dh
#         dpy[:,1] += dh
#         dny[:,1] -= dh

#         c = sdPoly(poly, p)
#         cpx = sdPoly(poly, p + dpx)
#         cnx = sdPoly(poly, p + dnx)
#         cpy = sdPoly(poly, p + dpy)
#         cny = sdPoly(poly, p + dny)

#         if inverted:
#             c = -c
#             cpx = -cpx
#             cnx = -cnx
#             cpy = -cpy
#             cny = -cny

#         grad = torch.zeros_like(p)
#         grad[:,0] = (cpx - cnx) / (2 * dh)
#         grad[:,1] = (cpy - cny) / (2 * dh)

#         gradLen = torch.linalg.norm(grad, dim =1)
#         grad[torch.abs(gradLen) > 1e-5] /= gradLen[torch.abs(gradLen)>1e-5,None]

#         return c, grad, cpx, cnx, cpy, cny

# @torch.jit.script
# def boundaryIntegralAndDer(poly, p, support : float, c, cpx, cnx, cpy, cny, dh : float = 1e-4):
#     k = boundaryKernelAnalytic(torch.clamp(c / support, min = -1, max = 1), c / support)   
#     kpx = boundaryKernelAnalytic(torch.clamp(cpx / support, min = -1, max = 1), c / support)
#     knx = boundaryKernelAnalytic(torch.clamp(cnx / support, min = -1, max = 1), c / support)  
#     kpy = boundaryKernelAnalytic(torch.clamp(cpy / support, min = -1, max = 1), c / support)  
#     kny = boundaryKernelAnalytic(torch.clamp(cny / support, min = -1, max = 1), c / support)   
        
#     kgrad = torch.zeros_like(p)
#     kgrad[:,0] = (kpx - knx) / (2 * dh)
#     kgrad[:,1] = (kpy - kny) / (2 * dh)
    
#     return k, kgrad
    
# @torch.jit.script
# def sdPolyDerAndIntegral(poly, p, support : float, masked : bool = False, inverted : bool = False):     
#     c, grad, cpx, cnx, cpy, cny = sdPolyDer(poly, p, dh = 1e-4, inverted = inverted)
#     k, kgrad = boundaryIntegralAndDer(poly, p, support, c, cpx, cnx, cpy, cny, dh = 1e-4)  
    
    
#     return c, grad, k, kgrad


# poly = simulationState['solidBC']['domainBoundary']['polygon']
# inverted = simulationState['solidBC']['domainBoundary']['inverted']
# p = simulationState['fluidPosition']
# support = solidBC.support

# with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],with_stack=True, profile_memory=True) as prof:    
#     for i in range(16):
#         with record_function("timing"): 
#             sdPolyDerAndIntegral(poly, p, support, inverted = inverted)
# #             solidBC.filterFluidNeighborhoods(sphSimulation.simulationState, sphSimulation)
        
        
# # print(prof.key_averages().table(sort_by='self_cpu_time_total'))
# print(prof.key_averages().table(sort_by='cpu_time_total'))


# prof.export_chrome_trace("trace.json")
# solidBC = solidBCModule()
# solidBC.initialize(sphSimulation.config, sphSimulation.simulationState)
# solidBC.filterFluidNeighborhoods(sphSimulation.simulationState, sphSimulation)
# sphSimulation.simulationState['fluidToGhostNeighbors'], sphSimulation.simulationState['ghostToFluidNeighbors'], sphSimulation.simulationState['ghostParticleBodyAssociation'], \
#     sphSimulation.simulationState['ghostParticlePosition'], sphSimulation.simulationState['ghostParticleDistance'], sphSimulation.simulationState['ghostParticleGradient'], \
#     sphSimulation.simulationState['ghostParticleKernelIntegral'], sphSimulation.simulationState['ghostParticleGradientIntegral'] = solidBC.search(sphSimulation.simulationState, sphSimulation)
# solidBC.density(sphSimulation.simulationState, sphSimulation)   
        
# with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],with_stack=True, profile_memory=True) as prof:    
#     for i in range(16):
#         with record_function("timing"): 
#             solidBC.filterFluidNeighborhoods(sphSimulation.simulationState, sphSimulation)
        
        
# # print(prof.key_averages().table(sort_by='self_cpu_time_total'))
# print(prof.key_averages().table(sort_by='cpu_time_total'))


# prof.export_chrome_trace("trace.json")
# solidBC = solidBCModule()
# solidBC.initialize(sphSimulation.config, sphSimulation.simulationState)
# solidBC.filterFluidNeighborhoods(sphSimulation.simulationState, sphSimulation)
# sphSimulation.simulationState['fluidToGhostNeighbors'], sphSimulation.simulationState['ghostToFluidNeighbors'], sphSimulation.simulationState['ghostParticleBodyAssociation'], \
#     sphSimulation.simulationState['ghostParticlePosition'], sphSimulation.simulationState['ghostParticleDistance'], sphSimulation.simulationState['ghostParticleGradient'], \
#     sphSimulation.simulationState['ghostParticleKernelIntegral'], sphSimulation.simulationState['ghostParticleGradientIntegral'] = solidBC.search(sphSimulation.simulationState, sphSimulation)
# solidBC.density(sphSimulation.simulationState, sphSimulation)   
        
    
# with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],with_stack=True, profile_memory=True) as prof:    
#     for i in range(16):
#         with record_function("timing"): 
#             solidBC.filterFluidNeighborhoods(sphSimulation.simulationState, sphSimulation)
        
        
# # print(prof.key_averages().table(sort_by='self_cpu_time_total'))
# print(prof.key_averages().table(sort_by='cpu_time_total'))


# prof.export_chrome_trace("trace.json")