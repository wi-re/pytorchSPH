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
def LinearCG(H, B, x0, i, j, tol=1e-5, verbose = False):    
    xk = x0
    rk = torch.zeros_like(x0)
    numParticles = rk.shape[0] // 2

    rk[::2]  += scatter(H[:,0,0] * xk[j * 2], i, dim=0, dim_size=numParticles, reduce= "add")
    rk[::2]  += scatter(H[:,0,1] * xk[j * 2 + 1], i, dim=0, dim_size=numParticles, reduce= "add")

    rk[1::2] += scatter(H[:,1,0] * xk[j * 2], i, dim=0, dim_size=numParticles, reduce= "add")
    rk[1::2] += scatter(H[:,1,1] * xk[j * 2 + 1], i, dim=0, dim_size=numParticles, reduce= "add")
    
    rk = rk - B
    
    pk = -rk
    rk_norm = torch.linalg.norm(rk)
    
    num_iter = 0

    if verbose:
        print('xk: ', x0)
        print('rk: ', rk)
        print('|rk|: ', rk_norm)
        print('pk: ', pk)


    while rk_norm > tol and num_iter < 32:
        apk = torch.zeros_like(x0)

        apk[::2]  += scatter(H[:,0,0] * pk[j * 2], i, dim=0, dim_size=numParticles, reduce= "add")
        apk[::2]  += scatter(H[:,0,1] * pk[j * 2 + 1], i, dim=0, dim_size=numParticles, reduce= "add")

        apk[1::2] += scatter(H[:,1,0] * pk[j * 2], i, dim=0, dim_size=numParticles, reduce= "add")
        apk[1::2] += scatter(H[:,1,1] * pk[j * 2 + 1], i, dim=0, dim_size=numParticles, reduce= "add")

        rkrk = torch.dot(rk, rk)
        
        alpha = rkrk / torch.dot(pk, apk)
        xk = xk + alpha * pk
        rk = rk + alpha * apk
        beta = torch.dot(rk, rk) / rkrk
        pk = -rk + beta * pk
        
        num_iter += 1

        rk_norm = torch.linalg.norm(rk)
        if verbose:
            print('iter: ', num_iter)
            print('\t|rk|: ', rk_norm)
            print('\talpha: ', alpha)
            
    return xk

@torch.jit.script
def x2c2(dInput, cr):
    zeroTensor = dInput.new_zeros(dInput.shape)
    dr = torch.abs(dInput)    
    dr[torch.abs(dr) < 1e-5] = 1
    
    d = torch.complex(dr, zeroTensor)
    c = torch.complex(cr, zeroTensor)
    
    a1 = ((-210*torch.cos(2*c)-75)*d**7+(-1260*torch.cos(2*c)-378)*d**5)*torch.log(torch.sqrt(1-d**2)+1)
    a2 = (75*d**7+378*d**5)*torch.log(torch.sqrt(1-d**2)-1)
    a3 = (210*torch.cos(2*c)*d**7+1260*torch.cos(2*c)*d**5)*torch.log(d)
    a4 = -24*torch.arccos(d)
    a5 = 0. #75.j*torch.arctan2(zeroTensor,torch.sqrt(1-dr**2)-1)*d**7
    a6 = torch.sqrt(1-d**2)*((1134*torch.cos(2*c)+746)*d**5+(392*torch.cos(2*c)+152)*d**3+(32-56*torch.cos(2*c))*d)
    a7 = 0. #378.j*torch.arctan2(zeroTensor,torch.sqrt(1-dr**2)-1)*d**5
    term = -(a1 + a2 + a3 + a4 + a5 + a6 + a7) /(24*np.pi)
    
    term[dInput < 0.] = 1 - term[dInput<0.]
    term[torch.abs(dInput) < 1e-5] = 1/2

    return term.real

@torch.jit.script
def x2cs(dInput, cr):
    dr = torch.abs(dInput)    
    dr[torch.abs(dr) < 1e-5] = 1
    
    d = dr + 0.j
    c = cr + 0.j
    
    a1 = (-105*torch.sin(2*c)*d**7-630*torch.sin(2*c)*d**5)*torch.log(torch.sqrt(1-d**2)+1)
    a2 = (105*torch.sin(2*c)*d**7+630*torch.sin(2*c)*d**5)*torch.log(d)
    a3 = torch.sqrt(1-d**2)*(567*torch.sin(2*c)*d**5+196*torch.sin(2*c)*d**3-28*torch.sin(2*c)*d)
    term = -(a1 + a2 + a3) /(12*np.pi)
    
    term[dInput < 0.] = 0 - term[dInput<0.]
    term[torch.abs(dInput) < 1e-5] = 0

    return term.real
@torch.jit.script
def x2s2(dInput, cr):
    zeroTensor = dInput.new_zeros(dInput.shape)
    dr = torch.abs(dInput)    
    dr[torch.abs(dr) < 1e-5] = 1
    
    d = dr + 0.j
    c = cr + 0.j
    
    a1 = ((75-210*torch.cos(2*c))*d**7+(378-1260*torch.cos(2*c))*d**5)*torch.log(torch.sqrt(1-d**2)+1)
    a2 = (-75*d**7-378*d**5)*torch.log(torch.sqrt(1-d**2)-1)
    a3 = (210*torch.cos(2*c)*d**7+1260*torch.cos(2*c)*d**5)*torch.log(d)
    a4 = 24*torch.arccos(d)
    a5 = 0. #-75.j*torch.arctan2(zeroTensor,torch.sqrt(1-d**2).real-1)*d**7
    a6 = torch.sqrt(1-d**2)*((1134*torch.cos(2*c)-746)*d**5+(392*torch.cos(2*c)-152)*d**3+(-56*torch.cos(2*c)-32)*d)
    a7 = 0. #-378.j*torch.arctan2(zeroTensor,torch.sqrt(1-dr**2)-1)*d**5
    term = (a1 + a2 + a3 + a4 + a5 + a6 + a7) /(24*np.pi)

    
    term[dInput < 0.] = 1 - term[dInput<0.]
    term[torch.abs(dInput) < 1e-5] = 1/2

    return term.real

def getCorrelationMatrix(distance, direction):
    angle = torch.atan2(direction[:,1], direction[:,0])    
    c2Term = x2c2(distance, angle)
    s2Term = x2s2(distance, angle)
    csTerm = x2cs(distance, angle)
    M = torch.stack([torch.stack([c2Term,csTerm]),torch.stack([csTerm,s2Term])]).transpose(0,-1).transpose(1,-1)
    return M


class implicitIterativeShiftModule(Module):
    def __init__(self):
        super().__init__('densityInterpolation', 'Evaluates density at the current timestep')
       
    def getParameters(self):
        return [
            Parameter('iishifting', 'enabled', 'bool', True, required = False, export = True, hint = ''),
            Parameter('iishifting', 'shiftIterations', 'int', 4, required = False, export = True, hint = ''),
            Parameter('iishifting', 'densityThreshold', 'float', 0.9, required = False, export = True, hint = '')
        ]
    
    def initialize(self, simulationConfig, simulationState):
        self.support = simulationConfig['particle']['support']
        self.active = simulationConfig['iishifting']['enabled']
        self.iterations = simulationConfig['iishifting']['shiftIterations']
        self.area = simulationConfig['particle']['area']
        
        self.dtype = simulationConfig['compute']['precision']
        self.device = simulationConfig['compute']['device']
        self.threshold = simulationConfig['iishifting']['densityThreshold']
        
        
    def solveShifting(self, simulationState, simulation, verbose = False):
    
        with record_function("sph - xsph correction"): 
            neighbors = simulationState['fluidNeighbors']
            i = neighbors[0]
            j = neighbors[1]

            fac = simulationState['fluidArea'][j] / simulationState['fluidDensity'][j]
            
            grad = wendlandGrad(simulationState['fluidRadialDistances'], simulationState['fluidDistances'], self.support)
            
            term = (fac)[:,None] * grad

            normals = self.support * scatter(term, i, dim=0, dim_size=simulationState['numParticles'], reduce="add")
#             syncQuantity(correction, config, simulationState)



        simulationState['fluidOmegas'] = self.area / simulationState['fluidDensity']
        simulation.periodicBC.syncQuantity(simulationState['fluidOmegas'], simulationState, simulation)

        K, J, H = evalKernel(simulationState['fluidOmegas'], simulationState['fluidPosition'], simulationState['fluidNeighbors'], simulationState['fluidDistances'], simulationState['fluidRadialDistances'], simulationState['numParticles'], self.support)

        JJ = scatter(J, simulationState['fluidNeighbors'][0], dim=0, dim_size=simulationState['numParticles'], reduce= "add")
        
        
        

#         if 'fluidToGhostNeighbors' in simulationState and simulationState['fluidToGhostNeighbors'] != None:
#             debugPrint(simulationState['boundaryGradient'])
#             JJ -= scatter(simulationState['boundaryGradient'], simulationState['fluidToGhostNeighbors'][0], dim=0, dim_size=simulationState['numParticles'],reduce='add')

#             source = source - simulationState['dt'] * torch.einsum('nd, nd -> n',  simulationState['fluidPredictedVelocity'],  boundaryTerm)

#         JJ -= simulationState['boundaryGradient']
    
#         JJ[simulationState['fluidSurfaceMask'] < 1.5] += normals[simulationState['fluidSurfaceMask'] < 1.5]
#         debugPrint(JJ)
#         debugPrint(normals)
    
#         debugPrint(JJ.shape)
#         debugPrint(normals.shape)


        surfNorm = torch.linalg.norm(normals, axis = 1)**2
        proj = torch.einsum('nd, nd -> n', JJ, normals)[:,None] * normals  
        surfMask = torch.logical_and(surfNorm > 1e-7, simulationState['fluidSurfaceMask'] < 1.5)
        JJ[surfMask] -= proj[surfMask] / surfNorm[surfMask,None]
        
        JJ[simulationState['fluidSurfaceMask'] < 0.5] = 0
    
        normals = simulationState['boundaryGradient']
        surfNorm = torch.linalg.norm(normals, axis = 1)**2
        proj = torch.einsum('nd, nd -> n', JJ, normals)[:,None] * normals  
        surfMask = surfNorm > 1e-7
        JJ[surfMask] -= proj[surfMask] / surfNorm[surfMask,None]
        
        JJ[simulationState['fluidSurfaceMask'] < 0.5] = 0
        
    
    
#         debugPrint(JJ.shape)
#         debugPrint(normals.shape)
#         debugPrint(simulationState['fluidSurfaceMask'].shape)
#         debugPrint(simulationState['fluidSurfaceMask'])
#         debugPrint(JJ[simulationState['fluidSurfaceMask'] < 1.5].shape)
#         debugPrint(normals)
#         JJ[simulationState['fluidSurfaceMask'] < 1] = 0
    
        simulation.periodicBC.syncQuantity(JJ, simulationState, simulation)


        B = torch.zeros(JJ.shape[0]*2, device = JJ.device, dtype=JJ.dtype)
        B[::2] = JJ[:,0]
        B[1::2] = JJ[:,1]


        i = simulationState['fluidNeighbors'][0]
        j = simulationState['fluidNeighbors'][1]

        x0 = torch.rand(simulationState['numParticles'] * 2).to(self.device).type(self.dtype) * self.support / 4
        diff = LinearCG(H, B, x0, i, j, verbose = verbose)

        dx = torch.zeros(J.shape[0], device = J.device, dtype=J.dtype)
        dy = torch.zeros(J.shape[0], device = J.device, dtype=J.dtype)
        dx = -diff[::2]
        dy = -diff[1::2]

        update = torch.vstack((dx,dy)).T
#         syncQuantity(update, config, state)
        return update
#         state['fluidUpdate'] = update

    def computeNormalizationMatrix(self, simulationState, simulation):
    #     global normaliza
        neighbors = simulationState['fluidNeighbors']
        i = neighbors[0]
        j = neighbors[1]
        volume = simulationState['fluidArea'][j]/simulationState['fluidDensity'][j]

        difference = simulationState['fluidPosition'][j] - simulationState['fluidPosition'][i]
        kernel = simulation.kernelGrad(simulationState['fluidRadialDistances'], simulationState['fluidDistances'], self.support)

        term = volume[:,None,None] * torch.einsum('nu,nv -> nuv', difference, kernel)

        normalizationMatrix = scatter(term, i, dim=0, dim_size=simulationState['numParticles'], reduce="add")

        if 'fluidToGhostNeighbors' in simulationState and simulationState['fluidToGhostNeighbors'] != None:
            neighbors = simulationState['fluidToGhostNeighbors']
            i = neighbors[0]
            b = neighbors[1]
            boundaryMatrices = getCorrelationMatrix(simulationState['ghostParticleDistance'], simulationState['ghostParticleGradient']) 
#             boundaryMatrices = boundaryMatrices * 2
            
            
            normalizationMatrix += scatter(boundaryMatrices,i, dim=0, dim_size=simulationState['numParticles'], reduce="add")
            L = torch.linalg.pinv(normalizationMatrix)

            ghostToFluidNeighbors = simulationState['ghostToFluidNeighbors']
            bf = ghostToFluidNeighbors[0]
            bb = ghostToFluidNeighbors[1]

            volume = simulationState['fluidArea'][bf]/simulationState['fluidDensity'][bf]
            difference = simulationState['fluidPosition'][bf] - simulationState['ghostParticlePosition'][bb]

            radialDistance = torch.linalg.norm(difference, axis = 1)
            distance = torch.clone(difference)

    #         debugPrint(radialDistance)
    #         debugPrint(distance)

    #         return L, normalizationMatrix, None, None


            distance[radialDistance > 1e-7] = distance[radialDistance > 1e-7] / radialDistance[radialDistance > 1e-7,None]
            kernel = simulation.kernelGrad(radialDistance, -distance, self.support)

            term = volume[:,None,None] * torch.einsum('nu,nv -> nuv', difference, kernel)

            boundaryM = scatter(term, bb, dim=0, dim_size=simulationState['ghostParticlePosition'].shape[0], reduce="add")
            boundaryM[:,0,0] += 1/2 #* 0.5
            boundaryM[:,1,1] += 1/2 #* 0.5
            
#             boundaryM = boundaryM * 2
#             debugPrint(boundaryM)
            boundaryL = torch.linalg.pinv(boundaryM)

            return L, normalizationMatrix, boundaryL, boundaryM

        else:
            L = torch.linalg.pinv(normalizationMatrix)
            return L, normalizationMatrix, None, None

    def computeNormal(self, simulationState, simulation):
        neighbors = simulationState['fluidNeighbors']
        i = neighbors[0]
        j = neighbors[1]
        volume = simulationState['fluidArea'][j]/simulationState['fluidDensity'][j]
        factor = simulationState['fluidLambda'][j] - simulationState['fluidLambda'][i]

    #     print(factor)

        kernel = simulation.kernelGrad(simulationState['fluidRadialDistances'], simulationState['fluidDistances'], self.support)

        correctedKernel = torch.bmm(simulationState['fluidL'][j], kernel[:,:,None])
        # print(correctedKernel.shape)

        term = -(volume * factor)[:,None] * correctedKernel[:,:,0]

        lambdaGrad = scatter(term, i, dim=0, dim_size=simulationState['numParticles'], reduce="add")

        if 'fluidToGhostNeighbors' in simulationState and simulationState['fluidToGhostNeighbors'] != None:
            neighbors = simulationState['fluidToGhostNeighbors']
            i = neighbors[0]
            b = neighbors[1]

            kernel = simulationState['ghostParticleGradientIntegral']
            correctedKernel = torch.bmm(simulationState['boundaryL'], kernel[:,:,None])
            factor = simulationState['boundaryLambda'] - simulationState['fluidLambda'][i]
            term = - factor[:,None] * correctedKernel[:,:,0]
    #         debugPrint(term)
            lambdaGrad += scatter(term, i, dim=0, dim_size=simulationState['numParticles'], reduce="add")

        lambdaGradNorm = torch.linalg.norm(lambdaGrad, axis=1)
        mask = lambdaGradNorm > 1e-3
        lambdaGrad[mask,:] = lambdaGrad[mask] / lambdaGradNorm[mask,None]

        return lambdaGrad

    def detectSurface(self, simulationState, simulation):    
        neighbors = simulationState['fluidNeighbors']
        i = neighbors[0, neighbors[0] != neighbors[1]]
        j = neighbors[1, neighbors[0] != neighbors[1]]

        gradients = simulationState['lambdaGrad'][i]
        distances = -simulationState['fluidDistances'][neighbors[0]!=neighbors[1]]

    #     debugPrint(i)
    #     debugPrint(j)
    #     debugPrint(gradients[i==0])
    #     debugPrint(distances[i==0])
    #     debugPrint(simulationState['fluidPosition'][j[i == 0]])
    #     debugPrint(simulationState['fluidPosition'][i[i == 0]])

        dotProducts = torch.einsum('nd, nd -> n', distances, gradients)
    #     debugPrint(dotProducts[i==0])
    #     dotProducts = torch.abs(dotProducts)

        scattered = scatter(dotProducts, i, dim = 0, dim_size = simulationState['numParticles'], reduce='max')
        simulationState['angleMin'] = torch.arccos(scatter(dotProducts, i, dim = 0, dim_size = simulationState['numParticles'], reduce='min'))
        simulationState['angleMax'] = torch.arccos(scatter(dotProducts, i, dim = 0, dim_size = simulationState['numParticles'], reduce='max'))

        
        if 'fluidToGhostNeighbors' in simulationState and simulationState['fluidToGhostNeighbors'] != None:
            neighbors = simulationState['fluidToGhostNeighbors']
            bf = neighbors[0]
            bb = neighbors[1]
            
            dotProducts = torch.einsum('nd, nd -> n', -simulation.simulationState['ghostParticleGradient'], simulationState['lambdaGrad'][bf])
            scattered2 = scatter(dotProducts, bf, dim = 0, dim_size = simulationState['numParticles'], reduce='max')
#             debugPrint(scattered.shape)
#             debugPrint(scattered2.shape)
#             scattered = torch.max(scattered, scattered2)
            
            

        scattered = torch.arccos(scattered)
        mask = scattered.new_zeros(scattered.shape)
        mask[ torch.logical_and(scattered > np.pi/6, simulationState['fluidLambda'] < 0.8)] = 1
#         mask[ scattered > np.pi/6] = 1

    
#         return mask
    #     i = neighbors[1]
    #     j = neighbors[0]

        mask2 = scatter(mask[j],i, dim=0, dim_size = mask.shape[0], reduce='max')

        mask3 = scatter(mask2[j],i, dim=0, dim_size = mask.shape[0], reduce='max')

        finalMask = scattered.new_zeros(scattered.shape)

#         finalMask[mask3 > 0] = 1/3
        finalMask[mask2 > 0] = 2/3
        finalMask[mask  > 0] = 1

        return finalMask

    def computeNormals(self, simulationState, simulation):      
        L, M, boundaryL, boundaryM = self.computeNormalizationMatrix(simulationState, simulation)

        simulationState['fluidL'] = L
        simulationState['boundaryL'] = boundaryL

        lambdas = torch.real(torch.linalg.eigvals(M))
        minEV = torch.min(lambdas, axis=1)[0]
        simulationState['fluidLambda'] = minEV

        if boundaryL is not None:
            boundaryLambdas = torch.real(torch.linalg.eigvals(boundaryM))
            boundaryMinEV = torch.min(boundaryLambdas, axis=1)[0]
            simulationState['boundaryLambda'] = boundaryMinEV

        lambdaGrad = self.computeNormal(simulationState, simulation,)
        
        simulationState['lambdaGrad'] = lambdaGrad

        simulationState['fluidSurfaceMask'] = self.detectSurface(simulationState,simulation)
#         delta = simulationState['fluidUpdate']
        
#         shiftAmount = simulationState['fluidUpdate']
        fluidLambda = simulationState['fluidLambda']
        surfaceMask = simulationState['fluidSurfaceMask']

#         zeroShift = shiftAmount.new_zeros(shiftAmount.shape)

#         normal = -simulationState['lambdaGrad']

#         debugPrint(normal)

#         normalOuter = torch.einsum('nu, nv -> nuv', normal, normal)
#         debugPrint(normalOuter)

#         idMatrix = torch.tensor([[1,0],[0,1]], dtype = normal.dtype, device = normal.device)

#         debugPrint(idMatrix)
#         debugPrint(idMatrix - normalOuter)
#         normalShift = torch.einsum('nuv, nu -> nu',idMatrix - normalOuter, shiftAmount)

#         debugPrint(shiftAmount)
#         debugPrint(normalShift)

#         zeroRegion = torch.logical_or(surfaceMask > 0.9, torch.logical_and(fluidLambda < 0.5, surfaceMask > 0.))
#         zeroRegion = torch.logical_and(fluidLambda < 0.5, surfaceMask > 0.)
        zeroRegion = surfaceMask > 0.7
#         surfRegion = torch.logical_and(fluidLambda >= 0.5, torch.logical_and(surfaceMask > 0., surfaceMask < 0.9))
        surfRegion = torch.logical_and(fluidLambda >= 0.5, surfaceMask > 0.)
        bulkRegion = surfaceMask < 0.1

#         zeroShift[zeroRegion] = 0
#         zeroShift[surfRegion] = normalShift[surfRegion]
#         zeroShift[bulkRegion] = shiftAmount[bulkRegion]


    
        simulationState['fluidSurfaceMask'][surfRegion] = 1
        simulationState['fluidSurfaceMask'][zeroRegion] = 0
        simulationState['fluidSurfaceMask'][bulkRegion] = 2
        
        
    def adjustShiftingAmount(self, simulationState, simulation):  
        delta = simulationState['fluidUpdate']
        
        shiftAmount = simulationState['fluidUpdate']
        fluidLambda = simulationState['fluidLambda']
        surfaceMask = simulationState['fluidSurfaceMask']

#         zeroShift = shiftAmount.new_zeros(shiftAmount.shape)

        normal = -simulationState['lambdaGrad']

#         debugPrint(normal)

        normalOuter = torch.einsum('nu, nv -> nuv', normal, normal)
#         debugPrint(normalOuter)

        idMatrix = torch.tensor([[1,0],[0,1]], dtype = normal.dtype, device = normal.device)

#         debugPrint(idMatrix)
#         debugPrint(idMatrix - normalOuter)
        normalShift = torch.einsum('nuv, nu -> nu',idMatrix - normalOuter, shiftAmount)

#         debugPrint(shiftAmount)
#         debugPrint(normalShift)

#         zeroRegion = torch.logical_or(surfaceMask > 0.9, torch.logical_and(fluidLambda < 0.5, surfaceMask > 0.))
#         zeroRegion = torch.logical_and(fluidLambda < 0.5, surfaceMask > 0.)
        zeroRegion = surfaceMask < 0.5
#         surfRegion = torch.logical_and(fluidLambda >= 0.5, torch.logical_and(surfaceMask > 0., surfaceMask < 0.9))
        surfRegion = surfaceMask < 1.5
        bulkRegion = surfaceMask > 1.5

#         zeroShift[zeroRegion] = 0
#         zeroShift[surfRegion] = normalShift[surfRegion]
#         zeroShift[bulkRegion] = shiftAmount[bulkRegion]

        shiftAmount[surfRegion] = normalShift[surfRegion] * 0.5
        shiftAmount[zeroRegion] = 0
        shiftAmount[surfRegion] = 0
#         shiftAmount[:] = 0
#         adjustedShift = 
        shiftLength = torch.linalg.norm(shiftAmount, axis = 1)
    
        shiftAmount[shiftLength > self.support * 0.1] = shiftAmount[shiftLength > self.support * 0.1] / shiftLength[shiftLength > self.support * 0.1,None] * self.support * 0.1
        
        simulationState['fluidUpdate'] = shiftAmount
    
    
        simulationState['fluidSurfaceMask'][surfRegion] = 1
        simulationState['fluidSurfaceMask'][zeroRegion] = 0
        simulationState['fluidSurfaceMask'][bulkRegion] = 2
        
    
    def applyShifting(self, simulationState, simulation):
        if not self.active:
            return

        # dots = detectSurface(simulationState, simulation)

        oldPositions = simulationState['fluidPosition'].detach().clone()

        
        for i in range(4):
            simulationState['fluidNeighbors'], simulationState['fluidDistances'], simulationState['fluidRadialDistances'] = \
                simulation.neighborSearch.search(simulationState, simulation)

            simulation.solidBC.filterFluidNeighborhoods(simulationState, simulation)
            
            simulation.simulationState['fluidToGhostNeighbors'], simulation.simulationState['ghostToFluidNeighbors'], simulation.simulationState['ghostParticleBodyAssociation'], \
                simulation.simulationState['ghostParticlePosition'], simulation.simulationState['ghostParticleDistance'], simulation.simulationState['ghostParticleGradient'], \
                simulation.simulationState['ghostParticleKernelIntegral'], simulation.simulationState['ghostParticleGradientIntegral'] = simulation.solidBC.search(sphSimulation.simulationState, sphSimulation)
            simulationState['fluidDensity'] = simulation.sphDensity.evaluate(simulationState, simulation)  
            simulationState['boundaryDensity'], simulationState['boundaryGradient'] = simulation.solidBC.density(simulationState, simulation)  
            simulationState['fluidDensity'] += simulationState['boundaryDensity']
            self.computeNormals(simulationState, simulation)
            simulation.periodicBC.syncQuantity(simulationState['fluidDensity'], simulationState, simulation)
            simulationState['fluidUpdate'] = self.solveShifting(simulationState, simulation)
            self.adjustShiftingAmount(simulationState, simulation)
            simulation.periodicBC.syncQuantity(simulationState['fluidUpdate'], simulationState, simulation)          
            
            
            simulationState['fluidPosition'] += simulationState['fluidUpdate']
            
#             simulation.periodicBC.enforcePeriodicBC(simulationState, simulation)
#             enforcePeriodicBC(config, state)
        simulationState['fluidUpdate'] = simulationState['fluidPosition'] - oldPositions
        simulationState['shiftedPositions'] = simulationState['fluidPosition'].detach().clone()
        simulationState['fluidPosition'] = oldPositions
        return simulationState['fluidUpdate']
        

        
# sphSimulation.simulationState['fluidPosition'] = initialPositions.detach().clone()
    
# iiShifting = implicitIterativeShiftModule()
# iiShifting.initialize(sphSimulation.config, sphSimulation.simulationState)
# iiShifting.applyShifting(sphSimulation.simulationState, sphSimulation)


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

# quiverData = sphSimulation.simulationState['fluidUpdate'].detach().cpu().numpy()
# qv = axis[0,0].quiver(positions[:,0], positions[:,1], quiverData[:,0], quiverData[:,1], \
#                       scale_units='xy', scale = 0.1) #scale = 2/ sphSimulation.config['particle']['support'], alpha=0.5)
# #                       scale_units='xy', scale = 2/ sphSimulation.config['particle']['support'], alpha=0.5)

# ax1_divider = make_axes_locatable(axis[0,0])
# cax1 = ax1_divider.append_axes("right", size="4%", pad="1%")
# cbar = fig.colorbar(sc, cax=cax1,orientation='vertical')
# cbar.ax.tick_params(labelsize=8) 

# fig.tight_layout()


# parsedConfig = tomli.loads(tomlConfig)
# sphSimulation = torchSPH(parsedConfig)
# sphSimulation.initializeSimulation()
# sphSimulation.timestep()

# fig, axis = sphSimulation.createPlot(plotScale = 2)

# state = sphSimulation.simulationState

# positions = state['fluidPosition'].detach().cpu().numpy()
# data = torch.linalg.norm(state['fluidUpdate'].detach(),axis=1).cpu().numpy()
# # data = state['fluidDensity'].detach().cpu().numpy()


# sc = axis[0,0].scatter(positions[:,0], positions[:,1], c = data, s = 4)
# axis[0,0].axis('equal')

# ax1_divider = make_axes_locatable(axis[0,0])
# cax1 = ax1_divider.append_axes("right", size="4%", pad="1%")
# cbar = fig.colorbar(sc, cax=cax1,orientation='vertical')
# cbar.ax.tick_params(labelsize=8) 

# quiverData = state['fluidUpdate'].detach().cpu().numpy()
# # qv = axis[0,0].quiver(positions[:,0], positions[:,1], quiverData[:,0], quiverData[:,1], \
# #                       scale_units='xy', scale = 0.1) #scale = 2/ sphSimulation.config['particle']['support'], alpha=0.5)

# quiverData = state['lambdaGrad'].detach().cpu().numpy()
# # qv = axis[0,0].quiver(positions[:,0], positions[:,1], quiverData[:,0], quiverData[:,1], \
# #                       scale_units='xy', scale = 2/ sphSimulation.config['particle']['support'], alpha=0.5)


# fig.suptitle('t=%2.4f[%4d] @ dt = %1.5fs, ptcls: %5d[%5d]'%(state['time'], state['timestep'], state['dt'],state['numParticles'],state['realParticles']))
# if 'densityErrors' in state and not 'divergenceErrors' in state:
#     fig.suptitle('t=%2.4f[%4d] @ dt = %1.5fs, ptcls: %5d[%5d], dfsph: [%3d]'%(state['time'], state['timestep'], state['dt'], state['numParticles'],state['realParticles'],len(state['densityErrors'])))
# if 'divergenceErrors' in state and not 'densityErrors' in state:
#     fig.suptitle('t=%2.4f[%4d] @ dt = %1.5fs, ptcls: %5d[%5d], dfsph: [%3d]'%(state['time'], state['timestep'], state['dt'],state['numParticles'],state['realParticles'],len(state['divergenceErrors'])))
# if 'densityErrors' in state and 'divergenceErrors' in state:
#     fig.suptitle('t=%2.4f[%4d] @ dt = %1.5fs, ptcls: %5d[%5d], dfsph: [%3d, %3d]'%(state['time'], state['timestep'], state['dt'],state['numParticles'],state['realParticles'],len(state['densityErrors']),len(state['divergenceErrors'])))

# fig.tight_layout()


# for i in range(512):
#     sphSimulation.timestep()

    
#     positions = state['fluidPosition'].detach().cpu().numpy()
#     data = torch.linalg.norm(state['fluidUpdate'].detach(),axis=1).cpu().numpy()
# #     data = state['fluidDensity'].detach().cpu().numpy()

#     sc.set_offsets(positions)
#     sc.set_array(data)
#     cbar.mappable.set_clim(vmin=np.min(data), vmax=np.max(data))
#     fig.canvas.draw()
#     fig.canvas.flush_events()
#     fig.suptitle('t=%2.4f[%4d] @ dt = %1.5fs, ptcls: %5d[%5d]'%(state['time'], state['timestep'], state['dt'],state['numParticles'],state['realParticles']))
#     if 'densityErrors' in state and not 'divergenceErrors' in state:
#         fig.suptitle('t=%2.4f[%4d] @ dt = %1.5fs, ptcls: %5d[%5d], dfsph: [%3d]'%(state['time'], state['timestep'], state['dt'], state['numParticles'],state['realParticles'],len(state['densityErrors'])))
#     if 'divergenceErrors' in state and not 'densityErrors' in state:
#         fig.suptitle('t=%2.4f[%4d] @ dt = %1.5fs, ptcls: %5d[%5d], dfsph: [%3d]'%(state['time'], state['timestep'], state['dt'],state['numParticles'],state['realParticles'],len(state['divergenceErrors'])))
#     if 'densityErrors' in state and 'divergenceErrors' in state:
#         fig.suptitle('t=%2.4f[%4d] @ dt = %1.5fs, ptcls: %5d[%5d], dfsph: [%3d, %3d]'%(state['time'], state['timestep'], state['dt'],state['numParticles'],state['realParticles'],len(state['densityErrors']),len(state['divergenceErrors'])))


#     if torch.any(torch.isnan(state['boundaryDensity'])) or torch.any(torch.isnan(state['boundaryGradient'])):
#         raise Exception('Simulation borked')
# simulationState = sphSimulation.simulationState

# # shiftAmount[bulkRegion] = shiftAmount[bulkRegion]


# fig, axis = sphSimulation.createPlot(plotScale = 2)

# positions = sphSimulation.simulationState['fluidPosition'].detach().cpu().numpy()
# colors = torch.linalg.norm(sphSimulation.simulationState['fluidUpdate'].detach(),axis=1).cpu().numpy()
# # colors = sphSimulation.simulationState['fluidLambda'].detach().cpu().numpy()


# sc = axis[0,0].scatter(positions[:,0], positions[:,1], c = colors, s = 2)
# axis[0,0].axis('equal')

# ax1_divider = make_axes_locatable(axis[0,0])
# cax1 = ax1_divider.append_axes("right", size="4%", pad="1%")
# cbar = fig.colorbar(sc, cax=cax1,orientation='vertical')
# cbar.ax.tick_params(labelsize=8) 

# fig.tight_layout()




# d = torch.linspace(-1,1,127)
# c = torch.linspace(-np.pi,np.pi,127)

# mesh_d, mesh_c = torch.meshgrid(d, c)
# fig = plt.figure(figsize=(9,9))

# # res = x2s2(mesh_d,mesh_c)# - x2c2(mesh_d, mesh_c)

# axis = fig.add_subplot(221, polar = True)
# axis.grid(False)
# pc = axis.pcolormesh(mesh_c, mesh_d, x2c2(mesh_d,mesh_c))
# # convert_polar_xticks_to_radians(axis)
# axis.set_yticklabels([])
# # axis.grid(True,alpha=0.25,c='black')
# fig.colorbar(pc)
# axis = fig.add_subplot(222, polar = True)
# axis.grid(False)
# pc = axis.pcolormesh(mesh_c, mesh_d, x2cs(mesh_d,mesh_c))
# # convert_polar_xticks_to_radians(axis)
# axis.set_yticklabels([])
# # axis.grid(True,alpha=0.25,c='black')
# fig.colorbar(pc)

# axis = fig.add_subplot(223, polar = True)
# axis.grid(False)
# pc = axis.pcolormesh(mesh_c, mesh_d, x2cs(mesh_d,mesh_c))
# # convert_polar_xticks_to_radians(axis)
# axis.set_yticklabels([])
# # axis.grid(True,alpha=0.25,c='black')
# fig.colorbar(pc)
# axis = fig.add_subplot(224, polar = True)
# axis.grid(False)
# pc = axis.pcolormesh(mesh_c, mesh_d, x2s2(mesh_d,mesh_c))
# # convert_polar_xticks_to_radians(axis)
# axis.set_yticklabels([])
# # axis.grid(True,alpha=0.25,c='black')
# fig.colorbar(pc)

# fig, axis = sphSimulation.createPlot(plotScale = 2, plotDomain = False)

# positions = sphSimulation.simulationState['fluidPosition'].detach().cpu().numpy()
# data = simulationState['fluidLambda'].detach().cpu().numpy()
# # axis[0,0].quiver(positions[:,0], positions[:,1], vectorData[:,0], vectorData[:,1])
# # sc = axis[0,0].scatter(positions[:,0], positions[:,1], c = data, s = 2)
# axis[0,0].axis('equal')


# positions = np.vstack((positions, sphSimulation.simulationState['ghostParticlePosition'].detach().cpu().numpy()))
# data = np.hstack((data, simulationState['boundaryLambda'].detach().cpu().numpy()))
# # axis[0,0].quiver(positions[:,0], positions[:,1], vectorData[:,0], vectorData[:,1])
# sc = axis[0,0].scatter(positions[:,0], positions[:,1], c = data, s = 4)
# axis[0,0].axis('equal')

# ax1_divider = make_axes_locatable(axis[0,0])
# cax1 = ax1_divider.append_axes("right", size="4%", pad="1%")
# cbar = fig.colorbar(sc, cax=cax1,orientation='vertical')
# cbar.ax.tick_params(labelsize=8) 

# fig.tight_layout()

# fig, axis = sphSimulation.createPlot(plotScale = 2)

# positions = sphSimulation.simulationState['fluidPosition'].detach().cpu().numpy()

# vectorData = simulationState['lambdaGrad'].detach().cpu().numpy()

# axis[0,0].quiver(positions[:,0], positions[:,1], vectorData[:,0], vectorData[:,1])

# sc = axis[0,0].scatter(positions[:,0], positions[:,1], c = data, s = 2)
# axis[0,0].axis('equal')

# ax1_divider = make_axes_locatable(axis[0,0])
# cax1 = ax1_divider.append_axes("right", size="4%", pad="1%")
# cbar = fig.colorbar(sc, cax=cax1,orientation='vertical')
# cbar.ax.tick_params(labelsize=8) 

# fig.tight_layout()
# dots  = simulationState['fluidSurfaceMask']

# fig, axis = sphSimulation.createPlot(plotScale = 2)

# positions = sphSimulation.simulationState['fluidPosition'].detach().cpu().numpy()
# # colors = torch.linalg.norm(sphSimulation.simulationState['fluidUpdate'].detach(),axis=1).cpu().numpy()
# # colors = sphSimulation.simulationState['fluidUpdate'].detach().cpu().numpy()
# colors = dots.detach().cpu().numpy()
# # colors = torch.linalg.norm(lambdaGrad.detach(),axis=1).cpu().numpy()

# # data = maxAngle.detach().cpu().numpy()
# data = dots.detach().cpu().numpy()
# # vectorData = lambdaGrad.detach().cpu().numpy()

# # axis[0,0].quiver(positions[:,0], positions[:,1], vectorData[:,0], vectorData[:,1])

# sc = axis[0,0].scatter(positions[:,0], positions[:,1], c = data, s = 2)
# axis[0,0].axis('equal')

# ax1_divider = make_axes_locatable(axis[0,0])
# cax1 = ax1_divider.append_axes("right", size="4%", pad="1%")
# cbar = fig.colorbar(sc, cax=cax1,orientation='vertical')
# cbar.ax.tick_params(labelsize=8) 

# fig.tight_layout()