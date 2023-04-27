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

@torch.jit.script
def k(dr, l2 : float = np.log(2)):
#     d = torch.complex(torch.abs(dr), torch.zeros_like(dr))
    d = torch.abs(dr)
    d[torch.abs(dr) < 1e-5] = 1
    
    d2 = d**2
    d3 = d * d2
    d5 = d3 * d2
    d7 = d5 * d2
    srt = torch.sqrt(1-d2)
    integral = ((
        (-15*d7-126*d5)*torch.log(2*srt+2)
        +(15*d7+126*d5)*torch.log(d)
        +6*torch.arccos(d)
        +15*l2*d7
        +srt*(97*d5+60*d3-22*d)
        +126*l2*d5
    )/(6*np.pi))
    integral[dr < 0] = 1 - integral[dr < 0]
    integral[torch.abs(dr) < 1e-5] = 1/2 
    
    gammaScale = 2.0
#     gamma = 1
    gamma = 1 + (1 - dr / 2) ** gammaScale   
    return integral * gamma
@torch.jit.script
def gradK(dr, l2 : float = np.log(2)):    
    d = torch.complex(torch.abs(dr), torch.zeros_like(dr))
    d[torch.abs(dr) < 1e-5] = 1    
    integral = (
        (-3675*d**10-47250*d**8-25200*d**6+3360*d**4)*torch.log(2j*torch.sqrt(d**2-1)+2)
        +(3675*d**10+47250*d**8+25200*d**6-3360*d**4)*torch.log(d)
        +3675*l2*d**10
        +torch.sqrt(d**2-1)*(29093j*d**8+48604j*d**6-5412j*d**4+640j*d**2-160j)
        +47250*l2*d**8
        +25200*l2*d**6
        -3360*l2*d**4
    )/(80*np.pi)
    integral = integral.real
    integral[torch.abs(dr) < 1e-5] = 2 / np.pi
    return integral
@torch.jit.script
def numGradK(dr, h:float = 1e-2):
    xl = k(dr - h)
#     xc = k(dr)
    xr = k(dr)
    return (xl - xr) / (h)
@torch.jit.script
def integralAndDer(dr, support : float):
    h = 1e-2
#     xl = k(dr - h)
    xc = k(dr)
    xr = k(torch.clamp(dr + h,0,1))
    
    integral = xc
    gradient = (xr - xc) / h / support
#     gradient = (xr - xl) / (2 * h)
    return integral, gradient

@torch.jit.script
def sdPoly(poly, p):    
    with record_function("sdPoly"): 
        N = len(poly)

        i = torch.arange(N, device = p.device, dtype = torch.int64)
        i2 = (i + 1) % N
        e = poly[i2] - poly[i]
        v = p - poly[i][:,None]

        ve = torch.einsum('npd, nd -> np', v, e)
        ee = torch.einsum('nd, nd -> n', e, e)

        pq = v - e[:,None] * torch.clamp(ve / ee[:,None], min = 0, max = 1)[:,:,None]

        d = torch.einsum('npd, npd -> np', pq, pq)
        d = torch.min(d, dim = 0).values

        wn = torch.zeros((N, p.shape[0]), device = p.device, dtype = torch.int64)

        cond1 = 0 <= v[i,:,1]
        cond2 = 0 >  v[i2,:,1]
        val3 = e[i,0,None] * v[i,:,1] - e[i,1,None] * v[i,:,0]

        c1c2 = torch.logical_and(cond1, cond2)
        nc1nc2 = torch.logical_and(torch.logical_not(cond1), torch.logical_not(cond2))

        wn[torch.logical_and(c1c2, val3 > 0)] += 1
        wn[torch.logical_and(nc1nc2, val3 < 0)] -= 1

        wn = torch.sum(wn,dim=0)
        s = torch.ones(p.shape[0], device = p.device, dtype = p.dtype)
        s[wn != 0] = -1

        return s * torch.sqrt(d)
@torch.jit.script
def sdPolyDer(poly, p, dh :float = 1e-4, inverted :bool = False):
    with record_function("sdPolyDer"): 
        dh = 1e-4
        dpx = torch.zeros_like(p)
        dnx = torch.zeros_like(p)
        dpy = torch.zeros_like(p)
        dny = torch.zeros_like(p)

        dpx[:,0] += dh
        dnx[:,0] -= dh
        dpy[:,1] += dh
        dny[:,1] -= dh

        c = sdPoly(poly, p)
        cpx = sdPoly(poly, p + dpx)
        cnx = sdPoly(poly, p + dnx)
        cpy = sdPoly(poly, p + dpy)
        cny = sdPoly(poly, p + dny)

        if inverted:
            c = -c
            cpx = -cpx
            cnx = -cnx
            cpy = -cpy
            cny = -cny

        grad = torch.zeros_like(p)
        grad[:,0] = (cpx - cnx) / (2 * dh)
        grad[:,1] = (cpy - cny) / (2 * dh)

        gradLen = torch.linalg.norm(grad, dim =1)
        grad[torch.abs(gradLen) > 1e-5] /= gradLen[torch.abs(gradLen)>1e-5,None]

        return c, grad, cpx, cnx, cpy, cny

@torch.jit.script
def sdPolyDerAndIntegral(poly, p, support : float, masked : bool = False, inverted : bool = False):     
    c, grad, cpx, cnx, cpy, cny = sdPolyDer(poly, p, dh = 1e-4, inverted = inverted)
    k = torch.zeros_like(c)
    kgrad = torch.zeros_like(c)
    k[c / support <= 1], kgrad[c/support <= 1] = integralAndDer(torch.clamp(c[c / support <= 1] / support,0,1), support)
    
#     k, kgrad = integralAndDer(torch.clamp(c / support,0,1), support)  
#     print('k', k)
#     print('kgrad', kgrad)
    return c, grad, k, grad * kgrad[:,None]

    
# polyDist2, polyDer2, polyInt2, polyGrad2 = sdPolyDerAndIntegral2(b['polygon'], simulationState['fluidPosition'], solidBC.support, inverted = b['inverted'])

@torch.jit.script
def getDistance(positions, minDomain, maxDomain):
    distanceMin = positions - minDomain
    distanceMax = maxDomain - positions
    distance = torch.hstack((distanceMin, distanceMax))
    distanceMin2 = torch.argmin(torch.abs(distance), dim = 1)
    polyDist = distance[torch.arange(distance.shape[0], device = distance.device), distanceMin2]
    return polyDist


@torch.jit.script
def domainDistance(positions, minDomain, maxDomain):
    distanceMin = positions - minDomain
    distanceMax = maxDomain - positions
    
    distance = torch.hstack((distanceMin, distanceMax))
    distanceMin2 = torch.argmin(torch.abs(distance), dim = 1)
    polyDist = distance[torch.arange(distance.shape[0], device = distance.device), distanceMin2]
    
    polyDer = torch.zeros((distance.shape[0],2), device = distance.device)

    dh = 1e-2
    dpx = torch.zeros_like(positions)
    dnx = torch.zeros_like(positions)
    dpy = torch.zeros_like(positions)
    dny = torch.zeros_like(positions)

    dpx[:,0] += dh
    dnx[:,0] -= dh
    dpy[:,1] += dh
    dny[:,1] -= dh

    c   = getDistance(positions, minDomain, maxDomain)
    cpx = getDistance(positions + dpx, minDomain, maxDomain)
    cnx = getDistance(positions + dnx, minDomain, maxDomain)
    cpy = getDistance(positions + dpy, minDomain, maxDomain)
    cny = getDistance(positions + dny, minDomain, maxDomain)

    grad = torch.zeros_like(positions)
    grad[:,0] = (cpx - cnx) / (2 * dh)
    grad[:,1] = (cpy - cny) / (2 * dh)

    gradLen = torch.linalg.norm(grad, dim =1)
    grad[torch.abs(gradLen) > 1e-5] /= gradLen[torch.abs(gradLen)>1e-5,None]

    # polyDer[distanceMin == 0, 0] = 1 
    # polyDer[distanceMin == 0, 1] = 0 
    # polyDer[distanceMin == 1, 0] = 0
    # polyDer[distanceMin == 1, 1] = 1
    # polyDer[distanceMin == 2, 0] = -1 
    # polyDer[distanceMin == 2, 1] = 0 
    # polyDer[distanceMin == 3, 0] = 0
    # polyDer[distanceMin == 3, 1] = -1
    
    return polyDist, grad

@torch.jit.script
def domainDistanceAndDer(p, minDomain, maxDomain):
    with record_function("domainDistanceAndDer"): 
        return domainDistance(p, minDomain, maxDomain)
@torch.jit.script
def domainDistanceAndDerAndIntegral(poly, p, support : float, masked : bool = False, inverted : bool = False):
    minDomain = poly[0]
    maxDomain = poly[2]
    c, grad = domainDistance(p, minDomain, maxDomain)
    k = torch.zeros_like(c)
    kgrad = torch.zeros_like(c)
    k[c / support <= 1], kgrad[c/support <= 1] = integralAndDer(torch.clamp(c[c / support <= 1] / support,0,1), support)
    
#     k, kgrad = integralAndDer(torch.clamp(c / support,0,1), support)  
#     print('k', k)
#     print('kgrad', kgrad)
    return c, grad, k, grad * kgrad[:,None]
    

@torch.jit.script
def computeFilterMaskDomain(positions, i,j, domainMin, domainMax):
    with record_function('solidBC - filter distance computation'):
        polyDist, polyDer = domainDistanceAndDer(positions, domainMin, domainMax)
        cp = positions - polyDist[:,None] * polyDer
        d = torch.einsum('nd,nd->n', polyDer, cp)
        neighDistances = torch.einsum('nd,nd->n', positions[j], polyDer[i]) - d[i]

        mask = neighDistances >= 0
        return mask

@torch.jit.script
def computeFilterMaskPoly(polygon, positions, i,j, inverted : bool, support: float):
    with record_function('solidBC - filter distance computation'):
        polyDist, polyDer, _, _,_,_ = sdPolyDer(polygon, positions, support, inverted = inverted)
        cp = positions - polyDist[:,None] * polyDer
        d = torch.einsum('nd,nd->n', polyDer, cp)
        neighDistances = torch.einsum('nd,nd->n', positions[j], polyDer[i]) - d[i]

        mask = neighDistances >= 0
        return mask
    

@torch.jit.script
def boundaryDistanceAndIntegral(bdy : str, poly, p, support : float, masked : bool = False, inverted : bool = False):
    if(bdy == 'domainBoundary'):
        return domainDistanceAndDerAndIntegral(poly, p, support, inverted = inverted)
    else:
        return sdPolyDerAndIntegral(poly, p, support, inverted = inverted)
    
@torch.jit.script
def computeFilterMask(bdy : str, poly, p, i, j, inverted : bool, support : float):
    if(bdy == 'domainBoundary'):
        return computeFilterMaskDomain(p, i, j, poly[0], poly[2])
    else:
        return computeFilterMaskPoly(poly, p, i,j, inverted, support)


@torch.jit.script
def computeBoundaryAccelTerm(fluidArea, fluidDensity, fluidRestDensity, fluidPressure2, pgPartial, ghostToFluidNeighbors2, fluidToGhostNeighbors, ghostParticleBodyAssociation, ghostParticleGradientIntegral, boundaryCounter : int, mlsPressure : bool = True, computeBodyForces : bool = True):
    with record_function("DFSPH - accel (boundary)"): 
        i = fluidToGhostNeighbors[0]
        b = ghostParticleBodyAssociation
        with record_function("DFSPH - accel (boundary)[pressure]"): 
            if mlsPressure:
                boundaryPressure = scatter(pgPartial * fluidPressure2[ghostToFluidNeighbors2[1]], ghostToFluidNeighbors2[0], dim=0, dim_size = ghostParticleBodyAssociation.shape[0], reduce='add')
            else:
                boundaryPressure = fluidPressure2[i]

        with record_function("DFSPH - accel (boundary)[factor]"): 
            fac = - fluidRestDensity[i]
            pi = fluidPressure2[i] / (fluidDensity[i] * fluidRestDensity[i])**2
            pb = boundaryPressure / (1. * fluidRestDensity[i])**2
            grad = ghostParticleGradientIntegral

        with record_function("DFSPH - accel (boundary)[scatter]"): 
            boundaryAccelTerm = scatter_sum((fac * (pi + pb))[:,None] * grad, i, dim = 0, dim_size = fluidArea.shape[0])
        with record_function("DFSPH - accel (boundary)[body]"): 
            if computeBodyForces:
                force = -boundaryAccelTerm * (fluidArea * fluidRestDensity)[:,None]
                boundaryPressureForce = scatter_sum(force[i], b, dim = 0, dim_size = boundaryCounter)
            else:
                boundaryPressureForce = torch.zeros((boundaryCounter,2), dtype = fluidPressure2.dtype, device = fluidArea.device)

        return boundaryPressure, boundaryAccelTerm, boundaryPressureForce

@torch.jit.script
def computeUpdatedPressureBoundarySum(fluidToGhostNeighbors: Optional[torch.Tensor], ghostParticleGradientIntegral : Optional[torch.Tensor], fluidPredAccel, dt : float):
    with record_function("DFSPH - pressure (boundary)"): 
        if fluidToGhostNeighbors is not None and ghostParticleGradientIntegral is not None:
            with record_function("DFSPH - update pressure kernel sum (boundary) [scatter]"): 
                boundaryTerm = scatter_sum(ghostParticleGradientIntegral, fluidToGhostNeighbors[0], dim=0, dim_size=fluidPredAccel.shape[0])

            with record_function("DFSPH - update pressure kernel sum (boundary) [einsum]"): 
                kernelSum = dt**2 * torch.einsum('nd, nd -> n', fluidPredAccel, boundaryTerm)

            return kernelSum
        else:
            return torch.zeros(fluidPredAccel.shape[0], dtype = fluidPredAccel.dtype, device = fluidPredAccel.device)

class sdfBoundaryModule(Module):
    def __init__(self):
        super().__init__('densityInterpolation', 'Evaluates density at the current timestep')
        
    def initialize(self, simulationConfig, simulationState):
        self.support = simulationConfig['particle']['support']
        self.active = True if 'solidBC' in simulationConfig else False
        self.maxNeighbors = simulationConfig['compute']['maxNeighbors']
        if not self.active:
            return
        self.numBodies = len(simulationConfig['solidBC'])
        self.boundaryObjects = simulationConfig['solidBC']
        simulationState['sdfBoundary'] = {}
        self.bodies =  simulationConfig['solidBC']
        # self.kernel, _ = getKernelFunctions(simulationConfig['kernel']['defaultKernel'])
        
        self.dtype = simulationConfig['compute']['precision']
        self.device = simulationConfig['compute']['device']  

        self.pressureScheme = simulationConfig['pressure']['boundaryPressureTerm'] 
        self.computeBodyForces = simulationConfig['simulation']['bodyForces'] 
        self.boundaryCounter = len(simulationConfig['solidBC']) if 'solidBC' in simulationConfig else 0
        self.relaxedJacobiOmega = simulationConfig['dfsph']['relaxedJacobiOmega'] if 'dfsph'in simulationConfig else 0.5
        self.backgroundPressure = simulationConfig['fluid']['backgroundPressure']
        
        self.domainMin = torch.tensor(simulationConfig['domain']['min'], device = self.device)
        self.domainMax = torch.tensor(simulationConfig['domain']['max'], device = self.device)
        
#         print(self.numBodies)
        
#         self.periodicX = simulationConfig['periodicBC']['periodicX']
#         self.periodicY = simulationConfig['periodicBC']['periodicY']
#         self.buffer = simulationConfig['periodicBC']['buffer']
#         self.domainMin = simulationConfig['domain']['virtualMin']
#         self.domainMax = simulationConfig['domain']['virtualMax']
#         self.dtype = simulationConfig['compute']['precision']
        
        

    def dfsphPrepareSolver(self, simulationState, simulation, density):
        if not(density) or self.fluidToGhostNeighbors == None:
            return 
        if self.pressureScheme == "deltaMLS":
            support = simulation.config['particle']['support']
            supports = torch.ones(self.ghostParticlePosition.shape[0], device = simulationState['fluidActualArea'].device, dtype = simulationState['fluidActualArea'].dtype) * support

            neighbors, distances, radialDistances = simulation.neighborSearch.searchExisting(self.ghostParticlePosition, supports * 2, simulationState, simulation, searchRadius = 2)
            self.ghostToFluidNeighbors2 = neighbors
            self.pgPartial = precomputeMLS(self.ghostParticlePosition, simulationState['fluidPosition'], simulationState['fluidArea'], simulationState['fluidDensity'], self.support * 2, neighbors, radialDistances)
    def dfsphBoundaryAccelTerm(self, simulationState, simulation, density):
        if not(density) or self.fluidToGhostNeighbors == None:
            return torch.zeros_like(simulationState['fluidPosition'])
        self.boundaryPressure, boundaryAccelTerm, self.boundaryPressureForce = \
                    computeBoundaryAccelTerm(simulationState['fluidArea'], simulationState['fluidDensity'], simulationState['fluidRestDensity'], simulationState['fluidPressure2'], self.pgPartial, \
                                             self.ghostToFluidNeighbors2, self.fluidToGhostNeighbors, self.ghostParticleBodyAssociation, self.ghostParticleGradientIntegral, \
                                             self.boundaryCounter, mlsPressure = self.pressureScheme == "deltaMLS", computeBodyForces = self.computeBodyForces)
        return boundaryAccelTerm

    def dfsphBoundaryPressureSum(self, simulationState, simulation, density):
        if not(density) or self.fluidToGhostNeighbors == None:
            return torch.zeros_like(simulationState['fluidDensity'])
        kernelSum = computeUpdatedPressureBoundarySum(self.fluidToGhostNeighbors, self.ghostParticleGradientIntegral, simulationState['fluidPredAccel'], simulationState['dt'])
        return kernelSum

    def dfsphBoundaryAlphaTerm(self, simulationState, simulation, density):
        placeholder1 = torch.zeros(simulationState['fluidDensity'].shape, device=self.device, dtype= self.dtype)
        placeholder2 = torch.zeros(simulationState['fluidPosition'].shape, device=self.device, dtype= self.dtype)
        if not(density) or self.fluidToGhostNeighbors == None:
            return placeholder2, placeholder1

        kSum1 = scatter(self.ghostParticleGradientIntegral, self.fluidToGhostNeighbors[0], dim=0, dim_size=simulationState['numParticles'],reduce='add')
        return kSum1, placeholder1

    def dfsphBoundarySourceTerm(self, simulationState, simulation, density):  
        if not(density) or self.fluidToGhostNeighbors == None:
            return torch.zeros_like(simulationState['fluidDensity'])         

        boundaryTerm = scatter(self.ghostParticleGradientIntegral, self.fluidToGhostNeighbors[0], dim=0, dim_size=simulationState['numParticles'],reduce='add')
                
        return - simulationState['dt'] * torch.einsum('nd, nd -> n',  simulationState['fluidPredictedVelocity'],  boundaryTerm)

    def evalBoundaryPressure(self, simulationState, simulation, density):
        raise Exception('Operation boundaryPressure not implemented for ', self.identifier)

    def evalBoundaryDensity(self, simulationState, simulation):
        density = torch.zeros(simulationState['fluidDensity'].shape, device=simulation.device, dtype= simulation.dtype)
        if self.fluidToGhostNeighbors == None:
            return density

        density = torch.zeros(simulationState['fluidDensity'].shape, device=simulation.device, dtype= simulation.dtype)
        gradient = torch.zeros(simulationState['fluidPosition'].shape, device=simulation.device, dtype= simulation.dtype)
        if self.fluidToGhostNeighbors != None:
            density = scatter(self.ghostParticleKernelIntegral, self.fluidToGhostNeighbors[0], dim = 0, dim_size = simulationState['numParticles'], reduce="add")
            gradient = scatter(self.ghostParticleGradientIntegral, self.fluidToGhostNeighbors[0], dim = 0, dim_size = simulationState['numParticles'], reduce="add")
            self.boundaryDensity = density 
            self.boundaryGradient = gradient

        return density
    def evalBoundaryFriction(self, simulationState, simulation):
        raise Exception('Operation boundaryFriction not implemented for ', self.identifier)
    def boundaryNeighborhoodSearch(self, simulationState, simulation):
        if not self.active:
            return
        with record_function('solidBC - neighborhood'):
            particleIndices = torch.arange(simulationState['numParticles'], device = simulation.device, dtype = torch.int64 )

            #counter for correct emission
            ghostParticleCounter = 0
            
            # Relationship from a particle index i to the corresponding ghost particle index
            fluidToGhostParticleRows = []
            fluidToGhostParticleCols = []
            
            # Neighborlists for the ghost particle i and its fluid neighbors j
            ghostParticleToFluidRows = []
            ghostParticleToFluidCols = []
            
            # Each ghost particle belongs to a body in solidBC
            ghostParticleBodyAssociation = []
            
            # State of the ghost particle, related to the fluid particle it was spawned from
            ghostParticlePosition = []
            ghostParticleDistance = []
            ghostParticleGradient = []
            ghostParticleKernelIntegral = []
            ghostParticleGradientIntegral = []
            
            for ib, bdy in enumerate(self.bodies):
                b = self.bodies[bdy]
                
#                 debugPrint(b['polygon'])
#                 debugPrint(self.support)
#                 debugPrint(b['inverted'])
                
                polyDist, polyDer, bIntegral, bGrad = boundaryDistanceAndIntegral(bdy, b['polygon'], simulationState['fluidPosition'], self.support, inverted = b['inverted'])

                adjacent = polyDist <= self.support
                polyDer = polyDer / torch.linalg.norm(polyDer,axis=1)[:,None]
                polyDist = polyDist / self.support
                if polyDer[adjacent].shape[0] == 0:
                    continue
                    
                i = particleIndices[adjacent]
                j = torch.ones(i.shape, device = simulation.device, dtype = torch.int64) *ib
                
                fluidToGhostParticleRows.append(i)
                fluidToGhostParticleCols.append(torch.arange(i.shape[0], device = i.device, dtype = i.dtype) + ghostParticleCounter)
                ghostParticleCounter += i.shape[0]
                
                ghostParticleBodyAssociation.append(j)
                
                pb = simulationState['fluidPosition'][adjacent] - polyDist[adjacent, None] * polyDer[adjacent,:] * self.support
                ghostParticlePosition.append(pb)
                
                row, col = radius(pb, simulationState['fluidPosition'], self.support, max_num_neighbors = self.maxNeighbors)

                ghostParticleToFluidRows.append(row)
                ghostParticleToFluidCols.append(col)
                
                ghostParticleDistance.append(polyDist[adjacent])
                ghostParticleGradient.append(polyDer[adjacent])
                ghostParticleKernelIntegral.append(bIntegral[adjacent])
                ghostParticleGradientIntegral.append(bGrad[adjacent])
                
            del particleIndices
            
            if ghostParticleCounter > 0:
                ghostParticlePosition = torch.cat(ghostParticlePosition)
                ghostParticleDistance = torch.cat(ghostParticleDistance)
                ghostParticleGradient = torch.cat(ghostParticleGradient)
                ghostParticleKernelIntegral = torch.cat(ghostParticleKernelIntegral)
                ghostParticleGradientIntegral = torch.cat(ghostParticleGradientIntegral)
                
                ghostParticleBodyAssociation = torch.cat(ghostParticleBodyAssociation)
                
                fluidToGhostParticleRows = torch.cat(fluidToGhostParticleRows)
                fluidToGhostParticleCols = torch.cat(fluidToGhostParticleCols)
                fluidToGhostNeighbors = torch.stack([fluidToGhostParticleRows, fluidToGhostParticleCols])
                
                ghostParticleToFluidRows = torch.cat(ghostParticleToFluidRows)
                ghostParticleToFluidCols = torch.cat(ghostParticleToFluidCols)
                ghostToFluidNeighbors = torch.stack([ghostParticleToFluidRows, ghostParticleToFluidCols])
                
                self.fluidToGhostNeighbors, self.ghostToFluidNeighbors, self.ghostParticleBodyAssociation, \
                 self.ghostParticlePosition, self.ghostParticleDistance, self.ghostParticleGradient, \
                 self.ghostParticleKernelIntegral, self.ghostParticleGradientIntegral = fluidToGhostNeighbors, ghostToFluidNeighbors, ghostParticleBodyAssociation,\
                    ghostParticlePosition, ghostParticleDistance, ghostParticleGradient, \
                    ghostParticleKernelIntegral, ghostParticleGradientIntegral
            else:
                self.fluidToGhostNeighbors, self.ghostToFluidNeighbors, self.ghostParticleBodyAssociation, \
                 self.ghostParticlePosition, self.ghostParticleDistance, self.ghostParticleGradient, \
                 self.ghostParticleKernelIntegral, self.ghostParticleGradientIntegral = None, None, None, None, None, None, None, None
    def boundaryFilterNeighborhoods(self, simulationState, simulation):       
        if self.active:
            for ib, bdy in enumerate(self.bodies):
#                 debugPrint(bdy)
                b = self.bodies[bdy]
                i = simulationState['fluidNeighbors'][0]
                j = simulationState['fluidNeighbors'][1]
                
#                 if bdy == 'domainBoundary':
#                     mask = computeFilterMaskDomain(simulationState['fluidPosition'], i, j, self.domainMin, self.domainMax)
#                 else:
                mask = computeFilterMask(bdy, b['polygon'], simulationState['fluidPosition'], i,j, b['inverted'], self.support)
                if not torch.any(torch.logical_not(mask)):
                    return

                with record_function('solidBC - filter filtering'):
                    i = i[mask]
                    j = j[mask]

                with record_function('solidBC - filter final step'):
#                     debugPrint(mask)
#                     debugPrint(i)
#                     debugPrint(j)
#                     debugPrint(simulationState['fluidDistances'][mask])
#                     debugPrint(simulationState['fluidRadialDistances'][mask])
                    neighbors = torch.vstack((i,j))
    
                    simulationState['fluidNeighbors'] = neighbors
                    simulationState['fluidDistances'] = simulationState['fluidDistances'][mask]
                    simulationState['fluidRadialDistances'] = simulationState['fluidRadialDistances'][mask]
    
    def getNormalizationMatrices(self, simulationState, simulation):  
        neighbors = self.fluidToGhostNeighbors
        i = neighbors[0]
        b = neighbors[1]
        boundaryMatrices = getCorrelationMatrix(self.ghostParticleDistance, self.ghostParticleGradient) 
    #             boundaryMatrices = boundaryMatrices * 2


        normalizationMatrix = scatter(boundaryMatrices,i, dim=0, dim_size=simulationState['numParticles'], reduce="add")
    #     debugPrint(normalizationMatrix)
    #     L = torch.linalg.pinv(normalizationMatrix)

        ghostToFluidNeighbors = self.ghostToFluidNeighbors
        bf = ghostToFluidNeighbors[0]
        bb = ghostToFluidNeighbors[1]

        volume = simulationState['fluidArea'][bf]/simulationState['fluidDensity'][bf]
        difference = simulationState['fluidPosition'][bf] - self.ghostParticlePosition[bb]

        radialDistance = torch.linalg.norm(difference, axis = 1)
        distance = torch.clone(difference)

    #         debugPrint(radialDistance)
    #         debugPrint(distance)

    #         return L, normalizationMatrix, None, None


        distance[radialDistance > 1e-7] = distance[radialDistance > 1e-7] / radialDistance[radialDistance > 1e-7,None]
        kernel = simulation.kernelGrad(radialDistance, -distance, self.support)

        term = volume[:,None,None] * torch.einsum('nu,nv -> nuv', difference, kernel)

        boundaryM = scatter(term, bb, dim=0, dim_size=self.ghostParticlePosition.shape[0], reduce="add")
        boundaryM[:,0,0] += 1/2 #* 0.5
        boundaryM[:,1,1] += 1/2 #* 0.5
        self.boundaryNormalizationMatrix = boundaryM
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