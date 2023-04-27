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

from .kernels import *

from typing import Dict, Optional

@torch.jit.script
def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())
    return src

@torch.jit.script
def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor:
    index = broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)

from scipy.optimize import minimize

def genParticlesCentered(minCoord, maxCoord, radius, support, packing, dtype = torch.float32, device = 'cpu'):
    area = np.pi * radius**2
    
    gen_position = lambda r, i, j: torch.tensor([r * i, r * j], dtype=dtype, device = device)
        
    diff = maxCoord - minCoord
    center = (minCoord + maxCoord) / 2
    requiredSlices = torch.div(torch.ceil(diff / packing / support).type(torch.int64), 2, rounding_mode='floor')
    
    generatedParticles = []
#     print(requiredSlices)
    for i in range(-requiredSlices[0]-1, requiredSlices[0]+2):
        for j in range(-requiredSlices[1]-1, requiredSlices[1]+2):
            p = center
            g = gen_position(packing * support,i,j)
            pos = p + g
            if pos[0] <= maxCoord[0] + support * 0.2 and pos[1] <= maxCoord[1] + support * 0.2 and \
             pos[0] >= minCoord[0] - support * 0.2 and pos[1] >= minCoord[1] - support * 0.2:
                generatedParticles.append(pos)
                
    return torch.stack(generatedParticles)

def genParticlesSphere(minCoord, maxCoord, radius, packing, support, dtype, device):
    with record_function('config - gen particles'):
        area = np.pi * radius**2
#         support = np.sqrt(area * config['targetNeighbors'] / np.pi)
        
        gen_position = lambda r, i, j: torch.tensor([r * i, r * j], dtype=dtype, device = device)
        
    #     packing *= support
        # debugPrint(minCoord)
        # debugPrint(maxCoord)
        
        rad = min(maxCoord[0] - minCoord[0], maxCoord[1] - minCoord[1]) / 2
        
        diff = maxCoord - minCoord
        requiredSlices = torch.div(torch.ceil(diff / packing / support).type(torch.int64), 2, rounding_mode = 'trunc') + 2
        xi = torch.arange(requiredSlices[0] ).type(dtype).to(device)
        xi = torch.hstack((-torch.flip(xi[1:],(0,)), xi))
        yi = torch.arange(requiredSlices[1] ).type(dtype).to(device)
        yi = torch.hstack((-torch.flip(yi[1:],(0,)), yi))
        
        # print(xi)
    
        xx, yy = torch.meshgrid(xi,yi, indexing = 'xy')
        positions = (packing * support) * torch.vstack((xx.flatten(), yy.flatten()))
        dist = torch.linalg.norm(positions ,dim=0)
        # debugPrint(dist)

#         debugPrint(rad)

        positions = positions[:,dist <= rad]
        dist = torch.linalg.norm(positions ,dim=0)
        
        positions[:] += (maxCoord[:,None] + minCoord[:,None])/2
        
        # debugPrint(torch.min(positions))
        # debugPrint(torch.max(positions))
        return positions.mT

def genParticles(minCoord, maxCoord, radius, packing, support, dtype, device):
    with record_function('config - gen particles'):
        area = np.pi * radius**2
#         support = np.sqrt(area * config['targetNeighbors'] / np.pi)
        
        gen_position = lambda r, i, j: torch.tensor([r * i, r * j], dtype=dtype, device = device)
        
        # debugPrint(minCoord)
        # debugPrint(maxCoord)
        # debugPrint(radius)
        # debugPrint(packing)
        # debugPrint(support)
    #     packing *= support
        # debugPrint(minCoord)
        # debugPrint(maxCoord)
        diff = maxCoord - minCoord
        requiredSlices = torch.ceil(diff / packing / support).type(torch.int64)
        xi = torch.arange(requiredSlices[0] ).type(dtype).to(device)
        yi = torch.arange(requiredSlices[1] ).type(dtype).to(device)
        
        xx, yy = torch.meshgrid(xi,yi, indexing = 'xy')
        positions = (packing * support) * torch.vstack((xx.flatten(), yy.flatten()))
        positions[:] += minCoord[:,None]
        # debugPrint(torch.min(positions))
        # debugPrint(torch.max(positions))
        return positions.mT

def evalBoundarySpacing(spacing, support, packing, radius, gamma, plot = False):
    x = spacing  * support
    particles = genParticles(torch.tensor([-2*support,x[0]]),torch.tensor([2*support,2* support + x[0]]), radius, packing, support, torch.float32, 'cpu')
    particleAreas = torch.ones(particles.shape[0]) * (np.pi * radius**2)
#     particleAreas = torch.ones(particles.shape[0]) * (packing * support)**2
    if plot:
        fig, axis = plt.subplots(1,1, figsize=(4 *  1.09, 4), squeeze = False)

    # axis[0,0].scatter(particles[:,0], particles[:,1], c = 'red',s = 4)

    centerPosition = torch.tensor([0,0])

    dists = particles - centerPosition
    dists = torch.linalg.norm(dists,axis=1)
    minDist = torch.argmin(dists)
    centerPosition = particles[minDist]
    if plot:
        axis[0,0].scatter(centerPosition[0], centerPosition[1], c = 'blue')

    minX = -2 * support
    maxX = 2 * support
    diff = maxX - minX

    requiredSlices = int(np.ceil(diff / packing / support))
#     debugPrint(requiredSlices)
    xi = torch.arange(requiredSlices).type(torch.float32)
    yi = torch.tensor([0]).type(torch.float32)
    xx, yy = torch.meshgrid(xi,yi, indexing = 'xy')

    bdyPositions = (packing * support) * torch.vstack((xx.flatten(), yy.flatten()))
    bdyPositions = bdyPositions.mT
    # debugPrint(bdyPositions)
    # debugPrint(particles)
    bdyPositions[:,0] += minX


    bdyDistances = torch.clamp(torch.linalg.norm(bdyPositions - bdyPositions[:,None], dim = 2) / support, min = 0, max = 1)
    # debugPrint(kernel(bdyDistances.flatten(), support).reshape(bdyDistances.shape))
    bdyArea = gamma / torch.sum(kernel(bdyDistances.flatten(), support).reshape(bdyDistances.shape), dim = 0)
    # debugPrint(bdyKernels.shape)

    p = torch.vstack((particles, bdyPositions))
    v = torch.hstack((particleAreas, bdyArea))

    # p = bdyPositions
    # v = bdyArea

    # sc = axis[0,0].scatter(p[:,0], p[:,1], c = v, s =4)
    # sc = axis[0,0].scatter(bdyPositions[:,0], bdyPositions[:,1], c = bdyKernels, s =4)


    fluidDistances = torch.clamp(torch.linalg.norm(particles - particles[:,None], dim = 2) / support, min = 0, max = 1)
    fluidDensity = torch.sum(kernel(fluidDistances, support) * particleAreas[:,None], dim = 0)


    fluidBdyDistances = torch.clamp(torch.linalg.norm(particles - bdyPositions[:,None], dim = 2) / support, min = 0, max = 1)
    fluidBdyDensity = torch.sum(kernel(fluidBdyDistances, support) * bdyArea[:, None], dim = 0)


    centerDensity = fluidDensity[minDist] + fluidBdyDensity[minDist]
    error = (1 - centerDensity)**2

    if plot:
        debugPrint(x)
        debugPrint(fluidDensity[minDist])
        debugPrint(fluidBdyDensity[minDist])
        debugPrint(error)
        axis[0,0].axis('equal')
        sc = axis[0,0].scatter(particles[:,0], particles[:,1], c = fluidBdyDensity, s = 4)

        ax1_divider = make_axes_locatable(axis[0,0])
        cax1 = ax1_divider.append_axes("right", size="4%", pad="1%")
        cbar = fig.colorbar(sc, cax=cax1,orientation='vertical')
        cbar.ax.tick_params(labelsize=8) 
        sc = axis[0,0].scatter(bdyPositions[:,0], bdyPositions[:,1], c = 'red', s = 4)
    return error

    # #     print(requiredSlices)
    #     generatedParticles = []
    #     for i in range(requiredSlices[0]+1):
    #         for j in range(requiredSlices[1]+1):
    #             p = minCoord
    #             g = gen_position(packing * support,i,j)
    #             pos = p + g
    #             if pos[0] <= maxCoord[0] + support * 0.2 and pos[1] <= maxCoord[1] + support * 0.2:
    #                 generatedParticles.append(pos)
    #     particles = torch.stack(generatedParticles)

    #     return particles



# def mlsInterpolation(simulationState, simulation, queryPositions, support):
#     with record_function("mls interpolation"): 
#         # queryPositions = simulationState['fluidPosition']
#         # queryPosition = pb
#         # support = simulation.config['particle']['support'] * 2

#         i, j = radius(simulationState['fluidPosition'], queryPositions, support, max_num_neighbors = 256)
#         neighbors = torch.stack([i, j], dim = 0)

#     #     debugPrint(neighbors)
#         # debugPrint(torch.min(neighbors[0]))
#         # debugPrint(torch.max(neighbors[0]))
#         # debugPrint(torch.min(neighbors[1]))
#         # debugPrint(torch.max(neighbors[1]))

#         distances = (simulationState['fluidPosition'][j] - queryPositions[i])
#         radialDistances = torch.linalg.norm(distances,axis=1)

#         distances[radialDistances < 1e-5,:] = 0
#         distances[radialDistances >= 1e-5,:] /= radialDistances[radialDistances >= 1e-5,None]
#         radialDistances /= support

#         kernel = wendland(radialDistances, support)

#         bij = simulationState['fluidPosition'][j] - queryPositions[i]
#         bij = torch.hstack((bij.new_ones((bij.shape[0]))[:,None], bij))
#     #     debugPrint(bij)

#         Mpartial = 2 * torch.einsum('nu, nv -> nuv', bij, bij) * \
#                 (simulationState['fluidArea'][j] / simulationState['fluidDensity'][j] * kernel)[:,None,None]

#         M = scatter(Mpartial, i, dim=0, dim_size = queryPositions.shape[0], reduce='add')
#         Minv = torch.linalg.pinv(M)
#     #     debugPrint(Minv)

#         e1 = torch.tensor([1,0,0], dtype=Minv.dtype, device=Minv.device)
#         Me1 = torch.matmul(Minv,e1)
#     #     debugPrint(Me1)


#         pGpartial = torch.einsum('nd, nd -> n', Me1[i], bij) * \
#             kernel * simulationState['fluidPressure2'][j] * (simulationState['fluidArea'][j] / simulationState['fluidDensity'][j])

#         pG = scatter(pGpartial, i, dim=0, dim_size = queryPositions.shape[0], reduce='add')
#     #     debugPrint(pG)

#         return pG


@torch.jit.script
def prepareMLSBoundaries(boundaryPositions, boundarySupports, neighbors, boundaryRadialDistances, fluidPosition, fluidActualArea, support):

    # boundaryPositions = state['akinciBoundary']['positions']
    # boundarySupports = state['akinciBoundary']['boundarySupport']

    bb = neighbors[0]
    bf = neighbors[1] #state['akinciBoundary']['boundaryToFluidNeighbors']
    # boundaryRadialDistances = state['akinciBoundary']['boundaryToFluidNeighborRadialDistances']

    k = kernel(boundaryRadialDistances, support)* fluidActualArea[bf]

    nominator = scatter((k)[:,None] * fluidPosition[bf], bb, dim=0, dim_size = boundaryPositions.shape[0], reduce = 'add')
    denominator = scatter((k), bb, dim=0, dim_size = boundaryPositions.shape[0], reduce = 'add')
    d = torch.clone(boundaryPositions)
    d[denominator > 1e-9] = nominator[denominator > 1e-9] / denominator[denominator > 1e-9,None]
    # debugPrint(state['fluidPosition'][bf] - d[bb])

    xbar =  fluidPosition[bf] - d[bb]

    prod = torch.einsum('nu, nv -> nuv', xbar, xbar) * k[:,None,None]

    Mpartial = scatter(prod, bb, dim = 0, dim_size = boundaryPositions.shape[0], reduce = 'add')

    M1 = torch.linalg.pinv(Mpartial)

    vec = xbar * k[:,None]
    bbar = torch.hstack((torch.ones_like(boundarySupports).unsqueeze(1), boundaryPositions - d))
    
    return M1, vec, bbar

@torch.jit.script
def evalMLSBoundaries(M1, vec, bbar, neighbors, boundaryRadialDistances, fluidPosition, fluidActualArea, fluidPressure, support):
    bb = neighbors[0]
    bf = neighbors[1] 
    k = kernel(boundaryRadialDistances, support)* fluidActualArea[bf]
    
    vecSum = scatter(vec * fluidPressure[bf,None], bb, dim = 0, dim_size = M1.shape[0], reduce = 'add')
    alphaP = scatter(fluidPressure[bf] * k, bb, dim = 0, dim_size = M1.shape[0], reduce = 'add')
    alphaS = scatter( k, bb, dim = 0, dim_size = M1.shape[0], reduce = 'add')

    alpha = alphaP
    alphaP[alphaS > 1e-6] = alphaP[alphaS > 1e-6] / alphaS[alphaS > 1e-6]

    c = torch.hstack((alpha.unsqueeze(1), torch.matmul(M1, vecSum.unsqueeze(2))[:,:,0]))
    pb = torch.einsum('nu, nu -> n', bbar, c)
    return pb


@torch.jit.script
def precomputeMLS(queryPositions, fluidPosition, fluidArea, fluidDensity, support, neighbors, radialDistances):
    with record_function("MLS - precomputeMLS"): 
        # queryPositions = simulationState['fluidPosition']
        # queryPosition = pb

#         i = neighbors[0]
#         j = neighbors[1]
# #         neighbors = torch.stack([i, j], dim = 0)

#     #     debugPrint(neighbors)
#         # debugPrint(torch.min(neighbors[0]))
#         # debugPrint(torch.max(neighbors[0]))
#         # debugPrint(torch.min(neighbors[1]))
#         # debugPrint(torch.max(neighbors[1]))

#         distances = (fluidPosition[j] - queryPositions[i])
#         radialDistances = torch.linalg.norm(distances,dim=1)

#         distances[radialDistances < 1e-5,:] = 0
#         distances[radialDistances >= 1e-5,:] /= radialDistances[radialDistances >= 1e-5,None]
#         radialDistances /= support
        i = neighbors[0]
        j = neighbors[1]

        kernel = kernel(radialDistances, support)

        bij = fluidPosition[j] - queryPositions[i]
        bij = torch.hstack((bij.new_ones((bij.shape[0]))[:,None], bij))
    #     debugPrint(bij)

        Mpartial = 2 * torch.einsum('nu, nv -> nuv', bij, bij) * \
                ((fluidArea / fluidDensity)[j] * kernel)[:,None,None]
        # print(Mpartial.shape)
        M = scatter(Mpartial, i, dim=0, dim_size = queryPositions.shape[0], reduce='add')
        # print(M.shape)
        # print(M)
        Minv = torch.linalg.pinv(M)
    #     debugPrint(Minv)

        e1 = torch.tensor([1,0,0], dtype=Minv.dtype, device=Minv.device)
        Me1 = torch.matmul(Minv,e1)
    #     debugPrint(Me1)


        pGpartial = torch.einsum('nd, nd -> n', Me1[i], bij) * \
            kernel * ((fluidArea / fluidDensity)[j])

#         pG = scatter(pGpartial, i, dim=0, dim_size = queryPositions.shape[0], reduce='add')
    #     debugPrint(pG)

        return pGpartial


@torch.jit.script
def pinv2x2(M):
    with record_function('Pseudo Inverse 2x2'):
        a = M[:,0,0]
        b = M[:,0,1]
        c = M[:,1,0]
        d = M[:,1,1]

        theta = 0.5 * torch.atan2(2 * a * c + 2 * b * d, a**2 + b**2 - c**2 - d**2)
        cosTheta = torch.cos(theta)
        sinTheta = torch.sin(theta)
        U = torch.zeros_like(M)
        U[:,0,0] = cosTheta
        U[:,0,1] = - sinTheta
        U[:,1,0] = sinTheta
        U[:,1,1] = cosTheta

        S1 = a**2 + b**2 + c**2 + d**2
        S2 = torch.sqrt((a**2 + b**2 - c**2 - d**2)**2 + 4* (a * c + b *d)**2)

        o1 = torch.sqrt((S1 + S2) / 2)
        o2 = torch.sqrt((S1 - S2) / 2)

        phi = 0.5 * torch.atan2(2 * a * b + 2 * c * d, a**2 - b**2 + c**2 - d**2)
        cosPhi = torch.cos(phi)
        sinPhi = torch.sin(phi)
        s11 = torch.sign((a * cosTheta + c * sinTheta) * cosPhi + ( b * cosTheta + d * sinTheta) * sinPhi)
        s22 = torch.sign((a * sinTheta - c * cosTheta) * sinPhi + (-b * sinTheta + d * cosTheta) * cosPhi)

        V = torch.zeros_like(M)
        V[:,0,0] = cosPhi * s11
        V[:,0,1] = - sinPhi * s22
        V[:,1,0] = sinPhi * s11
        V[:,1,1] = cosPhi * s22


        o1_1 = torch.zeros_like(o1)
        o2_1 = torch.zeros_like(o2)

        o1_1[torch.abs(o1) > 1e-5] = 1 / o1[torch.abs(o1) > 1e-5] 
        o2_1[torch.abs(o2) > 1e-5] = 1 / o2[torch.abs(o2) > 1e-5] 
        o = torch.vstack((o1_1, o2_1))
        S_1 = torch.diag_embed(o.mT, dim1 = 2, dim2 = 1)

        eigVals = torch.vstack((o1, o2)).mT
        eigVals[torch.abs(eigVals[:,1]) > torch.abs(eigVals[:,0]),:] = torch.flip(eigVals[torch.abs(eigVals[:,1]) > torch.abs(eigVals[:,0]),:],[1])

        return torch.matmul(torch.matmul(V, S_1), U.mT), eigVals



def plotWCSPHSimulation(simulationState, simulation):
    fig, axis = plt.subplots(2,5, figsize=(18 *  1.09, 6), squeeze = False)
    for axx in axis:
        for ax in axx:
            ax.axis('equal')
            ax.set_xlim(sphSimulation.config['domain']['virtualMin'][0], sphSimulation.config['domain']['virtualMax'][0])
            ax.set_ylim(sphSimulation.config['domain']['virtualMin'][1], sphSimulation.config['domain']['virtualMax'][1])
    #         ax.axvline(sphSimulation.config['domain']['min'][0], ls= '--', c = 'black')
    #         ax.axvline(sphSimulation.config['domain']['max'][0], ls= '--', c = 'black')
    #         ax.axhline(sphSimulation.config['domain']['min'][1], ls= '--', c = 'black')
    #         ax.axhline(sphSimulation.config['domain']['max'][1], ls= '--', c = 'black')

    def scatter(axis, fluidPositions, fluidData, boundaryPositions = None, boundaryData = None, label = None):
        positions = fluidPositions.detach().cpu().numpy()
        M = fluidData.detach().cpu().numpy()

        if boundaryPositions is not None and boundaryData is not None:
            bPositions = boundaryPositions.detach().cpu().numpy()
            bM = boundaryData.detach().cpu().numpy()

            positions = np.vstack((positions, bPositions))
            M = np.hstack((M, bM))
        elif boundaryPositions is not None and boundaryData is None:
            bPositions = boundaryPositions.detach().cpu().numpy()
            bM = torch.zeros(bPositions.shape[0]).detach().cpu().numpy()

            positions = np.vstack((positions, bPositions))
            M = np.hstack((M, bM))  


        sc = axis.scatter(positions[:,0], positions[:,1], c = M, s = 4)
        ax1_divider = make_axes_locatable(axis)
        cax1 = ax1_divider.append_axes("right", size="4%", pad="1%")
        cbar = fig.colorbar(sc, cax=cax1,orientation='vertical')
        cbar.ax.tick_params(labelsize=8) 
        if label is not None:
            axis.set_title(label)
        return sc, cbar
    
    plots = []
    plots.append(scatter(axis[0,0], fluidPositions = simulationState['fluidPosition'], boundaryPositions = simulation.boundaryModule.boundaryPositions, fluidData = simulationState['fluidDensity'], boundaryData = simulation.boundaryModule.boundaryDensity, label = 'Density'))
    plots.append(scatter(axis[1,0], fluidPositions = simulationState['fluidPosition'], boundaryPositions = simulation.boundaryModule.boundaryPositions, fluidData = sphSimulation.momentumModule.dpdt, boundaryData = sphSimulation.boundaryModule.dpdt, label = 'drho/dt'))

    plots.append(scatter(axis[0,1], fluidPositions = simulationState['fluidPosition'], boundaryPositions = simulation.boundaryModule.boundaryPositions, fluidData = sphSimulation.pressureModule.pressure, boundaryData = None, label = 'Pressure'))
    plots.append(scatter(axis[1,1], fluidPositions = simulationState['fluidPosition'], boundaryPositions = simulation.boundaryModule.boundaryPositions, fluidData = sphSimulation.densityDiffusionModule.densityDiffusion, boundaryData = None, label = 'rho diff'))

    plots.append(scatter(axis[0,2], fluidPositions = simulationState['fluidPosition'], boundaryPositions = simulation.boundaryModule.boundaryPositions, fluidData = sphSimulation.velocityDiffusionModule.velocityDiffusion[:,0], boundaryData = None, label = 'u_x diff'))
    plots.append(scatter(axis[1,2], fluidPositions = simulationState['fluidPosition'], boundaryPositions = simulation.boundaryModule.boundaryPositions, fluidData = sphSimulation.velocityDiffusionModule.velocityDiffusion[:,1], boundaryData = None, label = 'u_y diff'))

    plots.append(scatter(axis[0,3], fluidPositions = simulationState['fluidPosition'], boundaryPositions = simulation.boundaryModule.boundaryPositions, fluidData = sphSimulation.densityDiffusionModule.eigVals[:,0], boundaryData = sphSimulation.boundaryModule.eigVals[:,0], label = 'a_x'))
    plots.append(scatter(axis[1,3], fluidPositions = simulationState['fluidPosition'], boundaryPositions = simulation.boundaryModule.boundaryPositions, fluidData = sphSimulation.densityDiffusionModule.eigVals[:,1], boundaryData = sphSimulation.boundaryModule.eigVals[:,0], label = 'a_y'))
#     plots.append(scatter(axis[0,3], fluidPositions = simulationState['fluidPosition'], boundaryPositions = simulation.boundaryModule.boundaryPositions, fluidData = simulationState['fluidAcceleration'][:,0], boundaryData = None, label = 'a_x'))
#     plots.append(scatter(axis[1,3], fluidPositions = simulationState['fluidPosition'], boundaryPositions = simulation.boundaryModule.boundaryPositions, fluidData = simulationState['fluidAcceleration'][:,1], boundaryData = None, label = 'a_y'))

    plots.append(scatter(axis[0,4], fluidPositions = simulationState['fluidPosition'], boundaryPositions = simulation.boundaryModule.boundaryPositions, fluidData = simulationState['fluidVelocity'][:,0], boundaryData = None, label = 'u_x'))
    plots.append(scatter(axis[1,4], fluidPositions = simulationState['fluidPosition'], boundaryPositions = simulation.boundaryModule.boundaryPositions, fluidData = simulationState['fluidVelocity'][:,1], boundaryData = None, label = 'u_y'))

    fig.tight_layout()
    
    return fig, plots

def updateWCSPHPlot(fig, plots, simulationState, simulation):
        fluidPositions = simulationState['fluidPosition'].detach().cpu().numpy()  
        boundaryPositions = simulation.boundaryModule.boundaryPositions.detach().cpu().numpy()    
    
        for i, (sc, cbar) in enumerate(plots):
            M = []
            if i ==  0: M = simulationState['fluidDensity']
            if i ==  1: M = sphSimulation.deltaSPH.dpdt
            if i ==  2: M = sphSimulation.deltaSPH.pressure
            if i ==  3: M = sphSimulation.deltaSPH.densityDiffusion
            if i ==  4: M = sphSimulation.deltaSPH.velocityDiffusion[:,0]
            if i ==  5: M = sphSimulation.deltaSPH.velocityDiffusion[:,1]
            if i ==  6: M = simulationState['fluidAcceleration'][:,0]
            if i ==  7: M = simulationState['fluidAcceleration'][:,1]
            if i ==  8: M = simulationState['fluidVelocity'][:,0]
            if i ==  9: M = simulationState['fluidVelocity'][:,1]            
            M = M.detach().cpu().numpy()
                        
            bM = []
            if i ==  0: bM = simulation.boundaryModule.boundaryDensity
            if i ==  1: bM = simulation.boundaryModule.dpdt
            if i ==  2: bM = simulation.boundaryModule.pressure
            if i ==  3: bM = simulation.boundaryModule.densityDiffusion
            if i ==  4: bM = simulation.boundaryModule.velocityDiffusion
            if i ==  5: bM = simulation.boundaryModule.velocityDiffusion
            if i ==  6: bM = torch.zeros(simulation.boundaryModule.boundaryPositions.shape[0])
            if i ==  7: bM = torch.zeros(simulation.boundaryModule.boundaryPositions.shape[0])
            if i ==  8: bM = torch.zeros(simulation.boundaryModule.boundaryPositions.shape[0])
            if i ==  9: bM = torch.zeros(simulation.boundaryModule.boundaryPositions.shape[0])
            bM = bM.detach().cpu().numpy()
            
            positions = np.vstack((fluidPositions, boundaryPositions))
            data = np.hstack((M, bM))            
            
            sc.set_offsets(positions)
            sc.set_array(data)
            cbar.mappable.set_clim(vmin=np.min(data), vmax=np.max(data))
        fig.canvas.draw()
        fig.canvas.flush_events()

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

# positions = state['fluidPosition'].detach().cpu().numpy()
# # data = torch.linalg.norm(state['fluidUpdate'].detach(),axis=1).cpu().numpy()
# data = polyDer[:,0].detach().cpu().numpy()


# sc = axis[0,0].scatter(positions[:,0], positions[:,1], c = polyGrad[:,0].detach().cpu().numpy(), s = 4)
# ax1_divider = make_axes_locatable(axis[0,0])
# cax1 = ax1_divider.append_axes("right", size="4%", pad="1%")
# cbar = fig.colorbar(sc, cax=cax1,orientation='vertical')
# cbar.ax.tick_params(labelsize=8) 



def evalRadius(arg, packing, dtype, device):
    r = torch.tensor(arg, dtype = dtype, device = device)

    area = np.pi * r**2
    support = np.single(np.sqrt(area / np.pi * 20))
    
    minDomain = torch.tensor([\
            -2 * support,\
            -2 * support\
        ], device = device, dtype = dtype)
    maxDomain = torch.tensor([\
             2 * support,\
             2 * support\
        ], device = device, dtype = dtype)

    fluidPosition = genParticlesCentered(minDomain, maxDomain, \
                        arg, support, packing / support, \
                        dtype, device)

    fluidArea = torch.ones(fluidPosition.shape[0], device = device, dtype=dtype) * area
    centralPosition = torch.tensor([[0,0]], device = device, dtype=dtype)

    row, col = radius(centralPosition, fluidPosition, \
                      support, max_num_neighbors = 256)
    fluidNeighbors = torch.stack([row, col], dim = 0)

    fluidDistances = (centralPosition - fluidPosition[fluidNeighbors[0]])
    fluidRadialDistances = torch.linalg.norm(fluidDistances,axis=1)

    fluidRadialDistances /= support
    rho = scatter(\
            kernel(fluidRadialDistances, support) * fluidArea[fluidNeighbors[1]], \
            fluidNeighbors[1], dim=0, dim_size=centralPosition.size(0), reduce="add")
#     print(rho)

    return ((1 - rho)**2).detach().cpu().numpy()[0]