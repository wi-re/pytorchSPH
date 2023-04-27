
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import time
import torch
from torch_geometric.loader import DataLoader
import argparse
import yaml
from torch_geometric.nn import radius
from torch.optim import Adam
import torch.autograd.profiler as profiler
from torch.profiler import profile, record_function, ProfilerActivity

# from rbfConv import RbfConv
# from dataset import compressedFluidDataset, prepareData

import inspect
import re
def debugPrint(x):
    frame = inspect.currentframe().f_back
    s = inspect.getframeinfo(frame).code_context[0]
    r = re.search(r"\((.*)\)", s).group(1)
    print("{} [{}] = {}".format(r,type(x).__name__, x))
# %matplotlib notebook
import copy

import time
import torch
from torch_geometric.loader import DataLoader
from tqdm.notebook import trange, tqdm
import argparse
import yaml
from torch_geometric.nn import radius
from torch.optim import Adam
import torch.autograd.profiler as profiler
from torch.profiler import profile, record_function, ProfilerActivity

# from rbfConv import RbfConv
# from dataset import compressedFluidDataset, prepareData

import inspect
import re
def debugPrint(x):
    frame = inspect.currentframe().f_back
    s = inspect.getframeinfo(frame).code_context[0]
    r = re.search(r"\((.*)\)", s).group(1)
    print("{} [{}] = {}".format(r,type(x).__name__, x))


import tomli
from scipy.optimize import minimize
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt

import math
from scipy import interpolate

import numpy as np
# %matplotlib notebook
import matplotlib.pyplot as plt

import scipy.special

from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence

import ipywidgets as widgets
from ipywidgets import interact, interact_manual

# import triangle as tr
from scipy.optimize import minimize

# np
from itertools import product

# seed = 0


# import random 
# import numpy as np
# random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# np.random.seed(seed)
# # print(torch.cuda.device_count())
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# # print('running on: ', device)
# torch.set_num_threads(1)

# from joblib import Parallel, delayed

# # from cutlass import *
# # from rbfConv import *
# # from tqdm.notebook import tqdm

# # from datautils import *
# # # from sphUtils import *
# # from lossFunctions import *
# import math
# from scipy import interpolate

# import numpy as np
# %matplotlib notebook
# import matplotlib.pyplot as plt

# import scipy.special

# from numpy.random import MT19937
# from numpy.random import RandomState, SeedSequence

# import ipywidgets as widgets
# from ipywidgets import interact, interact_manual

# # import triangle as tr
# from scipy.optimize import minimize

# # np
# from itertools import product

import numpy as np


def interpolant(t):
    return t*t*t*(t*(t*6 - 15) + 10)


def generate_perlin_noise_2d(
        shape, res, tileable=(False, False), interpolant=interpolant, rng = np.random.default_rng(seed=42)
):
    """Generate a 2D numpy array of perlin noise.

    Args:
        shape: The shape of the generated array (tuple of two ints).
            This must be a multple of res.
        res: The number of periods of noise to generate along each
            axis (tuple of two ints). Note shape must be a multiple of
            res.
        tileable: If the noise should be tileable along each axis
            (tuple of two bools). Defaults to (False, False).
        interpolant: The interpolation function, defaults to
            t*t*t*(t*(t*6 - 15) + 10).

    Returns:
        A numpy array of shape shape with the generated noise.

    Raises:
        ValueError: If shape is not a multiple of res.
    """
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]]\
             .transpose(1, 2, 0) % 1
    # Gradients
    angles = 2*np.pi*rng.random((res[0]+1, res[1]+1))
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    if tileable[0]:
        gradients[-1,:] = gradients[0,:]
    if tileable[1]:
        gradients[:,-1] = gradients[:,0]
    gradients = gradients.repeat(d[0], 0).repeat(d[1], 1)
    g00 = gradients[    :-d[0],    :-d[1]]
    g10 = gradients[d[0]:     ,    :-d[1]]
    g01 = gradients[    :-d[0],d[1]:     ]
    g11 = gradients[d[0]:     ,d[1]:     ]
    # Ramps
    n00 = np.sum(np.dstack((grid[:,:,0]  , grid[:,:,1]  )) * g00, 2)
    n10 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]  )) * g10, 2)
    n01 = np.sum(np.dstack((grid[:,:,0]  , grid[:,:,1]-1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]-1)) * g11, 2)
    # Interpolation
    t = interpolant(grid)
    n0 = n00*(1-t[:,:,0]) + t[:,:,0]*n10
    n1 = n01*(1-t[:,:,0]) + t[:,:,0]*n11
    return np.sqrt(2)*((1-t[:,:,1])*n0 + t[:,:,1]*n1)


def generate_fractal_noise_2d(
        shape, res, octaves=1, persistence=0.5,
        lacunarity=2, tileable=(False, False),
        interpolant=interpolant, seed = 1337
):
    """Generate a 2D numpy array of fractal noise.

    Args:
        shape: The shape of the generated array (tuple of two ints).
            This must be a multiple of lacunarity**(octaves-1)*res.
        res: The number of periods of noise to generate along each
            axis (tuple of two ints). Note shape must be a multiple of
            (lacunarity**(octaves-1)*res).
        octaves: The number of octaves in the noise. Defaults to 1.
        persistence: The scaling factor between two octaves.
        lacunarity: The frequency factor between two octaves.
        tileable: If the noise should be tileable along each axis
            (tuple of two bools). Defaults to (False, False).
        interpolant: The, interpolation function, defaults to
            t*t*t*(t*(t*6 - 15) + 10).

    Returns:
        A numpy array of fractal noise and of shape shape generated by
        combining several octaves of perlin noise.

    Raises:
        ValueError: If shape is not a multiple of
            (lacunarity**(octaves-1)*res).
    """
    rng = np.random.default_rng(seed=seed)
    
    noise = np.zeros(shape)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * generate_perlin_noise_2d(
            shape, (frequency*res[0], frequency*res[1]), tileable, interpolant, rng
        )
        frequency *= lacunarity
        amplitude *= persistence
    return noise

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
#         dh = 1e-2
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
    
def buildSDF(poly, minCoord = [-1,-1], maxCoord = [1,1], n = 256, dh = 1e-2):
    
    x = np.linspace(minCoord[0],maxCoord[0],n)
    y = np.linspace(minCoord[1],maxCoord[1],n)

    xx, yy = np.meshgrid(x,y)

    sdf, sdfGrad, _, _, _, _ = sdPolyDer(torch.tensor(poly[:-1,:]), torch.tensor(np.vstack((yy.flatten(),xx.flatten()))).mT, dh = dh)
    
    return xx, yy, sdf, sdfGrad
def plotMesh(xx,yy,z, axis, fig):
    im = axis.pcolormesh(xx,yy,z)
    axis.axis('equal')
    ax1_divider = make_axes_locatable(axis)
    cax1 = ax1_divider.append_axes("bottom", size="7%", pad="2%")
    cbar = fig.colorbar(im, cax=cax1,orientation='horizontal')
    cbar.ax.tick_params(labelsize=8)
    
    
def createNoiseFunction(n = 256, res = 2, octaves = 2, lacunarity = 2, persistance = 0.5, seed = 1336):
    noise = generate_fractal_noise_2d(shape = (n,n), res = (res,res), octaves = octaves, persistence = persistance, lacunarity = lacunarity, tileable = (False, False), seed = seed)
#     noise = Octave(n, octaves = octaves, lacunarity = lacunarity, persistance = persistance, seed = seed)

#     noise[:,0] = noise[:,1] - noise[:,2] + noise[:,1]
#     noise[0,:] = noise[1,:] - noise[2,:] + noise[1,:]

#     noise = noise[:n,:n] / 255
    x = np.linspace(-1,1,n)
    y = np.linspace(-1,1,n)
    xx, yy = np.meshgrid(x,y)

    f = interpolate.RegularGridInterpolator((x, y), noise, bounds_error = False, fill_value = None, method = 'linear')
    
    return f, noise

def createVelocityField(f, n = 256, dh = 1e-4):
    x = np.linspace(-1,1,n)
    y = np.linspace(-1,1,n)

    xx, yy = np.meshgrid(x,y)

    z = f((xx, yy))
    zxp = f((xx + dh, yy))
    zxn = f((xx -dh, yy))
    zyp = f((xx, yy + dh))
    zyn = f((xx, yy - dh))
    yv = (zxp - zxn) / (2 * dh)
    xv = -(zyp - zyn) / (2 * dh)
#     print(xv)
#     print(yv)
    
    return np.stack((xv, yv), axis = 2), xx, yy, z

def createPotentialField(n = 256, res = 4, octaves = 2, lacunarity = 2, persistance = 0.5, seed = 1336):
    f, noise = createNoiseFunction(n = n, res = res, octaves = octaves, lacunarity = lacunarity, persistance = persistance, seed = seed)
#     noise = Octave(n, octaves = octaves, lacunarity = lacunarity, persistance = persistance, seed = seed)

#     noise[:,0] = noise[:,1] - noise[:,2] + noise[:,1]
#     noise[0,:] = noise[1,:] - noise[2,:] + noise[1,:]

#     noise = noise[:n,:n] / 255
    x = np.linspace(-1,1,n)
    y = np.linspace(-1,1,n)
    xx, yy = np.meshgrid(x,y)

#     f = interpolate.RegularGridInterpolator((x, y), noise, bounds_error = False, fill_value = None, method = 'linear')
    
    return xx,yy,noise

# def filterPotential(noise, sdf, d0 = 0.25):
#     r = sdf / d0
# #     ramped = r * r * (3 - 2 * r)
#     ramped = 15/8 * r - 10/8 * r**3 + 3/8 * r**5
# #     ramped = r
#     ramped[r >= 1] = 1
#     ramped[r <= 0] = 0
# #     ramped[r <= -1] = -1
    
#     return ramped * noise
#     # ramped = r
    
# def filterPotential(noise, sdf, d0 = 0.25):
#     r = sdf / d0 /2  + 0.5
# #     ramped = r * r * (3 - 2 * r)
#     ramped = 15/8 * r - 10/8 * r**3 + 3/8 * r**5
# #     ramped = r
#     ramped[r >= 1] = 1
#     ramped[r <= 0] = 0
# #     ramped[r <= -1] = -1
    
#     return ramped * noise
#     # ramped = r
    
def generateParticles(nd, nb, border = 3):
    if not isinstance(nd, np.ndarray):
        nd = np.array([nd,nd])
    if not isinstance(nb, np.ndarray):
        nb = np.array([nb,nb])
#     nd = 16
    nc = 2 * nd
#     nb = 32
    na = 2 * nb + nc
#     border = 3
    xi = np.arange(-border, na[0] + border, dtype = int) + border
    yi = np.arange(-border, na[1] + border, dtype = int) + border
    dx = 2 / (na[0] - 1) if na[0] > na[1] else 2 / (na[1] - 1)
    dy = dx
    px = xi * dx - 1 - border * dx
    
    
    py = yi * dy - 1 - border * dy
    # print(xi)
    # print(x)
    xx, yy = np.meshgrid(px,py)
    xxi, yyi = np.meshgrid(xi,yi)

    c = np.ones_like(xx)
#     print(xx.shape)

    c[xxi < border] = -1
    c[xxi >= na[0] + border] = -1
    c[yyi < border] = -1
    c[yyi >= na[1] + border] = -1
#     print(np.sum(c > 0) - 96**2)
    # print(96**2)

    maskA = xxi >= border + nb[0]
    maskB = yyi >= border + nb[1]
    maskAB = np.logical_and(maskA, maskB)

    maskC = xxi < border + nb[0] + nc[0]
    maskD = yyi < border + nb[1] + nc[1]
    maskCD = np.logical_and(maskC, maskD)

    mask = np.logical_and(maskAB, maskCD)
#     print(np.sum(mask))
    c[mask] = -1

    maskA = xxi >= 2 * border + nb[0]
    maskB = yyi >= 2 * border + nb[1]
    maskAB = np.logical_and(maskA, maskB)

    maskC = xxi < border + nb[0] + nc[0] - border
    maskD = yyi < border + nb[1] + nc[1] - border
    maskCD = np.logical_and(maskC, maskD)

    mask = np.logical_and(maskAB, maskCD)
#     print(np.sum(mask))
    c[mask] = 0.25
    # c[:,:] = -0.5


    minDomain = -1 - dx / 2
    minCenter = - nd * dx# - dx / 2
#     print(dx)
#     print(-nd * dx)
#     print(minCenter)


#     fig, axis = plt.subplots(1, 1, figsize=(6,6), sharex = False, sharey = False, squeeze = False)

    ptcls = np.vstack((xx[c > 0.5], yy[c>0.5])).transpose()
    bdyPtcls = np.vstack((xx[c < -0.5], yy[c <-0.5])).transpose()
    
    center = (np.max(bdyPtcls,axis=0) + np.min(bdyPtcls,axis=0))/2
#     print(center)
    ptcls = ptcls - center
    bdyPtcls = bdyPtcls - center
    minDomain = np.min(bdyPtcls,axis=0) + (border - .5) * dx
    
    return ptcls, bdyPtcls, minDomain, minCenter

def genNoisyParticles(nd = 8, nb = 16, border = 3, n = 256, res = 2, octaves = 4, lacunarity = 2, persistance = 0.25, seed = 1336, boundary = 0.25, dh = 1e-3):
    ptcls, bdyPtcls, minDomain, minCenter = generateParticles(nd, nb, border = border)

#     dh = 1e-3

#     boundary = 0.25

    c = -minCenter
    domainBoundary = np.array([[minDomain[0] + boundary,minDomain[1] + boundary],[-minDomain[0] - boundary,minDomain[1] + boundary], [-minDomain[0] - boundary,-minDomain[1] - boundary],[minDomain[0] + boundary,-minDomain[1] - boundary],[minDomain[0] + boundary,minDomain[1] + boundary]])
    centerBoundary = np.array([[-c[0],-c[1]],[c[0],-c[1]],[c[0],c[1]],[-c[0],c[1]],[-c[0],-c[1]]])

    _, _, polySDF, polySDFGrad = buildSDF(centerBoundary, n = n, dh = dh)
    _, _, domainSDF, domainSDFGrad = buildSDF(domainBoundary, n = n, dh = dh)
    # _, _, domainSDF, domainSDFGrad = buildSDF(np.array([[-1.0 ,-1 ],[1 ,-1 ],\
    #                                                     [1 ,1 ],[-1 ,1 ],[-1 ,-1 ]]), n = 256, dh = dh)

    # poly, shape = buildPolygon()
    # xx, yy, polySDF, polySDFGrad = buildSDF(poly, n = 256)
    s = (- domainSDF + boundary).numpy()
    s = s.reshape(polySDF.shape)
    # s = - domainSDF



    xx, yy, noise = createPotentialField(n = n, res = res, octaves = octaves, lacunarity = lacunarity, persistance = persistance, seed = seed)
    filtered = noise
    
    filtered = filterPotential(torch.tensor(filtered).flatten(), torch.tensor(s).flatten(), d0 = boundary ).numpy().reshape(noise.shape)
    if np.any(nd > 0):
        filtered = filterPotential(torch.tensor(filtered).flatten(), (polySDF).flatten(), d0 = boundary).numpy().reshape(noise.shape)
        filtered[polySDF.reshape(noise.shape) < 0] = 0

    x = np.linspace(-1,1,n)
    y = np.linspace(-1,1,n)
    f = interpolate.RegularGridInterpolator((x, y), filtered, bounds_error = False, fill_value = None, method = 'linear')

    velocityField, xx, yy, potential = createVelocityField(f, n = n, dh = 2 / (np.max(nd) + np.max(nb)) / 2)  
#     print(filtered)
    
    f = interpolate.RegularGridInterpolator((x, y), velocityField, bounds_error = False, fill_value = None, method = 'linear')
    vel = f((ptcls[:,0], ptcls[:,1]))
    
    
    domainBoundaryActual = np.array([[minDomain[0],minDomain[1]],[-minDomain[0],minDomain[1]], [-minDomain[0],-minDomain[1]],[minDomain[0],-minDomain[1]],[minDomain[0],minDomain[1]]])
    sdf, sdfDer, _, _, _, _ = sdPolyDer(torch.tensor(domainBoundaryActual[:-1]), torch.tensor(bdyPtcls), dh = 1e-2)
    domainPtcls = bdyPtcls[-sdf < 0]
    domainGhostPtcls = domainPtcls - 2 * (sdfDer[-sdf < 0] * (sdf[-sdf < 0,None])).numpy()

    csdf, csdfDer, _, _, _, _ = sdPolyDer(torch.tensor(centerBoundary[:-1]), torch.tensor(bdyPtcls), dh = 1e-2)
    centerPtcls = bdyPtcls[csdf < 0]
    centerGhostPtcls = centerPtcls - 2 * (csdfDer[csdf < 0] * (csdf[csdf < 0,None])).numpy()
    
    return ptcls, vel, domainPtcls, domainGhostPtcls, -sdf[-sdf < 0], -sdfDer[-sdf < 0], centerPtcls, centerGhostPtcls, csdf[csdf < 0], csdfDer[csdf < 0], minDomain, minCenter, xx, yy, filtered



def filterPotential(noise, sdf, d0 = 0.25):
#     r = sdf / d0 /2  + 0.5
    r = sdf / d0 / 0.5 - 1
#     ramped = r * r * (3 - 2 * r)
    ramped = 15/8 * r - 10/8 * r**3 + 3/8 * r**5
#     ramped = r
    ramped[r >= 1] = 1
    ramped[r <= -1] = -1
#     ramped[r <= 0] = 0
#     ramped[r <= -1] = -1
    
    return (ramped /2 + 0.5) * (noise)
    return (ramped /2 + 0.5) * torch.ones_like(noise)
    # ramped = r
from torch_geometric.nn import radius
from torch_scatter import scatter

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

@torch.jit.script
def kernelGrad(q,r,h):
    C = 7 / np.pi    
    return - r * C / h**3 * (20. * q * (1. -q)**3)[:,None]
    

@torch.jit.script
def kernel(q, h):
    C = 7 / np.pi
    b1 = torch.pow(1. - q, 4)
    b2 = 1.0 + 4.0 * q
    return b1 * b2 * C / h**2    

def evalPacking(arg, dtype, device, config):
    packing = torch.tensor(arg, dtype = dtype, device = device)

    minDomain = torch.tensor([\
            -2 * config['particle']['support'],\
            -2 * config['particle']['support']\
        ], device = device, dtype = dtype)
    maxDomain = torch.tensor([\
             2 * config['particle']['support'],\
             2 * config['particle']['support']\
        ], device = device, dtype = dtype)

    fluidPosition = genParticlesCentered(minDomain, maxDomain, \
                        config['particle']['radius'], config['particle']['support'], packing, \
                        dtype, device)

    fluidArea = torch.ones(fluidPosition.shape[0], device = device, dtype=dtype) * config['particle']['area']
    centralPosition = torch.tensor([[0,0]], device = device, dtype=dtype)

    row, col = radius(centralPosition, fluidPosition, \
                      config['particle']['support'], max_num_neighbors = 256)
    fluidNeighbors = torch.stack([row, col], dim = 0)

    fluidDistances = (centralPosition - fluidPosition[fluidNeighbors[0]])
    fluidRadialDistances = torch.linalg.norm(fluidDistances,axis=1)

    fluidRadialDistances /= config['particle']['support']
    rho = scatter(\
            kernel(fluidRadialDistances, config['particle']['support']) * fluidArea[fluidNeighbors[1]], \
            fluidNeighbors[1], dim=0, dim_size=centralPosition.size(0), reduce="add")
    print(rho)

    return ((1 - rho)**2).detach().cpu().numpy()[0], fluidPosition


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


def noisifyParticles(noiseSampler, allPtcls, area, support):
    row, col = radius(allPtcls, allPtcls, \
                      support, max_num_neighbors = 256)
    fluidNeighbors = torch.stack([row, col], dim = 0)

    i = fluidNeighbors[1]
    j = fluidNeighbors[0]
    
    fluidDistances = (allPtcls[fluidNeighbors[1]] - allPtcls[fluidNeighbors[0]])
    fluidRadialDistances = torch.linalg.norm(fluidDistances,axis=1)

    fluidRadialDistances /= support
    
    x_ij = allPtcls[i] - allPtcls[j]
    dist_ij = torch.linalg.norm(fluidDistances,axis=1)
    dir_ij = torch.clone(x_ij)
    dir_ij[dist_ij > 1e-5] = x_ij[dist_ij > 1e-5] / dist_ij[dist_ij > 1e-5,None]
    dist_ij = dist_ij / support
    
    
    rho = scatter(kernel(dist_ij, support) * area, i, dim=0, dim_size=allPtcls.size(0), reduce="add")
    
    potential = noiseSampler((allPtcls[:,0], allPtcls[:,1]))
    
    gradTerm = (area / rho[j]  *  (potential[j] - potential[i]))[:,None] * kernelGrad(dist_ij, dir_ij, support)
    potentialGradient = scatter(gradTerm, i, dim=0, dim_size=allPtcls.size(0), reduce="add")

    velocities = torch.zeros_like(potentialGradient)
    velocities[:,0] = potentialGradient[:,1]
    velocities[:,1] = -potentialGradient[:,0]
    
    gterm = area / rho[j]  * torch.einsum('nd, nd -> n', velocities[j] - velocities[i], kernelGrad(dist_ij, dir_ij, support))

    div = scatter(gterm, i, dim=0, dim_size=allPtcls.size(0), reduce="add")
    
    return velocities, rho, potential, div



# def filterNoise(filtered, minDomain, minCenter, boundary, nd, n, dh = 1e-2):
#     c = -minCenter
#     domainBoundary = np.array([[minDomain + boundary,minDomain + boundary],[-minDomain - boundary,minDomain + boundary], [-minDomain - boundary,-minDomain - boundary],[minDomain + boundary,-minDomain - boundary],[minDomain + boundary,minDomain + boundary]])
#     centerBoundary = np.array([[-c,-c],[c,-c],[c,c],[-c,c],[-c,-c]])

#     _, _, polySDF, polySDFGrad = buildSDF(centerBoundary, n = n, dh = dh)
#     _, _, domainSDF, domainSDFGrad = buildSDF(domainBoundary, n = n, dh = dh)
#     s = (- domainSDF + boundary).numpy()
#     s = s.reshape(polySDF.shape)
    
#     # xx, yy, noise = createPotentialField(n = n, res = res, octaves = octaves, lacunarity = lacunarity, persistance = persistance, seed = seed)
#     # filtered = noise
    
#     filtered = filterPotential(torch.tensor(filtered).flatten(), torch.tensor(s).flatten(), d0 = boundary ).numpy().reshape(filtered.shape)
#     if nd > 0:
#         filtered = filterPotential(torch.tensor(filtered).flatten(), torch.tensor(polySDF).flatten(), d0 = boundary).numpy().reshape(filtered.shape)
#         filtered[polySDF.reshape(filtered.shape) < 0] = 0
#     return filtered 

def filterNoise(filtered, minDomain, minCenter, boundary, nd, n , dh):
    c = -minCenter
    domainBoundary = np.array([[minDomain[0] + boundary,minDomain[1] + boundary],[-minDomain[0] - boundary,minDomain[1] + boundary], [-minDomain[0] - boundary,-minDomain[1] - boundary],[minDomain[0] + boundary,-minDomain[1] - boundary],[minDomain[0] + boundary,minDomain[1] + boundary]])
    centerBoundary = np.array([[-c[0],-c[1]],[c[0],-c[1]],[c[0],c[1]],[-c[0],c[1]],[-c[0],-c[1]]])

    _, _, polySDF, polySDFGrad = buildSDF(centerBoundary, n = n, dh = dh)
    _, _, domainSDF, domainSDFGrad = buildSDF(domainBoundary, n = n, dh = dh)
    s = (- domainSDF + boundary).numpy()
    s = s.reshape(polySDF.shape)
#     s = s.transpose()
    
    # xx, yy, noise = createPotentialField(n = n, res = res, octaves = octaves, lacunarity = lacunarity, persistance = persistance, seed = seed)
    # filtered = noise
    
    filtered = filterPotential((filtered).flatten(), torch.tensor(s).flatten(), d0 = boundary ).numpy().reshape(filtered.shape)
    if np.any(nd > 0):
        filtered = filterPotential((filtered).flatten(), (polySDF).flatten(), d0 = boundary).numpy().reshape(filtered.shape)
        filtered[polySDF.reshape(filtered.shape) < 0] = 0
    return filtered