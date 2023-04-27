# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# Helpful statement for debugging, prints the thing entered as x and the output, i.e.,
# debugPrint(1+1) will output '1+1 [int] = 2'
import inspect
import re
def debugPrint(x):
    frame = inspect.currentframe().f_back
    s = inspect.getframeinfo(frame).code_context[0]
    r = re.search(r"\((.*)\)", s).group(1)
    print("{} [{}] = {}".format(r,type(x).__name__, x))
    
    
import os
import os, sys
# sys.path.append(os.path.join('~/dev/pytorchSPH/', "lib"))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from tqdm import trange, tqdm
import yaml
import warnings
warnings.filterwarnings(action='once')
from datetime import datetime

import torch
from torch_geometric.nn import radius
from torch_geometric.nn import SplineConv, fps, global_mean_pool, radius_graph, radius
from torch_scatter import scatter

from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

import tomli
from scipy.optimize import minimize
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator
import matplotlib.ticker as mticker

import torch
# import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity


velocities = []

# r = 2
n = 6

# for i in range(n):
# #     for j in range(8):
#     theta_i = 2 * np.pi * i / 8
#     xi = r * np.cos(theta_i)
#     yi = r * np.sin(theta_i)
    
#     velocities.append(np.array([xi,yi]))
    
# #     debugPrint(xi)
# debugPrint(velocities)

seeds = np.random.randint(low = 0, high = 2**16, size = (n**2))
print('seeds:', seeds)
from src.deltaSPH import deltaSPHSimulation
from src.dfsph import dfsphSimulation
from tqdm import trange, tqdm


# def loadConfig(config, i, j):
#     with open(config, 'r') as file:
#         tomlConfig = file.read()
#     parsedConfig = tomli.loads(tomlConfig)
    
#     parsedConfig['emitter']['fluidL']['velocity'] = velocities[i]
#     parsedConfig['emitter']['fluidR']['velocity'] = velocities[j]
#     parsedConfig['export']['prefix'] = 'collision %d x %d' %(i,j)
    
#     simulationScheme = 'deltaSPH'
#     if 'simulation' in parsedConfig:
#         if 'scheme' in parsedConfig['simulation']:
#             simulationScheme = parsedConfig['simulation']['scheme']
            
#     if simulationScheme == 'deltaSPH' or simulationScheme == 'deltaPlus':
#         return parsedConfig, deltaSPHSimulation
#     if simulationScheme == 'dfsph':
#         return parsedConfig, dfsphSimulation


def loadConfig(config, seed):
    with open(config, 'r') as file:
        tomlConfig = file.read()
    parsedConfig = tomli.loads(tomlConfig)
    
    parsedConfig['generative']['seed'] = seed
    # parsedConfig['emitter']['fluidR']['velocity'] = velocities[j]
    # parsedConfig['export']['prefix'] = 'collision %d x %d' %(i,j)
    
    simulationScheme = 'deltaSPH'
    if 'simulation' in parsedConfig:
        if 'scheme' in parsedConfig['simulation']:
            simulationScheme = parsedConfig['simulation']['scheme']
            
    if simulationScheme == 'deltaSPH' or simulationScheme == 'deltaPlus':
        return parsedConfig, deltaSPHSimulation
    if simulationScheme == 'dfsph':
        return parsedConfig, dfsphSimulation

# config = 'configs/collision_dfsph.toml'
config = 'configs/generative.toml'

# for i in tqdm(range(n)):
#     for j in tqdm(range(n)):
#         parsedConfig, simulationModel = loadConfig(config,i,j)
#         sphSimulation = simulationModel(parsedConfig)
#         sphSimulation.initializeSimulation()

#         for t in tqdm(range(2000), leave = False):
#             sphSimulation.integrate()

#         sphSimulation.outFile.close()


for i in tqdm(seeds):
    parsedConfig, simulationModel = loadConfig(config,i)
    sphSimulation = simulationModel(parsedConfig)
    sphSimulation.initializeSimulation()

    for t in tqdm(range(3200), leave = False):
        sphSimulation.integrate()

    sphSimulation.outFile.close()