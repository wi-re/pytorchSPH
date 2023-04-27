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
from tqdm.notebook import trange, tqdm
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

from src.simulationBase import SPHSimulation
from src.kernels import kernel, kernelGradient, spikyGrad, wendland, wendlandGrad, cohesionKernel, getKernelFunctions
from src.util import *
from src.module import Module
from src.parameter import Parameter

# import modules to build simulation with
from src.modules.density import densityModule
from src.modules.neighborSearch import neighborSearchModule
from src.modules.akinciTension import akinciTensionModule
from src.modules.sdfBoundary import sdfBoundaryModule, sdPolyDerAndIntegral
from src.modules.akinciBoundary import akinciBoundaryModule
from src.modules.solidBoundary import solidBoundaryModule
from src.modules.periodicBC import periodicBCModule
from src.modules.velocityBC import velocityBCModule
from src.modules.implicitShifting import implicitIterativeShiftModule
from src.modules.gravity import gravityModule
from src.modules.xsph import xsphModule
from src.modules.dfsph import dfsphModule
from src.modules.deltaSPH import *
from src.modules.diffusion import diffusionModule
from src.modules.densityDiffusion import densityDiffusionModule
from src.modules.adaptiveDT import adaptiveTimeSteppingModule
from src.modules.deltaPlus import deltaPlusModule

from src.modules.momentum import momentumModule
from src.modules.pressure import pressureModule
from src.modules.laminar import laminarViscosityModule

# Weakly compressible SPH simulation based on deltaSPH
class deltaSPHSimulation(SPHSimulation):    
    # Initialization function that loads all necessary modules
    def __init__(self, config = tomli.loads('')):
        super().__init__(config)
        
        self.modules = []
        self.moduleParameters = []
        
        if self.verbose: print('Processing modules')
        # Default module imports that are always needed
        self.neighborSearch = neighborSearchModule()
        self.sphDensity = densityModule()
        self.periodicBC = periodicBCModule()
        self.deltaSPH = deltaSPHModule()
        self.velocityBC = velocityBCModule()
        self.gravityModule = gravityModule()
        self.adaptiveDT = adaptiveTimeSteppingModule()
        self.surfaceTension = akinciTensionModule()
        self.momentumModule = momentumModule()
        self.pressureModule = pressureModule()
        # Add modules to the module list
        self.modules.append(self.neighborSearch)
        self.modules.append(self.sphDensity)
        self.modules.append(self.periodicBC)
        self.modules.append(self.velocityBC)
        self.modules.append(self.deltaSPH)
        self.modules.append(self.momentumModule)
        self.modules.append(self.pressureModule)
        self.modules.append(self.gravityModule)
        self.modules.append(self.adaptiveDT)
        self.modules.append(self.surfaceTension)    
        
        # Conditional modules for artificial viscosity diffusion
        if self.config['diffusion']['velocityScheme'] == 'xsph':
            self.velocityDiffusionModule = xsphModule()
            self.modules.append(self.velocityDiffusionModule)            
        if self.config['diffusion']['velocityScheme'] == 'deltaSPH':
            self.velocityDiffusionModule = diffusionModule()
            self.modules.append(self.velocityDiffusionModule)
        # Density diffusion module loaded after velocity diffusion
        self.densityDiffusionModule = densityDiffusionModule()
        self.modules.append(self.densityDiffusionModule)
        # Laminar viscosity module for actual viscosity
        self.laminarViscosityModule = laminarViscosityModule()
        self.modules.append(self.laminarViscosityModule)
        # Enable shifting for deltaPlus
        if self.config['simulation']['scheme'] == 'deltaPlus' or (self.config['shifting']['scheme'] == 'deltaPlus' and self.config['shifting']['enabled'] == True):
            self.shiftingModule = deltaPlusModule()
            self.modules.append(self.shiftingModule)
        # Add boundary handling modules
        if self.config['simulation']['boundaryScheme'] == 'solid': 
            self.boundaryModule = solidBoundaryModule() 
            self.modules.append(self.boundaryModule)  
        if self.config['simulation']['boundaryScheme'] == 'SDF': 
            self.boundaryModule = sdfBoundaryModule() 
            self.modules.append(self.boundaryModule)  
        if self.config['simulation']['boundaryScheme'] == 'Akinci': 
            self.boundaryModule = akinciBoundaryModule() 
            self.modules.append(self.boundaryModule)  
        # Process parameters for all modules in sequence
        if self.verbose: print('Processing module parameters')
        for module in self.modules:    
            moduleParams =  module.getParameters()
            if moduleParams is not None:
                for param in moduleParams:
                    param.parseConfig(self.config)
                self.moduleParameters = self.moduleParameters + moduleParams
                
    def initializeSimulation(self):
        super().initializeSimulation()
        
    # Evaluate updates for a single timestep, returns dudt, dxdt and drhodt
    def timestep(self):
        step = ' 1 - Enforcing periodic boundary conditions'
        if self.verbose: print(step)
        with record_function(step):
            self.periodicBC.enforcePeriodicBC(self.simulationState, self)            
        step = ' 2 - Fluid neighborhood search'
        if self.verbose: print(step)
        with record_function(step):
            self.neighborSearch.search(self.simulationState, self)            
        step = ' 3 - Boundary neighborhood search'
        if self.verbose: print(step)
        with record_function(step):
            self.boundaryModule.boundaryFilterNeighborhoods(self.simulationState, self)
            self.boundaryModule.boundaryNeighborhoodSearch(self.simulationState, self)
        step = '4 - density  evaluation'
        if self.verbose: print(step)
        with record_function(step):             
            self.sphDensity.evaluate(self.simulationState, self)    
            self.boundaryModule.evalBoundaryDensity(self.simulationState, self)
        step = '5 - surface detection'
        if self.verbose: print(step)
        with record_function(step):           
            if hasattr(self, 'shiftingModule'):
                self.shiftingModule.detectSurface(self.simulationState, self)    
        step = ' 6 - Initializing acceleration'
        if self.verbose: print(step)
        with record_function(step):
            self.simulationState['fluidAcceleration'] = torch.zeros_like(self.simulationState['fluidVelocity'])            
        step = ' 7 - External force evaluation'
        if self.verbose: print(step)
        with record_function(step):
            self.gravityModule.evaluate(self.simulationState, self)        
        step = ' 8 - momentum equation'
        if self.verbose: print(step)
        with record_function(step):     
            self.momentumModule.computeDpDt(self.simulationState, self)            
        step = ' 9 - density diffusion'
        if self.verbose: print(step)
        with record_function(step):     
            self.densityDiffusionModule.computeDensityDiffusion(self.simulationState, self)            
        step = '10 - pressure evaluation'
        if self.verbose: print(step)
        with record_function(step):     
            self.pressureModule.computePressure(self.simulationState, self)
            self.pressureModule.computePressureAcceleration(self.simulationState, self)            
        step = '11 - velocity diffusion'
        if self.verbose: print(step)
        with record_function(step):     
            self.velocityDiffusionModule.evaluate(self.simulationState, self)    
        step = '12 - laminar viscosity'
        if self.verbose: print(step)
        with record_function(step):       
            self.laminarViscosityModule.computeLaminarViscosity(self.simulationState, self)   
            
        return self.simulationState['fluidAcceleration'], self.simulationState['fluidVelocity'], self.simulationState['dpdt'] if self.config['simulation']['densityScheme'] == 'continuum' else None
                    
            
