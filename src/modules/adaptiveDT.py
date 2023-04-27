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

class adaptiveTimeSteppingModule(Module):
    def getParameters(self):
        return [
            Parameter('timestep', 'min', 'float', 0.0001, required = False, export = True, hint = ''),
            Parameter('timestep', 'max', 'float', 0.01, required = False, export = True, hint = ''),
            Parameter('timestep', 'fixed', 'bool', True, required = False, export = True, hint = ''),
            
            Parameter('timestep', 'CFLNumber', 'float', 1.5, required = False, export = True, hint = ''),
            Parameter('timestep', 'viscosity', 'bool', False, required = False, export = True, hint = ''),
            Parameter('timestep', 'acceleration', 'bool', True, required = False, export = True, hint = ''),
            Parameter('timestep', 'acoustic', 'bool', True, required = False, export = True, hint = '')
        ]
        
    def __init__(self):
        super().__init__('densityInterpolation', 'Evaluates density at the current timestep')
    
    def initialize(self, simulationConfig, simulationState):
        self.support = simulationConfig['particle']['support']
        self.minTimestep = simulationConfig['timestep']['min']
        self.maxTimestep = simulationConfig['timestep']['max']
        self.fixedTimestep = simulationConfig['timestep']['fixed']
        
        self.CFLNumber = simulationConfig['timestep']['CFLNumber']
        self.viscosity = simulationConfig['timestep']['min']
        self.acceleration = simulationConfig['timestep']['min']
        self.acoustic = simulationConfig['timestep']['min']
        
        self.kinematicViscosity = simulationConfig['diffusion']['kinematic']

        self.scheme = simulationConfig['simulation']['scheme']
        self.c0 = simulationConfig['fluid']['c0']
#         self.kinematicViscosity = 0.01
#         self.speedOfSound = 1481
        
        
    def updateTimestep(self, simulationState, simulation):
        with record_function('adaptiveDT - updateTimestep'):
            if self.fixedTimestep:
                return simulationState['dt']
    #         debugPrint(simulationState['dt'])
            
            minSupport = torch.min(simulationState['fluidSupport'])
    #         debugPrint(minSupport)
            
            viscosityDt = (0.125 * minSupport **2 / self.kinematicViscosity).item()
    #         debugPrint(viscosityDt)
            
            accelDt = (0.25 * torch.min(torch.sqrt(simulationState['fluidSupport'] / torch.linalg.norm(simulationState['fluidAcceleration'],axis=-1)))).item()
    #         debugPrint(accelDt)
            if self.scheme == 'deltaSPH' or self.scheme == 'deltaPlus':
                velocityDt = (self.CFLNumber * minSupport / self.c0).item()
            else:
                velocityDt = (0.4 * minSupport / torch.max(torch.linalg.norm(simulationState['fluidVelocity'],axis=-1))).item()

    #         debugPrint(velocityDt)
            
            maximumTimestep = self.maxTimestep
            if self.viscosity:
                maximumTimestep = min(maximumTimestep, viscosityDt)
            if self.acceleration:
                maximumTimestep = min(maximumTimestep, accelDt)
            if self.acoustic and not np.isnan(velocityDt):
                maximumTimestep = min(maximumTimestep, velocityDt)
    #         debugPrint(maximumTimestep)
            
            targetDt = maximumTimestep if maximumTimestep > self.minTimestep else self.minTimestep
            currentDt = simulationState['dt']
            updatedDt = min(max(targetDt, currentDt * 0.5), currentDt * 1.05)
    #         debugPrint(targetDt)
    #         debugPrint(currentDt)
    #         debugPrint(updatedDt)
            return updatedDt
            
    
# adaptivedt = adaptiveTimeSteppingModule()
# adaptivedt.initialize(sphSimulation.config, sphSimulation.simulationState)
# adaptivedt.updateTimestep(sphSimulation.simulationState, sphSimulation)
# # dfsph.incompressibleSolver(sphSimulation.simulationState, sphSimulation)