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

class gravityModule(Module):
    def __init__(self):
        super().__init__('densityInterpolation', 'Evaluates density at the current timestep')
    
    def getParameters(self):
        return [
            Parameter('gravity', 'magnitude', 'float', 9.81, required = False, export = True, hint = ''),
            Parameter('gravity', 'direction', 'float array', [0, -1], required = False, export = True, hint = ''),
            Parameter('gravity', 'pointSource', 'bool', False, required = False, export = True, hint = ''),
            Parameter('gravity', 'potentialField', 'bool', True, required = False, export = True, hint = ''),
            Parameter('gravity', 'center', 'float array', [0, 0], required = False, export = True, hint = ''),
        ]
    def initialize(self, simulationConfig, simulationState):
        self.magnitude = simulationConfig['gravity']['magnitude']
        self.direction = simulationConfig['gravity']['direction']
        self.pointSource = simulationConfig['gravity']['pointSource']
        self.potentialField = simulationConfig['gravity']['potentialField']
        self.center = simulationConfig['gravity']['center']
        
        self.dtype = simulationConfig['compute']['precision']
        self.device = simulationConfig['compute']['device']
        return
    
    def evaluate(self, simulationState, simulation):
        with record_function('gravity - evaluate'):        
            if self.pointSource:
                difference = simulationState['fluidPosition'] - torch.tensor(self.center, dtype = self.dtype, device = self.device)
                distance = torch.linalg.norm(difference,axis=1)
                difference[distance > 1e-7] = difference[distance > 1e-7] / distance[distance > 1e-7, None]
                if self.potentialField:
                    simulationState['fluidAcceleration'] += -self.magnitude * difference * (distance)[:,None]
                else:
                    simulationState['fluidAcceleration'] += -self.magnitude * difference
            else:
                simulationState['fluidAcceleration'] += self.magnitude * torch.tensor(self.direction, device = self.device, dtype = self.dtype)
            
            # simulation.sync(simulationState['fluidAcceleration'])