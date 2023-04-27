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

class velocityBCModule(Module):
    def __init__(self):
        super().__init__('densityInterpolation', 'Evaluates density at the current timestep')
    
    def getParameters(self):
        return [
            Parameter('xsph', 'fluidViscosity', 'float', 0.01, required = False, export = True, hint = ''),
            Parameter('xsph', 'boundaryViscosity', 'float', 0.01, required = False, export = True, hint = '')
        ]
    def initialize(self, simulationConfig, simulationState):
        if 'velocitySource' not in simulationConfig:
            return
        self.support = simulationConfig['particle']['support']
        self.dtype = simulationConfig['compute']['precision']
        self.device = simulationConfig['compute']['device']
        
        simulationState['velocitySource'] = simulationConfig['velocitySource']
        return
    
    def enforce(self, simulationState, simulation):
        if not 'velocitySource' in simulationState:
            return
        with record_function('boundaryCondition[velocity] - enforcing'):
            simulationState['fluidGamma'] = torch.ones(simulationState['fluidArea'].shape, device=self.device, dtype=self.dtype)
            for s in simulationState['velocitySource']:
                source = simulationState['velocitySource'][s]
            #     print(source)
                velTensor = torch.tensor(source['velocity'], device=self.device, dtype=self.dtype)
                curSpeed = velTensor if source['rampTime']>0. else velTensor * np.clip(simulationState['time'] / source['rampTime'], a_min = 0., a_max = 1.)
            #     print(curSpeed)

                xmask = torch.logical_and(simulationState['fluidPosition'][:,0] >= source['min'][0], simulationState['fluidPosition'][:,0] <= source['max'][0])
                ymask = torch.logical_and(simulationState['fluidPosition'][:,1] >= source['min'][1], simulationState['fluidPosition'][:,1] <= source['max'][1])

                mask = torch.logical_and(xmask, ymask)

                active = torch.any(mask)
                # print(xmask)
                # print(ymask)
                # print(mask)
                # print(active)
            #     print(mask)
            #     print(torch.any(mask))
                mu = 3.5
                xr = (simulationState['fluidPosition'][:,0] - source['min'][0]) / (source['max'][0] - source['min'][0])

                if source['min'][0] < 0:
                    xr = 1 - xr

                gamma = (torch.exp(torch.pow(torch.clamp(xr,min = 0, max = 1), mu)) - 1) / (np.exp(1) - 1)

                # gamma = 1 - (torch.exp(torch.pow(xr,mu)) - 1) / (np.exp(1) - 1)
                simulationState['fluidGamma'] = torch.min(gamma, simulationState['fluidGamma'])
                if active:
                    # print(gamma.shape)
                    # gamma = gamma[mask]
                    simulationState['fluidVelocity'][mask,:] = simulationState['fluidVelocity'][mask,:] * (1 - gamma)[mask,None] + gamma[mask,None] * curSpeed
                    
                    simulation.sync(simulationState['fluidGamma'])
                    simulation.sync(simulationState['fluidVelocity'])

            #     print('\n')


        

# velBC = velocityBCModule()
# velBC.initialize(sphSimulation.config, sphSimulation.simulationState)
# velBC.enforce(sphSimulation.simulationState, sphSimulation)
        

# with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],with_stack=True, profile_memory=True) as prof:    
#     for i in range(16):
#         with record_function("fluid Friction"): 
#             velBC.enforce(sphSimulation.simulationState, sphSimulation)
        
# # print(prof.key_averages().table(sort_by='self_cpu_time_total'))
# print(prof.key_averages().table(sort_by='cpu_time_total'))