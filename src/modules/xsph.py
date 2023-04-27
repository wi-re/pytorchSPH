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
def computeFluidTerm(fluidCoefficient, area, density, restDensity, velocity, radialDistance, neighbors, support):
        i = neighbors[0]
        j = neighbors[1]

        fac = fluidCoefficient * restDensity[j] * area[j]
        rho_i = density[i] * restDensity[i]
        rho_j = density[j] * restDensity[j]

        v_ij = velocity[j] - velocity[i]

        k = kernel(radialDistance, support)

        term = (fac / (rho_i + rho_j) * 2. * k)[:,None] * v_ij

        correction = scatter_sum(term, i, dim=0, dim_size=area.shape[0])
        return correction


class xsphModule(Module):
    def __init__(self):
        super().__init__('densityInterpolation', 'Evaluates density at the current timestep')
    
    # def getParameters(self):
    #     return [
    #         # Parameter('xsph', 'fluidViscosity', 'float', 0.05, required = False, export = True, hint = ''),
    #         # Parameter('xsph', 'boundaryViscosity', 'float', 0.01, required = False, export = True, hint = '')
    #     ]
    def initialize(self, simulationConfig, simulationState):
        self.support = simulationConfig['particle']['support']
        self.dtype = simulationConfig['compute']['precision']
        
        self.fluidCoefficient = simulationConfig['diffusion']['alpha']
        self.boundaryCoefficient = simulationConfig['diffusion']['alpha'] if simulationConfig['diffusion']['boundaryDiffusion'] else 0
        self.boundaryScheme = simulationConfig['simulation']['boundaryScheme']
        return
    
    def fluidTerm(self, simulationState, simulation):
        with record_function("diffusion[xsph] - fluid"): 
            return computeFluidTerm(self.fluidCoefficient, simulationState['fluidArea'], simulationState['fluidDensity'], simulationState['fluidRestDensity'], simulationState['fluidVelocity'],simulationState['fluidRadialDistances'],simulationState['fluidNeighbors'], self.support)
            
            neighbors = simulationState['fluidNeighbors']
            i = neighbors[0]
            j = neighbors[1]

            fac = self.fluidCoefficient * simulationState['fluidRestDensity'][j] * simulationState['fluidArea'][j]
            rho_i = simulationState['fluidDensity'][i] * simulationState['fluidRestDensity'][i]
            rho_j = simulationState['fluidDensity'][j] * simulationState['fluidRestDensity'][j]

            v_ij = simulationState['fluidVelocity'][j] - simulationState['fluidVelocity'][i]

            k = wendland(simulationState['fluidRadialDistances'], self.support)

            term = (fac / (rho_i + rho_j) * 2. * k)[:,None] * v_ij

            correction = scatter(term, i, dim=0, dim_size=simulationState['numParticles'], reduce="add")
#             syncQuantity(correction, config, simulationState)
            simulationState['fluidVelocity'] += correction
            simulation.sync(simulationState['fluidVelocity'])
            return correction
    def boundaryTerm(self, simulationState, simulation):
        with record_function('diffusion[xsph] - boundary'):
            # print(state)
            # print(state['boundaryNeighbors'])

            
            if self.boundaryScheme == 'SDF' and 'fluidToGhostNeighbors' in simulationState['sdfBoundary'] and simulationState['sdfBoundary']['fluidToGhostNeighbors'] != None:
                neighbors = simulationState['sdfBoundary']['fluidToGhostNeighbors']
                i = neighbors[0]
                j = neighbors[1]
                b = simulationState['sdfBoundary']['ghostParticleBodyAssociation']
                
                sdfs = simulationState['sdfBoundary']['ghostParticleDistance']
                sdfgrads = simulationState['sdfBoundary']['ghostParticleGradient']

            #     print(i.shape)
            #     print(b.shape)
            #     print(sdfs.shape)
            #     print(sdfgrads.shape)

                fluidVelocity = simulationState['fluidVelocity'][i]

            #     print(fluidVelocity.shape)

                fluidVelocityOrthogonal = torch.einsum('nd, nd -> n', fluidVelocity, sdfgrads)[:,None] * sdfgrads
                fluidVelocityParallel = fluidVelocity - fluidVelocityOrthogonal
            #     print(fluidVelocity)
            #     print(fluidVelocityOrthogonal)
            #     print(fluidVelocityParallel)
                velocities = []
                for bb in simulationState['sdfBoundary']['bodies']:
                    sb = simulationState['sdfBoundary']['bodies'][bb]
                    if 'velocity' in sb:
                        velocities.append(torch.tensor(sb['velocity'],device=simulation.device,dtype=self.dtype))
                    else:
                        velocities.append(torch.tensor([0,0],device=simulation.device,dtype=self.dtype))

                boundaryVelocities = torch.stack(velocities)
                fac = self.boundaryCoefficient * simulationState['fluidRestDensity'][i]
                rho_i = simulationState['fluidDensity'][i] * simulationState['fluidRestDensity'][i]
                rho_b = simulationState['fluidRestDensity'][i]

                v_ib = boundaryVelocities[b] - fluidVelocityParallel

                k = simulationState['sdfBoundary']['ghostParticleKernelIntegral']

                term = (fac / (rho_i + rho_b))[:,None] * v_ib

                correction = scatter(term, i, dim = 0, dim_size=simulationState['numParticles'], reduce='add')
                # print(correction[i])

    #             state['fluidVelocity'] += correction
                force = -correction / simulationState['dt'] * (simulationState['fluidArea'] * simulationState['fluidRestDensity'])[:,None]
                simulationState['sdfBoundary']['boundaryFrictionForce'] = scatter(force[i], b, dim = 0, dim_size = len(simulationState['sdfBoundary']['bodies']), reduce = "add")

                return correction
        return torch.zeros_like(simulationState['fluidVelocity'])
                

        

# xsph = xsphModule()
# xsph.initialize(sphSimulation.config, sphSimulation.simulationState)
# xsph.fluidTerm(sphSimulation.simulationState, sphSimulation)
# frictionForces = xsph.boundaryTerm(sphSimulation.simulationState, sphSimulation)
# torch.max(frictionForces)


# with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],with_stack=True, profile_memory=True) as prof:    
#     for i in range(16):
#         with record_function("fluid Friction"): 
#             xsph.fluidTerm(sphSimulation.simulationState, sphSimulation)
#     for i in range(16):
#         with record_function("boundary Friction"): 
#             xsph.boundaryTerm(sphSimulation.simulationState, sphSimulation)
        
# # print(prof.key_averages().table(sort_by='self_cpu_time_total'))
# print(prof.key_averages().table(sort_by='cpu_time_total'))