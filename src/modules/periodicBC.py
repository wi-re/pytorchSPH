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

def createGhostParticlesKernel(positions, domainMin, domainMax, buffer, support, periodicX, periodicY):
    indices = torch.arange(positions.shape[0], dtype=torch.int64).to(positions.device)
    virtualMin = domainMin
    virtualMax = domainMax

    mask_xp = positions[:,0] >= virtualMax[0] - buffer * support
    mask_xn = positions[:,0] < virtualMin[0] + buffer * support
    mask_yp = positions[:,1] >= virtualMax[1] - buffer * support
    mask_yn = positions[:,1] < virtualMin[1] + buffer * support

    filter_xp = indices[mask_xp]
    filter_xn = indices[mask_xn]
    filter_yp = indices[mask_yp]
    filter_yn = indices[mask_yn]

    mask_xp_yp = torch.logical_and(mask_xp, mask_yp)
    mask_xp_yn = torch.logical_and(mask_xp, mask_yn)
    mask_xn_yp = torch.logical_and(mask_xn, mask_yp)
    mask_xn_yn = torch.logical_and(mask_xn, mask_yn)

    filter_xp_yp = indices[torch.logical_and(mask_xp, mask_yp)]
    filter_xp_yn = indices[torch.logical_and(mask_xp, mask_yn)]
    filter_xn_yp = indices[torch.logical_and(mask_xn, mask_yp)]
    filter_xn_yn = indices[torch.logical_and(mask_xn, mask_yn)]

    main = filter_xp.shape[0] + filter_xn.shape[0] + filter_yp.shape[0] + filter_yn.shape[0]
    corner = filter_xp_yp.shape[0] + filter_xp_yn.shape[0] + filter_xn_yp.shape[0] + filter_xn_yn.shape[0]

    ghosts_xp = torch.zeros((filter_xp.shape[0], positions.shape[1]), dtype = positions.dtype, device = positions.device)
    ghosts_xp[:,0] -=  virtualMax[0] - virtualMin[0]

    ghosts_yp = torch.zeros((filter_yp.shape[0], positions.shape[1]), dtype = positions.dtype, device = positions.device)
    ghosts_yp[:,1] -=  virtualMax[1] - virtualMin[1]

    ghosts_xn = torch.zeros((filter_xn.shape[0], positions.shape[1]), dtype = positions.dtype, device = positions.device)
    ghosts_xn[:,0] +=  virtualMax[0] - virtualMin[0]

    ghosts_yn = torch.zeros((filter_yn.shape[0], positions.shape[1]), dtype = positions.dtype, device = positions.device)
    ghosts_yn[:,1] +=  virtualMax[1] - virtualMin[1]


    ghosts_xp_yp = torch.zeros((filter_xp_yp.shape[0], positions.shape[1]), dtype = positions.dtype, device = positions.device)
    ghosts_xp_yp[:,0] -=  virtualMax[0] - virtualMin[0]
    ghosts_xp_yp[:,1] -=  virtualMax[1] - virtualMin[1]

    ghosts_xp_yn = torch.zeros((filter_xp_yn.shape[0], positions.shape[1]), dtype = positions.dtype, device = positions.device)
    ghosts_xp_yn[:,0] -=  virtualMax[0] - virtualMin[0]
    ghosts_xp_yn[:,1] +=  virtualMax[1] - virtualMin[1]

    ghosts_xn_yp = torch.zeros((filter_xn_yp.shape[0], positions.shape[1]), dtype = positions.dtype, device = positions.device)
    ghosts_xn_yp[:,0] +=  virtualMax[0] - virtualMin[0]
    ghosts_xn_yp[:,1] -=  virtualMax[1] - virtualMin[1]

    ghosts_xn_yn = torch.zeros((filter_xn_yn.shape[0], positions.shape[1]), dtype = positions.dtype, device = positions.device)
    ghosts_xn_yn[:,0] +=  virtualMax[0] - virtualMin[0]
    ghosts_xn_yn[:,1] +=  virtualMax[1] - virtualMin[1]

    filters = []
    offsets = []
    if periodicX:
        filters.append(filter_xp)
        filters.append(filter_xn)
        offsets.append(ghosts_xp)
        offsets.append(ghosts_xn)
    if periodicY:
        filters.append(filter_yp)
        filters.append(filter_yn)
        offsets.append(ghosts_yp)
        offsets.append(ghosts_yn)
    if periodicX and periodicY:
        filters.append(filter_xp_yp)
        filters.append(filter_xp_yn)
        filters.append(filter_xn_yp)
        filters.append(filter_xn_yn)
        offsets.append(ghosts_xp_yp)
        offsets.append(ghosts_xp_yn)
        offsets.append(ghosts_xn_yp)
        offsets.append(ghosts_xn_yn)

    return filters, offsets

class periodicBCModule(Module):
    def __init__(self):
        super().__init__('densityInterpolation', 'Evaluates density at the current timestep')
        
    def initialize(self, simulationConfig, simulationState):
        self.support = simulationConfig['particle']['support']
        self.periodicX = simulationConfig['periodicBC']['periodicX']
        self.periodicY = simulationConfig['periodicBC']['periodicY']
        self.buffer = simulationConfig['periodicBC']['buffer']
        self.domainMin = simulationConfig['domain']['min']
        self.domainMax = simulationConfig['domain']['max']
        self.virtualMin = simulationConfig['domain']['virtualMin']
        self.virtualMax = simulationConfig['domain']['virtualMax']
        self.dtype = simulationConfig['compute']['precision']
        
    def resetState(self, simulationState):
        if not 'realParticles' in simulationState:
            simulationState['realParticles'] = simulationState['numParticles']
#         print('Old Particle Count: ', simulationState['numParticles'] )

        with record_function('periodicBC - enforce BC I'):
            realParticles = self.filterVirtualParticles(simulationState['fluidPosition'], simulationState)
        with record_function('periodicBC - enforce BC II'):
            for arr in simulationState:
                if not torch.is_tensor(simulationState[arr]):
                    continue
                if simulationState[arr].shape[0] == simulationState['numParticles']:
                    simulationState[arr] = simulationState[arr][realParticles]

        with record_function('periodicBC - enforce BC III'):
            simulationState['numParticles'] = simulationState['fluidPosition'].shape[0]
            if 'realParticles' in simulationState:
                if simulationState['realParticles'] != simulationState['fluidPosition'].shape[0]:
                    print('panik, deleted or removed actual particles at time', simulationState['time'])

            simulationState['realParticles'] = simulationState['fluidPosition'].shape[0]
    #         print('After pruning: ', simulationState['numParticles'] )

        simulationState.pop('ghostIndices', None)
        simulationState.pop('ghosts', None) 
        simulationState['numParticles'] = simulationState['realParticles']


    def filterVirtualParticles(self, positions, state):    
        with record_function('boundaryCondition[periodic] - filtering'):

            counter = torch.zeros(state['numParticles'], dtype=torch.int64).to(positions.device)
            uidCounter = scatter(torch.ones(state['numParticles'], dtype=torch.int64).to(positions.device), state['UID'], dim = 0, dim_size=state['realParticles'])

            if self.periodicX:
                counter[positions[:,0] < self.domainMin[0]] = -1
                counter[positions[:,0] >= self.domainMax[0]] = -1
            if self.periodicY:        
                counter[positions[:,1] < self.domainMin[1]] = -1    
                counter[positions[:,1] >= self.domainMax[1]] = -1

            deletionCounter = scatter(counter, state['UID'], dim = 0, dim_size=state['realParticles'])
            actualCounter = uidCounter + deletionCounter
            problematicUIDs = state['UID'][:state['realParticles']][actualCounter != 1]
            indices = torch.ones(state['numParticles'], dtype = torch.int64, device=positions.device) * -1
            indices[counter != -1] = state['UID'][counter != -1]

            tempUIDs = torch.arange(state['numParticles'], dtype=torch.int64, device=positions.device)
            for uid in problematicUIDs:
                relevantIndices = tempUIDs[state['UID'] == uid]
                relevantPositions = positions[relevantIndices,:]
                clippedPositions = positions[relevantIndices,:]
                clippedPositions[:,0] = torch.clamp(clippedPositions[:,0], min = self.domainMin[0], max = self.domainMax[0])
                clippedPositions[:,1] = torch.clamp(clippedPositions[:,1], min = self.domainMin[1], max = self.domainMax[1])
                distances = torch.linalg.norm(clippedPositions - relevantPositions, axis =1)
                iMin = torch.argmin(distances)
                for i in range(relevantIndices.shape[0]):
                    indices[relevantIndices[i]] = state['UID'][relevantIndices[i]] if i == iMin else -1
                    positions[relevantIndices[i]] = clippedPositions[i] if i == iMin else positions[relevantIndices[i]]

            indices = tempUIDs[indices != -1]
            args = torch.argsort(state['UID'][indices])
            indices = indices[args]

            return indices

        
    def createGhostParticles(self, positions):
        with record_function('boundaryCondition[periodic] - creating ghost particles'):
            return createGhostParticlesKernel(positions, self.domainMin, self.domainMax, self.buffer, self.support, self.periodicX, self.periodicY)
            indices = torch.arange(positions.shape[0], dtype=torch.int64).to(positions.device)
            virtualMin = self.domainMin
            virtualMax = self.domainMax

            mask_xp = positions[:,0] >= virtualMax[0] - self.buffer * self.support
            mask_xn = positions[:,0] < virtualMin[0] + self.buffer * self.support
            mask_yp = positions[:,1] >= virtualMax[1] - self.buffer * self.support
            mask_yn = positions[:,1] < virtualMin[1] + self.buffer * self.support

            filter_xp = indices[mask_xp]
            filter_xn = indices[mask_xn]
            filter_yp = indices[mask_yp]
            filter_yn = indices[mask_yn]

            mask_xp_yp = torch.logical_and(mask_xp, mask_yp)
            mask_xp_yn = torch.logical_and(mask_xp, mask_yn)
            mask_xn_yp = torch.logical_and(mask_xn, mask_yp)
            mask_xn_yn = torch.logical_and(mask_xn, mask_yn)

            filter_xp_yp = indices[torch.logical_and(mask_xp, mask_yp)]
            filter_xp_yn = indices[torch.logical_and(mask_xp, mask_yn)]
            filter_xn_yp = indices[torch.logical_and(mask_xn, mask_yp)]
            filter_xn_yn = indices[torch.logical_and(mask_xn, mask_yn)]

            main = filter_xp.shape[0] + filter_xn.shape[0] + filter_yp.shape[0] + filter_yn.shape[0]
            corner = filter_xp_yp.shape[0] + filter_xp_yn.shape[0] + filter_xn_yp.shape[0] + filter_xn_yn.shape[0]

            ghosts_xp = torch.zeros((filter_xp.shape[0], positions.shape[1]), dtype = self.dtype, device = positions.device)
            ghosts_xp[:,0] -=  virtualMax[0] - virtualMin[0]

            ghosts_yp = torch.zeros((filter_yp.shape[0], positions.shape[1]), dtype = self.dtype, device = positions.device)
            ghosts_yp[:,1] -=  virtualMax[1] - virtualMin[1]

            ghosts_xn = torch.zeros((filter_xn.shape[0], positions.shape[1]), dtype = self.dtype, device = positions.device)
            ghosts_xn[:,0] +=  virtualMax[0] - virtualMin[0]

            ghosts_yn = torch.zeros((filter_yn.shape[0], positions.shape[1]), dtype = self.dtype, device = positions.device)
            ghosts_yn[:,1] +=  virtualMax[1] - virtualMin[1]


            ghosts_xp_yp = torch.zeros((filter_xp_yp.shape[0], positions.shape[1]), dtype = self.dtype, device = positions.device)
            ghosts_xp_yp[:,0] -=  virtualMax[0] - virtualMin[0]
            ghosts_xp_yp[:,1] -=  virtualMax[1] - virtualMin[1]

            ghosts_xp_yn = torch.zeros((filter_xp_yn.shape[0], positions.shape[1]), dtype = self.dtype, device = positions.device)
            ghosts_xp_yn[:,0] -=  virtualMax[0] - virtualMin[0]
            ghosts_xp_yn[:,1] +=  virtualMax[1] - virtualMin[1]

            ghosts_xn_yp = torch.zeros((filter_xn_yp.shape[0], positions.shape[1]), dtype = self.dtype, device = positions.device)
            ghosts_xn_yp[:,0] +=  virtualMax[0] - virtualMin[0]
            ghosts_xn_yp[:,1] -=  virtualMax[1] - virtualMin[1]

            ghosts_xn_yn = torch.zeros((filter_xn_yn.shape[0], positions.shape[1]), dtype = self.dtype, device = positions.device)
            ghosts_xn_yn[:,0] +=  virtualMax[0] - virtualMin[0]
            ghosts_xn_yn[:,1] +=  virtualMax[1] - virtualMin[1]

            filters = []
            offsets = []
            if self.periodicX:
                filters.append(filter_xp)
                filters.append(filter_xn)
                offsets.append(ghosts_xp)
                offsets.append(ghosts_xn)
            if self.periodicY:
                filters.append(filter_yp)
                filters.append(filter_yn)
                offsets.append(ghosts_yp)
                offsets.append(ghosts_yn)
            if self.periodicX and self.periodicY:
                filters.append(filter_xp_yp)
                filters.append(filter_xp_yn)
                filters.append(filter_xn_yp)
                filters.append(filter_xn_yn)
                offsets.append(ghosts_xp_yp)
                offsets.append(ghosts_xp_yn)
                offsets.append(ghosts_xn_yp)
                offsets.append(ghosts_xn_yn)

            return filters, offsets
        
    def enforcePeriodicBC(self, simulationState, simulation):
        with record_function('boundaryCondition[periodic] - enforce BC'):
            if self.periodicX or self.periodicX:
                if not 'realParticles' in simulationState:
                    simulationState['realParticles'] = simulationState['numParticles']
        #         print('Old Particle Count: ', simulationState['numParticles'] )
        
                with record_function('periodicBC - enforce BC I'):
                    realParticles = self.filterVirtualParticles(simulationState['fluidPosition'], simulationState)
                with record_function('periodicBC - enforce BC II'):
                    for arr in simulationState:
                        if not torch.is_tensor(simulationState[arr]):
                            continue
                        if simulationState[arr].shape[0] == simulationState['numParticles']:
                            simulationState[arr] = simulationState[arr][realParticles]

                with record_function('periodicBC - enforce BC III'):
                    simulationState['numParticles'] = simulationState['fluidPosition'].shape[0]
                    if 'realParticles' in simulationState:
                        if simulationState['realParticles'] != simulationState['fluidPosition'].shape[0]:
                            print('panik, deleted or removed actual particles at time', simulationState['time'])

                    simulationState['realParticles'] = simulationState['fluidPosition'].shape[0]
            #         print('After pruning: ', simulationState['numParticles'] )




                with record_function('periodicBC - enforce BC IV'):
                    ghostIndices, ghostOffsets = self.createGhostParticles(simulationState['fluidPosition'])

                    ghostIndices = torch.cat(ghostIndices)
                    ghostOffsets = torch.vstack(ghostOffsets)
                with record_function('periodicBC - enforce BC V'):
                    realParticles = self.filterVirtualParticles(simulationState['fluidPosition'], simulationState)
                    
                with record_function('periodicBC - enforce BC VI'):
                    for arr in simulationState:
                        if not torch.is_tensor(simulationState[arr]):
                            continue
                        if simulationState[arr].shape[0] == simulationState['numParticles']:
                            if arr == 'fluidPosition':
                                simulationState[arr] = torch.cat((simulationState[arr],simulationState[arr][ghostIndices] + ghostOffsets))
                            else:
                                simulationState[arr] = torch.cat((simulationState[arr],simulationState[arr][ghostIndices]))

                with record_function('periodicBC - enforce BC VII'):
                    simulationState['numParticles'] = simulationState['fluidPosition'].shape[0]
            #         print('New Particle Count: ', simulationState['numParticles'] )

                    ones = torch.ones(simulationState['realParticles'], dtype = torch.int64, device=simulation.device) * -1
                    simulationState['ghostIndices'] = torch.cat((ones, ghostIndices))
                    simulationState['ghosts'] = ghostIndices
        
    def syncToGhost(self, qty, simulationState, simulation):
        with record_function('boundaryCondition[periodic] - syncing quantity'):
            if self.periodicX or self.periodicX:
                ghosts = simulationState['ghosts']
                qty[simulationState['numParticles'] - ghosts.shape[0]:] = qty[simulationState['ghosts']]

# periodicBC = periodicBCModule()
# periodicBC.initialize(sphSimulation.config, sphSimulation.simulationState)
# periodicBC.enforcePeriodicBC(sphSimulation.simulationState, sphSimulation)
# periodicBC.syncQuantity(sphSimulation.simulationState['fluidDensity'], sphSimulation.simulationState, sphSimulation)



# with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],with_stack=False, profile_memory=True) as prof:    
#     for i in range(16):
#         with record_function("periodicBC"): 
#             periodicBC.enforcePeriodicBC(sphSimulation.simulationState, sphSimulation)
        
# # print(prof.key_averages().table(sort_by='self_cpu_time_total'))
# print(prof.key_averages().table(sort_by='cpu_time_total'))

# prof.export_chrome_trace("trace.json")