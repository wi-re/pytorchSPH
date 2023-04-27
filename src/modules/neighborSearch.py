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


from sympy import nextprime
from typing import Dict, Optional
from torch.utils.cpp_extension import load

neighborSearch = load(name="neighborSearch", sources=["cppSrc/neighSearch.cpp", "cppSrc/neighSearch_cuda.cu"], verbose=False, extra_cflags=['-fopenmp', '-O3', '-march=native'])

@torch.jit.script
def sortPositions(queryParticles, querySupport, supportScale :float = 1.0, qMin : Optional[torch.Tensor]  = None, qMax : Optional[torch.Tensor]  = None):
    with record_function("sort"): 
        with record_function("sort - bound Calculation"): 
            hMax = torch.max(querySupport)
            if qMin is None:
                qMin = torch.min(queryParticles,dim=0)[0] - hMax * supportScale
            else:
                qMin = qMin  - hMax * supportScale
            if qMax is None:
                qMax = torch.max(queryParticles,dim=0)[0] + 2 * hMax * supportScale
            else:
                qMax = qMax + 2 * hMax * supportScale
        with record_function("sort - index Calculation"): 
            qExtent = qMax - qMin
            cellCount = torch.ceil(qExtent / (hMax * supportScale)).to(torch.int32)
            indices = torch.ceil((queryParticles - qMin) / hMax).to(torch.int32)
            linearIndices = indices[:,0] + cellCount[0] * indices[:,1]
        with record_function("sort - actual argsort"): 
            sortingIndices = torch.argsort(linearIndices)
        with record_function("sort - sorting data"): 
            sortedLinearIndices = linearIndices[sortingIndices]
            sortedPositions = queryParticles[sortingIndices,:]
            sortedSupport = querySupport[sortingIndices]
    return sortedPositions, sortedSupport, sortedLinearIndices, sortingIndices, \
            int(cellCount[0]), qMin, float(hMax)


@torch.jit.script
def constructHashMap(sortedPositions, sortedSupport, sortedIndices, sort, hashMapLength : int, cellCount : int):
    with record_function("hashmap"): 
        # First create a list of occupied cells and then create a cumsum to figure out where each cell starts in the data
        with record_function("hashmap - cell cumulation"): 
            cellIndices, cellCounters = torch.unique_consecutive(sortedIndices, return_counts=True, return_inverse=False)
            cellCounters = cellCounters.to(torch.int32)
            cumCell = torch.hstack((torch.tensor([0], device = cellIndices.device, dtype=cellCounters.dtype),torch.cumsum(cellCounters,dim=0)))[:-1].to(torch.int32)
            
        # Now compute the hash indices of all particles by reversing the linearIndices            
        with record_function('hashmap - compute indices'): 
            xIndices = cellIndices % cellCount
            yIndices = torch.div(cellIndices, cellCount, rounding_mode='trunc')
            hashedIndices = (xIndices * 3 + yIndices * 5) % hashMapLength
        # Sort the hashes and use unique consecutive to find hash collisions. Then resort the cell indices based on the hash indices
        with record_function('hashmap - sort hashes'): 
            hashIndexSorting = torch.argsort(hashedIndices)
        with record_function('hashmap - collision detection'): 
            hashMap, hashMapCounters = torch.unique_consecutive(hashedIndices[hashIndexSorting], return_counts=True, return_inverse=False)
            hashMapCounters = hashMapCounters.to(torch.int32)
            cellIndices = cellIndices[hashIndexSorting]
            cellSpan = cumCell[hashIndexSorting]
            cumCell = cellCounters[hashIndexSorting]
        # Now construct the hashtable
        with record_function('hashmap - hashmap construction'):
            hashTable = hashMap.new_ones(hashMapLength,2) * -1
            hashTable[:,1] = 0
            hashMap64 = hashMap.to(torch.int64)
            hashTable[hashMap64,0] = torch.hstack((torch.tensor([0], device = cellIndices.device, dtype=cellIndices.dtype),torch.cumsum(hashMapCounters,dim=0)))[:-1].to(torch.int32) #torch.cumsum(hashMapCounters, dim = 0) #torch.arange(hashMap.shape[0], device=hashMap.device)
            hashTable[hashMap64,1] = hashMapCounters
    return hashTable, cellIndices, cumCell, cellSpan

# @torch.jit.script
def constructNeighborhoods(queryPositions, querySupports, hashMapLength :int = -1, supportScale : float = 1.0, minCoord : Optional[torch.Tensor]  = None, maxCoord : Optional[torch.Tensor]  = None, searchRadius : int = 1):
    with record_function('sortPositions'):
        sortedPositions, sortedSupport, sortedLinearIndices, sortingIndices, cellCount, qMin, hMax = sortPositions(queryPositions, querySupports, 1.0, minCoord, maxCoord)
    if hashMapLength == -1:
        hashMapLength = nextprime(queryPositions.shape[0])
#     return None, None, None, None, None
    with record_function('constructHashMap'):
        hashTable, cellLinearIndices, cellOffsets, cellParticleCounters = constructHashMap(sortedPositions, sortedSupport, sortedLinearIndices, sortingIndices, hashMapLength, cellCount)
        sortingIndices = sortingIndices.to(torch.int32)

    # print(hashMapLength)
    # print('sortedPositions', sortedPositions.shape, sortedPositions.dtype, sortedPositions)
    # print('sortedSupport', sortedSupport.shape, sortedSupport.dtype, sortedSupport)
    # print('sortedLinearIndices', sortedLinearIndices.shape, sortedLinearIndices.dtype, sortedLinearIndices)
    # print('cellCount', cellCount)
    # print('hashTable', hashTable.shape, hashTable.dtype, hashTable)
    # print('cellLinearIndices', cellLinearIndices.shape, cellLinearIndices.dtype, cellLinearIndices)
    # print('cellOffsets', cellOffsets.shape, cellOffsets.dtype, cellOffsets)
    # print('cellParticleCounters', cellParticleCounters.shape, cellParticleCounters.dtype, cellParticleCounters)

    # return None
    with record_function('buildNeighborList'):
        rows, cols = neighborSearch.buildNeighborList(sortedPositions, sortedSupport, hashTable, cellLinearIndices, cellOffsets, cellParticleCounters, sortingIndices, cellCount, hashMapLength, searchRadius)
    
    # return None
    return rows.to(torch.int64), cols.to(torch.int64), None, None, (qMin, hMax, sortingIndices, sortedPositions, sortedSupport), (hashTable, hashMapLength), (cellLinearIndices, cellOffsets, cellParticleCounters, cellCount)
    

def constructNeighborhoodsPreSorted(queryPositions, querySupports, particleState, hashMap, cellMap, searchRadius : int = 1):
    qMin, hMax, sortingIndices, sortedPositions, sortedSupport = particleState
    hashTable, hashMapLength = hashMap
    cellLinearIndices, cellOffsets, cellParticleCounters, cellCount = cellMap

    # print(qMin.shape, qMin.dtype)
    # print(sortedPositions.shape, sortedPositions.dtype)
    # print(sortedSupport.shape, sortedSupport.dtype)
    # print(queryPositions.shape, querySupports.dtype)
    # debugPrint(hMax)
    # debugPrint()

    with record_function('buildNeighborList'):
        rows, cols = neighborSearch.buildNeighborListUnsortedPerParticle(queryPositions, querySupports, sortedPositions, sortedSupport, hashTable, cellLinearIndices, cellOffsets, cellParticleCounters, sortingIndices, cellCount, hashMapLength, qMin.type(torch.float32), np.float32(hMax), searchRadius)
        
    
    return rows.to(torch.int64), cols.to(torch.int64)
    


def constructNeighborhoodsCUDA(queryPositions, querySupports, hashMapLength :int = -1, supportScale : float = 1.0, minCoord : Optional[torch.Tensor]  = None, maxCoord : Optional[torch.Tensor]  = None, searchRadius : int = 1):
    with record_function('sortPositions'):
        sortedPositions, sortedSupport, sortedLinearIndices, sortingIndices, cellCount, qMin, hMax = sortPositions(queryPositions, querySupports, 1.0, minCoord, maxCoord)
    if hashMapLength == -1:
        hashMapLength = nextprime(queryPositions.shape[0])
#     return None, None, None, None, None
    with record_function('constructHashMap'):
        hashTable, cellLinearIndices, cellOffsets, cellParticleCounters = constructHashMap(sortedPositions, sortedSupport, sortedLinearIndices, sortingIndices, hashMapLength, cellCount)
    sortingIndices = sortingIndices.to(torch.int32)
# 

    with record_function('buildNeighborList'):
        # ctr, offsets, i, j = neighborSearch.buildNeighborListCUDA(queryPositions, querySupports, sortedPositions, sortedSupport, hashTable, cellLinearIndices, cellOffsets, cellParticleCounters, sortingIndices, qMin, hMax, cellCount, hashMapLength, searchRadius)
        i, j = neighborSearch.buildNeighborListCUDA(queryPositions, querySupports, sortedPositions, sortedSupport, hashTable, cellLinearIndices, cellOffsets, cellParticleCounters, sortingIndices, qMin, hMax, cellCount, hashMapLength, searchRadius)
    with record_function('finalize'):        
        j, jj = torch.sort(j, dim = 0, stable = True)
        i = i[jj]
        i, ii = torch.sort(i, dim = 0, stable = True)
        j = j[ii]
       
    return i.to(torch.int64), j.to(torch.int64), None, None, (qMin, hMax, sortingIndices, sortedPositions, sortedSupport), (hashTable, hashMapLength), (cellLinearIndices, cellOffsets, cellParticleCounters, cellCount)

    return i.to(torch.int64), j.to(torch.int64), ctr, offsets, (qMin, hMax, sortingIndices, sortedPositions, sortedSupport), (hashTable, hashMapLength), (cellLinearIndices, cellOffsets, cellParticleCounters, cellCount)

def constructNeighborhoodsPreSortedCUDA(queryPositions, querySupports, particleState, hashMap, cellMap, searchRadius : int = 1):
    qMin, hMax, sortingIndices, sortedPositions, sortedSupport = particleState
    hashTable, hashMapLength = hashMap
    cellLinearIndices, cellOffsets, cellParticleCounters, cellCount = cellMap
    with record_function('buildNeighborList'):
        rows, cols = neighborSearch.buildNeighborListCUDA(queryPositions, querySupports, sortedPositions, sortedSupport, hashTable, cellLinearIndices, cellOffsets, cellParticleCounters, sortingIndices, qMin.type(torch.float32), np.float32(hMax), cellCount, hashMapLength, searchRadius)
        
    
    return rows.to(torch.int64), cols.to(torch.int64)
    
    
class neighborSearchModule(Module):
    def __init__(self):
        super().__init__('densityInterpolation', 'Evaluates density at the current timestep')
        
    def getParameters(self):
        return [
            Parameter('neighborSearch', 'gradientThreshold', 'float', 1e-7, required = False, export = True, hint = ''),
            Parameter('neighborSearch', 'supportScale', 'float', 1.0, required = False, export = True, hint = ''),
            Parameter('neighborSearch', 'sortNeighborhoods', 'bool', True, required = False, export = True, hint = '')
        ]
        
    def initialize(self, simulationConfig, simulationState):
        self.support = simulationConfig['particle']['support']
        self.maxNeighbors = simulationConfig['compute']['maxNeighbors']
        self.threshold = simulationConfig['neighborSearch']['gradientThreshold']
        self.supportScale = simulationConfig['neighborSearch']['supportScale']
        self.sortNeighborhoods = simulationConfig['neighborSearch']['sortNeighborhoods']
        
        self.dtype = simulationConfig['compute']['precision']
        self.device = simulationConfig['compute']['device']

        self.minDomain = simulationConfig['domain']['virtualMin']
        self.maxDomain = simulationConfig['domain']['virtualMax']
        
    def resetState(self, simulationState):
        simulationState.pop('fluidNeighbors', None)
        simulationState.pop('fluidDistances', None)
        simulationState.pop('fluidRadialDistances', None)

    def search(self, simulationState, simulation):
        with record_function("neighborhood - fluid neighbor search"): 
            queryPositions = simulationState['fluidPosition']
            querySupports = simulationState['fluidSupport']

            # _ = constructNeighborhoods(queryPositions, querySupports, -1, minCoord = torch.tensor(self.minDomain), maxCoord = torch.tensor(self.maxDomain))


            if queryPositions.is_cuda:
                row, col, ctr, offsets, self.sortInfo, self.hashMap, self.cellMap = constructNeighborhoodsCUDA(queryPositions, querySupports, -1, minCoord = torch.tensor(self.minDomain,device=self.device,dtype=self.dtype), maxCoord = torch.tensor(self.maxDomain,device=self.device,dtype=self.dtype))
            else:
                row, col, ctr, offsets, self.sortInfo, self.hashMap, self.cellMap = constructNeighborhoods(queryPositions, querySupports, -1, minCoord = torch.tensor(self.minDomain), maxCoord = torch.tensor(self.maxDomain))

            fluidNeighbors = torch.stack([row, col], dim = 0)

            fluidDistances = (simulationState['fluidPosition'][fluidNeighbors[0]] - simulationState['fluidPosition'][fluidNeighbors[1]])
            fluidRadialDistances = torch.linalg.norm(fluidDistances,axis=1)

            fluidDistances[fluidRadialDistances < self.threshold,:] = 0
            fluidDistances[fluidRadialDistances >= self.threshold,:] /= fluidRadialDistances[fluidRadialDistances >= self.threshold,None]
            fluidRadialDistances /= self.support

            simulationState['fluidNeighbors'] = fluidNeighbors
            simulationState['fluidDistances'] = fluidDistances
            simulationState['fluidRadialDistances'] = fluidRadialDistances

            return fluidNeighbors, fluidDistances, fluidRadialDistances
        
    def searchExisting(self, queryPositions, querySupports, simulationState, simulation, searchRadius :int = 1):
        with record_function("neighborhood - searching existing"): 
            # queryPositions = simulationState['fluidPosition'].to('cpu')
            # querySupports = simulationState['fluidSupport'].to('cpu')
            if queryPositions.is_cuda:
                rows, cols = constructNeighborhoodsPreSortedCUDA(queryPositions, querySupports,  self.sortInfo, self.hashMap, self.cellMap, searchRadius = searchRadius)
            else:
                rows, cols = constructNeighborhoodsPreSorted(queryPositions, querySupports,  self.sortInfo, self.hashMap, self.cellMap, searchRadius = searchRadius)
            # rows = rows.to(self.device)
            # cols = cols.to(self.device)
            
#             row, col = radius(simulationState['fluidPosition'], simulationState['fluidPosition'], self.support, max_num_neighbors = self.maxNeighbors)
            fluidNeighbors = torch.stack([rows, cols], dim = 0)

            fluidDistances = (queryPositions[fluidNeighbors[0]] - simulationState['fluidPosition'][fluidNeighbors[1]])
            fluidRadialDistances = torch.linalg.norm(fluidDistances,axis=1)

            fluidDistances[fluidRadialDistances < self.threshold,:] = 0
            fluidDistances[fluidRadialDistances >= self.threshold,:] /= fluidRadialDistances[fluidRadialDistances >= self.threshold,None]
            fluidRadialDistances /= querySupports[rows]

            return fluidNeighbors, fluidDistances, fluidRadialDistances
