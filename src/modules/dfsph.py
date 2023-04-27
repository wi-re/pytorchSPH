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

from ..kernels import kernel, spikyGrad, kernelGradient
from ..module import Module
from ..parameter import Parameter
from ..util import *


@torch.jit.script
def computeFluidAcceleration(fluidArea, fluidDensity, fluidRestDensity, fluidPressure2, fluidNeighbors, fluidDistances, fluidRadialDistances, support):
    with record_function("DFSPH - accel (fluid)"): 
        i = fluidNeighbors[0]
        j = fluidNeighbors[1]
        with record_function("DFSPH - accel (fluid) [gradient]"): 
            grad = spikyGrad(fluidRadialDistances, fluidDistances, support)

        with record_function("DFSPH - accel (fluid) [factor]"): 
            fac = -(fluidArea * fluidRestDensity)[j]
            p = fluidPressure2 / (fluidDensity * fluidRestDensity)**2
            pi = p[i]
            pj = p[j]
        with record_function("DFSPH - accel (fluid) [mul]"): 
            term = (fac * (pi + pj))[:,None] * grad
        with record_function("DFSPH - accel (fluid) [scatter]"): 
            fluidAccelTerm = scatter_sum(term, i, dim=0, dim_size=fluidArea.shape[0])
        return fluidAccelTerm
@torch.jit.script
def computeUpdatedPressureFluidSum(fluidActualArea, fluidPredAccel, fluidNeighbors, fluidRadialDistances, fluidDistances, support, dt):
    with record_function("DFSPH - pressure (fluid)"): 
        i = fluidNeighbors[0]
        j = fluidNeighbors[1]
        with record_function("DFSPH - pressure (fluid) [gradient]"): 
            grad = spikyGrad(fluidRadialDistances, fluidDistances, support)

        with record_function("DFSPH - pressure (fluid) [factor]"): 
            fac = dt**2 * fluidActualArea[j]
            aij = fluidPredAccel[i] - fluidPredAccel[j]
        with record_function("DFSPH - pressure (fluid) [scatter]"): 
            kernelSum = scatter_sum(torch.einsum('nd, nd -> n', fac[:,None] * aij, grad), i, dim=0, dim_size=fluidActualArea.shape[0])

        return kernelSum
@torch.jit.script
def computeAlphaFluidTerm(fluidArea, fluidRestDensity, fluidActualArea, fluidNeighbors, fluidRadialDistances, fluidDistances, support):
    with record_function("DFSPH - alpha (fluid)"): 
        i = fluidNeighbors[0]
        j = fluidNeighbors[1]
        with record_function("DFSPH - alpha (fluid) [gradient]"): 
            grad = spikyGrad(fluidRadialDistances, fluidDistances, support)
            grad2 = torch.einsum('nd, nd -> n', grad, grad)

        with record_function("DFSPH - alpha (fluid) [term]"): 
            term1 = fluidActualArea[j][:,None] * grad
            term2 = (fluidActualArea**2 / (fluidArea * fluidRestDensity))[j] * grad2

        with record_function("DFSPH - alpha (fluid) [scatter]"): 
            kSum1 = scatter_sum(term1, i, dim=0, dim_size=fluidArea.shape[0])
            kSum2 = scatter_sum(term2, i, dim=0, dim_size=fluidArea.shape[0])
            
        return kSum1, kSum2
@torch.jit.script
def computeAlphaFinal(kSum1, kSum2, dt, fluidArea, fluidActualArea, fluidRestDensity):
    with record_function("DFSPH - alpha (final)"): 
        fac = - dt **2 * fluidActualArea
        mass = fluidArea * fluidRestDensity
        alpha = fac / mass * torch.einsum('nd, nd -> n', kSum1, kSum1) + fac * kSum2
        alpha = torch.clamp(alpha, -1, -1e-7)
        return alpha

@torch.jit.script
def computeSourceTermFluid(fluidActualArea, fluidPredictedVelocity, fluidNeighbors, fluidRadialDistances, fluidDistances, support, dt : float):
    with record_function("DFSPH - source (fluid)"): 
        i = fluidNeighbors[0]
        j = fluidNeighbors[1]
        with record_function("DFSPH - source (fluid) [term]"): 
            fac = - dt * fluidActualArea[j]
            vij = fluidPredictedVelocity[i] - fluidPredictedVelocity[j]
        with record_function("DFSPH - source (fluid) [gradient]"): 
            grad = spikyGrad(fluidRadialDistances, fluidDistances, support)
            prod = torch.einsum('nd, nd -> n', vij, grad)

        with record_function("DFSPH - source (fluid) [scatter]"): 
            source = scatter_sum(fac * prod, i, dim=0, dim_size=fluidActualArea.shape[0])

        return source


class dfsphModule(Module):
    def getParameters(self):
        return [
            Parameter('dfsph', 'minDensitySolverIterations', 'int', 2, required = False, export = True, hint = ''),
            Parameter('dfsph', 'minDivergenceSolverIterations', 'int', 2, required = False, export = True, hint = ''),
            Parameter('dfsph', 'maxDensitySolverIterations', 'int', 256, required = False, export = True, hint = ''),
            Parameter('dfsph', 'maxDivergenceSolverIterations', 'int', 8, required = False, export = True, hint = ''),
            Parameter('dfsph', 'densityThreshold', 'float', 1e-3, required = False, export = True, hint = ''),
            Parameter('dfsph', 'divergenceThreshold', 'float', 1e-2, required = False, export = True, hint = ''),
            Parameter('dfsph', 'divergenceSolver', 'bool', True, required = False, export = True, hint = ''),
            Parameter('dfsph', 'relaxedJacobiOmega', 'float', 0.5, required = False, export = True, hint = '')
        ]
        
    def __init__(self):
        super().__init__('densityInterpolation', 'Evaluates density at the current timestep')
    
    def initialize(self, simulationConfig, simulationState):
        self.support = simulationConfig['particle']['support']
        # self.kernel, self.gradientKernel = getKernelFunctions(simulationConfig['kernel']['defaultKernel'])
        
        
        self.minDensitySolverIterations = simulationConfig['dfsph']['minDensitySolverIterations']
        self.minDivergenceSolverIterations = simulationConfig['dfsph']['minDivergenceSolverIterations']
        self.maxDensitySolverIterations = simulationConfig['dfsph']['maxDensitySolverIterations']
        self.maxDivergenceSolverIterations = simulationConfig['dfsph']['maxDivergenceSolverIterations']
        self.densityThreshold = simulationConfig['dfsph']['densityThreshold']
        self.divergenceThreshold = simulationConfig['dfsph']['divergenceThreshold']
#         self.divergenceSolver - simulationConfig['dfsph']['divergenceSolver']
        self.relaxedJacobiOmega = simulationConfig['dfsph']['relaxedJacobiOmega']  if 'dfsph'in simulationConfig else 0.5
    
        self.backgroundPressure = simulationConfig['fluid']['backgroundPressure']
        
        self.boundaryScheme = simulationConfig['simulation']['boundaryScheme']
        self.boundaryCounter = len(simulationConfig['solidBC']) if 'solidBC' in simulationConfig else 0

        self.pressureScheme = simulationConfig['pressure']['boundaryPressureTerm'] 
        self.computeBodyForces = simulationConfig['simulation']['bodyForces'] 
        
        
    def computeAlpha(self, simulationState, simulation, density = True):
        with record_function("DFSPH - alpha"): 
            kSum1, kSum2 = computeAlphaFluidTerm(simulationState['fluidArea'], simulationState['fluidRestDensity'], simulationState['fluidActualArea'], simulationState['fluidNeighbors'], simulationState['fluidRadialDistances'], simulationState['fluidDistances'], self.support)

            bdykSum1, bdykSum2 = simulation.boundaryModule.dfsphBoundaryAlphaTerm(simulationState, simulation, density)
            kSum1 += bdykSum1
            kSum2 += bdykSum2
            
            return computeAlphaFinal(kSum1, kSum2, simulationState['dt'], simulationState['fluidArea'], simulationState['fluidActualArea'], simulationState['fluidRestDensity'])
        

        
    def computeSourceTerm(self, simulationState, simulation, density = True):
        with record_function("DFSPH - source"): 
            source = computeSourceTermFluid(simulationState['fluidActualArea'], simulationState['fluidPredictedVelocity'], simulationState['fluidNeighbors'], simulationState['fluidRadialDistances'], simulationState['fluidDistances'], self.support, simulationState['dt'])
            
            source = source + simulation.boundaryModule.dfsphBoundarySourceTerm(simulationState, simulation, density)

                                
            return 1. - simulationState['fluidDensity'] + source if density else source            
        
        
    def computeUpdatedPressure(self, simulationState, simulation, density = True):
        with record_function("DFSPH - pressure"): 
            kernelSum = computeUpdatedPressureFluidSum(simulationState['fluidActualArea'], simulationState['fluidPredAccel'], simulationState['fluidNeighbors'], simulationState['fluidRadialDistances'], simulationState['fluidDistances'], self.support, simulationState['dt'])

            bdyKernelSum = simulation.boundaryModule.dfsphBoundaryPressureSum(simulationState, simulation, density)
            kernelSum += bdyKernelSum
            # kernelSum = -kernelSum


            residual = kernelSum - simulationState['fluidSourceTerm']
            self.relaxedJacobiOmega
            pressure = simulationState['fluidPressure'] - 0.3 * residual / simulationState['fluidAlpha']
            pressure = torch.clamp(pressure, min = 0.) if density else pressure
            if density and self.backgroundPressure:
                pressure = torch.clamp(pressure, min = (5**2) * simulationState['fluidRestDensity'])
            if torch.any(torch.isnan(pressure)) or torch.any(torch.isinf(pressure)):
                raise Exception('Pressure solver became unstable!')

            return pressure, residual

        
        
    def computeAcceleration(self, simulationState, simulation, density = True):
        with record_function("DFSPH - accel"):
            fluidAccelTerm = computeFluidAcceleration(simulationState['fluidArea'], simulationState['fluidDensity'], simulationState['fluidRestDensity'], simulationState['fluidPressure2'], simulationState['fluidNeighbors'], simulationState['fluidDistances'], simulationState['fluidRadialDistances'], self.support)

            return fluidAccelTerm + simulation.boundaryModule.dfsphBoundaryAccelTerm(simulationState, simulation, density)




    def densitySolve(self, simulationState, simulation):
        with record_function("DFSPH - densitySolve"): 
            errors = []
            i = 0
            error = 0.
            minIters = self.minDensitySolverIterations
#             minIters = 32
            if 'densityErrors' in simulationState:
                minIters = max(minIters, len(simulationState['densityErrors'])*0.75)

            while((i < minIters or \
                    error > self.densityThreshold) and \
                    i <= self.maxDensitySolverIterations):
                
                with record_function("DFSPH - densitySolve (iteration)"): 
                    with record_function("DFSPH - densitySolve (computeAccel)"): 
                        simulationState['fluidPredAccel'] = self.computeAcceleration(simulationState, simulation, True)
                        # simulation.sync(simulationState['fluidPredAccel'])
                        # simulation.periodicBC.syncQuantity(simulationState['fluidPredAccel'], simulationState, simulation)
                        simulationState['fluidPressure'][:] = simulationState['fluidPressure2'][:]

                    with record_function("DFSPH - densitySolve (updatePressure)"): 
                        simulationState['fluidPressure2'], simulationState['residual'] = self.computeUpdatedPressure(simulationState, simulation, True)             
                        # simulation.sync(simulationState['fluidPressure2'])
                        # simulation.periodicBC.syncQuantity(simulationState['fluidPressure2'], simulationState, simulation)
                        # debugPrint(self.boundaryScheme)
                        # debugPrint(self.pressureScheme)
                        if self.pressureScheme == 'PBSPH':
                            boundaryError = torch.sum(torch.clamp(simulation.boundaryModule.boundaryResidual, min = -self.densityThreshold))
                            fluidError = torch.sum(torch.clamp(simulationState['residual'], min = -self.densityThreshold))
                            error = (fluidError + boundaryError) / (simulation.boundaryModule.numPtcls + simulationState['residual'].shape[0])
                            # debugPrint(boundaryError)
                        else:
                            error = torch.mean(torch.clamp(simulationState['residual'], min = -self.densityThreshold))# * simulationState['fluidArea'])
                        
                    errors.append((error).item())
                    i = i + 1
            simulationState['densityErrors'] = errors
            simulationState['densitySolverPressure'] = simulationState['fluidPressure']
            return errors

    def divergenceSolve(self, simulationState, simulation):
        with record_function("DFSPH - divergenceSolve"): 
            errors = []
            i = 0
            error = 0.
            while((i < self.minDivergenceSolverIterations or error > self.divergenceThreshold) and i <= self.maxDivergenceSolverIterations):
                
                with record_function("DFSPH - divergenceSolve (iteration)"): 
                    with record_function("DFSPH - divergenceSolve (computeAccel)"): 
                        simulationState['fluidPredAccel'] = self.computeAcceleration(simulationState, simulation, False)
                        # simulation.sync(simulationState['fluidPredAccel'])
                        # simulation.periodicBC.syncQuantity(simulationState['fluidPredAccel'], simulationState, simulation)
                        simulationState['fluidPressure'][:] = simulationState['fluidPressure2'][:]

                    with record_function("DFSPH - divergenceSolve (updatePressure)"): 
                        simulationState['fluidPressure2'], simulationState['residual'] = self.computeUpdatedPressure(simulationState, simulation, False)                    
                        # simulation.sync(simulationState['fluidPressure2'])
                        # simulation.periodicBC.syncQuantity(simulationState['fluidPressure2'], simulationState, simulation)
                        error = torch.mean(torch.clamp(simulationState['residual'], min = -self.divergenceThreshold))# * simulationState['fluidArea'])
                        
                    errors.append((error).item())
                    i = i + 1
            simulationState['divergenceErrors'] = errors
            simulationState['divergenceSolverPressure'] = simulationState['fluidPressure']
            return errors


    def DFSPH(self, simulationState, simulation, density = True): 
        with record_function("DFSPH - solver"): 
            with record_function("DFSPH - predict velocity"): 
                simulationState['fluidPredAccel'] = torch.zeros(simulationState['fluidPosition'].shape, dtype = simulationState['fluidPosition'].dtype, device = simulationState['fluidPosition'].device)
                simulationState['fluidPredictedVelocity'] = simulationState['fluidVelocity'] + simulationState['dt'] * simulationState['fluidAcceleration']
                simulationState['fluidActualArea'] = simulationState['fluidArea'] / simulationState['fluidDensity']

            with record_function("DFSPH - compute alpha"): 
                simulation.boundaryModule.dfsphPrepareSolver(simulationState, simulation, density)

                simulationState['fluidAlpha'] = self.computeAlpha(simulationState, simulation, density)
                # simulation.sync(simulationState['fluidAlpha'])
                # simulation.periodicBC.syncQuantity(simulationState['fluidAlpha'], simulationState, simulation)

            with record_function("DFSPH - compute source"):
                simulationState['fluidSourceTerm'] = self.computeSourceTerm(simulationState, simulation, density)
                # simulation.sync(simulationState['fluidSourceTerm'])
                # simulation.periodicBC.syncQuantity(simulationState['fluidSourceTerm'], simulationState, simulation)
                
            with record_function("DFSPH - initialize pressure"):
                simulationState['fluidPressure2'] =  torch.zeros(simulationState['numParticles'], dtype = simulationState['fluidPosition'].dtype, device = simulationState['fluidPosition'].device)

                # if 'densitySolverPressure' in simulationState and density:
                    # simulationState['fluidPressure2'] =  simulationState['densitySolverPressure'] * 0.5
                # else:
                simulationState['fluidPressure2'] = torch.zeros(simulationState['numParticles'], dtype = simulationState['fluidPosition'].dtype, device = simulationState['fluidPosition'].device)
                simulationState['fluidPressure'] = torch.zeros(simulationState['numParticles'], dtype = simulationState['fluidPosition'].dtype, device = simulationState['fluidPosition'].device)
                
                # simulation.sync(simulationState['fluidPressure2'])
                # simulation.periodicBC.syncQuantity(simulationState['fluidPressure2'], simulationState, simulation)
                totalArea = torch.sum(simulationState['fluidArea'])


            with record_function("DFSPH - solver step"):
                if density:
                    errors = self.densitySolve(simulationState, simulation)
                else:
                    errors = self.divergenceSolve(simulationState, simulation)

                
            with record_function("DFSPH - compute accel"):
                simulationState['fluidPredAccel'] = self.computeAcceleration(simulationState, simulation, density)
                # simulation.sync(simulationState['fluidPredAccel'])
                # simulation.periodicBC.syncQuantity(simulationState['fluidPredAccel'], simulationState, simulation)
                simulationState['fluidPressure'][:] = simulationState['fluidPressure2'][:]

                simulationState['fluidPredictedVelocity'] += simulationState['dt'] * simulationState['fluidPredAccel']

            return errors
        
    def incompressibleSolver(self, simulationState, simulation):
        with record_function("DFSPH - incompressibleSolver"): 
            return self.DFSPH(simulationState, simulation, True)
    def divergenceSolver(self, simulationState, simulation):
        with record_function("DFSPH - divergenceSolver"): 
            return self.DFSPH(simulationState, simulation, False)