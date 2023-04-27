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

import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator
import matplotlib.ticker as mticker






class deltaSPHModule(Module):
    def getParameters(self):
        return [
            # Parameter('deltaSPH', 'alpha', 'float', 0.01, required = False, export = True, hint = ''),
            # Parameter('deltaSPH', 'delta', 'float', 0.1, required = False, export = True, hint = ''),
            # Parameter('deltaSPH', 'c0', 'float', -1.0, required = False, export = True, hint = ''),
            # Parameter('deltaSPH', 'gamma', 'float', 7.0, required = False, export = True, hint = ''),
            # Parameter('deltaSPH', 'beta', 'float', -1.0, required = False, export = True, hint = ''),
            Parameter('deltaSPH', 'HughesGrahamCorrection', 'bool', True, required = False, export = True, hint = '')
        ]
        
    def __init__(self):
        super().__init__('densityInterpolation', 'Evaluates density at the current timestep')
    
    def initialize(self, simulationConfig, simulationState):
        self.support = simulationConfig['particle']['support']    
        self.backgroundPressure = simulationConfig['fluid']['backgroundPressure']
        self.restDensity = simulationConfig['fluid']['restDensity']
        self.gamma = simulationConfig['pressure']['gamma']
        
        self.boundaryScheme = simulationConfig['simulation']['boundaryScheme']
        self.boundaryCounter = len(simulationConfig['solidBC']) if 'solidBC' in simulationConfig else 0

        self.pressureScheme = simulationConfig['pressure']['fluidPressureTerm'] 
        self.computeBodyForces = simulationConfig['simulation']['bodyForces'] 
        
        self.dtype = simulationConfig['compute']['precision']
        self.device = simulationConfig['compute']['device'] 
        
        self.alpha = simulationConfig['diffusion']['alpha']
        self.delta = simulationConfig['diffusion']['delta'] 
        dx = simulationConfig['particle']['support'] * simulationConfig['particle']['packing']
        c0 = 10.0 * np.sqrt(2.0*9.81*0.3)
        h0 = simulationConfig['particle']['support']
        dt = 0.25 * h0 / (1.1 * c0)
        if simulationConfig['fluid']['c0'] < 0:
            simulationConfig['fluid']['c0'] = c0
        
        self.c0 = simulationConfig['fluid']['c0']
        self.eps = self.support **2 * 0.1
        

        
        
    # def computeTerms(self, simulationState, simulation):
        # with record_function('deltaSPH - compute terms'):
            # self.computeNormalizationMatrices(simulationState, simulation)
            # self.computeRenormalizedDensityGradient(simulationState, simulation)
            # self.computeDensityDiffusion(simulationState, simulation)
            # self.computeDpDt(simulationState, simulation)
            # self.computePressure(simulationState, simulation)
            # # self.computeVelocityDiffusion(simulationState, simulation)
            # self.computePressureAcceleration(simulationState, simulation)
        
    # def integrate(self, simulationState, simulation):
        # with record_function('deltaSPH - integration'):
            # simulationState['fluidAcceleration'] += self.pressureAccel + self.velocityDiffusion
            # simulationState['fluidVelocity'] += simulationState['dt'] * simulationState['fluidAcceleration']
            # simulationState['fluidPosition'] += simulationState['dt'] * simulationState['fluidVelocity']
            # simulationState['fluidDensity'] += simulationState['dt'] * self.dpdt / self.restDensity
            # simulation.boundaryModule.integrate(simulationState, simulation)
        

    def plotState(self, simulationState, simulation):
        fig, axis = plt.subplots(3,6, figsize=(22, 12), squeeze = False, sharex = True, sharey = True)
        for axx in axis:
            for ax in axx:
                ax.axis('equal')
                ax.set_xlim(simulation.config['domain']['virtualMin'][0], simulation.config['domain']['virtualMax'][0])
                ax.set_ylim(simulation.config['domain']['virtualMin'][1], simulation.config['domain']['virtualMax'][1])
                ax.axvline(simulation.config['domain']['min'][0], ls= '--', c = 'black')
                ax.axvline(simulation.config['domain']['max'][0], ls= '--', c = 'black')
                ax.axhline(simulation.config['domain']['min'][1], ls= '--', c = 'black')
                ax.axhline(simulation.config['domain']['max'][1], ls= '--', c = 'black')

#         positions = simulationState['fluidPosition']
#         M = deltaModule.dpdt

        # bPositions = simulation.boundaryModule.boundaryPositions
        # bM = simulation.boundaryModule.boundaryL
        # bPositions = simulation.boundaryModule.ghostParticlePosition
        # bM = boundaryNormalizationMatrix

        # positions = torch.vstack((positions, bPositions)).detach().cpu().numpy()
        # M = torch.vstack((M, bM)).detach().cpu().numpy()

        # positions = bPositions.detach().cpu().numpy()
        # M = bM.detach().cpu().numpy()

        def scatterPlot(axis, positions, data,title = None):
            if title:
                axis.set_title(title)
            sc = axis.scatter(positions[:,0], positions[:,1], c = data, s = 0.25)
            ax1_divider = make_axes_locatable(axis)
            cax1 = ax1_divider.append_axes("right", size="4%", pad="1%")
            cbar = fig.colorbar(sc, cax=cax1,orientation='vertical')
            cbar.ax.tick_params(labelsize=8) 
            return sc, cbar

        plots = []

        plots.append(scatterPlot(axis[0,0], simulationState['fluidPosition'].detach().cpu().numpy(), self.normalizationMatrix[:,0,0].detach().cpu().numpy(), 'L^-1[0,0]'))
        plots.append(scatterPlot(axis[0,1], simulationState['fluidPosition'].detach().cpu().numpy(), self.normalizationMatrix[:,0,1].detach().cpu().numpy(), 'L^-1[0,1]'))
        plots.append(scatterPlot(axis[1,0], simulationState['fluidPosition'].detach().cpu().numpy(), self.normalizationMatrix[:,1,0].detach().cpu().numpy(), 'L^-1[1,0]'))
        plots.append(scatterPlot(axis[1,1], simulationState['fluidPosition'].detach().cpu().numpy(), self.normalizationMatrix[:,1,1].detach().cpu().numpy(), 'L^-1[1,1]'))
        
#         plots.append(scatterPlot(axis[0,0], simulationState['fluidPosition'].detach().cpu().numpy(), self.fluidL[:,0,0].detach().cpu().numpy(), 'L^-1[0,0]'))
#         plots.append(scatterPlot(axis[0,1], simulationState['fluidPosition'].detach().cpu().numpy(), self.fluidL[:,0,1].detach().cpu().numpy(), 'L^-1[0,1]'))
#         plots.append(scatterPlot(axis[1,0], simulationState['fluidPosition'].detach().cpu().numpy(), self.fluidL[:,1,0].detach().cpu().numpy(), 'L^-1[1,0]'))
#         plots.append(scatterPlot(axis[1,1], simulationState['fluidPosition'].detach().cpu().numpy(), self.fluidL[:,1,1].detach().cpu().numpy(), 'L^-1[1,1]'))

#         plots.append(scatterPlot(axis[2,0], simulationState['fluidPosition'].detach().cpu().numpy(), self.renormalizedDensityGradient[:,0].detach().cpu().numpy(), '<Vrho>_x'))
#         plots.append(scatterPlot(axis[2,1], simulationState['fluidPosition'].detach().cpu().numpy(), self.renormalizedDensityGradient[:,1].detach().cpu().numpy(), '<Vrho>_y'))

        plots.append(scatterPlot(axis[2,0], simulationState['fluidPosition'].detach().cpu().numpy(), self.eigVals[:,0].detach().cpu().numpy(), 'ev0'))
        plots.append(scatterPlot(axis[2,1], simulationState['fluidPosition'].detach().cpu().numpy(), self.eigVals[:,1].detach().cpu().numpy(), 'ev1'))

        plots.append(scatterPlot(axis[2,2], simulationState['fluidPosition'].detach().cpu().numpy(), self.densityDiffusion.detach().cpu().numpy(), 'rho diff'))
        plots.append(scatterPlot(axis[2,3], simulationState['fluidPosition'].detach().cpu().numpy(), self.dpdt.detach().cpu().numpy(), 'dpdt'))
        plots.append(scatterPlot(axis[2,4], simulationState['fluidPosition'].detach().cpu().numpy(), self.pressure.detach().cpu().numpy(), 'p'))

        plots.append(scatterPlot(axis[0,2], simulationState['fluidPosition'].detach().cpu().numpy(), self.velocityDiffusion[:,0].detach().cpu().numpy(), 'v diff_x'))
        plots.append(scatterPlot(axis[1,2], simulationState['fluidPosition'].detach().cpu().numpy(), self.velocityDiffusion[:,1].detach().cpu().numpy(), 'v diff_y'))

        plots.append(scatterPlot(axis[0,3], simulationState['fluidPosition'].detach().cpu().numpy(), self.pressureAccel[:,0].detach().cpu().numpy(), 'vp_x'))
        plots.append(scatterPlot(axis[1,3], simulationState['fluidPosition'].detach().cpu().numpy(), self.pressureAccel[:,1].detach().cpu().numpy(), 'vp_y'))
        
        plots.append(scatterPlot(axis[0,4], simulationState['fluidPosition'].detach().cpu().numpy(), simulationState['fluidAcceleration'][:,0].detach().cpu().numpy(), 'dv/dt_x'))
        plots.append(scatterPlot(axis[1,4], simulationState['fluidPosition'].detach().cpu().numpy(), simulationState['fluidAcceleration'][:,1].detach().cpu().numpy(), 'dv/dt_y'))

        plots.append(scatterPlot(axis[0,5], simulationState['fluidPosition'].detach().cpu().numpy(), simulationState['fluidVelocity'][:,0].detach().cpu().numpy(), 'v_x'))
        plots.append(scatterPlot(axis[1,5], simulationState['fluidPosition'].detach().cpu().numpy(), simulationState['fluidVelocity'][:,1].detach().cpu().numpy(), 'v_y'))
        
        plots.append(scatterPlot(axis[2,5], simulationState['fluidPosition'].detach().cpu().numpy(), simulationState['fluidDensity'].detach().cpu().numpy(), 'rho'))

        fig.tight_layout()

        self.plots = plots
        self.fig = fig
#         debugPrint(plots)
        
    def updatePlots(self, simulationState, simulation):
        positions = simulationState['fluidPosition'].detach().cpu().numpy()
        for i, (sc, cbar) in enumerate(self.plots):
            data = []
#             if i ==  0: data = self.fluidL[:,0,0].detach().cpu().numpy()
#             if i ==  1: data = self.fluidL[:,0,1].detach().cpu().numpy()
#             if i ==  2: data = self.fluidL[:,1,0].detach().cpu().numpy()
#             if i ==  3: data = self.fluidL[:,1,1].detach().cpu().numpy()
            if i ==  0: data = self.fluidNormalizationMatrix[:,0,0].detach().cpu().numpy()
            if i ==  1: data = self.fluidNormalizationMatrix[:,0,1].detach().cpu().numpy()
            if i ==  2: data = self.fluidNormalizationMatrix[:,1,0].detach().cpu().numpy()
            if i ==  3: data = self.fluidNormalizationMatrix[:,1,1].detach().cpu().numpy()
#             if i ==  4: data = self.renormalizedDensityGradient[:,0].detach().cpu().numpy()
#             if i ==  5: data = self.renormalizedDensityGradient[:,1].detach().cpu().numpy()
            if i ==  4: data = self.eigVals[:,0].detach().cpu().numpy()
            if i ==  5: data = self.eigVals[:,1].detach().cpu().numpy()
            if i ==  6: data = self.densityDiffusion.detach().cpu().numpy()
            if i ==  7: data = self.dpdt.detach().cpu().numpy()                
            if i ==  8: data = self.pressure.detach().cpu().numpy()
            if i ==  9: data = self.velocityDiffusion[:,0].detach().cpu().numpy()
            if i == 10: data = self.velocityDiffusion[:,1].detach().cpu().numpy()
            if i == 11: data = self.pressureAccel[:,0].detach().cpu().numpy()
            if i == 12: data = self.pressureAccel[:,1].detach().cpu().numpy()
            if i == 13: data = simulationState['fluidAcceleration'][:,0].detach().cpu().numpy()
            if i == 14: data = simulationState['fluidAcceleration'][:,1].detach().cpu().numpy()
            if i == 15: data = simulationState['fluidVelocity'][:,0].detach().cpu().numpy()
            if i == 16: data = simulationState['fluidVelocity'][:,1].detach().cpu().numpy()
            if i == 17: data = simulationState['fluidDensity'].detach().cpu().numpy()
            
            sc.set_offsets(positions)
            sc.set_array(data)
            cbar.mappable.set_clim(vmin=np.min(data), vmax=np.max(data))
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


        