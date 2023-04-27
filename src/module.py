# Basic parent class for modules
# This class is used to build a set of functions that always exist to allow for some simple polymorphism
# The default behavior is to not do anything
class Module():
    def getParameters(self):
        return None
    def initialize(self, config, state):
        return
    def finalize(self):
        return
    def resetState(self, simulationState):
        return
    def saveState(self, perennialState, copy):
        return
    def exportState(self, simulationState, simulation, group, mask):
        return
    def setupSimulationState(self, perennialState):
        return
    def __init__(self, identifier, moduleDescription):
        self.description = moduleDescription
        self.name = identifier
        return

# Boundary modules ahve more features that are required either by a DFSPH Solver or by deltaSPH
# So these functions serve as stubs that throw an exception if a function is not implemented
# as the underlying solvers cannot function without an implementation of these functions
class BoundaryModule(Module):
    def getParameters(self):
        return None
    def initialize(self, config, state):
        return
    def finalize(self):
        return
    def __init__(self, identifier, moduleDescription):
        super().__init__(identifier, moduleDescription)
        return
    def dfsphPrepareSolver(self, simulationState, simulation, density):
        raise Exception('Operation dfsphPrepareSolver not implemented for ', self.identifier)
    def dfsphBoundaryAccelTerm(self, simulationState, simulation, density):
        raise Exception('Operation dfsphBoundaryAccelTerm not implemented for ', self.identifier)
    def dfsphBoundaryPressureSum(self, simulationState, simulation, density):
        raise Exception('Operation dfsphBoundaryPressureSum not implemented for ', self.identifier)
    def dfsphBoundaryAlphaTerm(self, simulationState, simulation, density):
        raise Exception('Operation dfsphBoundaryAlphaTerm not implemented for ', self.identifier)
    def dfsphBoundarySourceTerm(self, simulationState, simulation, density):
        raise Exception('Operation dfsphBoundarySourceTerm not implemented for ', self.identifier)
    def evalBoundaryPressure(self, simulationState, simulation, density):
        raise Exception('Operation boundaryPressure not implemented for ', self.identifier)
    def evalBoundaryDensity(self, simulationState, simulation):
        raise Exception('Operation boundaryDensity not implemented for ', self.identifier)
    def evalBoundaryFriction(self, simulationState, simulation):
        raise Exception('Operation boundaryFriction not implemented for ', self.identifier)
    def boundaryNeighborhoodSearch(self, simulationState, simulation):
        raise Exception('Operation boundaryNeighborsearch not implemented for ', self.identifier)
    def boundaryFilterNeighborhoods(self, simulationState, simulation):
        return # Default behavior here is do nothing so no exception needs to be thrown as this function is optional
