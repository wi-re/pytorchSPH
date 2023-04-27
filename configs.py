incompressibleDamBreakConfig = """
[xsph]
fluidViscosity = 0.01
boundaryViscosity = 0.01

[pressure]
kappa = 1.5
gamma = 7.0

[timestep]
fixed = false
max = 0.005

[dfsph]
minDensitySolverIterations = 2
minDivergenceSolverIterations = 2
maxDensitySolverIterations = 256
maxDivergenceSolverIterations = 8
densityThreshold = 1e-3
divergenceThreshold = 1e-2
divergenceSolver = false
backgroundPressure = false
relaxedJacobiOmega = 0.5

[domain]
min = [-2, -2]
max = [2, 2]
adjustParticle = true
adjustDomain = true

[periodicBCs]
periodicX = true

[velocitySources]
[velocitySources.one]
min = [1,-2]
max = [2,2]
velocity = [0,1]

[emitter.fluidBulk]
fillDomain = false
min = [-0.5,-2]
max = [0.5,1.0]
velocity = [ 0.0, 0.0]
adjust = true

[emitter2.fluidBulk]
fillDomain = false
min = [-2,-2]
max = [-1.5, -1.5]
velocity = [ 0.0, 0.0]
adjust = true

[compute]
device='cpu'

[particle]
radius = 0.01

[simulation]
boundaryScheme = 'solid'
pressureTerm1 = 'PBSPH'
pressureTerm = 'ghostMLS'
pressureTerm2 = 'deltaMLS'
pressureTerm3 = 'MLSPressure'
verbose = false

[akinciBoundary]
beta = 0.125
gamma = 0.7
""" 