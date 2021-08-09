using StaticArrays
using LinearAlgebra
using SuiteSparse
using SparseArrays
using LoopVectorization
using IterativeSolvers
using Preconditioners
using HDF5
using TimerOutputs
# using MKL
# using Pardiso

import Statistics: mean

include("src/Grid/mesher.jl")
include("src/Rheology/Rheology.jl")
include("src/Stokes/StokesSolver.jl")
# include("src/Stokes/StokesPenalty.jl")
include("src/ThermalDiffusion/ThermalSolver.jl")
include("src/Stress/FSE.jl")
include("src/Stress/StressCalculation.jl")
include("src/IterativeSolvers.jl")
include("src/PardisoSolvers.jl")
include("src/Utilities.jl")
include("src/Algebra/Quadrature.jl")
include("src/Algebra/Sparsity.jl")
include("src/Algebra/Algebra.jl")
include("src/IO/IOmanager.jl")
include("src/Setup/Setup.jl")
include("src/Rheology/anisotropy.jl")
include("src/PiC/tsearch.jl")
include("src/PiC/PiC.jl")
include("src/Advection/Advection.jl")

include("src/Grid/coloring.jl")
include("src/Grid/DofHandler.jl")
include("src/ThermalDiffusion/AssemblerThermal.jl")
include("src/Stokes/AssemblerStokes.jl")
include("src/Setup/boundary_conditions.jl")
include("src/PiC/CubicInterpolations.jl")