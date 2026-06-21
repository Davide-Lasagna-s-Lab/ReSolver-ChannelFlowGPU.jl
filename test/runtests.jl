using Test,
      CUDA

using NSEBase,
      ReSolverChannelFlow,
      ReSolverChannelFlowGPU,
      LinearAlgebra,
      FDGrids

# include("test_fields.jl")
include("test_fft.jl")
include("test_derivatives.jl")
# include("test_dot.jl")
# include("test_galerkin.jl")
include("test_equations.jl")
