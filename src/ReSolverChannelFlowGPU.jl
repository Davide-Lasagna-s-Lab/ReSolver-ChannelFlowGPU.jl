module ReSolverChannelFlowGPU

using CUDA,
      LinearAlgebra

using CUDA: i32

__init__() = @assert CUDA.functional(true)

using NSEBase,
      ReSolverChannelFlow

export TwoStage, Atomic, Shared

"""
    optimal_threads(kernel!, args...; max_threads=nothing) -> Int

Query the hardware-optimal thread count for a CUDA kernel using
`launch_configuration`. Compiles the kernel without launching it,
queries the occupancy, and returns the optimal thread count capped
at `max_threads` if provided.

# Example
```julia
threads = optimal_threads(_ddx_kernel!, out, u, sz, nelem,
            im*one(T), wavenumber_scale(u, 1), Val(1), Val(ORDER))
```
"""
function optimal_threads(kernel!, args...; max_threads=nothing)
    k = @cuda launch=false kernel!(args...)
    threads = launch_configuration(k.fun).threads
    isnothing(max_threads) ? threads : min(threads, max_threads)
end

@inline @generated function _linear_to_cart(idx, sz::NTuple{D, Int32}) where {D}
    return quote
        rem = idx - 1i32
        $(ntuple(d -> d < D ? quote
            $(Symbol(:i_, d)) = rem % sz[$d] + 1i32
            rem = rem ÷ sz[$d]
        end : quote
            $(Symbol(:i_, D)) = rem + 1i32
        end, Val(D))...)
        return CartesianIndex($(ntuple(d -> Symbol(:i_, d), Val(D))...))
    end
end

using Adapt
function Adapt.adapt_structure(to, g::ChannelGrid{S, T}) where {S, T}
    y = Adapt.adapt_structure(to, g.y)
    D₁ = Adapt.adapt_structure(to, g.D₁)
    D₂ = Adapt.adapt_structure(to, g.D₂)
    D₁⁺ = Adapt.adapt_structure(to, g.D₁⁺)
    D₂⁺ = Adapt.adapt_structure(to, g.D₂⁺)
    ws = Adapt.adapt_structure(to, g.ws)
    α = Adapt.adapt_structure(to, g.α)
    β = Adapt.adapt_structure(to, g.β)
    return ChannelGrid{S, T}(y, D₁, D₂, D₁⁺, D₂⁺, ws, α, β)
end

Adapt.adapt_structure(to, u::FTField) =
    FTField(Adapt.adapt_structure(to, grid(u)), Adapt.adapt_structure(to, parent(u)))

Adapt.adapt_structure(to, u::VectorField{N}) where {N} =
    VectorField([Adapt.adapt_structure(to, u[n]) for n in 1:N]...)

function Adapt.adapt_structure(to, a::ProjectedField)
    g = Adapt.adapt_structure(to, grid(a))
    data = Adapt.adapt_structure(to, parent(a))
    mds = Adapt.adapt_structure(to, modes(a))
    return ProjectedField(g, data, mds)
end


CUDA.cu(g::ChannelGrid{S, T}) where {S, T} = ChannelGrid{S, Float32}(CUDA.cu(g.y),
                                                                     CUDA.cu(g.D₁), CUDA.cu(g.D₂),
                                                                     CUDA.cu(g.D₁⁺), CUDA.cu(g.D₂⁺),
                                                                     CUDA.cu(g.ws),
                                                                     Float32(g.α), Float32(g.β))

CUDA.cu(u::FTField)                  =        FTField(CUDA.cu(grid(u)), CUDA.cu(parent(u)))
CUDA.cu(u::Field)                    =          Field(CUDA.cu(grid(u)), CUDA.cu(parent(u)))
CUDA.cu(u::VectorField{N}) where {N} =    VectorField([CUDA.cu(u[n]) for n in 1:N]...)
CUDA.cu(a::ProjectedField)           = ProjectedField(CUDA.cu(grid(a)), CUDA.cu(parent(a)), CUDA.cu(modes(a)))

FTFieldGPU(g::ChannelGrid{S}, ::Type{T}=Float32) where {S, T} =
    FTField(CUDA.cu(g), CUDA.zeros(Complex{T}, S[1], (S[2] >> 1) + 1, S[3], S[4]))
FieldGPU(g::ChannelGrid{S}, ::Type{T}=Float32; kwargs...) where {S, T} =
    Field(CUDA.cu(g), CUDA.zeros(T, size(g)); kwargs...)
ProjectedFieldGPU(g::ChannelGrid, modes) = throw(NSEBase.NotImplementedError(g, modes))

include("fft.jl")
include("derivatives.jl")
include("galerkin.jl")
include("dot.jl")
# include("operators.jl")

end
