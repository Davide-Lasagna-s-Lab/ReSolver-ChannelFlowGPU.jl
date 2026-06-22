module ReSolverChannelFlowGPU

using CUDA,
      LinearAlgebra,
      Adapt

using CUDA: i32

# TODO: benchmark residual
# TODO: actual tests for construction, galerkin, and dot auto-tuning

__init__() = @assert CUDA.functional(true)

using NSEBase,
      ReSolverChannelFlow,
      FDGrids

export show_tuning_info
export CuFFTPlans
export DotTwoStage, DotAtomic, DotShared
export ProjectBroadcast, ProjectLoop, ProjectShared
export ExpandBroadcast, ExpandModal
export FTFieldGPU, FieldGPU, ProjectedFieldGPU

const TUNING_INFO = Ref(false)

"""
    show_tuning_info(show_info::Bool)

Toggle extra information when performing kernel tuning for Galerkin methods
and dot product of `ProjectedField`.
"""
show_tuning_info(show_info::Bool) = TUNING_INFO[] = show_info


const LAUNCH_PARAMS = Dict{Tuple{Type, NTuple}, Int32}()

function get_launch_params(kernel_f::F, kernel_args...) where {F}
    key = (F, map(typeof, kernel_args))
    get!(LAUNCH_PARAMS, key) do
        kernel = @cuda launch=false kernel_f(kernel_args...)
        Int32(CUDA.launch_configuration(kernel.fun).threads)
    end
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
    FTField(Adapt.adapt_structure(to, NSEBase.grid(u)), Adapt.adapt_structure(to, parent(u)))

Adapt.adapt_structure(to, u::VectorField{N}) where {N} =
    VectorField(ntuple(n -> Adapt.adapt_structure(to, u[n]), Val(N))...)

function Adapt.adapt_structure(to, a::ProjectedField)
    g = Adapt.adapt_structure(to, NSEBase.grid(a))
    data = Adapt.adapt_structure(to, parent(a))
    mds = Adapt.adapt_structure(to, modes(a))
    return ProjectedField(g, data, mds)
end


CUDA.cu(g::ChannelGrid{S}) where {S} = ChannelGrid{S, Float32}(CUDA.cu(g.y),
                                                               CUDA.cu(g.D₁), CUDA.cu(g.D₂),
                                                               CUDA.cu(g.D₁⁺), CUDA.cu(g.D₂⁺),
                                                               CUDA.cu(g.ws),
                                                               Float32(g.α), Float32(g.β))

CUDA.cu(u::FTField)                  =        FTField(CUDA.cu(NSEBase.grid(u)), CUDA.cu(parent(u)))
CUDA.cu(u::Field)                    =          Field(CUDA.cu(NSEBase.grid(u)), CUDA.cu(parent(u)))
CUDA.cu(u::VectorField{N}) where {N} =    VectorField([CUDA.cu(u[n]) for n in 1:N]...)
CUDA.cu(a::ProjectedField)           = ProjectedField(CUDA.cu(NSEBase.grid(a)), CUDA.cu(parent(a)), CUDA.cu(modes(a)))

FTFieldGPU(g::ChannelGrid{S}, ::Type{T}=Float32) where {S, T} =
    FTField(CUDA.cu(g), CUDA.zeros(Complex{T}, S[2], (S[1] >> 1) + 1, S[3], S[4]))
FieldGPU(g::ChannelGrid{S}, ::Type{T}=Float32; kwargs...) where {S, T} =
    Field(CUDA.cu(g), CUDA.zeros(T, size(g)); kwargs...)
ProjectedFieldGPU(g::ChannelGrid, modes) = throw(NSEBase.NotImplementedError(g, modes))

# ! when moving all this stuff to NSEBase.jl, a new type parameter for AbstractGrid would be useful to be able to specialise the constructors
# ! or maybe a new type (like DecomposedGrid) that stores the original grid and can be used for such dispatch
NSEBase.FTField(grid::ChannelGrid{<:Any, T, <:CuArray}) where {T} = NSEBase.FTField(grid, CUDA.zeros(Complex{T}, transform_size(grid)))
NSEBase.Field(grid::ChannelGrid{<:Any, T, <:CuArray}; dealias=true) where {T} = NSEBase.Field(grid, CUDA.zeros(T, NSEBase.get_padded_size(size(grid), NSEBase.fft_dims(grid))))
function NSEBase.ProjectedField(grid::ChannelGrid{<:Any, T, <:CuArray}, modes) where {T}
    Nm = size(modes[1], 1)
    return ProjectedField(grid,
                          CUDA.zeros(Complex{T}, Nm,
                                transform_size(grid)[collect(fft_dims(grid))]...),
                          modes)
end

include("fft.jl")
include("derivatives.jl")
include("galerkin.jl")
include("dot.jl")

end
