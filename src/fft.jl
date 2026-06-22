# CUDA FFT plans.

struct CuFFTPlans{D, T, ORDER, PLAN, IPLAN, CA}
     plan::PLAN
    iplan::IPLAN
    cache::CA
     norm::T

    # TODO: replace unused keyword arguments with kwargs...
    function CuFFTPlans(size::Dims{D},
                       order::NTuple{H, Int},
                            ::Type{T}=Float32;
                       flags                      =nothing,
                 padded_size::Union{Nothing, Dims}=nothing) where {D, H, T}
        all(1 ≤ d ≤ D for d in order) || throw(ArgumentError("order indices must be in 1:$D, got $order"))
        allunique(order)              || throw(ArgumentError("order indices must be unique, got $order"))

        grid_size = if !isnothing(padded_size)
            length(padded_size) == D ||
                throw(ArgumentError("padded_size must have $D elements, got $(length(padded_size))"))
            all(padded_size[d] >= size[d] for d in order) ||
                throw(ArgumentError("padded_size must be ≥ size along each transformed dimension"))
            padded_size
        else
            NSEBase.get_padded_size(size, order)
        end

        spectral_array   = CUDA.zeros(Complex{T}, NSEBase._get_transform_size(grid_size, order[1]))
        physical_array   = CUDA.zeros(T, grid_size)
        norm             = T(1/prod(grid_size[i] for i in order))

        plan  = cuFFT.plan_rfft( physical_array,                      order)
        iplan = cuFFT.plan_brfft(spectral_array, grid_size[order[1]], order)

        return new{D, T, order, typeof(plan), typeof(iplan), typeof(spectral_array)}(plan, iplan, spectral_array, norm)
    end
end

CuFFTPlans(g::NSEBase.AbstractGrid{T}; kwargs...) where {T} = CuFFTPlans(size(g), NSEBase.fft_dims(g), T; kwargs...)

NSEBase.FFTPlans(g::ChannelGrid{<:Any, <:Any, <:CuArray}; kwargs...) = CuFFTPlans(g; kwargs...)

# ------------------------ #
# in-place transformations #
# ------------------------ #
function (f::CuFFTPlans)(û::VectorField{N, <:FTField},
                         u::VectorField{N, <:Field};
                       add::Bool=false) where {N}
    for n in 1:N
        f(û[n], u[n]; add=add)
    end
    return û
end
(f::CuFFTPlans)(û::FTField, u::Field; add::Bool=false) = (f(parent(û), parent(u), add); return û)

(f::CuFFTPlans{<:Any, T})(û::AbstractArray{Complex{T}},
                          u::AbstractArray{        T},
                        add::Bool) where {T} =
    _forward_transform!(û, u, f, add)

function _forward_transform!(û, u, f::CuFFTPlans{D, <:Any, ORDER}, add::Bool) where {D, ORDER}
    cuFFT.unsafe_execute!(f.plan, u, f.cache)
    f.cache .*= f.norm
    add ? NSEBase._add_from_padded!(û, f.cache, ORDER) : NSEBase._copy_from_padded!(û, f.cache, ORDER)
    return û
end

function (f::CuFFTPlans)(u::VectorField{N, <:Field},
                         û::VectorField{N, <:FTField}) where {N}
    for n in 1:N
        f(u[n], û[n])
    end
    return u
end
(f::CuFFTPlans)(u::Field, û::FTField) = (f(parent(u), parent(û)); return u)

(f::CuFFTPlans{<:Any, T})(u::AbstractArray{        T},
                          û::AbstractArray{Complex{T}}) where {T} =
    _backward_transform!(u, û, f)

function _backward_transform!(u, û, f::CuFFTPlans{D, <:Any, ORDER}) where {D, ORDER}
    NSEBase._apply_mask!(f.cache)
    NSEBase._copy_to_padded!(f.cache, û, ORDER)
    cuFFT.unsafe_execute!(f.iplan, f.cache, u)
    return u
end

#--------------------------- #
# dealiasing utility methods #
#--------------------------- #
# ! only needed because NSEBase._apply_mask! is defined for Array rather than AbstractArray
NSEBase._apply_mask!(cache::CuArray{T}) where {T} = (cache .= zero(T); return cache)

"""
Launch `_loopblk_kernel!` for a single contiguous block.
`ar` and `br` are NTuple{D, UnitRange{Int}} — constructed on the host,
never passed to the device.
"""
@inline function NSEBase._loopblk!(dest::CuArray,
                                     ar::NTuple{D},
                                    src::CuArray,
                                     br::NTuple{D},
                                   vadd::Val) where {D}
    # All range arithmetic happens here on the host
    dest_starts = Int32.(ntuple(k ->  first(ar[k]) - 1           , Val(D)))
    dest_sizes  = Int32.(ntuple(k -> length(ar[k])               , Val(D)))
    src_offsets = Int32.(ntuple(k ->  first(br[k]) - first(ar[k]), Val(D)))

    kernel_args = (dest, src, dest_starts, dest_sizes, src_offsets, vadd)
    nthreads, blocks = get_launch_params(_loopblk_kernel!, Int32(prod(dest_sizes)), kernel_args...)
    @cuda threads=nthreads blocks=blocks _loopblk_kernel!(kernel_args...)
    return nothing
end

"""
    _loopblk_kernel!(dest, src, dest_starts, dest_sizes, src_offsets, ::Val{ADD})

GPU kernel: each thread handles one CartesianIndex in the destination block.
`dest_starts` is an `NTuple{D, Int}` of 0-based start indices for the destination block.
`dest_sizes` is an `NTuple{D, Int}` of element counts in each dimension.
`src_offsets` is an `NTuple{D, Int}` where `src_idx = dest_idx + src_offset`.
No heap allocation: all index arithmetic is register-based.
"""
@generated function _loopblk_kernel!(dest, src,
                                     dest_starts::NTuple{D, Int32},
                                      dest_sizes::NTuple{D, Int32},
                                     src_offsets::NTuple{D, Int32},
                                                ::Val{ADD}) where {D, ADD}
    # Unroll the D-dimensional linear-index → CartesianIndex decomposition
    # entirely at code-generation time. The emitted kernel has no loops over
    # dimensions — just straight-line index arithmetic.
    index_exprs = quote
        linear = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
        linear > prod(dest_sizes) && return nothing
    end

    # Generate the strided decomposition: recover each dimension's local index
    # from `linear` using precomputed strides.
    decomp = Expr(:block)
    push!(decomp.args, :(rem_idx = linear - 1i32))
    for d in 1:D
        if d < D
            push!(decomp.args, quote
                $(Symbol(:loc_, d)) = rem_idx % dest_sizes[$d] + 1i32
                rem_idx = rem_idx ÷ dest_sizes[$d]
            end)
        else
            push!(decomp.args, :($(Symbol(:loc_, d)) = rem_idx + 1i32))
        end
    end

    # Build the CartesianIndex expressions for dest and src
    dest_idx = :(CartesianIndex($(
        [:(dest_starts[$d] + $(Symbol(:loc_, d))) for d in 1:D]...
    )))
    src_idx  = :(CartesianIndex($(
        [:(dest_starts[$d] + $(Symbol(:loc_, d)) + src_offsets[$d]) for d in 1:D]...
    )))

    # The assignment — branch eliminated by Val{ADD} at code-generation time
    assign = ADD ? :(@inbounds dest[$dest_idx] += src[$src_idx]) :
                   :(@inbounds dest[$dest_idx]  = src[$src_idx])

    return quote
        $index_exprs
        $decomp
        $assign
        return nothing
    end
end
