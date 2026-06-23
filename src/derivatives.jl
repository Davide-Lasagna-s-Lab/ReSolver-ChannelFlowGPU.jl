# GPU kernels for derivatives of FTFields wrapping CuArrays.

# ! when this is transferred to NSEBase.jl I need to find a way to get rid of this method
function NSEBase.ddx!(out::FTField{G, <:CuArray},
                        u::FTField{G, <:CuArray},
                         ::Val{1};
                  adjoint::Bool=false) where {G<:AbstractChannelGrid}
    LinearAlgebra.mul!(parent(out), adjoint ? NSEBase.grid(u).D₁⁺ : NSEBase.grid(u).D₁, parent(u), Val(1))
    return out
end

function NSEBase.ddx!(out::F,
                        u::F,
                         ::Val{DIM};
                  adjoint::Bool=false) where {DIM, T, D, AXES, ORDER, G<:AbstractGrid{T, D, AXES, ORDER}, F<:Union{FTField{G, <:CuArray}, ProjectedField{G, <:Any, <:CuArray}}}
    # kernel arguments
    sz     = Int32.(size(u))
    nelem  = Int32(prod(sz))
    _ddx_sign  = adjoint ? -1im*one(T) : 1im*one(T)
    _ddx_scale = NSEBase.wavenumber_scale(NSEBase.grid(u), DIM)

    # launch kernel
    kernel_args = (parent(out), parent(u), sz, nelem, _ddx_sign, _ddx_scale, Val(Int32(DIM)), Val(Int32.(ORDER)))
    nthreads = get_launch_params(_ddx_kernel!, kernel_args...)
    @cuda threads=nthreads blocks=Int32(cld(nelem, nthreads)) _ddx_kernel!(kernel_args...)

    return out
end

function _ddx_kernel!(out::CuDeviceArray, u::CuDeviceArray, sz::NTuple, nelem::Int32, _ddx_sign, _ddx_scale, ::Val{DIM}, ::Val{ORDER}) where {DIM, ORDER}
    idx = (blockIdx().x - 1i32)*blockDim().x + threadIdx().x
    idx > nelem && return nothing

    I = _linear_to_cart(idx, sz)
    n = _get_freq(I, sz, Val(DIM), Val(ORDER))

    @inbounds out[I] = _ddx_sign*n*_ddx_scale*u[I]
    return nothing
end

@inline @generated function _get_freq(I, sz, ::Val{DIM}, ::Val{ORDER}) where {DIM, ORDER}
    return if DIM == ORDER[1]
        :(I[$DIM] - 1)
    else
        :(ifelse(I[$DIM] ≤ (sz[$DIM] >> 1) + 1, I[$DIM] - 1, I[$DIM] - 1 - sz[$DIM]))
    end
end


function NSEBase.add_homogeneous_laplacian!(out::FTField{G, <:CuArray},
                                              u::FTField{G, <:CuArray}) where {G<:AbstractGrid}
    # kernel arguments
    sz     = Int32.(size(u))
    nelem  = Int32(prod(sz))
    scales = map(d -> NSEBase.wavenumber_scale(NSEBase.grid(u), d), spatial_fft_dims(NSEBase.grid(u)))

    # launch kernel
    kernel_args = (parent(out), parent(u), sz, nelem, scales, Val(Int32.(spatial_fft_dims(NSEBase.grid(u)))), Val(Int32(NSEBase.fft_dims(NSEBase.grid(u))[1])))
    nthreads = get_launch_params(_add_homogeneous_laplacian_kernel!, kernel_args...)
    @cuda threads=nthreads blocks=Int32(cld(nelem, nthreads)) _add_homogeneous_laplacian_kernel!(kernel_args...)

    return out
end
NSEBase.add_homogeneous_laplacian!(out::VectorField{N, F}, u::VectorField{N, F}; nthreads=nothing) where {N, G, F<:FTField{G, <:CuArray}} =
    (for n in 1:N; add_homogeneous_laplacian!(out[n], u[n]; nthreads=nthreads); end; return out)

function _add_homogeneous_laplacian_kernel!(out::CuDeviceArray, u::CuDeviceArray, sz::NTuple, nelem::Int32, scales, ::Val{SPATIAL_ORDER}, ::Val{RFFT_DIM}) where {SPATIAL_ORDER, RFFT_DIM}
    idx = (blockIdx().x - 1i32)*blockDim().x + threadIdx().x
    idx > nelem && return nothing

    I  = _linear_to_cart(idx, sz)
    k² = _get_laplacian_freq(I, sz, scales, Val(SPATIAL_ORDER), Val(RFFT_DIM))

    @inbounds out[I] -= k²*u[I]
    return nothing
end

@inline @generated function _get_laplacian_freq(I, sz, scales, ::Val{SPATIAL_ORDER}, ::Val{RFFT_DIM}) where {SPATIAL_ORDER, RFFT_DIM}
    terms = map(enumerate(SPATIAL_ORDER)) do (i, d)
        n = if d == RFFT_DIM
            :(Int32(I[$d] - 1))
        else
            :(ifelse(I[$d] ≤ (sz[$d] >> 1) + 1, Int32(I[$d] - 1), Int32(I[$d] - 1 - sz[$d])))
        end
        :(scales[$i]*$n)
    end

    sum_expr = mapreduce(
        t      -> :($t^2),
        (a, b) -> :($a + $b),
        terms
    )

    return :(Float32($sum_expr))
end
