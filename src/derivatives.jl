# GPU kernels for derivatives of FTFields wrapping CuArrays.

# TODO: tuning global parameters to store data, reducing allocations, and storing optimal threads

NSEBase.ddx_1!(out, u; adjoint=false, nthreads=nothing) = ddx!(out, u, Val(2); adjoint=adjoint, nthreads=nthreads)
function NSEBase.ddx_2!(out::FTField{G}, u::FTField{G}; adjoint::Bool=false, nthreads=nothing) where {G<:ChannelGrid}
    mul!(parent(out), adjoint ? NSEBase.grid(u).D₁⁺ : NSEBase.grid(u).D₁, parent(u), Val(1); nthreads=nthreads)
    return out
end
NSEBase.ddx_3!(out, u; adjoint=false, nthreads=nothing) = ddx!(out, u, Val(3); adjoint=adjoint, nthreads=nthreads)
NSEBase.ddx_4!(out, u; adjoint=false, nthreads=nothing) = ddx!(out, u, Val(4); adjoint=adjoint, nthreads=nthreads)

function NSEBase.ddx!(out::F,
                        u::F,
                         ::Val{DIM};
                  adjoint::Bool=false,
                 nthreads::TH  =nothing) where {DIM, T, D, AXES, ORDER, G<:AbstractGrid{T, D, AXES, ORDER}, M, F<:Union{FTField{G, <:CuArray}, ProjectedField{G, M, <:CuArray}}, TH<:Union{Nothing, Int}}
    (isnothing(DIM) || isnothing(AXES[DIM])) && return :(return out)
    DIM ∉ ORDER && return :(throw(NSEBase.NotImplementedError(NSEBase.grid(u), Val($DIM))))

    # get sizes
    sz     = Int32.(size(u))
    nelem  = Int32(prod(sz))

    # spectral derivative parameters
    _ddx_sign  = adjoint ? -1im*one(T) : 1im*one(T)
    _ddx_scale = wavenumber_scale(NSEBase.grid(u), DIM)

    # get optimal threads if not assigned during call
    _nthreads, _blocks = if TH <: Nothing
        _nthreads = optimal_threads(_ddx_kernel!,
                                    parent(out), parent(u), sz, nelem,
                                    _ddx_sign, _ddx_scale,
                                    Val(Int32(DIM)), Val(Int32.(ORDER)),
                                    max_threads=nelem)
        _blocks = cld(nelem, _nthreads)
        _nthreads, _blocks
    else
        nthreads, cld(nelem, nthreads)
    end

    # launch kernel
    @cuda threads=_nthreads blocks=_blocks _ddx_kernel!(
        parent(out), parent(u), sz, nelem, _ddx_sign, _ddx_scale, Val(Int32(DIM)), Val(Int32.(ORDER))
    )

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

initialise_ddx!(out::F, u::F, ::Val{DIM}) where {T, D, AXES, ORDER, G<:AbstractGrid{T, D, AXES, ORDER}, F<:Union{FTField{G}, ProjectedField{G}}, DIM} =
    optimal_threads(_ddx_kernel!,
                    out, u, Int32.(size(u)), Int32(prod(size(u))),
                    one(Complex{T}), NSEBase.wavenumber_scale(NSEBase.grid(u), DIM),
                    Val(Int32(DIM)), Val(Int32.(ORDER));
                    max_threads=nelem)

initialise_ddx!(out::F, u::F, ::Val{1}; adjoint=false) where {T, D, AXES, ORDER, G<:AbstractGrid{T, D, AXES, ORDER}, F<:Union{FTField{G}, ProjectedField{G}}} =
    FDGrids.optimal_forward_threads(parent(out), 
                                    adjoint ? NSEBase.grid(u).D₁⁺ : NSEBase.grid(u).D₁,
                                    parent(u),
                                    Val(1))


function NSEBase.laplacian!(out::FTField{G, <:CuArray}, u::FTField{G, <:CuArray}; nthreads=nothing, kwargs...) where {G}
    NSEBase.inhomogeneous_laplacian!(out, u; nthreads=nthreads, kwargs...)
    NSEBase.add_homogeneous_laplacian!(out, u; nthreads=nthreads)
end

function NSEBase.inhomogeneous_laplacian!(out::FTField{G, <:CuArray}, u::FTField{G, <:CuArray}; adjoint::Bool=false, nthreads=nothing) where {G<:AbstractChannelGrid{<:Any, <:Any}}
    LinearAlgebra.mul!(parent(out), adjoint ? NSEBase.grid(u).D₂⁺ : NSEBase.grid(u).D₂, parent(u), Val(1), nthreads=nthreads)
    return out
end

function NSEBase.add_homogeneous_laplacian!(out::FTField{G, <:CuArray},
                                              u::FTField{G, <:CuArray};
                                       nthreads::TH=nothing) where {G, TH<:Union{Nothing, Int}}
    # get sizes
    sz     = Int32.(size(u))
    nelem  = Int32(prod(sz))

    # spectral derivative parameters
    _scales = map(d -> wavenumber_scale(NSEBase.grid(u), d), spatial_fft_dims(NSEBase.grid(u)))

    # get optimal threads if not assigned during call
    _nthreads, _blocks = if TH <: Nothing
        _nthreads = optimal_threads(_add_homogeneous_laplacian_kernel!,
                                    parent(out), parent(u), sz, nelem,
                                    _scales,
                                    Val(Int32.(spatial_fft_dims(NSEBase.grid(u)))), Val(Int32(NSEBase.fft_dims(NSEBase.grid(u))[1])),
                                    max_threads=nelem)
        _blocks = cld(nelem, _nthreads)
        _nthreads, _blocks
    else
        nthreads, cld(nelem, nthreads)
    end

    # launch kernel
    @cuda threads=_nthreads blocks=_blocks _add_homogeneous_laplacian_kernel!(
        parent(out), parent(u), sz, nelem, _scales, Val(Int32.(spatial_fft_dims(NSEBase.grid(u)))), Val(Int32(NSEBase.fft_dims(NSEBase.grid(u))[1]))
    )

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
