# GPU kernels for derivatives of FTFields wrapping CuArrays.

ddx_x!(out, u; adjoint=false, nthreads=256) = ddx!(out, u, Val(2); adjoint=adjoint, nthreads=nthreads)
ddx_y!(out::FTField{G}, u::FTField{G}; adjoint::Bool=false, nthreads=nothing) where {G<:ChannelGrid} =
    mul!(out, adjoint ? grid(u).Dya : grid(u).Dy, u; nthreads=nthreads)
ddx_z!(out, u; adjoint=false, nthreads=256) = ddx!(out, u, Val(3); adjoint=adjoint, nthreads=nthreads)
  dds!(out, u; adjoint=false, nthreads=256) = ddx!(out, u, Val(4); adjoint=adjoint, nthreads=nthreads)

@generated function ddx!(out::FTField{G},
                           u::FTField{G},
                            ::Val{DIM};
                     adjoint::Bool=false,
                    nthreads::TH  =nothing) where {DIM, T, D, AXES, ORDER, G<:AbstractGrid{T, D, AXES, ORDER}, TH<:Union{Nothing, Int}}
    (isnothing(DIM) || isnothing(AXES[DIM])) && return :(return out)
    DIM ∉ ORDER && return :(throw(NSEBase.NotImplementedError(grid(u), Val($DIM))))

    # Compile kernel without launching to query optimal thread count
    thread_block_expr = if TH <: Nothing
        quote
            optimal_threads(_ddx_kernel!,
                            out, u, sz, nelem,
                            _ddx_sign, _ddx_scale,
                            Val($DIM), Val($ORDER),
                            max_threads=nelem)
        end
    else
        quote
            _nthreads = nthreads
            _blocks   = cld(nelem, nthreads)
        end
    end

    return quote
        sz     = Int32.(size(u))
        nelem  = Int32(prod(sz))

        _ddx_sign  = adjoint ? -one(Complex{T}) : one(Complex{T})
        _ddx_scale = wavenumber_scale(u, $DIM)

        $thread_block_expr

        @cuda threads=_nthreads blocks=_blocks _ddx_kernel!(
            out, u, sz, nelem, _ddx_sign, _ddx_scale, Val($DIM), Val($ORDER)
        )
    end
end


function _ddx_kernel!(out, u, sz, nelem, _ddx_sign, _ddx_scale, ::Val{DIM}, ::Val{ORDER}) where {DIM, ORDER}
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
        :(I[$DIM] ≤ (sz[$DIM] >> 1) + 1 ? I[$DIM] - 1 : I[$DIM] - 1 - sz[$DIM])
    end
end
