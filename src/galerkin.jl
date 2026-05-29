# Galerkin methods used to convert between ProjectedField and FTField
# completely on device.

# -------------- #
# project method #
# -------------- #
abstract type ProjectMethod end

"""
    ProjectBroadcast{T} <: ProjectMethod

Galerkin projection method for GPU-resident `ProjectedField` and `VectorField`
arrays using higher-order array abstractions provided by CUDA.jl.

Computes the Galerkin projection of a vector field onto a set of orthonormal
modes by accumulating the reduction for each wall-normal location and mode
number for every frequency.

The quadrature weights are moved from the device to the host in preperation
for them being used on the host-side function during the broadcasting operation.

# Fields
- `ws::Vector{T}`: quadrature weights from the grid copied onto the host.

# Constructor
    ProjectBroadcast(a::ProjectedField)

- `a`: projected field the projection is to be assigned

# Example
```julia
method = ProjectBroadcast(a)
project!(a, b, method)
```
"""
struct ProjectBroadcast{T} <: ProjectMethod
    ws::Vector{T}

    ProjectBroadcast(a::ProjectedField{<:AbstractGrid{T}}) where {T} =
        new{T}(Array(grid(a).ws))
end

"""
    ProjectLoop{D} <: ProjectMethod

Galerkin projection method that performs the reduction at each frequency on a
single thread.

Each thread computes a single accumulated sum as the result of the inner-product
between a single mode and channel profile at a given frequency and mode number.
The accumulated sum on each thread is then assigned to a single element of the
output projected field. The optimal number of threads for the computation are
estimated upon construction of this method and re-used for projection of the same
input types.

# Fields
- `sz::NTuple{D, Int32}`: size of `ProjectedField` arrays
- `nelem::Int32`: total number of elements of `ProjectedField` arrays (`nelem=prod(sz)`)
- `threads`::Int32: number of GPU threads per block
- `blocks::Int32`: number of GPU blocks assigned for kernel

# Constructor
    ProjectLoop(a::ProjectedField, u::VectorField)

- `a`: a representative `ProjectedField` used to determine array size and
  query optimal thread count. Not mutated.
- `u`: a corresponding representative `VectorField` used to determine number
  of velocity components used for the projection. Not mutated.

# Example
```julia
method = ProjectLoop(a, u)
project!(a, b, method)
```
"""
struct ProjectLoop{D} <: ProjectMethod
        sz::NTuple{D, Int32}
     nelem::Int32
   threads::Int32
    blocks::Int32

    function ProjectLoop(a::ProjectedField{<:ChannelGrid{S}},
                         u::VectorField{N};
                   threads::Union{Nothing, Int}=nothing) where {N, S}
        pa = parent(a)
        sz = Int32.(size(pa))
        nelem = Int32(prod(sz))
        _threads = if isnothing(threads)
            optimal_threads(_project_loop_kernel!,
                            pa, modes(a), u, grid(u).ws,
                            sz, nelem,
                            Val(Int32(N)), Val(Int32(S[2]));
                            max_threads=nelem)
        else
            threads
        end
        new{length(sz)}(sz, nelem, Int32(_threads), Int32(cld(nelem, _threads)))
    end
end

"""
    ProjectShared{D} <: ProjectMethod

Galerkin projection method that performs the reduction at each frequency on a
single thread.

Very similar to [`ProjectLoop`](@ref) with the addition of assigning the
quadrature weights to static shared memory. Might be favourable if number of
global reads to the quadrature weights is a bottleneck in the `ProjectLoop` method.

# Fields
- `sz::NTuple{D, Int32}`: size of `ProjectedField` arrays
- `nelem::Int32`: total number of elements of `ProjectedField` arrays (`nelem=prod(sz)`)
- `threads`::Int32: number of GPU threads per block
- `blocks::Int32`: number of GPU blocks assigned for kernel

# Constructor
    ProjectShared(a::ProjectedField, u::VectorField)

- `a`: a representative `ProjectedField` used to determine array size and
  query optimal thread count. Not mutated.
- `u`: a corresponding representative `VectorField` used to determine number
  of velocity components used for the projection. Not mutated.

# Example
```julia
method = ProjectSahred(a, u)
project!(a, b, method)
```
"""
struct ProjectShared{D} <: ProjectMethod
        sz::NTuple{D, Int32}
     nelem::Int32
   threads::Int32
    blocks::Int32

    function ProjectShared(a::ProjectedField{<:ChannelGrid{S}},
                           u::VectorField{N};
                     threads::Union{Nothing, Int}=nothing) where {N, S}
        pa = parent(a)
        sz = Int32.(size(pa))
        nelem = Int32(prod(sz))
        _threads = if isnothing(threads)
            optimal_threads(_project_shared_kernel!,
                            pa, modes(a), u, grid(u).ws,
                            sz, nelem,
                            Val(Int32(N)), Val(Int32(S[2]));
                            max_threads=nelem)
        else
            threads
        end
        new{length(sz)}(sz, nelem, Int32(_threads), Int32(cld(nelem, _threads)))
    end
end

struct ProjectWarp <: ProjectMethod end


# --------------------------------------- #
# auto-tuning for optimal method dispatch #
# --------------------------------------- #
const PROJECT_METHODS = Dict{Tuple{Type, Type}, ProjectMethod}()

"""
    project_method(a::ProjectedField{G, M, <:CuArray},
                   u::VectorField{N, <:FTField{G, <:CuArray}}) -> ProjectMethod

Return the `ProjectMethod` associated with the concrete type of `a` and `u`,
auto-tuning if these types have not been seen before. Results are cached in
`PROJECT_METHODS` keyed on `(typeof(a), typeof(u))`.

    project_method(a) -> cached_method or autotune_project(a)
"""
function project_method(a::ProjectedField{G, M, <:CuArray},
                        u::VectorField{N, <:FTField{G, <:CuArray}}) where {G, M, N}
    get!(PROJECT_METHODS, (typeof(a), typeof(u))) do
        autotune_project(a, u)
    end
end

"""
    autotune_project(a::ProjectedField{G, M, <:CuArray{T}},
                     u::VectorField{N, <:FTField{G, <:CuArray}}) -> ProjectMethod

Benchmark all available `ProjectMethod` implementations against a dummy field of
the same types as `a` and `u` and return the fastest. Each candidate is warmed up
once to trigger compilation before timing. Timing uses `CUDA.@elapsed` to measure
device-side execution time, taking the minimum over 5 trials to reduce noise.

Logs the winner and all trial times via `@info`.

    autotune_project(a) -> best::ProjectMethod
"""
function autotune_project(a, u)
    # Construct all candidate methods
    candidates = ProjectMethod[
        ProjectBroadcast(a, u),
        ProjectLoop(a, u),
        ProjectShared(a, u),
    ]

    # Warmup all candidates — triggers compilation
    for method in candidates
        project!(a, u, method)
    end
    CUDA.synchronize()

    # Time each candidate
    times = map(candidates) do method
        minimum(1:5) do _
            CUDA.@elapsed project!(a, u, method)
        end
    end

    best = candidates[argmin(times)]
    @info "Auto-tuned dot product" typeof(a) best=typeof(best) times_ns=times
    return best
end

"""
    initialise_project!(a::ProjectedField, u::VectorField)

Eagerly auto-tune and cache the optimal `ProjectMethod` for the concrete types of
`a` and `u`. After this call, all `project!(a, b)` invocations for fields of the same
type will use the cached optimal method with no auto-tuning overhead.

Should be called once per field type during program initialisation, before
entering any performance-critical loops.

# Example
```julia
a = ProjectedField(...)
u = VectorField(...)
initialise_project!(a, u)   # benchmarks all methods, caches the winner

# all subsequent calls use the cached optimal method
for i in 1:nsteps
    project!(similar(a), u)
end
```

See also: [`reset_project_cache!`](@ref)
"""
function initialise_project!(a::ProjectedField, u::VectorField)
    project_method(a, u)
    return nothing
end

"""
    reset_project_cache!()
    reset_project_cache!(a::ProjectedField, u::VectorField)

Clear the auto-tune cache for all field types, or for the specific types of `a`
and `u`. The next `project!` call after resetting will trigger auto-tuning again.

Useful when benchmarking different methods explicitly, or after moving to a
different GPU with different performance characteristics.

# Example
```julia
reset_project_cache!()     # clear all cached methods
reset_project_cache!(a, u) # clear only the method cached for (typeof(a), typeof(u))
```

See also: [`initialise_dot!`](@ref)
"""
reset_project_cache!() = empty!(PROJECT_METHODS)
reset_project_cache!(::P, ::V) where {P<:ProjectedField, V<:VectorField} = delete!(PROJECT_METHODS, (P, V))


# ---------------------------- #
# top-level projection methods #
# ---------------------------- #
"""
    project!(a::ProjectedField{G, M, <:CuArray},
             u::VectorField{N, <:FTField{G, <:CuArray}}) -> a

Compute the Galerkin projection of the GPU-resident `VectorField` onto a set of
orthonormal modes stored in the provided `ProjectedField`, using auto-tuned optimal
method for their concrete type.

The method used is determined by `project_method(a)`, which auto-tunes on the first
call and returns the cached result on all subsequent calls. Call
`initialise_project!(a, u)` before entering performance-critical loops to ensure the
auto-tuning cost is paid upfront.

# Arguments
- `a`: `ProjectedField` on the grid `G` and `CuArray` storage.
- `u`: `VectorField` of `FTField` on the same grid `G` and `CuArray` storage.

# Returns
- `a` with each element asigned the accumulated constribution of `u` for each mode

# Example
```julia
initialise_project!(a, u)
project!(a, u)   # uses cached optimal method, returns `a`
```

```julia
project!(a, u)   # auto-tune optimal method and cache for later use, returns `a`
```

See also: [`initialise_project!`](@ref), [`ProjectBroadcast`](@ref),
[`ProjectLoop`](@ref), [`ProjectShared`](@ref)
"""
NSEBase.project!(a::ProjectedField{G, M, <:CuArray},
                 u::VectorField{N, <:FTField{G, <:CuArray}}) where {G<:AbstractGrid, M, N} =
    NSEBase.project!(a, u, project_method(a, y))

"""
    project!(a::ProjectedField{G, M, <:CuArray},
             u::VectorField{N, <:FTField{G, <:CuArray}},
             method::ProjectMethod) -> a

Compute the Galerkin projection of the GPU-resident `VectorField` onto a set of
orthonormal modes stored in the provided `ProjectedField`. Bypasses the auto-tune
cache entirely.

Useful for benchmarking specific methods or for one-off computations where
constructing and caching a method is not warranted.

# Arguments
- `a`: `ProjectedField` on the grid `G` and `CuArray` storage.
- `u`: `VectorField` of `FTField` on the same grid `G` and `CuArray` storage.
- `method`: a pre-constructed `ProjectMethod` — one of `ProjectBroadcast`,
`ProjectLoop`, or `ProjectShared`.

# Returns
- `a` with each element asigned the accumulated constribution of `u` for each mode

# Example
```julia
method = ProjectLoop(a, u)
project!(a, b, method)
```
"""
NSEBase.project!(a::ProjectedField{G, M, <:CuArray},
                 u::VectorField{N, <:FTField{G, <:CuArray}},
            method::ProjectMethod) where {N, S, T, G<:ChannelGrid{S, T}, M} = _project!(a, u, method)

"""
    _project!(a::ProjectedField,
              u::VectorField,
              cache::ProjectBroadcast) -> a

Compute Galerkin projection of `u` onto modes stored in `a` using higher-order
array abstractions provided by CUDA.jl.
"""
function _project!(a::ProjectedField{G},
                   u::VectorField{N},
               cache::ProjectBroadcast{T}) where {N, S, T, G<:ChannelGrid{S, T}}
    pa  = parent(a)
    fill!(pa, zero(T))

    ws = cache.ws
    @inbounds begin
        for n in 1:N
            for m in axes(a, 1)
                for ny in 1:S[2]
                    mode_ny = view(modes(a)[n], ny, m, :, :, :)
                    profs_ny = view(u[n], ny, :, :, :)
                    pa_m = view(pa, m, :, :, :)
                    @. pa_m += ws[ny]*dot(mode_ny, profs_ny)
                end
            end
        end
    end

    return a
end

"""
    _project!(a::ProjectedField,
              u::VectorField,
              cache::ProjectLoop) -> a

Compute Galerkin projection of `u` onto modes stored in `a` using
[_project_loop_kernel!](@ref).
"""
function _project!(a::ProjectedField{<:ChannelGrid{S}},
                   u::VectorField{N},
               cache::ProjectLoop) where {N, S}
    sz      = cache.sz
    nelem   = cache.nelem
    threads = cache.threads
    blocks  = cache.blocks

    @cuda threads=threads blocks=blocks _project_loop_kernel!(
        parent(a), modes(a), u, grid(a).ws, sz, nelem, Val(Int32(N)), Val(Int32(S[2]))
    )

    return a
end

"""
    _project!(a::ProjectedField,
              u::VectorField,
              cache::ProjectShared) -> a

Compute Galerkin projection of `u` onto modes stored in `a` using
[_project_shared_kernel!](@ref).
"""
function _project!(a::ProjectedField{<:ChannelGrid{S}},
                   u::VectorField{N},
               cache::ProjectShared) where {N, S}
    sz      = cache.sz
    nelem   = cache.nelem
    threads = cache.threads
    blocks  = cache.blocks

    @cuda threads=threads blocks=blocks _project_shared_kernel!(
        parent(a), modes(a), u, grid(a).ws, sz, nelem, Val(Int32(N)), Val(Int32(S[2]))
    )

    return a
end

"""
    _project!(a::ProjectedField,
              u::VectorField,
              cache::ProjectWarp) -> a

Compute Galerkin projection of `u` onto modes stored in `a` using
[_project_warp_kernel!](@ref).

Not currently functional since the underlying kernel has a bug.
"""
function _project!(a::ProjectedField{<:ChannelGrid{S}}, u::VectorField{N}, ::ProjectWarp) where {N, S}
    throw(error("this method is broken - use a different one instead"))
    sz = Int32.(size(a))
    nelem = Int32(prod(sz))
    threads = 256
    blocks = cld(nelem, threads)

    @cuda threads=threads blocks=blocks _project_warp_kernel!(
        parent(a), modes(a), u, grid(a).ws, sz, nelem, Val(Int32(N)), Val(Int32(S[2]))
    )

    return a
end


# ------------------ #
# projection kernels #
# ------------------ #
"""
    _project_loop_kernel!(a, modes, u, ws, sz, nelem, ::Val{N}, ::Val{Ny})

GPU kernel: compute projection for each element of `a` on each thread using loops,
which are statically unrolled.
"""
function _project_loop_kernel!(a::CuDeviceArray, modes::NTuple, u::VectorField, ws::CuDeviceArray,
                               sz::NTuple, nelem::Int32, ::Val{N}, ::Val{Ny}) where {N, Ny}
    idx = (blockIdx().x - 1i32)*blockDim().x + threadIdx().x
    idx > nelem && return nothing

    # get cartesian index
    I = _linear_to_cart(idx, sz)

    # each thread computes the full inner reduction for its index
    acc = zero(eltype(a))
    for n in 1:N
        for ny in 1:Ny
            J = CartesianIndex(ny,       I[2], I[3], I[4])
            K = CartesianIndex(ny, I[1], I[2], I[3], I[4])
            acc += ws[ny]*dot(modes[n][K], u[n][J])
        end
    end

    # assign to output
    @inbounds a[I] = acc
    return nothing
end

"""
    _project_shared_kernel!(a, modes, u, ws, sz, nelem, ::Val{N}, ::Val{Ny})

GPU kernel: compute projection for each element of `a` on each thread using loops,
which are statically unrolled. The quadrature weights are assigned to static shared
memory to try to optimise global memory reads.
"""
function _project_shared_kernel!(a::CuDeviceArray, modes::NTuple, u::VectorField, ws::CuDeviceArray,
                                 sz::NTuple, nelem::Int32, ::Val{N}, ::Val{Ny}) where {N, Ny}
    # load weights into shared memory so all threads in the block share it
    ws_shared = @cuStaticSharedMem(Float32, Ny)
    tid = threadIdx().x
    if tid ≤ Ny
        @inbounds ws_shared[tid] = ws[tid]
    end
    sync_threads()

    idx = (blockIdx().x - 1i32)*blockDim().x + threadIdx().x
    idx > nelem && return nothing

    # get cartesian index
    I = _linear_to_cart(idx, sz)

    # each thread computes the full inner reduction for its index
    acc = zero(eltype(a))
    for n in 1:N
        for ny in 1:Ny
            J = CartesianIndex(ny,       I[2], I[3], I[4])
            K = CartesianIndex(ny, I[1], I[2], I[3], I[4])
            acc += ws[ny]*dot(modes[n][K], u[n][J])
        end
    end

    # the mode index I[1] selects the row of the projection
    @inbounds a[I] = acc
    return nothing
end

"""
    _project_warp_kernel!(a, modes, u, ws, sz, nelem, ::Val{N}, ::Val{Ny})

GPU kernel: compute projection for each element of `a` using warp level primitives.
The final result is accumulated using `CUDA.shfl_down_sync` to reduce the result in
a single warp to single number.
"""
function _project_warp_kernel!(a::CuDeviceArray, modes::NTuple, u::VectorField, ws::CuDeviceArray,
                               sz::NTuple, nelem::Int32, ::Val{N}, ::Val{Ny}) where {N, Ny}
    # each warp handles one output element
    warp_id = (threadIdx().x - 1i32)÷32i32 + 1i32
    lane_id = (threadIdx().x - 1i32)%32i32 + 1i32
    warps_per_block = blockDim().x÷32i32
    idx = (blockIdx().x - 1i32)*warps_per_block + warp_id

    # get cartesian index
    acc = zero(eltype(a))
    valid = idx ≤ nelem

    # distribute inner reduction accross lanes
    if valid
        I = _linear_to_cart(idx, sz)
        total = N*Ny
        lane = lane_id - 1i32
        while lane < total
            n  = lane÷Ny + 1i32
            ny = lane%Ny + 1i32
            J = CartesianIndex(ny,       I[2], I[3], I[4])
            K = CartesianIndex(ny, I[1], I[2], I[3], I[4])
            @inbounds acc += ws[ny]*dot(modes[n][K], u[n][J])
            lane += 32i32
        end
    end

    # warp reduction using shuffle
    acc += CUDA.shfl_down_sync(0xffffffff, acc, 16i32)
    acc += CUDA.shfl_down_sync(0xffffffff, acc,  8i32)
    acc += CUDA.shfl_down_sync(0xffffffff, acc,  4i32)
    acc += CUDA.shfl_down_sync(0xffffffff, acc,  2i32)
    acc += CUDA.shfl_down_sync(0xffffffff, acc,  1i32)

    if lane_id == 1i32 && valid
        @inbounds a[I] = acc
    end
    return nothing
end


# ------------- #
# expand method #
# ------------- #
# TODO: this
# TODO: benchmark residual
# TODO: test FFT plans
# TODO: test derivatives
expand!(u::VectorField{N, <:FTField{G, <:CuArray}},
        a::ProjectedField{G, M, <:CuArray}) where {N, G, M} = nothing
