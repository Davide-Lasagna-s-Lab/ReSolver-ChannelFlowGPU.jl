# Inner-product for ProjectedField stored on device.

# ---------------- #
# reduction method #
# ---------------- #
abstract type DotMethod end

"""
    DotTwoStage{A} <: DotMethod

Two-stage dot product method for GPU-resident `ProjectedField` arrays.

Computes the weighted Hermitian inner product by first forming an elementwise
weighted product into a pre-allocated intermediate array, then reducing with
`sum`. The weight accounts for the rfft Hermitian symmetry: indices along the
rfft dimension with `i=1` receive weight 1, all others receive weight 2. The
result is divided by 2 to give the correct inner product.

Pre-allocating both the weight and intermediate arrays at construction time
avoids any allocation cost at the call site.

# Fields
- `cache::NTuple{2, A}`: tuple of `(weights, intermediate)` — both `CuArray`s
  of the same size and element type as the input fields.

# Constructor
    DotTwoStage(sz::NTuple{D, Int}, order, ::Type{T}=Float32)

- `sz`:    size of the underlying parent array
- `order`: FFT dimension ordering — `order[1]` is the rfft dimension
- `T`:     real element type, defaults to `Float32`

# Example
```julia
method = DotTwoStage(size(parent(a)), fft_dims(grid(a)), Float32)
dot(a, b, method)
```
"""
struct DotTwoStage{A} <: DotMethod
    cache::NTuple{2, A}

    function DotTwoStage(sz::NTuple{D, Int}, order::NTuple, ::Type{T}=Float32) where {D, T}
        weight_vec   = CuArray{T}([i == 1 ? one(T) : T(2) for i in 1:sz[order[1]]])
        shape        = ntuple(d -> d == order[1] ? sz[order[1]] : 1, D)
        weights      = CUDA.zeros(T, sz...)
        weights     .= reshape(weight_vec, shape)
        intermediate = CUDA.zeros(T, sz...)
        new{typeof(weights)}((weights, intermediate))
    end
end
DotTwoStage(a::ProjectedField) = DotTwoStage(size(a), NSEBase.fft_dims(grid(a)), real(eltype(a)))

"""
    DotAtomic{THREADS, D, A} <: DotMethod

DotAtomic accumulation dot product method for GPU-resident `ProjectedField` arrays.

Each thread computes one weighted elementwise product and accumulates its
contribution into a single scalar result via `CUDA.@atomic`. The optimal thread
count is queried from the hardware via `launch_configuration` at construction
time and encoded as the type parameter `THREADS`, ensuring static dispatch with
no runtime overhead per call.

Best suited for small-to-medium problem sizes where atomic contention on the
single accumulator is not a bottleneck. For large arrays the `DotShared` or
`DotTwoStage` methods are likely faster — use `initialise_dot!` to auto-tune.

# Fields
- `result::A`: pre-allocated single-element `CuArray` accumulator.
- `sz::NTuple{D, Int32}`: size of `ProjectedField` arrays
- `nelem::Int32`: total number of elements of `ProjectedField` arrays (`nelem=prod(sz)`)
- `blocks::Int32`: number of GPU blocks assigned for kernel

# Constructor
    DotAtomic(a::ProjectedField, ::Type{T}=Float32)

- `a`: a representative `ProjectedField` used to determine array size and
  query optimal thread count. Not mutated.
- `T`: real element type, defaults to `Float32`.

# Example
```julia
method = DotAtomic(a, Float32)
dot(a, b, method)
```
"""
struct DotAtomic{D, A} <: DotMethod
    result::A
        sz::NTuple{D, Int32}
     nelem::Int32
   threads::Int32
    blocks::Int32

    function DotAtomic(a::ProjectedField, ::Type{T}=Float32; threads::Union{Nothing, Int}=nothing) where {T}
        pa = parent(a)
        result = CUDA.zeros(T, 1)
        sz = Int32.(size(pa))
        nelem = Int32(prod(sz))
        _threads = if isnothing(threads)
            optimal_threads(_dot_atomic_kernel!,
                            result, pa, pa,
                            nelem, sz;
                            max_threads=nelem)
        else
            threads
        end
        new{length(sz), typeof(result)}(result, sz, nelem, Int32(_threads), cld(nelem, _threads))
    end
end

"""
    DotShared{THREADS, D, A} <: DotMethod

DotShared memory tree-reduction dot product method for GPU-resident `ProjectedField`
arrays.

Each thread block performs a local tree reduction into shared memory, then
contributes a single atomic add to the global result — reducing atomic contention
from `O(N)` to `O(N/THREADS)` compared to the `DotAtomic` method. The shared memory
size is determined by `THREADS` which is encoded as a type parameter and must be
a power of 2.

The optimal thread count is determined at construction time via
`launch_configuration` with shared memory pressure accounted for, and rounded
down to the nearest power of 2 as required by the tree reduction algorithm.

Best suited for medium problem sizes. For very large arrays the `DotTwoStage` method
may still win due to its optimised memory access pattern — use `initialise_dot!`
to auto-tune.

# Fields
- `result::A`: pre-allocated single-element `CuArray` accumulator.
- `sz::NTuple{D, Int32}`: size of `ProjectedField` arrays
- `nelem::Int32`: total number of elements of `ProjectedField` arrays (`nelem=prod(sz)`)
- `blocks::Int32`: number of GPU blocks assigned for kernel

# Constructor
    DotShared(a::ProjectedField, ::Type{T}=Float32)

- `a`: a representative `ProjectedField` used to determine array size and
  query optimal thread count. Not mutated.
- `T`: real element type, defaults to `Float32`.

# Example
```julia
method = DotShared(a, Float32)
dot(a, b, method)
```
"""
struct DotShared{THREADS, D, A} <: DotMethod
    result::A
        sz::NTuple{D, Int32}
     nelem::Int32
    blocks::Int32

    function DotShared(a::ProjectedField, ::Type{T}=Float32; threads::Union{Nothing, Int}=nothing) where {T}
        pa = parent(a)
        result = CUDA.zeros(T, 1)
        sz = Int32.(size(pa))
        nelem = Int32(prod(sz))
        _threads = if isnothing(threads)
                kernel  = @cuda launch=false _dot_shared_kernel!(
                    result, pa, pa, nelem, sz, Val(256)  # dummy Val — replaced below
                )
                threads = let config = launch_configuration(kernel.fun;
                                        shmem = t -> t * sizeof(T))
                    # round down to nearest power of 2 — required for tree reduction
                    prev_pow2 = 2^floor(Int, log2(config.threads))
                    min(prev_pow2, nelem)
            end
        else
            threads
        end
        new{_threads, length(sz), typeof(result)}(result, sz, nelem, cld(nelem, _threads))
    end
end


# --------------------------------------- #
# auto-tuning for optimal method dispatch #
# --------------------------------------- #
const DOT_METHODS = Dict{Type, DotMethod}()

"""
    dot_method(a::ProjectedField{G, M, <:CuArray}) -> DotMethod

Return the `DotMethod` associated with the concrete type of `a`, auto-tuning
if this type has not been seen before. Results are cached in `DOT_METHODS`
keyed on `typeof(a)`.

    dot_method(a) -> cached_method or autotune_dot(a)
"""
function dot_method(a::ProjectedField{G, M, <:CuArray}) where {G, M}
    get!(DOT_METHODS, typeof(a)) do
        autotune_dot(a)
    end
end

"""
    autotune_dot(a::ProjectedField{G, M, <:CuArray{T}}) -> DotMethod

Benchmark all available `DotMethod` implementations against a dummy field of
the same type as `a` and return the fastest. Each candidate is warmed up once
to trigger compilation before timing. Timing uses `CUDA.@elapsed` to measure
device-side execution time, taking the minimum over 5 trials to reduce noise.

Logs the winner and all trial times via `@info`.

    autotune_dot(a) -> best::DotMethod
"""
# TODO: allow option to turn off info
function autotune_dot(a::ProjectedField{G, M, <:CuArray{T}}) where {G, M, T}
    pa = parent(a)
    sz = size(pa)
    b  = similar(a)  # dummy field for benchmarking

    # Construct all candidate methods
    candidates = DotMethod[
        DotTwoStage(sz, NSEBase.fft_dims(grid(a)), real(T)),
        DotAtomic(a, real(T)),
        DotShared(a, real(T)),
    ]

    # Warmup all candidates — triggers compilation
    for method in candidates
        dot(a, b, method)
    end
    CUDA.synchronize()

    # Time each candidate
    times = map(candidates) do method
        minimum(1:5) do _
            CUDA.@elapsed dot(a, b, method)
        end
    end

    best = candidates[argmin(times)]
    @info "Auto-tuned dot product" typeof(a) best=typeof(best) times_ns=times
    return best
end


"""
    initialise_dot!(a::ProjectedField)

Eagerly auto-tune and cache the optimal `DotMethod` for the concrete type of
`a`. After this call, all `dot(a, b)` invocations for fields of the same type
will use the cached optimal method with no auto-tuning overhead.

Should be called once per field type during program initialisation, before
entering any performance-critical loops.

# Example
```julia
a = ProjectedField(...)
b = ProjectedField(...)
initialise_dot!(a)   # benchmarks all methods, caches the winner

# all subsequent calls use the cached optimal method
for i in 1:nsteps
    s = dot(a, b)
end
```

See also: [`reset_dot_cache!`](@ref)
"""
function initialise_dot!(a::ProjectedField)
    dot_method(a)  # triggers auto-tune and caches result
    return nothing
end

"""
    reset_dot_cache!()
    reset_dot_cache!(a::ProjectedField)

Clear the auto-tune cache for all field types, or for the specific type of `a`.
The next `dot` call after resetting will trigger auto-tuning again.

Useful when benchmarking different methods explicitly, or after moving to a
different GPU with different performance characteristics.

# Example
```julia
reset_dot_cache!()     # clear all cached methods
reset_dot_cache!(a)    # clear only the method cached for typeof(a)
```

See also: [`initialise_dot!`](@ref)
"""
reset_dot_cache!() = empty!(DOT_METHODS)
reset_dot_cache!(::P) where {P<:ProjectedField} = delete!(DOT_METHODS, P)


# ------------------------------- #
# top-level inner-product methods #
# ------------------------------- #
"""
    dot(a::ProjectedField{G, M, <:CuArray},
        b::ProjectedField{G, M, <:CuArray}) -> T

Compute the weighted Hermitian inner product of two GPU-resident `ProjectedField`
arrays using the auto-tuned optimal method for their concrete type.

The inner product accounts for the rfft Hermitian symmetry by assigning weight 1
to the first index along the rfft dimension and weight 2 to all others, then
dividing the total by 2. This gives the correct energy-norm inner product for
pseudo-spectral representations of real-valued fields.

The method used is determined by `dot_method(a)`, which auto-tunes on the first
call and returns the cached result on all subsequent calls. Call
`initialise_dot!(a)` before entering performance-critical loops to ensure the
auto-tuning cost is paid upfront.

# Arguments
- `a`, `b`: `ProjectedField`s on the same grid `G` and `CuArray` storage.

# Returns
- A host-side scalar of the real element type `T`.

# Example
```julia
initialise_dot!(a)
s = dot(a, b)   # uses cached optimal method, returns Float32
```

See also: [`initialise_dot!`](@ref), [`DotTwoStage`](@ref), [`DotShared`](@ref),
[`DotAtomic`](@ref)
"""
LinearAlgebra.dot(a::ProjectedField{G, M, A},
                  b::ProjectedField{G, M, A}) where {G<:AbstractGrid, M, A<:CuArray} =
    dot(a, b, dot_method(a))

"""
    dot(a::ProjectedField{G, M, <:CuArray},
        b::ProjectedField{G, M, <:CuArray},
        method::DotMethod) -> T

Compute the weighted Hermitian inner product of `a` and `b` using the
explicitly supplied `method`. Bypasses the auto-tune cache entirely.

Useful for benchmarking specific methods or for one-off computations where
constructing and caching a method is not warranted.

# Arguments
- `a`, `b`: `ProjectedField`s on the same grid with `CuArray` storage.
- `method`: a pre-constructed `DotMethod` — one of `DotTwoStage`, `DotShared`,
  or `DotAtomic`.

# Returns
- A host-side scalar of the real element type `T`.

# Example
```julia
method = DotTwoStage(size(parent(a)), fft_dims(grid(a)), Float32)
s = dot(a, b, method)
```
"""
LinearAlgebra.dot(a::ProjectedField{G, M, <:CuArray},
                  b::ProjectedField{G, M, <:CuArray},
             method::DotMethod) where {G<:AbstractGrid, M} =
    _dot(parent(a), parent(b), method)

"""
    _dot(a, b, cache::DotTwoStage) -> T

Two-stage weighted reduction: elementwise weighted product into `intermediate`,
then `sum`. Stays entirely on device until the final scalar transfer.
"""
function _dot(a::CuArray, b::CuArray, cache::DotTwoStage)
    weights, intermediate = cache.cache
 @. intermediate          = weights*real(dot(a, b))
    return sum(intermediate)/2
end

"""
    _dot(a::CuArray{T}, b::CuArray{T}, cache::DotAtomic{THREADS}) -> T

Single-pass reduction using per-thread `CUDA.@atomic` accumulation into a
scalar. Thread count `THREADS` is a compile-time constant from the type parameter.
"""
function _dot(a::CuArray{T}, b::CuArray{T}, cache::DotAtomic) where {T}
    sz      = cache.sz
    nelem   = cache.nelem
    result  = cache.result
    threads = cache.threads
    blocks  = cache.blocks
    CUDA.fill!(result, zero(T))

    @cuda threads=threads blocks=blocks _dot_atomic_kernel!(
        result, a, b, nelem, sz
    )

    return Array(result)[1]/2
end

"""
    _dot(a::CuArray{T}, b::CuArray{T}, cache::DotShared{THREADS}) -> T

Two-level reduction: shared memory tree reduction within each block, then one
`CUDA.@atomic` per block into the scalar result. `THREADS` must be a power of 2.
"""
function _dot(a::CuArray{T}, b::CuArray{T}, cache::DotShared{THREADS}) where {T, THREADS}
    sz     = cache.sz
    nelem  = cache.nelem
    result = cache.result
    blocks = cache.blocks
    CUDA.fill!(result, zero(T))

    @cuda threads=THREADS blocks=blocks _dot_shared_kernel!(
        result, a, b, nelem, sz, Val(THREADS)
    )

    return Array(result)[1]/2
end


# ----------- #
# dot kernels #
# ----------- #
"""
    _dot_atomic_kernel!(result, a, b, nelem, sz)

GPU kernel: each thread computes one weighted elementwise product and
accumulates into `result` via `CUDA.@atomic`. Weight is 1 for `I[2]==1`,
2 otherwise.
"""
# ! doesn't contain transform ORDER information, so is cannot be transplanted to NSEBase.jl as is
function _dot_atomic_kernel!(result, a::CuDeviceArray, b::CuDeviceArray, nelem::Int32, sz::NTuple)
    idx = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
    idx > nelem && return nothing

    I           = _linear_to_cart(idx, sz)
    w           = I[2] == 1i32 ? one(Float32) : Float32(2)
    contrib     = w * real(dot(@inbounds(a[I]), @inbounds(b[I])))

    CUDA.@atomic result[] += contrib
    return nothing
end

"""
    _dot_shared_kernel!(result, a, b, nelem, sz, ::Val{THREADS})

GPU kernel: shared memory tree reduction within each block followed by a single
`CUDA.@atomic` per block. `THREADS` must be a power of 2 and match the launch
thread count. Shared memory is statically allocated as `THREADS` × `sizeof(T)`.
"""
function _dot_shared_kernel!(result, a::CuDeviceArray, b::CuDeviceArray, nelem::Int32, sz::NTuple, ::Val{THREADS}) where {THREADS}
    shared = @cuStaticSharedMem(Float32, THREADS)
    tid    = threadIdx().x
    idx    = (blockIdx().x - 1i32) * blockDim().x + tid

    shared[tid] = if idx <= nelem
        I = _linear_to_cart(idx, sz)
        w = I[2] == 1i32 ? one(Float32) : Float32(2)
        w * real(dot(@inbounds(a[I]), @inbounds(b[I])))
    else
        zero(Float32)
    end
    sync_threads()

    stride = Int32(THREADS) >> 1i32
    while stride > 0i32
        if tid <= stride
            shared[tid] += shared[tid + stride]
        end
        sync_threads()
        stride >>= 1i32
    end

    tid == 1i32 && CUDA.@atomic result[] += shared[1]
    return nothing
end
