using CUDA
using LinearAlgebra
using BenchmarkTools
using StyledStrings

using NSEBase,
      ReSolverChannelFlow,
      ReSolverChannelFlowGPU

# ============================================================
# Benchmark configuration
# ============================================================
const BENCHMARK_SIZES = [
    # (S[1], S[2], S[3], S[4], M, N) — channel modes, nx, nz, nt, velocity components
    (17, 4, 17,   9, 5, 2),
    (33, 4, 33,  17, 5, 3),
    (33, 8, 65,  17, 5, 3),
    (65, 16, 129, 33, 8, 3),
]
using Random
Random.seed!(0)
# ============================================================
# Helper to construct fields for a given size
# ============================================================
using InteractiveUtils
function make_fields(S, M, N)
    # Construct grid — adjust constructor to match your API
    g = ChannelGrid(zeros(S[2]), S[1], S[3], S[4],
                    2π, 2π,
                    zeros(S[2], S[2]), zeros(S[2], S[2]),
                    zeros(S[2], S[2]), zeros(S[2], S[2]),
                    rand(S[2]))

    # ProjectedField — CuArray backed
    a = ProjectedField(g, ntuple(_ -> randn(ComplexF64, S[2], M, (S[1] >> 1) + 1, S[3], S[4]), N))

    # VectorField of N FTFields
    u = VectorField([FTField(g) for _ in 1:N]...)

    # Fill with random data
    a .= randn(ComplexF64, size(parent(a)))

    return u, a
end

# ============================================================
# Construct all methods for a given field pair
# ============================================================
function make_methods(u, a)
    Dict(
        "Broadcast" => ReSolverChannelFlowGPU.ExpandBroadcast(),
        "Modal 1"   => ReSolverChannelFlowGPU.ExpandModal(CUDA.cu(u), CUDA.cu(a), false),
        "Modal 2"   => ReSolverChannelFlowGPU.ExpandModal(CUDA.cu(u), CUDA.cu(a), true),
    )
end

# ============================================================
# Correctness check
# ============================================================
function check_correctness(u, a, methods)
    println("  Correctness check:")

    # Compute reference with CPU method
    ref_u  = similar(u)
    expand!(ref_u, a)
    ref = copy.(parent(ref_u))

    # move data to device
    ud = CUDA.cu(u); ad = CUDA.cu(a)

    # compute projection using methods
    names  = collect(keys(methods))
    for name in names
        out = similar(ud)
        expand!(out, ad, methods[name])
        CUDA.synchronize()
        result = ntuple(n -> Array(parent(out[n])), length(u))
        diff = max(ntuple(n -> maximum(abs, result[n] .- ref[n]), length(u))...)
        diff < 1f-4 ? printstyled("    ✓", color=:green) : printstyled("    ✗", color=:red)
        println(" $name vs CPU: max diff = $diff")
    end
end

# ============================================================
# Detailed benchmark for a single size
# ============================================================
function benchmark_single(u, a, methods, S, N)
    println("\n  Detailed benchmark (S=$S, M=$(size(a, 1)), N=$N, T=Float32):")
    println("  ", rpad("Method", 16),
                  rpad("Device (μs)", 14),
                  rpad("Host (μs)", 14),
                  rpad("Allocs (B)", 12),
                  rpad("Speed-up", 12))
    println("  ", "-"^64)

    # benchmark CPU method
    expand!(u, a) # warm-up
    ts = @belapsed expand!($u, $a) seconds=2

    # move data to device
    ud = CUDA.cu(u); ad = CUDA.cu(a)

    for (name, method) in methods
        # Warmup
        expand!(ud, ad, method)
        CUDA.synchronize()

        # Device-side time — pure GPU execution
        t_device = minimum(1:20) do _
            CUDA.@elapsed expand!(ud, ad, method)
        end

        # Host-side time — full round trip including launch overhead
        t_host = @belapsed begin
            CUDA.@sync expand!($ud, $ad, $method)
        end seconds=2

        # Allocation check
        expand!(ud, ad, method)  # ensure compiled
        CUDA.synchronize()
        allocs = @allocated begin
            expand!(ud, ad, method)
            CUDA.synchronize()
        end

        println("  ", rpad(name, 16),
                      rpad(round(t_device*1e6, digits=3), 14),
                      rpad(round(t_host*1e6,   digits=3), 14),
                      rpad(allocs, 12),
                      rpad(round(ts/t_host, digits=2), 12))
    end
end

# ============================================================
# Scaling sweep
# ============================================================
function benchmark_scaling(sizes)
    println("\n\nScaling sweep (device time, μs):")

    # Collect all method names from first size
    S0, M0, N0   = first(sizes)[1:4], first(sizes)[5], first(sizes)[6]
    u0, a0       = make_fields(S0, M0, N0)
    method_names = collect(keys(make_methods(u0, a0)))

    # Header
    header = rpad("(S..., M, N)", 30)
    for name in method_names
        header *= rpad(name * " (μs)", 16)
    end
    header *= rpad("BW (GB/s)", 12)
    header *= rpad("Speed-up", 12)
    println(header)
    println("-"^(30 + 16*length(method_names) + 21))

    for config in sizes
        S = config[1:4]
        M = config[5]
        N = config[6]

        u, a    = make_fields(S, M, N)
        ud, ad  = CUDA.cu(u), CUDA.cu(a)
        methods = make_methods(ud, ad)

        # Warmup all methods
        expand!(u, a)
        for method in values(methods)
            expand!(ud, ad, method)
        end
        CUDA.synchronize()

        # Time each method
        ts = @belapsed expand!($u, $a) seconds=2
        times = map(method_names) do name
            minimum(1:20) do _
                CUDA.@elapsed expand!(ud, ad, methods[name])
            end
        end

        # Effective bandwidth for fastest method:
        # reads N FTFields + modes, writes one ProjectedField
        # each element is Complex{T} = 2*sizeof(T) bytes
        bytes_read    = N * prod(S) * sizeof(Complex{Float32})   # u
        bytes_read   += N * prod(S) * sizeof(Complex{Float32})   # modes
        bytes_written = prod(size(parent(a))) * sizeof(Complex{Float32})
        total_bytes   = bytes_read + bytes_written
        best_bw       = total_bytes / minimum(times) / 1e9

        row = rpad(string(config), 30)
        for t in times
            row *= rpad(round(t*1e6, digits=3), 16)
        end
        row *= rpad(round(best_bw, digits=1), 12)
        row *= rpad(round(ts/minimum(times), digits=2), 12)
        println(row)
    end
end

# ============================================================
# Main entry point
# ============================================================
function run_expand_benchmarks()
    println("="^70)
    println("ProjectedField projection benchmark")
    println("="^70)

    # ── detailed benchmark at reference size ──────────────────────────── #
    S_ref, M_ref, N_ref = (33, 16, 65, 17), 5, 3
    println("\nReference size: S=$S_ref, M=$M_ref, N=$N_ref")

    u_ref, a_ref = make_fields(S_ref, M_ref, N_ref)
    methods_ref  = make_methods(u_ref, a_ref)

    check_correctness(u_ref, a_ref, methods_ref)
    benchmark_single(u_ref, a_ref, methods_ref, S_ref, N_ref)

    # ── scaling sweep ─────────────────────────────────────────────────── #
    benchmark_scaling(BENCHMARK_SIZES)
end

run_expand_benchmarks()
