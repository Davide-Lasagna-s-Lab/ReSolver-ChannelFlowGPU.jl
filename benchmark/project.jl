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
    a = ProjectedField(g, ntuple(_ -> randn(ComplexF64, M, S[2], (S[1] >> 1) + 1, S[3], S[4]), N))

    # VectorField of N FTFields
    u = VectorField([FTField(g) for _ in 1:N]...)

    # Fill with random data
    for n in 1:N
        parent(u[n]) .= randn(ComplexF64, size(parent(u[n])))
    end

    return a, u
end

# ============================================================
# Construct all methods for a given field pair
# ============================================================
function make_methods(a, u)
    Dict(
        "Broadcast" => ReSolverChannelFlowGPU.ProjectBroadcast(CUDA.cu(a)),
        "Loop"      => ReSolverChannelFlowGPU.ProjectLoop(CUDA.cu(a)),
        "Shared"    => ReSolverChannelFlowGPU.ProjectShared(CUDA.cu(a)),
        # "Warp"      => ReSolverChannelFlowGPU.ProjectWarp(),
    )
end

# ============================================================
# Correctness check
# ============================================================
function check_correctness(a, u, methods)
    println("  Correctness check:")

    # Compute reference with CPU method
    ref_a  = similar(a)
    project!(ref_a, u)
    ref = copy(parent(ref_a))

    # move data to device
    ad = CUDA.cu(a); ud = CUDA.cu(u)

    # compute projection using methods
    names  = collect(keys(methods))
    for name in names
        out = similar(ad)
        project!(out, ud, methods[name])
        CUDA.synchronize()
        result = Array(parent(out))
        diff   = maximum(abs, result .- ref)
        diff < 1f-4 ? printstyled("    ✓", color=:green) : printstyled("    ✗", color=:red)
        println(" $name vs CPU: max diff = $diff")
    end
end

# ============================================================
# Detailed benchmark for a single size
# ============================================================
function benchmark_single(a, u, methods, S, N, T)
    println("\n  Detailed benchmark (S=$S, N=$N, T=$T):")
    println("  ", rpad("Method", 16),
                  rpad("Device (μs)", 14),
                  rpad("Host (μs)", 14),
                  rpad("Allocs (B)", 12),
                  rpad("Speed-up", 12))
    println("  ", "-"^64)

    # benchmark CPU method
    project!(a, u) # warm-up
    ts = @belapsed project!($a, $u) seconds=2

    # move data to device
    ad = CUDA.cu(a); ud = CUDA.cu(u)

    for (name, method) in methods
        # Warmup
        project!(ad, ud, method)
        CUDA.synchronize()

        # Device-side time — pure GPU execution
        t_device = minimum(1:20) do _
            CUDA.@elapsed project!(ad, ud, method)
        end

        # Host-side time — full round trip including launch overhead
        t_host = @belapsed begin
            CUDA.@sync project!($ad, $ud, $method)
        end seconds=2

        # Allocation check
        project!(ad, ud, method)  # ensure compiled
        CUDA.synchronize()
        allocs = @allocated begin
            project!(ad, ud, method)
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
function benchmark_scaling(sizes, T)
    println("\n\nScaling sweep (device time, μs):")

    # Collect all method names from first size
    S0, M0, N0   = first(sizes)[1:4], first(sizes)[5], first(sizes)[6]
    a0, u0       = make_fields(S0, M0, N0)
    method_names = collect(keys(make_methods(a0, u0)))

    # Header
    header = rpad("(S..., N)", 30)
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

        a, u    = make_fields(S, M, N)
        ad, ud  = CUDA.cu(a), CUDA.cu(u)
        methods = make_methods(ad, ud)

        # Warmup all methods
        project!(a, u)
        for method in values(methods)
            project!(ad, ud, method)
        end
        CUDA.synchronize()

        # Time each method
        ts = @belapsed project!($a, $u) seconds=2
        times = map(method_names) do name
            minimum(1:20) do _
                CUDA.@elapsed project!(ad, ud, methods[name])
            end
        end

        # Effective bandwidth for fastest method:
        # reads N FTFields + modes, writes one ProjectedField
        # each element is Complex{T} = 2*sizeof(T) bytes
        bytes_read    = N * prod(S) * sizeof(Complex{T})   # u
        bytes_read   += N * prod(S) * sizeof(Complex{T})   # modes
        bytes_written = prod(size(parent(a))) * sizeof(Complex{T})
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
function run_project_benchmarks()
    println("="^70)
    println("ProjectedField projection benchmark")
    println("="^70)

    # ── detailed benchmark at reference size ──────────────────────────── #
    S_ref, M_ref, N_ref = (33, 16, 65, 17), 5, 3
    println("\nReference size: S=$S_ref, M=$M_ref, N=$N_ref")

    a_ref, u_ref = make_fields(S_ref, M_ref, N_ref)
    methods_ref  = make_methods(a_ref, u_ref)

    check_correctness(a_ref, u_ref, methods_ref)
    benchmark_single(a_ref, u_ref, methods_ref, S_ref, N_ref, Float32)

    # ── scaling sweep ─────────────────────────────────────────────────── #
    benchmark_scaling(BENCHMARK_SIZES, Float32)
end

run_project_benchmarks()
