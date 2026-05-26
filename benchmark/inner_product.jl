using CUDA
using LinearAlgebra
using BenchmarkTools

import CUDA: i32

using NSEBase,
      ReSolverChannelFlow,
      ReSolverChannelFlowGPU

# ============================================================
# Benchmark
# ============================================================
function run_benchmarks()
    # Problem size: (ny, nx, nz, nt)
    S = (16, 33, 33, 33)
    M = 4
    T = Float32
    g = ChannelGrid(zeros(S[1]), S[2:end]...,
                    2π, 2π,
                    zeros(S[1], S[1]), zeros(S[1], S[1]),
                    zeros(S[1]);
                    adjoint_diff=false)
    a = ProjectedField(g, randn(M, (S[2] >> 1) + 1, S[3], S[4]),
                          zeros(3*S[1], M, (S[2] >> 1) + 1, S[3], S[4]))
    b = ProjectedField(g, randn(M, (S[2] >> 1) + 1, S[3], S[4]),
                          zeros(3*S[1], M, (S[2] >> 1) + 1, S[3], S[4]))
    ad = CUDA.cu(a)
    bd = CUDA.cu(b)

    # Construct methods once — no allocation cost in benchmark loop
    twostage_m = TwoStage(size(a), NSEBase.fft_dims(g), T)
    atomic_m   = Atomic(ad, T)
    shared_m   = Shared(ad, T)

    methods      = [twostage_m, atomic_m, shared_m]
    method_names = ["TwoStage", "Atomic", "Shared"]

    # warm-up
    println("Warming up...")
    dot(a, b)
    for method in methods
        dot(ad, bd, method)
    end
    CUDA.synchronize()

    # correctness check
    rs = dot(a, b)
    results = [dot(ad, bd, m) for m in methods]
    println("\nCorrectness check (S=$S):")
    println("  ", rpad("Single CPU:", 12), rs)
    for (name, r) in zip(method_names, results)
        println("  ", rpad("$name: ", 12), r)
    end
    println("  Max pairwise difference: ",
        maximum(abs, [results[i] - results[j]
                      for i in eachindex(results)
                      for j in i+1:length(results)])/abs(rs))

    # single-size detailed benchmark
    println("\nDetailed benchmark (S=$S, T=$T):")
    ts = @belapsed(dot($a, $b))
    allocs = @allocated(dot(a, b))
    println("\n  Single CPU")
    println("      Host time (μs): ", round(ts * 1e6, digits=3))
    println("     Allocations (B): ", allocs)
    for (name, method) in zip(method_names, methods)
        println("\n  $name")

        # Device-side time only — excludes host overhead and scalar transfer
        t_device = minimum(1:20) do _
            CUDA.@elapsed dot(ad, bd, method)
        end

        # Full end-to-end time — includes launch overhead and scalar transfer
        t_host = @belapsed begin
            CUDA.@sync dot($ad, $bd, $method)
        end seconds=2

        # Allocation check — should be near zero after fixes
        allocs = @allocated begin
            dot(ad, bd, method)
            CUDA.synchronize()
        end

        println("    Device time (μs): ", round(t_device * 1e6, digits=3))
        println("      Host time (μs): ", round(t_host   * 1e6, digits=3))
        println("     Allocations (B): ", allocs)
    end

    # scaling sweep
    sizes = [
        (8,  9, 25,   7),
        (8, 25, 33,  13),
        (8, 33, 65,  25),
        (8, 65, 129, 65),
    ]
    Ms = [4, 4, 4, 8]

    println("\n\nScaling sweep (device time only, μs):")
    println(rpad("Size", 20),
            rpad("TwoStage", 13),
            rpad("Atomic", 13),
            rpad("Shared", 12),
            rpad("Bandwidth (GB/s)", 20),
            "GPU Speed-up")
    println("-"^91)

    for (M, sz) in zip(Ms, sizes)
        g = ChannelGrid(zeros(sz[1]), sz[2:end]...,
                        2π, 2π,
                        zeros(sz[1], sz[1]), zeros(sz[1], sz[1]),
                        zeros(sz[1]);
                        adjoint_diff=false)
        a_ = ProjectedField(g, randn(M, (sz[2] >> 1) + 1, sz[3], sz[4]),
                               zeros(3*sz[1], M, (sz[2] >> 1) + 1, sz[3], sz[4]))
        b_ = ProjectedField(g, randn(M, (sz[2] >> 1) + 1, sz[3], sz[4]),
                               zeros(3*sz[1], M, (sz[2] >> 1) + 1, sz[3], sz[4]))
        ad_ = CUDA.cu(a_)
        bd_ = CUDA.cu(b_)

        m_twostage = TwoStage(size(a_), NSEBase.fft_dims(g), T)
        m_atomic   = Atomic(ad_, T)
        m_shared   = Shared(ad_, T)

        local_methods = [m_twostage, m_atomic, m_shared]

        # Warmup
        dot(a_, b_)
        for m in local_methods
            dot(ad_, bd_, m)
        end
        CUDA.synchronize()

        ts = @belapsed(dot($a_, $b_))
        times = map(local_methods) do method
            minimum(1:20) do _
                CUDA.@elapsed dot(ad_, bd_, method)
            end
        end

        # Effective memory bandwidth for the fastest method:
        # reads 2 CuArrays of ComplexF32 = 2 * prod(sz) * 8 bytes
        best_time   = minimum(times)
        bytes_read  = 2*prod(sz)*sizeof(Complex{T})
        bandwidth   = bytes_read/best_time/1e9

        println(rpad(string(sz), 20),
                rpad(round(times[1]*1e6, digits=3), 13),
                rpad(round(times[2]*1e6, digits=3), 13),
                rpad(round(times[3]*1e6, digits=3), 12),
                rpad(round(bandwidth, digits=1),    20),
                round(ts/best_time, digits=2))
    end
end

run_benchmarks()
