@testset verbose=true "Dot product                 " begin
    # construct grid and modes
    Ny = 8; Nx = 9; Nz = 9; Nt = 9
    g = ChannelGrid(zeros(Ny), Nx, Nz, Nt,
                    2π, 2π,
                    zeros(Ny, Ny), zeros(Ny, Ny),
                    zeros(Ny, Ny), zeros(Ny, Ny),
                    zeros(Ny))
    M = 5
    Ψ = ntuple(n -> zeros(ComplexF64, M, Ny, (Nx >> 1) + 1, Nz, Nt), 1)

    # construct projected fields
    a = ProjectedField(g, randn(M, (Nx >> 1) + 1, Nz, Nt), Ψ); ad = CUDA.cu(a)
    b = ProjectedField(g, randn(M, (Nx >> 1) + 1, Nz, Nt), Ψ); bd = CUDA.cu(b)

    @testset "methods are correct" begin
        # initialise dot product methods
        method_twostage = DotTwoStage(ad)
        method_atomic   = DotAtomic(ad)
        method_shared   = DotShared(ad)

        # test result
        res_host = dot(a, b)
        @test abs(res_host - dot(ad, bd, method_twostage)) < 2e-4
        @test abs(res_host - dot(ad, bd, method_atomic))   < 2e-4
        @test abs(res_host - dot(ad, bd, method_shared))   < 2e-4
    end

    # second dummy field for testin auto-tuning
    g2 = ChannelGrid(zeros(Ny), 3, 1, 1,
                    2π, 2π,
                    zeros(Ny, Ny), zeros(Ny, Ny),
                    zeros(Ny, Ny), zeros(Ny, Ny),
                    zeros(Ny))
    Ψ2 = ntuple(n -> zeros(ComplexF64, 2, Ny, (3 >> 1) + 1, 1, 1), 1)
    ad2 = CUDA.cu(ProjectedField(g2, randn(2, (3 >> 1) + 1, 1, 1), Ψ))

    @testset "auto-tuning" begin
        @test isempty(ReSolverChannelFlowGPU.DOT_METHODS)
        dot_method(ad)
        @test length(ReSolverChannelFlowGPU.DOT_METHODS) == 1
        reset_dot_cache!()
        @test isempty(ReSolverChannelFlowGPU.DOT_METHODS)
        dot_method(ad)
        dot_method(ad2)
        @test length(ReSolverChannelFlowGPU.DOT_METHODS) == 2
        reset_dot_cache!(ad2)
        @test length(ReSolverChannelFlowGPU.DOT_METHODS) == 1
        dot_method(ad)
        @test length(ReSolverChannelFlowGPU.DOT_METHODS) == 1
        dot_method(ad2)
        reset_dot_cache!()
        @test isempty(ReSolverChannelFlowGPU.DOT_METHODS)
    end
end
