@testset verbose=true "Dot product                 " begin
    @testset "methods are correct" begin
        # construct grid and modes
        Ny = 8; Nx = 9; Nz = 9; Nt = 9
        g = ChannelGrid(zeros(Ny), Nx, Nz, Nt,
                        2π, 2π,
                        zeros(Ny, Ny), zeros(Ny, Ny),
                        zeros(Ny, Ny), zeros(Ny, Ny),
                        zeros(Ny))
        M = 5
        Ψ = ntuple(n -> zeros(ComplexF32, M, Ny, (Nx >> 1) + 1, Nz, Nt), 1)

        # construct projected fields
        a = ProjectedField(g, randn(M, (Nx >> 1) + 1, Nz, Nt), Ψ); ad = CUDA.cu(a)
        b = ProjectedField(g, randn(M, (Nx >> 1) + 1, Nz, Nt), Ψ); bd = CUDA.cu(b)

        # initialise dot product methods
        method_broadcast = DotTwoStage(ad)
        method_atomic    = DotAtomic(ad)
        method_shared    = DotShared(ad)

        # test result
        res_host = dot(a, b)
        @test abs(res_host - dot(ad, bd, method_broadcast)) < 5e-5
        @test abs(res_host - dot(ad, bd, method_atomic))    < 5e-5
        @test abs(res_host - dot(ad, bd, method_shared))    < 5e-5
    end
    @testset "auto-tuning" begin
        # TODO: this
    end
end
