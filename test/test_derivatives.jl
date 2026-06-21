@testset "Field derivatives           " begin
    # function definitions
    u_fun(y, x, z, t)      = (1 - y^2)*cos(4π*x)*exp(cos(5.8*z))*atan(sin(2π*t))
    dudx_fun(y, x, z, t)   = -4π*(1 - y^2)*sin(4π*x)*exp(cos(5.8*z))*atan(sin(2π*t))
    d2udx2_fun(y, x, z, t) = -(4π)^2*(1 - y^2)*cos(4π*x)*exp(cos(5.8*z))*atan(sin(2π*t))
    dudy_fun(y, x, z, t)   = -2*y*cos(4π*x)*exp(cos(5.8*z))*atan(sin(2π*t))
    d2udy2_fun(y, x, z, t) = -2*cos(4π*x)*exp(cos(5.8*z))*atan(sin(2π*t))
    dudz_fun(y, x, z, t)   = -5.8*(1 - y^2)*cos(4π*x)*sin(5.8*z)*exp(cos(5.8*z))*atan(sin(2π*t))
    d2udz2_fun(y, x, z, t) = (5.8^2)*(1 - y^2)*cos(4π*x)*(sin(5.8*z)^2 - cos(5.8*z))*exp(cos(5.8*z))*atan(sin(2π*t))
    lapl_fun(y, x, z, t)   = d2udx2_fun(y, x, z, t) + d2udy2_fun(y, x, z, t) + d2udz2_fun(y, x, z, t)
    dudt_fun(y, x, z, t)   = 2π*((1 - y^2)*cos(4π*x)*exp(cos(5.8*z))*cos(2π*t))/(sin(2π*t)^2 + 1)

    # construct grid
    Ny = 32; Nx = 15; Nz = 33; Nt = 51
    y, ws = FDGrids.grid(Ny, -1, 1, MappedGrid(1))
    D₁ = DiffMatrix(y, 3, 1)
    D₂ = DiffMatrix(y, 3, 2)
    g = ChannelGrid(y, Nx, Nz, Nt,
                    2π, 5.8,
                    D₁,
                    D₂,
                    adjoint(D₁, ws),
                    adjoint(D₂, ws),
                    ws)

    # test values of derivatives
    u = CUDA.cu(FFT(Field(g, u_fun)))
    @test Array(parent(NSEBase.ddx_1!(    FTFieldGPU(g), u))) ≈ FFT(Field(g, dudx_fun))
    @test Array(parent(NSEBase.ddx_2!(    FTFieldGPU(g), u))) ≈ FFT(Field(g, dudy_fun))
    @test Array(parent(NSEBase.ddx_3!(    FTFieldGPU(g), u))) ≈ FFT(Field(g, dudz_fun))
    @test Array(parent(NSEBase.laplacian!(FTFieldGPU(g), u))) ≈ FFT(Field(g, lapl_fun))

    # test time derivative of projected field
    M = 10
    Ψ₁ = zeros(ComplexF64, M, Ny, (Nx >> 1) + 1, Nz, Nt)
    for nt in 1:Nt, nz in 1:Nz, nx in 1:(Nx >> 1) + 1
        Ψ₁[:, :, nx, nz, nt] .= (qr(randn(ComplexF64, Ny, M)).Q[:, 1:M])'
    end
    for m in 1:M
        NSEBase.apply_symmetry!(@view(Ψ₁[m, :, :, :, :]), (2, 3, 4))
        Ψ₁[m, :, 1, 1, 1] .= real.(Ψ₁[m, :, 1, 1, 1])
    end
    Ψ = (Ψ₁,)
    a = CUDA.cu(project(FFT(VectorField(g, u_fun)), Ψ))
    @test Array(parent(NSEBase.ddx_4!(similar(a), a))) ≈ project(FFT(VectorField(g, dudt_fun)), Ψ)
end
