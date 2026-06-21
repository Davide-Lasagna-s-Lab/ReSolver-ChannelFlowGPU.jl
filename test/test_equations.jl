@testset "GPU equations               " begin
    # construct grid
    Ny = 32; Nx = 15; Nz = 33; Nt = 51
    y, ws = FDGrids.grid(Ny, -1, 1, MappedGrid(1))
    D₁ = DiffMatrix(y, 3, 1)
    D₂ = DiffMatrix(y, 3, 2)
    g = CUDA.cu(ChannelGrid(y, Nx, Nz, Nt,
                            2π, 5.8,
                            D₁,
                            D₂,
                            adjoint(D₁, ws),
                            adjoint(D₂, ws),
                            ws))

    # construct Couette and Poiseuille equations
    Re = 10
    Ro = Float32(0.5)
    op = @test_nowarn ReSolverChannelFlow.PlaneCouetteFlow(g, Re; Ro=Ro)

    # check if types are correct and memory is correct place
    @test op.nl isa CartesianPrimitive3DNSE{Float32, <:CuFFTPlans, <:FTField{<:AbstractGrid, <:CuArray}, <:Field{<:AbstractGrid, <:CuArray}, CoriolisForce{Float32}}
    @test op.ln isa CartesianPrimitive3DLNSE{AdjointDiscrete, Float32, <:CuFFTPlans, <:FTField{<:AbstractGrid, <:CuArray}, <:Field{<:AbstractGrid, <:CuArray}, CoriolisForce{Float32}}
    @test op.base isa Tuple{<:CuArray, Nothing, Nothing}
    @test eltype(op.cache1) <: FTField{<:AbstractGrid, <:CuArray}
    @test eltype(op.cache2) <: FTField{<:AbstractGrid, <:CuArray}

    # check if computation completes
    M   = 5
    Ψ   = ntuple(n -> CUDA.randn(ComplexF32, M, Ny, (Nx >> 1) + 1, Nz, Nt), 3)
    a   = ProjectedField(g, Ψ)
    b   = ProjectedField(g, Ψ)
    out = ProjectedField(g, Ψ)
    u = VectorField(g)
    ReSolverChannelFlowGPU.initialise_project!(a, u)
    ReSolverChannelFlowGPU.initialise_expand!(u, a)
    @test_nowarn op(out, a)
    @test_nowarn op(out, a, b)
end
