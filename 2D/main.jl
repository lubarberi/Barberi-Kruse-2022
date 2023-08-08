using CUDA
using JLD
using FFTW
using Printf
using Roots
using Random

include("kernels.jl")

function main()
    @inbounds begin
        # Initialize dynamical fields
        C      = CuArray{Float64}(zeros(Float64, Nx, Ny))         # Actomyosin
        Np     = CuArray{Float64}(zeros(Float64, Nx, Ny))         # N_+ = N_a + N_i
        Nm     = CuArray{Float64}(zeros(Float64, Nx, Ny))         # N_m = N_a - N_i
        V      = CuArray{Float64}(zeros(Float64, Nx, Ny, 2))      # Velocity (1 = x, 2 = y)
        Π      = CuArray{Float64}(zeros(Float64, Nx, Ny))         # Non-viscous stress field

        # Initialize RHS (s.t. ∂t A = RHS_A)
        RHS_C  = CuArray{Float64}(zeros(Float64, Nx, Ny)) 
        RHS_Np = CuArray{Float64}(zeros(Float64, Nx, Ny))
        RHS_Nm = CuArray{Float64}(zeros(Float64, Nx, Ny))
        # Initialize divergences ∂(AV)
        ∂CV    = CuArray{Float64}(zeros(Float64, Nx, Ny, 2))
        ∂NpV   = CuArray{Float64}(zeros(Float64, Nx, Ny, 2))
        ∂NmV   = CuArray{Float64}(zeros(Float64, Nx, Ny, 2))
        # Initialize divergences div_AV = ∇⋅(A𝐕)
        div_CV    = CuArray{Float64}(zeros(Float64, Nx, Ny))
        div_NpV   = CuArray{Float64}(zeros(Float64, Nx, Ny))
        div_NmV   = CuArray{Float64}(zeros(Float64, Nx, Ny))
        # Initialize laplacians ΔA = ∇⋅∇A
        ΔC    = CuArray{Float64}(zeros(Float64, Nx, Ny))
        ΔNp   = CuArray{Float64}(zeros(Float64, Nx, Ny))
        ΔNm   = CuArray{Float64}(zeros(Float64, Nx, Ny))

        # Factors for FFT derivatives (analogous to 1D code in this repository)
        factor_∂x = CUDA.zeros(ComplexF64, length(kx), Nx)
        factor_∂y = CUDA.zeros(ComplexF64, length(kx), Ny)
        factor_Δ = CUDA.zeros(Float64, length(kx), Nx)

        @cuda threads = block_dim blocks = gridFFT_dim kernel_compute_FFTderivative_factors!(factor_∂x, factor_∂y, factor_Δ, kx, ky, kx2, ky2)

        # FFT
        fV = CUDA.zeros(ComplexF64, Lkx, Ny, 2)
        fΠ = CUDA.zeros(ComplexF64, Lkx, Ny)

        # Homogeneous Steady State (HSS), used in initial conditions below
        # Solving EqnHSS = 0 for Na gives Na at HSS
        EqnHSS(Na) = - (Ωd0 + Ωd * A * Na / Kd) * Na + Ω0 * (1 - Na) * (1 + Ω * Na^2)
        NaHSS = find_zero(EqnHSS, (0, 1))
        # Ni and C at HSS are calculated from NaHSS
        NiHSS = 1 - NaHSS
        CHSS  = (A / Kd) * NaHSS

        # Initial Conditions
        # Initialize bump (2D Gaussian)
        ZeroMeanBump = CuArray{Float64}(zeros(Nx, Ny))
        get_ZeroMeanBump!(ZeroMeanBump, L, 2^0.5, Δx, Δy)       #Arguments: ZeroMeanBump, Center of Gaussian, Variance, Δx, Δy
        # Initialize stripe (1D Gaussian along x, translationally invariant along y)
        ZeroMeanStripe = CuArray{Float64}(zeros(Nx, Ny))
        get_ZeroMeanStripe!(ZeroMeanStripe, L, 4^0.5, Δx, Δy)   #Arguments: ZeroMeanStripe, Center of Gaussian, Variance, Δx, Δy
        # Initialize noise
        seedd = 123                                             # Seed of rand. number generator
        Random.seed!(seedd)                                     # Seeds random number generator
        ε   = 0.01                                              # Noise amplitude, small number
        ZeroMeanNoise = CuArray{Float64}(rand(Float64, Nx, Ny)) # Matrix of weak noise, made of random numbers between ε*[-1, 1]
        @. ZeroMeanNoise = ε * (-1 + 2 * ZeroMeanNoise)
        get_ZeroMeanNoise!(ZeroMeanNoise, L, Δx, Δy)            # Noisy vector with zero average, s.t. HSS + Noise conserves tot. number of molecules
        # Uncomment if desired IC is homogeneous steady state (HSS) + bump + weak noise
        @. C[:,:]  = CHSS          * (1 + ZeroMeanBump + ZeroMeanNoise)
        @. Nm[:,:] = (2*NaHSS - 1) * (1 + ZeroMeanBump + ZeroMeanNoise)
        @. Np[:,:] =                 (1 + ZeroMeanBump + ZeroMeanNoise)
        # Uncomment if desired IC is homogeneous steady state (HSS) + stripe + weak noise
        #@. C[:,:]  = CHSS          * (1 + ZeroMeanStripe + ZeroMeanNoise)
        #@. Nm[:,:] = (2*NaHSS - 1) * (1 + ZeroMeanStripe + ZeroMeanNoise)
        #@. Np[:,:] =                 (1 + ZeroMeanStripe + ZeroMeanNoise)
        # Uncomment if desired IC is homogeneous steady state (HSS) + weak noise
        #@. C[:,:]  = CHSS          * (1 + ZeroMeanNoise)
        #@. Nm[:,:] = (2*NaHSS - 1) * (1 + ZeroMeanNoise)
        #@. Np[:,:] =                  1 + ZeroMeanNoise

        println("Simulation starts...")

        CUDA.@time for t in 0:Nt
            if (t%PrintEvery==0)
                any(isnan, C) && return 1
                numm = @sprintf("%08d", t)  #print time iteration step, number format
                global nm = "data.jld"
                save(nm, "actin", Array(C), "active_nucleator", Array((Np .+ Nm)/2), "inactive_nucleator", Array((Np .- Nm)/2), "velocity", Array(V))
                println("Now at step: ", t, " of ", Nt)
            end
            EulerForward!(Δt, V, Π, fV, fΠ, FT, IFT, Lkx, factor_∂x, factor_∂y, factor_Δ, Z, B, C, Np, Nm, ΔC, ΔNp, ΔNm, ∂CV, ∂NpV, ∂NmV, div_CV, div_NpV, div_NmV, RHS_C, RHS_Np, RHS_Nm, Dc, Dap, Dam, A, Kd, Ω0, Ω, Ωd, Ωd0, ZeroMeanBump)
        end
    end

    return nothing
end

# Launch simulation
main()