include("InputParameters.jl")

function kernel_stripe!(stripe, L, VarStripe, Δx)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    @inbounds stripe[i,j] = exp(-((i*Δx-L/2)^2)/(2*(VarStripe^2)))
    return nothing
end

@inline function get_ZeroMeanStripe!(stripe, L, VarStripe, Δx, Δy)
    @cuda threads = block_dim blocks = grid_dim kernel_stripe!(stripe, L, VarStripe, Δx)
    stripe .-= sum(stripe)*Δx*Δy/(L^2)
end

function kernel_bump!(bump, L, VarBump, Δx, Δy)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    @inbounds bump[i,j] = exp(-((i*Δx-L/2)^2 + (j*Δy-L/2)^2)/(2*(VarBump^2)))
    return nothing
end

@inline function get_ZeroMeanBump!(bump, L, VarBump, Δx, Δy)
    @cuda threads = block_dim blocks = grid_dim kernel_bump!(bump, L, VarBump, Δx, Δy)
    bump .-= sum(bump)*Δx*Δy/(L^2)
end

@inline function get_ZeroMeanNoise!(noise, L, Δx, Δy)
    noise .-= sum(noise)*Δx*Δy/(L^2)
end

function kernel_compute_FFTderivative_factors!(factor_∂x, factor_∂y, factor_Δ, kx, ky, kx2, ky2) # Indexing (i,j)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    if i>length(kx)
        return nothing
    end
    factor_∂x[i,j] = im*kx[i]
    factor_∂y[i,j] = im*ky[j]
    factor_Δ[i,j] = -kx2[i]-ky2[j]
    return nothing
end 

function kernel_compute_FFT_v!(fV, fΠ_, kx_, ky_, Lkx)  # Indexing (i,j)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    if i>Lkx
        return nothing
    end
    kx = kx_[i];     kx2 = kx*kx
    ky = ky_[j];     ky2 = ky*ky

    fΠ = fΠ_[i, j]
    
    fV[i,j,1] = im * kx * fΠ / (1 + 2*(kx2 + ky2))
    fV[i,j,2] = im * ky * fΠ / (1 + 2*(kx2 + ky2))

    return nothing
end

function kernel_compute_RHS!(C_, Np_, Nm_, ΔC_, ΔNp_, ΔNm_, ∂CV_, ∂NpV_, ∂NmV_, div_CV, div_NpV, div_NmV, RHS_C, RHS_Np, RHS_Nm, Dc, Dap, Dam, A, Kd, Ω0, Ω, Ωd, Ωd0)  # Indexing (i,j)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    C  = C_[i,j]
    Np = Np_[i,j]
    Nm = Nm_[i,j]

    ΔC = ΔC_[i, j]
    ΔNp = ΔNp_[i, j]
    ΔNm = ΔNm_[i, j]

    div_CV =  ∂CV_[i,j,1]  + ∂CV_[i,j,2]
    div_NpV = ∂NpV_[i,j,1] + ∂NpV_[i,j,2]
    div_NmV = ∂NmV_[i,j,1] + ∂NmV_[i,j,2]
    
    RHS_C[i,j]  = - div_CV  + Dc * ΔC + A * 0.5 * (Np + Nm) - Kd * C
    RHS_Np[i,j] = - div_NpV + 0.5 * (Dap * ΔNp + Dam * ΔNm)
    RHS_Nm[i,j] = - div_NmV + 0.5 * (Dam * ΔNp + Dap * ΔNm) + Ω0 * ( 1 + Ω * 0.25 * (Np + Nm)^2) * (Np - Nm) - (Ωd0 + Ωd * C) * (Np + Nm)

    return nothing
end

function compute_RHS!(V, Π, fV, fΠ, FT, IFT, Lkx, factor_∂x, factor_∂y, factor_Δ, Z, B, C, Np, Nm, ΔC, ΔNp, ΔNm, ∂CV, ∂NpV, ∂NmV, div_CV, div_NpV, div_NmV, RHS_C, RHS_Np, RHS_Nm, Dc, Dap, Dam, A, Kd, Ω0, Ω, Ωd, Ωd0)
    
    @. Π[:,:] = Z*(C[:,:]^2) - B*(C[:,:]^3)

    @inbounds @views begin
        fΠ[:,:] .= FT * Π[:,:]
    end

    @cuda threads = block_dim blocks = gridFFT_dim kernel_compute_FFT_v!(fV, fΠ, kx, ky, Lkx)

    @inbounds @views begin
        V[:,:,1]  .= IFT * fV[:,:,1]
        V[:,:,2]  .= IFT * fV[:,:,2]
    end

    @inbounds @views begin
        ∂CV[:,:,1]  .= IFT * (factor_∂x .* (FT * (C[:,:] .* V[:,:,1])))
        ∂CV[:,:,2]  .= IFT * (factor_∂y .* (FT * (C[:,:] .* V[:,:,2])))

        ∂NpV[:,:,1] .= IFT * (factor_∂x .* (FT * (Np[:,:] .* V[:,:,1])))
        ∂NpV[:,:,2] .= IFT * (factor_∂y .* (FT * (Np[:,:] .* V[:,:,2])))

        ∂NmV[:,:,1] .= IFT * (factor_∂x .* (FT * (Nm[:,:] .* V[:,:,1])))
        ∂NmV[:,:,2] .= IFT * (factor_∂y .* (FT * (Nm[:,:] .* V[:,:,2])))
    end

    @inbounds @views begin
        ΔC[:,:,1]  .= IFT * (factor_Δ .* (FT * C[:,:]))
        ΔNp[:,:,1] .= IFT * (factor_Δ .* (FT * Np[:,:]))
        ΔNm[:,:,1] .= IFT * (factor_Δ .* (FT * Nm[:,:]))
    end

    @cuda threads = block_dim blocks = grid_dim kernel_compute_RHS!(C, Np, Nm, ΔC, ΔNp, ΔNm, ∂CV, ∂NpV, ∂NmV, div_CV, div_NpV, div_NmV, RHS_C, RHS_Np, RHS_Nm, Dc, Dap, Dam, A, Kd, Ω0, Ω, Ωd, Ωd0)
end

function kernel_EulerForward!(Δt, C, Np, Nm, RHS_C, RHS_Np, RHS_Nm, ZeroMeanBump)  # Indexing (i,j)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    C[i,j]  += Δt * RHS_C[i,j]
    Np[i,j] += Δt * RHS_Np[i,j]
    Nm[i,j] += Δt * RHS_Nm[i,j]

    return nothing
end

function EulerForward!(Δt, V, Π, fV, fΠ, FT, IFT, Lkx, factor_∂x, factor_∂y, factor_Δ, Z, B, C, Np, Nm, ΔC, ΔNp, ΔNm, ∂CV, ∂NpV, ∂NmV, div_CV, div_NpV, div_NmV, RHS_C, RHS_Np, RHS_Nm, Dc, Dap, Dam, A, Kd, Ω0, Ω, Ωd, Ωd0, ZeroMeanBump)
    # This funct. propagates dynamical fields in time using Euler forward
    # Calculate right hand sides
    compute_RHS!(V, Π, fV, fΠ, FT, IFT, Lkx, factor_∂x, factor_∂y, factor_Δ, Z, B, C, Np, Nm, ΔC, ΔNp, ΔNm, ∂CV, ∂NpV, ∂NmV, div_CV, div_NpV, div_NmV, RHS_C, RHS_Np, RHS_Nm, Dc, Dap, Dam, A, Kd, Ω0, Ω, Ωd, Ωd0)
    # Update fields with their values at t + Δt
    @cuda threads = block_dim blocks = grid_dim kernel_EulerForward!(Δt, C, Np, Nm, RHS_C, RHS_Np, RHS_Nm, ZeroMeanBump)
    return nothing
end