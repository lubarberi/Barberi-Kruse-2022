# Libraries
using Plots
using LinearAlgebra
using Random
using FFTW
using JLD
using Dates
using Roots
using NumericalIntegration
using SparseArrays
#using BenchmarkTools

# Set current directory as working directory
cd(@__DIR__)

# If absent, create Data/ folder
if isdir("Data") == false
    mkdir("Data")
end

include("functions.jl")

function main()
    include("InputParameters.jl")

    # Initialize dynamical fields
    Fields = zeros(Float64, 3, Nx)          # Fields[i, :] = actomyosin (i=1), active nucleator (i=2), inactive nucleator (i=3)
    v      = zeros(Float64, Nx)             # Velocity

    # Initialize auxiliary fields used in AdaptiveTimeStep!() of functions.jl
    Fields_Aux          = zeros(Float64, 3, Nx)
    # Δt-propagated fields
    Fields_LongStep     = zeros(Float64, 3, Nx)
    v_LongStep          = zeros(Float64, Nx)
    # 1 x 0.5*Δt-propagated fields
    Fields_ShortStepAux = zeros(Float64, 3, Nx)
    v_ShortStepAux      = zeros(Float64, Nx)
    # 2 x 0.5*Δt-propagated fields
    Fields_ShortStep    = zeros(Float64, 3, Nx)
    v_ShortStep         = zeros(Float64, Nx)

    # Initialize stress and right hand sides of dynamical Eqns
    ∂xΠ = zeros(Float64, Nx-2)    # Derivative of non-viscous stress Π
    RHS = zeros(Float64, 3, Nx)   # ∂t Fields[i, :] = RHS[i, :]

    # Initialize derivatives
    ∂xFields  = zeros(Float64, 3, Nx)
    ∂xFieldsv = zeros(Float64, 3, Nx)
    ∂xxFields = zeros(Float64, 3, Nx)

    # Force-balance boundary value problem on discretized grid,
    # solved as v[2:Nx-1] = A \ ∂xΠ, w/ v[1] = v[Nx] = 0 for no-flux BCs.
    # Second derivative of viscous stress appriximated w/ standard central difference
    A    = SymTridiagonal((2/(Δx^2) + 1) .* ones(Nx-2), -1/(Δx^2) .* ones(Nx-3))
    # To approximate second derivative of viscous stress w/ 5-point stencil, uncomment below
    #A = zeros(Nx-2, Nx-2)
    #A[1,1]         = (2 + Δx) ; A[1,2]         = -1                  ;
    #A[2,1]         = -16/12   ; A[2,2]         = (30 + 12 * Δx^2)/12 ; A[2,3]         = -16/12              ; A[2,4]       = 1/12
    #A[end-1,end-3] = 1/12     ; A[end-1,end-2] = -16/12              ; A[end-1,end-1] = (30 + 12 * Δx^2)/12 ; A[end-1,end] = -16/12
    #A[end, end-1]  = -1       ; A[end, end]    = (2 + Δx)            ;
    #for i in 3:Nx-4
    #    A[i, i-2] = 1/12                ; A[i, i+2] = 1/12
    #    A[i, i-1] = -16/12              ; A[i, i+1] = -16/12
    #    A[i, i]   = (30 + 12 * Δx^2)/12
    #end
    #A[:,:] = A[:,:] ./ (Δx^2)
    #A = sparse(A)

    # Finite Difference operator for 1st derivative
    # Forward (backward) difference at 1 (Nx) ; Centered difference at 2, Nx-1 ; 5-point stencil at 3...Nx-2
    FDC1 = zeros(Nx, Nx)
    FDC1[1,1]       = -1/Δx     ; FDC1[1,2]     = 1/Δx ;
    FDC1[2,1]       = -1/(2*Δx) ; FDC1[2,3]     = 1/(2*Δx) ;
    FDC1[Nx-1,Nx-2] = -1/(2*Δx) ; FDC1[Nx-1,Nx] = 1/(2*Δx) ;
    FDC1[Nx,Nx-1]   = -1/Δx     ; FDC1[Nx,Nx]   = 1/Δx ;
    for i in 3:Nx-2
        FDC1[i, i-2] =  1 / (12 * Δx)
        FDC1[i, i-1] = -8 / (12 * Δx)
        FDC1[i, i+1] =  8 / (12 * Δx)
        FDC1[i, i+2] = -1 / (12 * Δx)
    end
    FDC1 = sparse(FDC1)
    # Initialize Finite Difference operator for 2nd Derivative
    # Forward (backward) difference at 1 (Nx) ; Centered difference at 2...Nx-1
    FDC2 = zeros(Nx, Nx)
    FDC2[1,1]     = 1/(Δx^2) ; FDC2[1,2]     = -2/(Δx^2) ; FDC2[1,3]   = 1/(Δx^2) ;
    FDC2[Nx,Nx-2] = 1/(Δx^2) ; FDC2[Nx,Nx-1] = -2/(Δx^2) ; FDC2[Nx,Nx] = 1/(Δx^2) ;
    for i in 2:Nx-1
        FDC2[i, i-1] =  1 / (Δx^2)
        FDC2[i, i]   = -2 / (Δx^2)
        FDC2[i, i+1] =  1 / (Δx^2)
    end
    FDC2 = sparse(FDC2)

    MaxError = 0    # Stores the estimated error in the AdaptiveTimeStep! function cycles

    # Initialize vectors where fields are saved for plots
    SavedFields = zeros(3, Nx, plotframes)
    Savedv      = zeros(Nx, plotframes)

    # Initialize time evolution parameters
    ThisFrame    = 1                   # Used to save system state in SavedFields and Savedv matrices, ∈ [1, plotframes]
    LastSaveTime = 0                   # Last time system state was saved, in non-dim. units
    CurrentTime  = 0                   # Real time at current time frame
    ΔtVec        = [MaxTimeStep, 0]    # ΔtVec[1(2)] is Δt_old (Δt)

    # Initial conditions (from InputParameters.jl)
    Fields[:,:] .= FieldICs[:,:]

    # Save initial conditions
    for i in 1:3
        SavedFields[i, :, 1] .= Fields[i, :]
    end
    Savedv[:, 1]  .= v

    println("Simulation starts...")
    # Simulation runs either until FinalTime reached, or until Δt drops below tolerance
    while (CurrentTime < FinalTime && ΔtVec[1] > MinTimeStep)
        # Double latest time step not to end up with very small Δt_old,
        # provided Δt never less than MaxTimeStep (stability of upwind scheme)
        ΔtVec[2] = 2*ΔtVec[1] < MaxTimeStep ? 2*ΔtVec[1] : ΔtVec[1]
        # Propagate fields
        AdaptiveTimeStep!(ErrorTolerance, ΔtVec, Nx, Pars, v, Fields, Fields_Aux, ∂xFields, v_LongStep, Fields_LongStep, v_ShortStepAux, Fields_ShortStepAux, v_ShortStep, Fields_ShortStep, ∂xΠ, A, ∂xFieldsv, ∂xxFields, RHS, FDC1, FDC2, MaxError)
        CurrentTime += ΔtVec[1]  # Update current time (Δt_old is the adapted time step)
        # Save system state if a time interval ≃ FrameTimeStep has passed since last save
        if CurrentTime - LastSaveTime >= FrameTimeStep
            LastSaveTime = CurrentTime
            PrintCurrentTime = round(CurrentTime, digits = 3) # This is to limit the digits of CurrentTime when printing
            println("Reached time $PrintCurrentTime of $FinalTime")
            ThisFrame += 1
            Savedv[:, ThisFrame]  = v
            for i in 1:3
                SavedFields[i, :, ThisFrame] .= Fields[i, :]
            end
        end
        # Export saved data every 10, or if FinalTime reached, or if Δt drops below MinTimeStep
        if (mod(ThisFrame,10) == 0 || CurrentTime >= FinalTime || ΔtVec[1] <= MinTimeStep)
            # Print message if Δt drops below tolerance
            if ΔtVec[1] <= MinTimeStep
                println("Simulation stops at frame $ThisFrame/$plotframes. Time step dropped below $MinTimeStep")
            end
            save(DataFileName, "actomyosin", SavedFields[1, :, 1:ThisFrame], "active_nucleator", SavedFields[2, :, 1:ThisFrame], "inactive_nucleator", SavedFields[3, :, 1:ThisFrame], "velocity", Savedv[:, 1:ThisFrame])
            #println("Data saved, simulation is over.")
        end
    end

    return nothing
end

main()
