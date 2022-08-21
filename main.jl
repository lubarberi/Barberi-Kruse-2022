# Libraries
using Plots
using LinearAlgebra
using Random
using FFTW
using JLD
using Dates
using Roots
using NumericalIntegration

# Set current directory as working directory
cd(@__DIR__)

# If absent, create Data/ folder
if isdir("Data") == false
    mkdir("Data")
end

include("functions.jl")
include("InputParameters.jl")

function main(Nx, Δx, FrameTimeStep, plotframes, FinalTime, ErrorTolerance, FieldICs, MinTimeStep, DataFileName)

    # Set up FFT tools to calculate derivatives
    k = 2*pi*rfftfreq(Nx, 1/Δx)             # Initialize reciprocal grid
    P = plan_rfft(x)                        # When applied to a vector, P calculates its FFT (provided the vector length is length(x))
    Pinv = inv(P)                           # Works like P, but inverts FFT
    # We calculate the n_th derivative of vector X as Pinv * ( (P * X) .* FactorNthDer), with:
    Factor1stDer = im .* k
    Factor2ndDer = - k .* k

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
    σ   = zeros(Float64, Nx)      # Stress field
    RHS = zeros(Float64, 3, Nx)   # ∂t Fields[i, :] = RHS[i, :]

    # Initialize derivatives
    ∂xFieldsv = zeros(Float64, 3, Nx)
    ∂xx       = zeros(Float64, 3, Nx)

    # Initialize vectors where fields are saved for plots
    SavedFields = zeros(3, Nx, plotframes)
    Savedv      = zeros(Nx, plotframes)

    # Initialize time evolution parameters
    ThisFrame    = 1             # Used to save system state in SavedFields and Savedv matrices, ∈ [1, plotframes]
    LastSaveTime = 0             # Last time system state was saved, in non-dim. units
    CurrentTime  = 0             # Real time at current time frame
    ΔtVec        = [0.01, 0.0]   # ΔtVec[1(2)] is Δt_old (Δt)

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
        ΔtVec[2] = 2*ΔtVec[1]       # Double latest time step not to end up with very small Δt
        # Propagate fields
        AdaptiveTimeStep!(ErrorTolerance, ΔtVec, Pars, v, Fields, Fields_Aux, v_LongStep, Fields_LongStep, v_ShortStepAux, Fields_ShortStepAux, v_ShortStep, Fields_ShortStep, ∂xx, ∂xFieldsv, σ, RHS, k, P, Pinv, Factor1stDer, Factor2ndDer)
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
        # Export saved data if either FinalTime reached, or if Δt drops below MinTimeStep
        if (CurrentTime >= FinalTime || ΔtVec[1] <= MinTimeStep)
            # Print message if Δt drops below tolerance
            if ΔtVec[1] <= MinTimeStep
                println("Simulation stops at frame $ThisFrame/$plotframes. Time step dropped below $MinTimeStep")
            end
            save(DataFileName, "actomyosin", SavedFields[1, :, 1:ThisFrame], "active_nucleator", SavedFields[2, :, 1:ThisFrame], "inactive_nucleator", SavedFields[3, :, 1:ThisFrame], "velocity", Savedv[:, 1:ThisFrame])
            println("Data saved, simulation is over.")
        end
    end

    return nothing
end

main(
    Nx,
    Δx,
    FrameTimeStep,
    plotframes,
    FinalTime,
    ErrorTolerance,
    FieldICs,
    MinTimeStep,
    DataFileName
)