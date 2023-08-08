function CalculateRHS!(Pars, v, Fields, ∂xx, ∂xFieldsv, σ, RHS, k, P, Pinv, Factor1stDer, Factor2ndDer)
    # This funct. calculates RHS of dynamical equations, written as ∂t (density) = RHS
    # Calculate 2nd derivatives w/ FFTs
    for i in 1:3
        ∂xx[i,:] .= Pinv * ((P * Fields[i,:]) .* Factor2ndDer)
    end
    # Calculate velocity
    σ[:] .= @. Pars["Z"] * (Fields[1,:] ^ 2) - Pars["B"] * (Fields[1,:] ^ 3)   # Calculate stress
    v[:] .= Pinv * ( (P * σ) .* (im .* k) ./ (1 .+ k.*k) )                     # Calculate velocity w/ FFTs
    # Calculate advective currents w/ FFTs
    for i in 1:3
        ∂xFieldsv[i,:] = Pinv * ((P * (Fields[i,:] .* v)) .* Factor1stDer)
    end
    # Calculate right hand sides
    RHS[1,:] .= @. - ∂xFieldsv[1,:] + Pars["Dc"] * ∂xx[1,:] - Pars["Kd"] * Fields[1,:] + Pars["A"] * Fields[2,:]
    RHS[2,:] .= @. - ∂xFieldsv[2,:] + Pars["Da"] * ∂xx[2,:] - (Pars["Ωd0"] + Pars["Ωd"] * Fields[1,:]) * Fields[2,:] + Pars["Ω0"] * Fields[3,:] * (1 + Pars["Ω"] * Fields[2,:]^2)
    RHS[3,:] .= @. - ∂xFieldsv[3,:] + ∂xx[3,:]              + (Pars["Ωd0"] + Pars["Ωd"] * Fields[1,:]) * Fields[2,:] - Pars["Ω0"] * Fields[3,:] * (1 + Pars["Ω"] * Fields[2,:]^2)
end

function EulerForward!(Δt, Pars, v, Fields, ∂xx, ∂xFieldsv, σ, RHS, k, P, Pinv, Factor1stDer, Factor2ndDer)
    # This funct. propagates dynamical fields in time using Euler forward
    # Calculate right hand sides
    CalculateRHS!(Pars, v, Fields, ∂xx, ∂xFieldsv, σ, RHS, k, P, Pinv, Factor1stDer, Factor2ndDer)
    # Update fields with their values at t + Δt
    Fields[:,:] .= @. Fields[:,:] + Δt * RHS[:,:]
    return nothing
end

function MidpointMethod!(Δt, Pars, v, Fields, Fields_Aux, ∂xx, ∂xFieldsv, σ, RHS, k, P, Pinv, Factor1stDer, Factor2ndDer)
    # This funct. propagates dynamical fields in time using the midpoint method
    # Save fields at step t
    Fields_Aux[:,:] .= Fields[:,:]
    # Update fields with their values at midpoint (i.e., at t + Δt/2)
    EulerForward!(0.5*Δt, Pars, v, Fields, ∂xx, ∂xFieldsv, σ, RHS, k, P, Pinv, Factor1stDer, Factor2ndDer)
    # Calculate RHS at midpoint by using updated fields
    CalculateRHS!(Pars, v, Fields, ∂xx, ∂xFieldsv, σ, RHS, k, P, Pinv, Factor1stDer, Factor2ndDer)
    # Update fields with their values at t + Δt
    Fields[:,:] .= @. Fields_Aux[:,:] + Δt * RHS[:,:]
    return nothing
end

function AdaptiveTimeStep!(ErrorTolerance, ΔtVec, Pars, v, Fields, Fields_Aux, v_LongStep, Fields_LongStep, v_ShortStepAux, Fields_ShortStepAux, v_ShortStep, Fields_ShortStep, ∂xx, ∂xFieldsv, σ, RHS, k, P, Pinv, Factor1stDer, Factor2ndDer)
    # First run
    # Propagate fields once by Δt
    Fields_LongStep[:,:] .= Fields[:,:]
    MidpointMethod!(ΔtVec[2], Pars, v_LongStep, Fields_LongStep, Fields_Aux, ∂xx, ∂xFieldsv, σ, RHS, k, P, Pinv, Factor1stDer, Factor2ndDer)
    # Propagate fields twice by 0.5*Δt, first integration
    Fields_ShortStep[:,:] .= Fields[:,:]
    MidpointMethod!(0.5*ΔtVec[2], Pars, v_ShortStep, Fields_ShortStep, Fields_Aux, ∂xx, ∂xFieldsv, σ, RHS, k, P, Pinv, Factor1stDer, Factor2ndDer)
    # Save current half step fields for next run
    v_ShortStepAux[:]  .= v_ShortStep
    Fields_ShortStepAux[:,:] .= Fields_ShortStep[:,:]
    # Propagate fields twice by 0.5*Δt, second integration
    MidpointMethod!(0.5*ΔtVec[2], Pars, v_ShortStep, Fields_ShortStep, Fields_Aux, ∂xx, ∂xFieldsv, σ, RHS, k, P, Pinv, Factor1stDer, Factor2ndDer)
    # Calculate maximum relative error
    MaxError = findmax(abs.((Fields_LongStep .- Fields_ShortStep) ./ Fields_LongStep))[1]
    # Proceed by halving Δt until MaxError < ErrorTolerance
    while MaxError >= ErrorTolerance
        # Halven time step
        ΔtVec[2] = 0.5*ΔtVec[2]
        # LongStep is one half step in previous run (i.e. "Aux" fields)
        v_LongStep[:]  .= v_ShortStepAux
        Fields_LongStep[:,:] .= Fields_ShortStepAux[:,:]
        # Propagate fields twice by 0.5*Δt, first integration
        Fields_ShortStep[:,:] .= Fields[:,:]
        MidpointMethod!(0.5*ΔtVec[2], Pars, v_ShortStep, Fields_ShortStep, Fields_Aux, ∂xx, ∂xFieldsv, σ, RHS, k, P, Pinv, Factor1stDer, Factor2ndDer)
        # Save current half step fields for next run
        v_ShortStepAux[:]  .= v_ShortStep
        Fields_ShortStepAux[:,:] .= Fields_ShortStep[:,:]
        # Propagate fields twice by 0.5*Δt, second integration
        MidpointMethod!(0.5*ΔtVec[2], Pars, v_ShortStep, Fields_ShortStep, Fields_Aux, ∂xx, ∂xFieldsv, σ, RHS, k, P, Pinv, Factor1stDer, Factor2ndDer)
        # Calculate maximum relative error
        MaxError = findmax(abs.(Fields_LongStep .- Fields_ShortStep) ./ Fields_LongStep)[1]
    end
    # Update Δt_old
    ΔtVec[1] = ΔtVec[2]
    # Update fields
    Fields[:,:] .= Fields_LongStep[:,:]
    v[:]  .= v_LongStep
    return nothing
end
