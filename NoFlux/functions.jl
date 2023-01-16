function CalculateRHS!(Pars, Nx, Fields, ∂xFields, ∂xΠ, v, A, ∂xFieldsv, ∂xxFields, RHS, FDC1, FDC2)
    # This funct. calculates RHS of dynamical equations, written as ∂t (density) = RHS
    @inbounds begin
        # Calculate field derivatives
        for i in 1:3
            ∂xFields[i,:] = FDC1 * Fields[i,:]
            ∂xxFields[i,:] = FDC2 * Fields[i,:]
        end
        # Calculate velocity
        # No-flux boundary condition is v[1] = v[Nx] = 0.
        # v[1] and v[Nx] are never changed from their starting, null value
        ∂xΠ[:] .= @. (2 * Pars["Z"] * Fields[1,2:Nx-1] - 3 * Pars["B"] * Fields[1,2:Nx-1] ^ 2) * ∂xFields[1,2:Nx-1]
        v[2:Nx-1] = A \ ∂xΠ
        # Calculate advective terms, like ∂x(cv), using upwind scheme
        for i in 1:3
            ∂xFieldsv[i,1] = v[2] < 0 ? - Fields[i,2] * v[2] : 0
            ∂xFieldsv[i,Nx] = v[Nx-1] > 0 ? Fields[i,Nx-1] * v[Nx-1] : 0
        end
        for i in 1:3, j in 2:Nx-1
            ∂xFieldsv[i,j] = -abs(v[j]) * Fields[i,j]
            ∂xFieldsv[i,j] += v[j-1] > 0 ? Fields[i,j-1] * v[j-1] : 0
            ∂xFieldsv[i,j] += v[j+1] < 0 ? - Fields[i,j+1] * v[j+1] : 0
        end
        ∂xFieldsv[:,:] .= ∂xFieldsv[:,:] ./ Δx
        # Calculate right hand sides of dynamical eqns
        RHS[1,2:Nx-1] .= @. ∂xFieldsv[1,2:Nx-1] + Pars["Dc"] * ∂xxFields[1,2:Nx-1] - Pars["Kd"] * Fields[1,2:Nx-1] + Pars["A"] * Fields[2,2:Nx-1]
        RHS[2,2:Nx-1] .= @. ∂xFieldsv[2,2:Nx-1] + Pars["Da"] * ∂xxFields[2,2:Nx-1] - Pars["Ωd"] * Fields[1,2:Nx-1] * Fields[2,2:Nx-1] + Pars["Ω0"] * Fields[3,2:Nx-1] * (1 + Pars["Ω"] * Fields[2,2:Nx-1]^2)
        RHS[3,2:Nx-1] .= @. ∂xFieldsv[3,2:Nx-1] +              ∂xxFields[3,2:Nx-1] + Pars["Ωd"] * Fields[1,2:Nx-1] * Fields[2,2:Nx-1] - Pars["Ω0"] * Fields[3,2:Nx-1] * (1 + Pars["Ω"] * Fields[2,2:Nx-1]^2)
        # No-flux boundary conditions imply ∂xFields[:,1] = ∂xFields[:,Nx] = 0.
        # This is enforced as Fields[:,1] = Fields[:,2] and Fields[:,Nx] = Fields[:,Nx-1],
        # through the choice of RHS below 
        RHS[:,1] = RHS[:,2];
        RHS[:,Nx] = RHS[:,Nx-1];
    end
end

function EulerForward!(Δt, Nx, Pars, Fields, ∂xFields, ∂xΠ, v, A, ∂xFieldsv, ∂xxFields, RHS, FDC1, FDC2)
    # This funct. propagates dynamical fields in time using Euler forward
    # Calculate right hand sides
    CalculateRHS!(Pars, Nx, Fields, ∂xFields, ∂xΠ, v, A, ∂xFieldsv, ∂xxFields, RHS, FDC1, FDC2)
    # Update fields with their values at t + Δt
    Fields[:,:] .= @. Fields[:,:] + Δt * RHS[:,:]
    return nothing
end

function MidpointMethod!(Δt, Nx, Pars, Fields, Fields_Aux, ∂xFields, ∂xΠ, v, A, ∂xFieldsv, ∂xxFields, RHS, FDC1, FDC2)
    # This funct. propagates dynamical fields in time using the midpoint method
    # Save fields at step t
    Fields_Aux[:,:] .= Fields[:,:]
    # Update fields with their values at midpoint (i.e., at t + Δt/2)
    EulerForward!(0.5*Δt, Nx, Pars, Fields, ∂xFields, ∂xΠ, v, A, ∂xFieldsv, ∂xxFields, RHS, FDC1, FDC2)
    # Calculate RHS at midpoint by using updated fields
    CalculateRHS!(Pars, Nx, Fields, ∂xFields, ∂xΠ, v, A, ∂xFieldsv, ∂xxFields, RHS, FDC1, FDC2)
    # Update fields with their values at t + Δt
    Fields[:,:] .= @. Fields_Aux[:,:] + Δt * RHS[:,:]
    return nothing
end

function AdaptiveTimeStep!(ErrorTolerance, ΔtVec, Nx, Pars, v, Fields, Fields_Aux, ∂xFields, v_LongStep, Fields_LongStep, v_ShortStepAux, Fields_ShortStepAux, v_ShortStep, Fields_ShortStep, ∂xΠ, A, ∂xFieldsv, ∂xxFields, RHS, FDC1, FDC2, MaxError)
    # First run
    # Propagate fields once by Δt
    Fields_LongStep[:,:] .= Fields[:,:]
    MidpointMethod!(ΔtVec[2], Nx, Pars, Fields_LongStep, Fields_Aux, ∂xFields, ∂xΠ, v_LongStep, A, ∂xFieldsv, ∂xxFields, RHS, FDC1, FDC2)
    # Propagate fields twice by 0.5*Δt, first integration
    Fields_ShortStep[:,:] .= Fields[:,:]
    MidpointMethod!(0.5*ΔtVec[2], Nx, Pars, Fields_ShortStep, Fields_Aux, ∂xFields, ∂xΠ, v_ShortStep, A, ∂xFieldsv, ∂xxFields, RHS, FDC1, FDC2)
    # Save current half step fields for next run
    v_ShortStepAux[:]  .= v_ShortStep[:]
    Fields_ShortStepAux[:,:] .= Fields_ShortStep[:,:]
    # Propagate fields twice by 0.5*Δt, second integration
    MidpointMethod!(0.5*ΔtVec[2], Nx, Pars, Fields_ShortStep, Fields_Aux, ∂xFields, ∂xΠ, v_ShortStep, A, ∂xFieldsv, ∂xxFields, RHS, FDC1, FDC2)
    # Calculate maximum relative error
    MaxError = findmax(abs.((Fields_LongStep .- Fields_ShortStep) ./ Fields_LongStep))[1]
    # Proceed by halving Δt until MaxError < ErrorTolerance
    while MaxError >= ErrorTolerance
        # Halven time step
        ΔtVec[2] = 0.5*ΔtVec[2]
        # LongStep is one half step in previous run (i.e. "Aux" fields)
        v_LongStep[:]  .= v_ShortStepAux[:]
        Fields_LongStep[:,:] .= Fields_ShortStepAux[:,:]
        # Propagate fields twice by 0.5*Δt, first integration
        Fields_ShortStep[:,:] .= Fields[:,:]
        MidpointMethod!(0.5*ΔtVec[2], Nx, Pars, Fields_ShortStep, Fields_Aux, ∂xFields, ∂xΠ, v_ShortStep, A, ∂xFieldsv, ∂xxFields, RHS, FDC1, FDC2)
        # Save current half step fields for next run
        v_ShortStepAux[:]  .= v_ShortStep
        Fields_ShortStepAux[:,:] .= Fields_ShortStep[:,:]
        # Propagate fields twice by 0.5*Δt, second integration
        MidpointMethod!(0.5*ΔtVec[2], Nx, Pars, Fields_ShortStep, Fields_Aux, ∂xFields, ∂xΠ, v_ShortStep, A, ∂xFieldsv, ∂xxFields, RHS, FDC1, FDC2)
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
