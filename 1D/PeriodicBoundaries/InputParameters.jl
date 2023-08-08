# All quantities are expressed in non-dimensional units
# Space grid
L = 10*π                    # System size
Nx   = 512                  # Number of grid nodes
Δx   = L / (Nx-1)           # Grid spacing
x = [-L/2 + i*Δx for i = 0:Nx-1]   # Space vector

# Model parameters (saved in a dictionary)
Pars = Dict{String, Float64}()
Pars["Dc"]  = 1e-2
Pars["Da"]  = 1e-1
Pars["A"]   = 1
Pars["Kd"]  = 1
Pars["Ω0"]  = 0.5
Pars["Ω"]   = 12
Pars["Ωd"]  = 10
Pars["Ωd0"] = 0
Pars["Z"]   = 9.3
Pars["B"]   = Pars["Z"]

# Homogeneous Steady State (used in initial conditions below)
# Solving EqnHSS = 0 for Na gives Na at HSS
EqnHSS(Na) = - (Pars["Ωd0"] + Pars["Ωd"] * Pars["A"] * Na / Pars["Kd"]) * Na + Pars["Ω0"] * (1 - Na) * (1 + Pars["Ω"] * Na^2)
NaHSS = find_zero(EqnHSS, (0, 1))
# Ni and C at HSS are readily calculated from NaHSS
NiHSS = 1 - NaHSS
CHSS = (Pars["A"] / Pars["Kd"]) * NaHSS

# Initial Conditions (either HSS + weak noise or HSS + localized bump)
# Parameters of noisy IC
seedd = 123                                     # Seed of rand. number generator
Random.seed!(seedd)                             # Seeds random number generator
ε   = 0.01                                      # Noise amplitude, small number
Noise = ε .* (-1 .+ 2 .* rand(Float64, Nx))     # Nx-long vector of weak noise, made of random numbers between ε*[-1, 1]
ZeroMeanNoise = Noise .- integrate(x, Noise)/L  # Noisy vector with zero average, s.t. HSS + Noise conserves tot. number of molecules
# Parameters of localized square bump IC
Bump = zeros(Nx)                                              # Initialize bump as a Nx-long vector of zeros
Bump_center = 0                                               # Bump will be centered at X = 0
for i in 1:Nx                                                 # Bump has width = 10 and height 1 
    if (x[i] >= Bump_center - 5 && x[i] <= Bump_center + 5)   #
        Bump[i] = 1                                           #
    end                                                       #
end                                                           #
ZeroMeanBump = Bump .- integrate(x, Bump)/L                   # Remove average from bump, same reason as for noisy IC above
# Initialize IC
FieldICs      = zeros(Float64, 3, Nx) # Matrix of initial conditions (see Fields in main.jl)
# Uncomment if desired IC is HSS + localized square bump
FieldICs[1,:] .= CHSS  .* (1 .+ ZeroMeanBump)
FieldICs[2,:] .= NaHSS .* (1 .+ ZeroMeanBump)
FieldICs[3,:] .= NiHSS .* (1 .+ ZeroMeanBump)
# Uncomment if desired IC is HSS + weak noise
#FieldICs[1,:] .= CHSS  .* (1 .+ ZeroMeanNoise)
#FieldICs[2,:] .= NaHSS .* (1 .+ ZeroMeanNoise)
#FieldICs[3,:] .= NiHSS .* (1 .+ ZeroMeanNoise)

# Final time reached by simulation
FinalTime = 10

# Adaptive timestep parameters
MinTimeStep = 1e-15         # Simulation stops if time step drops below MinTimeStep
ErrorTolerance = 1e-10      # Error tolerance between Δt and 0.5*Δt steps

# Saving parameters
# Output file parameters
FrameTimeStep = 0.1                                  # Save state of system with time intervals = FrameTimeStep (in non-dim. units)
plotframes = floor(Int64, FinalTime / FrameTimeStep) # Total number of system states saved during the simulation
dt = Dates.format(now(), "yyyymmdd")                 # String of today's date
DataFileName = "Data/"*dt*"-Data.jld"                # Name of output file
