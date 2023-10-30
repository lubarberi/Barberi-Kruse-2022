# All quantities are expressed in non-dimensional units
# Initialize a square spatial grid [-L/2 ; L/2] x [-L/2 ; L/2]
L  = 10*π                                            # System size
N  = 512                                             # Number of grid node
Nx = N; Ny = N
Δx = L / N                                           # Grid spacing
Δy = L / N
x  = CuArray{Float64}([-L/2 + i*Δx for i = 0:N-1])   # Space vector
y  = CuArray{Float64}([-L/2 + i*Δy for i = 0:N-1])   # Space vector
# Initialize reciprocal space grid
kx = CuArray{Float64}(2*pi*rfftfreq(Nx, 1/Δx)); Lkx = length(kx)
ky = CuArray{Float64}(2*pi*fftfreq(Ny, 1/Δy))
kx2 = kx.*kx
ky2 = ky.*ky
# FFT tools to calculate derivatives
FT  = plan_rfft(CUDA.ones(Float64, Nx, Ny))    # Operator that Fourier transforms a matrix
IFT = inv(FT)                                  # like FT, but Inverse Fourier transforms

# GPU parameters -- real space mapping
WrapsT = 16                                   # N. of wraps per block
Bx = ceil(Int, N/WrapsT)                      # N. of blocks along x
By = ceil(Int, N/WrapsT)                      # N. of blocks along y
Nx = WrapsT * Bx
Ny = WrapsT * Bx
block_dim = (WrapsT, WrapsT)                  # Dimension of each block
grid_dim = (Bx, By)                           # Dimension of grid
# GPU parameter -- reciprocal space mapping
# Note that we first do rfft along x, then fft along y, hence the asymmetric grid
# The +1 is for the k=0 mode
gridFFT_dim = (div(Bx,2)+1, By)

# Integration time step
Δt = 1e-4

# Model parameters
Dc  = 1e-2
Da  = 1e-1
Di  = 1
Dap = Da + Di
Dam = Da - Di
A   = 1
Kd  = 1
Ω0  = 0.5
Ωd  = 10
Ωd0 = 10
Ω   = 10
Z   = 10
B   = Z

# Final time reached by simulation
FinalTime = 1e4

#= # Adaptive timestep parameters
MinTimeStep = 1e-15         # Simulation stops if time step drops below MinTimeStep
ErrorTolerance = 1e-10      # Error tolerance between Δt and 0.5*Δt steps =#

# Saving parameters
# Output file parameters
Nt = Int(floor(FinalTime / Δt))
PrintEvery = Int(floor(Nt / 100))                 # Save state of system with time intervals = FrameTimeStep (in non-dim. units)
TotPrints = floor(Int64, FinalTime / PrintEvery)  # Total number of system states saved during the simulation
