# Barberi-Kruse-2022
This repo provides the code used in the article [_"Localized states in active fluids"_](https://arxiv.org/abs/2209.02581), Barberi &amp; Kruse, arXiv:2209.02581.

Folders content: 

- 1D/stabilitydiagram.jl >> numerical calculation of the instability boundary of the homogeneous steady state. The result is featured in the stability diagram of the Supplementary Material.

- 1D/PeriodicBoundaries/main.jl (depends on InputParameters.jl and functions.jl) >> numerical integration of the dynamical equations with periodic boundary conditions. To change simulation parameters, please edit InputParameters.jl;

- 1D/NoFlux/main.jl (depends on InputParameters.jl and functions.jl) >> numerical integration of the dynamical equations with no-flux boundary conditions. To change simulation parameters, please edit InputParameters.jl;

- 2D/main.jl (depends on InputParameters.jl and kernels.jl) >> numerical integration of the dynamical equations in 2D, with periodic boundary conditions. To change simulation parameters, please edit InputParameters.jl

Please note that the 1D code runs on CPU, while the 2D code runs on GPU.