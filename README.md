# Barberi-Kruse-2022
This repo provides the code used in the article "Localized states in active fluids", Barberi &amp; Kruse, arXiv:2209.02581.

- stabilitydiagram.jl >> numerical calculation of the Turing- and Hopf- instability lines of the homogeneous steady state, illustrated in the Supplementary Material.

In the folder PeriodicBoundaries:
- main.jl (depends on InputParameters.jl and functions.jl) >> numerical integration of the dynamical equations with periodic boundary conditions. To change simulation parameters, please edit InputParameters.jl;

In the folder NoFlux:
- main.jl (depends on InputParameters.jl and functions.jl) >> numerical integration of the dynamical equations with no-flux boundary conditions. To change simulation parameters, please edit InputParameters.jl;
