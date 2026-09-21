[![Build Status](https://github.com/ODINN-SciML/Huginn.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ODINN-SciML/Huginn.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/ODINN-SciML/Huginn.jl/branch/main/graph/badge.svg)](https://app.codecov.io/gh/ODINN-SciML/Huginn.jl)
[![CompatHelper](https://github.com/ODINN-SciML/Huginn.jl/actions/workflows/CompatHelper.yml/badge.svg)](https://github.com/ODINN-SciML/Huginn.jl/actions/workflows/CompatHelper.yml)

<img src="https://github.com/JordiBolibar/Huginn.jl/blob/main/data/Huginn_logo-20.png" width="250">

## About Huginn.jl

Huginn.jl is a package containing all the glacier ice flow models and solvers for [ODINN.jl](https://github.com/ODINN-SciML/ODINN.jl). For now, we have implemented a 2D Shallow Ice Approximation (SIA). The package architecture makes it pretty straighforward to add other ice flow models. It also provides an accessible API to be easily used by other glacier models, such as the Open Global Glacier Model ([OGGM](https://github.com/OGGM/oggm)).

The ice flow PDE is discretized in space with the method of lines and integrated with [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl). The main entry point is the `Prediction` simulation: `run!(prediction)` solves the ice thickness evolution of every glacier, and stores thickness, surface elevation, velocity and mass balance time series in `Results` objects.

Huginn is part of the ODINN ecosystem, where each package has a narrow role:

  - [Gungnir](https://github.com/ODINN-SciML/Gungnir) (Python): preprocesses OGGM glacier and climate data.
  - [Sleipnir.jl](https://github.com/ODINN-SciML/Sleipnir.jl): core data structures (glaciers, climate, parameters, laws, results).
  - [Muninn.jl](https://github.com/ODINN-SciML/Muninn.jl): surface mass balance models.
  - **Huginn**: ice flow models and PDE solvers (this package).
  - [ODINN.jl](https://github.com/ODINN-SciML/ODINN.jl): differentiable pipeline for UDE training and inversions.

## Use Huginn directly or ODINN.jl?

Use Huginn on its own to run forward ice flow simulations without UDE training or inversion, or to validate a new ice flow model against an analytical solution (e.g. Halfar). Install [ODINN.jl](https://github.com/ODINN-SciML/ODINN.jl) if you need gradients through the ice flow solver, functional inversions or the full training pipeline. It wraps Huginn's forward solver and adds the adjoint infrastructure.

## Installing Huginn

> `Huginn.jl` requires Julia v1.11.

In order to install `Huginn` in a given environment, just do in the REPL:
```julia
julia> ] # enter Pkg mode
(@v1.11) pkg> activate MyEnvironment # or activate whatever path for the Julia environment
(MyEnvironment) pkg> add Huginn
```

Huginn re-exports Muninn and Sleipnir, so a single `using Huginn` gives access to the three packages. The preprocessed glacier data are downloaded automatically the first time Sleipnir is precompiled, see the [Sleipnir README](https://github.com/ODINN-SciML/Sleipnir.jl#data-preprocessing).

## How to use Huginn

The following example runs a forward simulation for one glacier between 2000 and 2020. The temperature-index mass balance model does not need any hand-set parameters: it is automatically calibrated per glacier against the geodetic mass balance of [Hugonnet et al. (2021)](https://doi.org/10.1038/s41586-021-03436-z) when the `Prediction` is built (`calibrate_MB = true` in `SimulationParameters`, the default).

```julia
using Huginn

# Multiprocessing is disabled for local runs. The Hugonnet et al. observation
# period (2000-2020) is used as the simulation tspan.
params = Parameters(
    simulation = SimulationParameters(
        tspan = (2000.0, 2020.0),
        multiprocessing = false,
        use_MB = true,
        rgi_paths = get_rgi_paths()
    )
)

# Initializing the glacier also loads its Hugonnet geodetic mass balance
glaciers = initialize_glaciers(["RGI60-11.03638"], params)

# Ice flow model (SIA2D) and temperature-index mass balance model
model = Model(
    iceflow = SIA2Dmodel(params),
    mass_balance = TImodel1(params)
)

# Building the Prediction calibrates the mass balance model against the observations
prediction = Prediction(model, glaciers, params)
run!(prediction)

# Time series of ice thickness for the first glacier
results = prediction.results[1]
@show size(results.H[end]) # final ice thickness grid
```

The results are also saved in `data/results/predictions` of the active project. For more examples, see the [forward simulation tutorial](https://odinn-sciml.github.io/ODINN.jl/dev/forward_simulation/) and the [SMB calibration tutorial](https://odinn-sciml.github.io/ODINN.jl/dev/smb_calibration/). Huginn's own page is [here](https://odinn-sciml.github.io/ODINN.jl/dev/Packages/huginn/), the full list of types and functions is in the [API reference](https://odinn-sciml.github.io/ODINN.jl/dev/API/api_huginn/), and the steps to add a new ice flow model (e.g. SSA) are in [Extending ODINN](https://odinn-sciml.github.io/ODINN.jl/dev/extending/).

## Contributing and community

Contributions are welcome. You can report bugs and request features in the [issues](https://github.com/ODINN-SciML/Huginn.jl/issues) tab, or open a pull request against `main` from a fork. See [How to contribute](https://odinn-sciml.github.io/ODINN.jl/dev/contribute/) and the [Code of conduct](https://odinn-sciml.github.io/ODINN.jl/dev/code_of_conduct/) for the guidelines shared across the ODINN ecosystem.

## How to cite

If you use Huginn, please cite the ODINN paper published in [Geoscientific Model Development](https://gmd.copernicus.org/articles/16/6671/2023/gmd-16-6671-2023.html):
```
@article{bolibar_sapienza_universal_2023,
	title = {Universal differential equations for glacier ice flow modelling},
	author = {Bolibar, J. and Sapienza, F. and Maussion, F. and Lguensat, R. and Wouters, B. and P\'erez, F.},
	journal = {Geoscientific Model Development},
	volume = {16},
	year = {2023},
	number = {22},
	pages = {6671--6687},
	url = {https://gmd.copernicus.org/articles/16/6671/2023/},
	doi = {10.5194/gmd-16-6671-2023}
}
```
