[![Build Status](https://github.com/ODINN-SciML/Huginn.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ODINN-SciML/Huginn.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/ODINN-SciML/Huginn.jl/branch/main/graph/badge.svg)](https://app.codecov.io/gh/ODINN-SciML/Huginn.jl)
[![CompatHelper](https://github.com/ODINN-SciML/Huginn.jl/actions/workflows/CompatHelper.yml/badge.svg)](https://github.com/ODINN-SciML/Huginn.jl/actions/workflows/CompatHelper.yml)

<img src="https://github.com/JordiBolibar/Huginn.jl/blob/main/data/Huginn_logo-20.png" width="250">

Huginn.jl is a package containing all the glacier ice flow models and solvers for [ODINN.jl](https://github.com/ODINN-SciML/ODINN.jl). For now, we have implemented a 2D Shallow Ice Approximation (SIA). The package architecture makes it pretty straighforward to add other ice flow models. It also provides an accessible API to be easily used by other glacier models, such as the Open Global Glacier Model ([OGGM](https://github.com/OGGM/oggm)). This enables OGGM to access the high-performance glacier ice flow solvers in Julia in a seamless manner, via JuliaCall and PyCall.jl.  
