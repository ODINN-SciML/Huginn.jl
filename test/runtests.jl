import Pkg
Pkg.activate(dirname(Base.current_project()))

using Revise
using Test
using JLD2
using Plots
using Infiltrator
using OrdinaryDiffEq

using Huginn

include("utils_test.jl")
include("params_construction.jl")
include("halfar.jl")
include("PDE_UDE_solve.jl")
include("mass_conservation.jl")

# Activate to avoid GKS backend Plot issues in the JupyterHub
ENV["GKSwstype"]="nul"

@testset "Solver parameters construction with specified variables" params_constructor_specified()

@testset "Solver parameters construction with default variables" params_constructor_default()

@testset "Halfar Solution" halfar_test(; rtol=0.02, atol=1.0)

@testset "PDE solving integration tests" pde_solve_test(; rtol=0.01, atol=0.01, save_refs=false, MB=false, fast=true)

@testset "Conservation of Mass - Flat Bed" unit_mass_flatbed_test(; rtol=1.0e-7)

