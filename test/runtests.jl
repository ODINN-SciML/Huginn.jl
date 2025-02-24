import Pkg
Pkg.activate(dirname(Base.current_project()))

using Revise
using Test
using JLD2
using Plots
using Infiltrator
using OrdinaryDiffEq
using CairoMakie
using Huginn

include("utils_test.jl")
include("params_construction.jl")
include("halfar.jl")
include("PDE_UDE_solve.jl")
include("mass_conservation.jl")
include("plotting.jl")

# Activate to avoid GKS backend Plot issues in the JupyterHub
ENV["GKSwstype"]="nul"

@testset "PDE solving integration tests w/o MB" pde_solve_test(; rtol=0.01, atol=0.01, save_refs=false, MB=false, fast=true)

@testset "PDE solving integration tests w/ MB" pde_solve_test(; rtol=0.01, atol=0.01, save_refs=false, MB=true, fast=true)

@testset "Run TI models in place" TI_run_test!(false; rtol=1e-5, atol=1e-5)

@testset "Solver parameters construction with specified variables" params_constructor_specified()

@testset "Solver parameters construction with default variables" params_constructor_default()

@testset "Halfar Solution (in-place)" halfar_test(; rtol=0.02, atol=1.0, inplace=true)

@testset "Halfar Solution (out-of-place)" halfar_test(; rtol=0.02, atol=1.0, inplace=false)

@testset "Conservation of Mass - Flat Bed" unit_mass_flatbed_test(; rtol=1.0e-7)

@testset "Conservation of Mass - Non Flat Bed" unit_mass_nonflatbed_test(; rtol=1.0e-7)

@testset "Glacier Plotting" plot_analysis_flow_parameters_test()

@testset "Video plot test" make_thickness_video_test()
