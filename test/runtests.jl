import Pkg
Pkg.activate(dirname(Base.current_project()))

if !parse(Bool, get(ENV, "CI", "false"))
    using Revise
    const printDebug = true
else
    const printDebug = false
end
using Test
using JLD2
using Plots
using Infiltrator
using OrdinaryDiffEq
using CairoMakie
using Random
using JET
using ForwardDiff
using Huginn

include("utils_test.jl")
include("params_construction.jl")
include("halfar.jl")
include("PDE_solve.jl")
include("mass_conservation.jl")
include("plotting.jl")

# Activate to avoid GKS backend Plot issues in the JupyterHub
ENV["GKSwstype"]="nul"

@testset "Running all tests" begin

@testset "PDE solving integration tests" begin
    @testset "w/o MB w/o laws" pde_solve_test(; rtol=0.01, atol=0.01, save_refs=false, MB=false, fast=true, laws=nothing)

    @testset "w/  MB w/o laws" pde_solve_test(; rtol=0.01, atol=0.01, save_refs=false, MB=true, fast=true, laws=nothing)

    @testset "w/  MB w/  scalar laws" pde_solve_test(; rtol=0.01, atol=0.01, save_refs=false, MB=true, fast=true, laws=:scalar, callback_laws=false)

    @testset "w/  MB w/  scalar callback laws" pde_solve_test(; rtol=0.01, atol=0.01, save_refs=false, MB=true, fast=true, laws=:scalar, callback_laws=true)

    @testset "w/  MB w/  matrix  laws" pde_solve_test(; rtol=0.01, atol=0.01, save_refs=false, MB=true, fast=true, laws=:matrix, callback_laws=false)

    @testset "w/  MB w/  matrix callback laws" pde_solve_test(; rtol=0.01, atol=0.01, save_refs=false, MB=true, fast=true, laws=:matrix, callback_laws=true)
end

@testset "Run TI models in-place" TI_run_test!(false; rtol=1e-5, atol=1e-5)

@testset "Solver parameters construction with specified variables" params_constructor_specified()

@testset "Solver parameters construction with default variables" params_constructor_default()

@testset "Analytical Halfar solution is correct" unit_halfar_is_solution()

# @testset "Halfar Solutions" halfar_test()

@testset "Conservation of Mass - Flat Bed" unit_mass_flatbed_test(; rtol=1.0e-7)

@testset "Conservation of Mass - Non Flat Bed" unit_mass_nonflatbed_test(; rtol=1.0e-7)

@testset "Glacier Plotting" plot_analysis_flow_parameters_test()

end
