import Pkg
Pkg.activate(dirname(Base.current_project()))

const GROUP = get(ENV, "GROUP", "All")
if !parse(Bool, get(ENV, "CI", "false"))
    using Revise
    const printDebug = true
else
    const printDebug = false
end
using Test
using JLD2
using Plots
using Dates
using Infiltrator
using OrdinaryDiffEq
using CairoMakie
using Random
using JET
using ForwardDiff
using MLStyle
using Huginn
using Huginn: Parameters, Model
using Sleipnir: DummyClimate2D, ScalarCacheNoVJP, MatrixCacheNoVJP
using Aqua

include("utils_test.jl")
include("params_construction.jl")
include("halfar.jl")
include("PDE_solve.jl")
include("mass_conservation.jl")
include("laws.jl")
include("plotting.jl")
include("Aqua.jl")

# Activate to avoid GKS backend Plot issues in the JupyterHub
ENV["GKSwstype"]="nul"

@info "Running group $(GROUP)"

@testset "Running all tests" begin
    if GROUP == "All" || GROUP == "Core1"
        @testset "PDE solving integration tests" begin
            @testset "w/o MB w/o laws" pde_solve_test(;
                rtol = 0.01, atol = 0.01, save_refs = false, MB = false, fast = true)
            @testset "w/  MB w/o laws" pde_solve_test(;
                rtol = 0.01, atol = 0.01, save_refs = false, MB = true, fast = true)
            @testset "w/  MB w/  A scalar laws" pde_solve_test(;
                rtol = 0.01, atol = 0.01, save_refs = false, MB = true,
                fast = true, laws_A = :scalar, callback_laws = false)
            @testset "w/  MB w/  A scalar callback laws" pde_solve_test(;
                rtol = 0.01, atol = 0.01, save_refs = false, MB = true,
                fast = true, laws_A = :scalar, callback_laws = true)
            @testset "w/  MB w/  A matrix  laws" pde_solve_test(;
                rtol = 0.01, atol = 0.01, save_refs = false, MB = true,
                fast = true, laws_A = :matrix, callback_laws = false)
            @testset "w/  MB w/  A matrix callback laws" pde_solve_test(;
                rtol = 0.01, atol = 0.01, save_refs = false, MB = true,
                fast = true, laws_A = :matrix, callback_laws = true)
            @testset "w/  MB w/  C scalar laws" pde_solve_test(;
                rtol = 0.01, atol = 0.01, save_refs = false, MB = true,
                fast = true, laws_C = :scalar, callback_laws = false)
            @testset "w/  MB w/  C scallar callback laws" pde_solve_test(;
                rtol = 0.01, atol = 0.01, save_refs = false, MB = true,
                fast = true, laws_C = :scalar, callback_laws = true)
        end
    end

    if GROUP == "All" || GROUP == "Core2"
        @testset "Ground truth generation" ground_truth_generation()
    end

    if GROUP == "All" || GROUP == "Core3"
        @testset "Run TI models in-place" TI_run_test!(false; rtol = 1e-5, atol = 1e-5)
    end

    if GROUP == "All" || GROUP == "Core4"
        @testset "Solver parameters construction" begin
            @testset "With specified variables" params_constructor_specified(false)
            @testset "With default variables" params_constructor_default(false)
        end
    end

    if GROUP == "All" || GROUP == "Core5"
        @testset "Analytical Halfar solution is correct" unit_halfar_is_solution()
        @testset "Halfar Solutions" halfar_test()
    end

    if GROUP == "All" || GROUP == "Core6"
        @testset "Mass Conservation" begin
            @testset "Flat Bed" unit_mass_flatbed_test(; rtol = 1.0e-7)
            @testset "Non Flat Bed" unit_mass_nonflatbed_test(; rtol = 1.0e-7)
        end
    end

    if GROUP == "All" || GROUP == "Core7"
        @testset "Laws" begin
            @testset "Constructors" begin
                laws_constructor_default()
                laws_constructor_specified()
            end
            @testset "Law Inputs" begin
                test_iTopoRough()
                test_iCPDD()
            end
            @testset "Laws" begin
                test_SyntheticC()
            end
        end
    end

    if GROUP == "All" || GROUP == "Core8"
        @testset "Glacier Plotting" plot_analysis_flow_parameters_test()
    end

    if GROUP == "All" || GROUP == "Aqua"
        @testset "Aqua" test_Aqua()
    end
end
