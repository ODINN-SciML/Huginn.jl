# Regenerates the mass balance dependent PDE references.
#
# Run it with the test environment active, so that JET and the other test dependencies
# resolve:
#
#     julia --project=<test env> test/regen_PDE_refs.jl
#
# It rewrites test/data/PDE/PDE_refs_MB.jld2, PDE_refs_MB_lawA.jld2 and PDE_refs_MB_lawC.jld2.
# PDE_refs_noMB.jld2 carries no mass balance and is left alone. MB_ref.jld2 and
# H_w_MB_ref.jld2 come from MB_timestep! and apply_MB_mask! called directly, with no solver
# and no scheme, so they are untouched as well.
#
# The law cases share a file: four of them write PDE_refs_MB_lawA and two write
# PDE_refs_MB_lawC. They run here in the same order as the suite, each checking against what
# it just wrote, so the real check is the suite afterwards: it holds every case against the
# file the last of them left behind.

using Test
using Dates
using JLD2
using JET
using OrdinaryDiffEq
using ForwardDiff
using MLStyle
using Huginn
using Huginn: Parameters, Model
using Sleipnir: DummyClimate2D, ScalarCacheNoVJP, MatrixCacheNoVJP

include("utils_test.jl")
include("PDE_solve.jl")

# Mirrors the mass balance cases of the "PDE solving integration tests" testset
const MB_CASES = (
    (;),
    (; laws_A = :scalar, callback_laws = false),
    (; laws_A = :scalar, callback_laws = true),
    (; laws_A = :matrix, callback_laws = false),
    (; laws_A = :matrix, callback_laws = true),
    (; laws_C = :scalar, callback_laws = false),
    (; laws_C = :scalar, callback_laws = true)
)

for case in MB_CASES
    pde_solve_test(; rtol = 0.01, atol = 0.01, save_refs = true, MB = true, case...)
end

println("Regenerated the mass balance PDE references")
