export SolverParameters

using Sleipnir: label, sep, field, val, hint, check

"""
A mutable struct that holds parameters for the solver.

    SolverParameters{F <: AbstractFloat, I <: Integer, ST <: OrdinaryDiffEqCore.OrdinaryDiffEqAdaptiveAlgorithm}

# Fields

  - `solver::ST`: The algorithm used for solving differential equations.
  - `reltol::F`: The relative tolerance for the solver.
  - `abstol::F`: The absolute tolerance for the solver, in metres of ice. Binds in practice: `reltol * H` sits far below `abstol` over most of a glacier, so `abstol` alone sets the error floor and the step size. Solver error accumulates roughly as `t^1.4`, so longer runs want a tighter value — see [`effective_abstol`](@ref).
  - `scale_abstol::Bool`: Whether `abstol` is tightened in proportion to run length, since solver error accumulates (see [`effective_abstol`](@ref)). Set `false` to use `abstol` exactly as given.
  - `adaptive::Bool`: Whether the solver chooses its own step size. Adaptive stepping makes the solution discontinuous in the parameters — an arbitrarily small change can flip which steps are accepted — which is harmless for a forward run but makes finite differences meaningless. Gradient checks want `adaptive = false`; production runs don't.
  - `dt::F`: Fixed step size used when `adaptive` is `false`, in years. Ignored otherwise.
  - `step::F`: The step size that controls at which frequency the results must be saved.
  - `tstops::Vector{F}`: Optional vector of time points where the solver should stop to store the results.
  - `save_everystep::Bool`: Flag indicating whether to save the solution at every step computed by the solver.
  - `progress::Bool`: Flag indicating whether to show progress during the solving process.
  - `progress_steps::I`: The number of steps between progress updates.
  - `maxiters::I`: Maximum number of iterations to perform in the iceflow solver.
"""
mutable struct SolverParameters{F <: AbstractFloat, I <: Integer,
    ST <: OrdinaryDiffEqCore.OrdinaryDiffEqAdaptiveAlgorithm} <: AbstractParameters
    solver::ST
    reltol::F
    abstol::F
    scale_abstol::Bool
    adaptive::Bool
    dt::F
    step::F
    tstops::Vector{F}
    save_everystep::Bool
    progress::Bool
    progress_steps::I
    maxiters::I
end

"""
Constructs a `SolverParameters` object with the specified parameters or using default values.

    SolverParameters(;
        solver::OrdinaryDiffEqCore.OrdinaryDiffEqAdaptiveAlgorithm = RDPK3Sp35(),
        reltol::F = 1e-12,
        abstol::F = 1e-3,
        scale_abstol::Bool = true,
        adaptive::Bool = true,
        dt::F = 1.0/120.0,
        step::F = 1.0/12.0,
        tstops::Vector{Sleipnir.Float} = Vector{Sleipnir.Float}(),
        save_everystep = false,
        progress::Bool = true,
        progress_steps::I = 10,
        maxiters::I = Int(1e5),
    ) where {F <: AbstractFloat, I <: Integer}

# Arguments

  - `solver::OrdinaryDiffEqCore.OrdinaryDiffEqAdaptiveAlgorithm`: The ODE solver algorithm to use. Defaults to `RDPK3Sp35()`.
  - `reltol::F`: The relative tolerance for the solver. Defaults to `1e-12`.
  - `abstol::F`: The absolute tolerance, in metres of ice, and the one that binds in practice. Defaults to `1e-3`, which suits runs of a few years.
  - `scale_abstol::Bool`: Whether to tighten `abstol` in proportion to the run length, since solver error accumulates. Defaults to `true`.
  - `adaptive::Bool`: Whether the solver picks its own step size. Defaults to `true`; gradient checks against finite differences need `false`.
  - `dt::F`: Fixed step in years, used only when `adaptive` is `false`. Defaults to `1.0/120.0`.
  - `step::F`: The step size that controls at which frequency the solution should be computed and returned in the results.
    Defaults to `1.0/12.0` (i.e. a month).
  - `tstops::Vector{Sleipnir.Float}`: Optional vector of time points where the solver should stop. Defaults to an empty vector.
  - `save_everystep::Bool`: Whether to save the solution at every step computed by the solver. Defaults to `false`.
  - `progress::Bool`: Whether to show progress during the solving process. Defaults to `true`.
  - `progress_steps::I`: The number of steps between progress updates. Defaults to `10`.
  - `maxiters::I`: Maximum number of iterations to perform in the iceflow solver. Defaults to `1e5`.

# Returns

  - `solver_parameters`: A `SolverParameters` object constructed with the specified parameters.
"""
function SolverParameters(;
        solver::OrdinaryDiffEqCore.OrdinaryDiffEqAdaptiveAlgorithm = RDPK3Sp35(),
        reltol::F = 1e-12,
        abstol::F = 1e-3,
        scale_abstol::Bool = true,
        adaptive::Bool = true,
        dt::F = 1.0/120.0,
        step::F = 1.0/12.0,
        tstops::Vector{Sleipnir.Float} = Vector{Sleipnir.Float}(),
        save_everystep = false,
        progress::Bool = true,
        progress_steps::I = 10,
        maxiters::I = Int(1e5)
) where {F <: AbstractFloat, I <: Integer}
    # Build the solver parameters based on input values
    return SolverParameters{Sleipnir.Float, Sleipnir.Int, typeof(solver)}(
        solver,
        Sleipnir.Float(reltol),
        Sleipnir.Float(abstol),
        scale_abstol,
        adaptive,
        Sleipnir.Float(dt),
        Sleipnir.Float(step),
        Sleipnir.Float.(tstops),
        save_everystep,
        progress,
        Sleipnir.Int(progress_steps),
        Sleipnir.Int(maxiters)
    )
end

function Base.:(==)(a::SolverParameters, b::SolverParameters)
    a.solver == b.solver && a.reltol == b.reltol && a.abstol == b.abstol &&
        a.scale_abstol == b.scale_abstol &&
        a.adaptive == b.adaptive && a.dt == b.dt &&
        a.step == b.step &&
        a.tstops == b.tstops && a.save_everystep == b.save_everystep &&
        a.progress == b.progress &&
        a.progress_steps == b.progress_steps && a.maxiters == b.maxiters
end

# Display setup
Base.show(io::IO, ::MIME"text/plain", params::SolverParameters) = Base.show(io, params)
function Base.show(io::IO, params::SolverParameters)
    pad = 12

    println(io, "SolverParameters")

    # Algorithm
    label(io, "  Algorithm", pad)
    val(io, "$(nameof(typeof(params.solver)))")
    sep(io)
    field(io, "reltol");
    print(io, " = ");
    val(io, "$(params.reltol)")
    sep(io)
    field(io, "abstol");
    print(io, " = ");
    val(io, "$(params.abstol)")
    sep(io)
    field(io, "maxiters");
    print(io, " = ");
    val(io, "$(params.maxiters)")
    println(io)

    # Output
    label(io, "  Output", pad)
    field(io, "step");
    print(io, " = ");
    val(io, "$(round(params.step; digits=4))");
    hint(io, " yr")
    sep(io)
    field(io, "tstops");
    print(io, " = ")
    n = length(params.tstops)
    n == 0 ? hint(io, "(empty)") : hint(io, "$n $(n == 1 ? "entry" : "entries")")
    sep(io)
    print(io, check(params.save_everystep));
    field(io, "save_everystep")
    println(io)

    # Progress
    label(io, "  Progress", pad)
    print(io, check(params.progress));
    field(io, "progress")
    if params.progress
        hint(io, " (every $(params.progress_steps) steps)")
    end
    println(io)
end

function Parameters(;
        physical::PhysicalParameters = PhysicalParameters(),
        simulation::SimulationParameters = SimulationParameters(),
        solver::SolverParameters = SolverParameters()
)

    # Build the parameters based on all the subtypes of parameters
    parameters = Sleipnir.Parameters{
        typeof(physical), typeof(simulation), Nothing, typeof(solver), Nothing}(
        physical, simulation,
        nothing, solver, nothing)

    enable_multiprocessing(parameters)

    return parameters
end

"""
    define_callback_steps(tspan::Tuple{F, F}, step::F) where {F <: AbstractFloat}

Defines the times to stop for the DiscreteCallback given a step and a timespan.

# Arguments

  - `tspan::Tuple{F, F}`: A tuple representing the start and end times.
  - `step::F`: The step size for generating the callback steps.

# Returns

  - `Vector{F}`: A vector of callback steps within the specified time span.
"""
function define_callback_steps(tspan::Tuple{F, F}, step::F) where {F <: AbstractFloat}
    tstops = collect(range(tspan[1], tspan[2], step = step))
    if tstops[end] !== tspan[2]
        push!(tstops, tspan[2])
    end
    return tstops
end
