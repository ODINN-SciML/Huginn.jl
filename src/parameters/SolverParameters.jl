export SolverParameters

"""
A mutable struct that holds parameters for the solver.

    SolverParameters{F <: AbstractFloat, I <: Integer}

# Fields
- `solver::OrdinaryDiffEqCore.OrdinaryDiffEqAdaptiveAlgorithm`: The algorithm used for solving differential equations.
- `reltol::F`: The relative tolerance for the solver.
- `step::F`: The step size that controls at which frequency the results must be saved.
- `tstops::Vector{F}`: Optional vector of time points where the solver should stop to store the results.
- `save_everystep::Bool`: Flag indicating whether to save the solution at every step computed by the solver.
- `progress::Bool`: Flag indicating whether to show progress during the solving process.
- `progress_steps::I`: The number of steps between progress updates.
- `maxiters::I`: Maximum number of iterations to perform in the iceflow solver.
"""
mutable struct SolverParameters{F <: AbstractFloat, I <: Integer} <: AbstractParameters
    solver::OrdinaryDiffEqCore.OrdinaryDiffEqAdaptiveAlgorithm
    reltol::F
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
    step::F = 1.0/12.0,
    tstops::Vector{Sleipnir.Float} = Vector{Sleipnir.Float}(),
    save_everystep = false,
    progress::Bool = true,
    progress_steps::I = 10,
    maxiters::I = Int(1e5),
) where {F <: AbstractFloat, I <: Integer}
    # Build the solver parameters based on input values
    return SolverParameters(
        solver,
        Sleipnir.Float(reltol),
        Sleipnir.Float(step),
        Sleipnir.Float.(tstops),
        save_everystep,
        progress,
        Sleipnir.Int(progress_steps),
        Sleipnir.Int(maxiters),
    )
end

Base.:(==)(a::SolverParameters, b::SolverParameters) = a.solver == b.solver && a.reltol == b.reltol && a.step == b.step &&
                                      a.tstops == b.tstops && a.save_everystep == b.save_everystep && a.progress == b.progress &&
                                      a.progress_steps == b.progress_steps && a.maxiters == b.maxiters

function Parameters(;
    physical::PhysicalParameters = PhysicalParameters(),
    simulation::SimulationParameters = SimulationParameters(),
    solver::SolverParameters = SolverParameters(),
)

    # Build the parameters based on all the subtypes of parameters
    parameters = Sleipnir.Parameters(physical, simulation,
                                     nothing, solver, nothing, nothing)

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
