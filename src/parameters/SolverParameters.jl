export SolverParameters, Parameters

"""
    SolverParameters{F <: AbstractFloat, I <: Integer}

A mutable struct that holds parameters for the solver.

# Fields
- `solver::OrdinaryDiffEq.OrdinaryDiffEqAdaptiveAlgorithm`: The algorithm used for solving differential equations.
- `reltol::F`: The relative tolerance for the solver.
- `step::F`: The step size for the solver.
- `tstops::Union{Nothing, Vector{F}}`: Optional vector of time points where the solver should stop for the callbacks.
- `save_everystep::Bool`: Flag indicating whether to save the solution at every step.
- `progress::Bool`: Flag indicating whether to show progress during the solving process.
- `progress_steps::I`: The number of steps between progress updates.
"""
mutable struct SolverParameters{F <: AbstractFloat, I <: Integer} <: AbstractParameters
    solver::OrdinaryDiffEq.OrdinaryDiffEqAdaptiveAlgorithm
    reltol::F
    step::F
    tstops::Union{Nothing,Vector{F}} 
    save_everystep::Bool
    progress::Bool
    progress_steps::I
end

"""
    SolverParameters(;
        solver::OrdinaryDiffEq.OrdinaryDiffEqAdaptiveAlgorithm = RDPK3Sp35(),
        reltol::Float64 = 1e-7
        )
Initialize the parameters for the numerical solver.
Keyword arguments
=================
    - `solver`: solver to use from DifferentialEquations.jl
    - `reltol`: Relative tolerance for the solver
"""
function SolverParameters(;
            solver::OrdinaryDiffEq.OrdinaryDiffEqAdaptiveAlgorithm = RDPK3Sp35(),
            reltol::F = 1e-12,
            step::F = 1.0/12.0,
            tstops::Union{Nothing,Vector{F}} = nothing,
            save_everystep = false,
            progress::Bool = true,
            progress_steps::I = 10
            ) where {F <: AbstractFloat, I <: Integer}
    # Build the solver parameters based on input values
    solver_parameters = SolverParameters(solver, reltol, 
                                         step, tstops,
                                         save_everystep, progress, progress_steps)

    return solver_parameters
end

Base.:(==)(a::SolverParameters, b::SolverParameters) = a.solver == b.solver && a.reltol == b.reltol && a.step == b.step &&
                                      a.tstops == b.tstops && a.save_everystep == b.save_everystep && a.progress == b.progress &&
                                      a.progress_steps == b.progress_steps

"""
Parameters(;
        physical::PhysicalParameters = PhysicalParameters(),
        simulation::SimulationParameters = SimulationParameters(),
        solver::SolverParameters = SolverParameters()
        )
Initialize Huginn parameters

Keyword arguments
=================

"""

function Parameters(;
    physical::PhysicalParameters = PhysicalParameters(),
    simulation::SimulationParameters = SimulationParameters(),
    solver::SolverParameters = SolverParameters()
    )

    # Build the parameters based on all the subtypes of parameters
    parameters = Sleipnir.Parameters(physical, simulation,
                                     nothing, solver, nothing, nothing)

    if parameters.simulation.multiprocessing
        enable_multiprocessing(parameters)
    end

    return parameters
end


"""
    define_callback_steps(tspan::Tuple{Float64, Float64}, step::Float64)

Defines the times to stop for the DiscreteCallback given a step
"""
function define_callback_steps(tspan::Tuple{Float64, Float64}, step::Float64)
    tmin_int = Int(tspan[1])
    tmax_int = Int(tspan[2])+1
    tstops = range(tmin_int+step, tmax_int, step=step) |> collect
    tstops = filter(x->( (Int(tspan[1])<x) & (x<=Int(tspan[2])) ), tstops)
    return tstops
end