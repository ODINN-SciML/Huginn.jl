export SolverParameters, Parameters

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
        simulation::SimulationParameters = SimulationParameters()
        physical::PhysicalParameters = PhysicalParameters()
        OGGM::OGGMparameters = OGGMparameters(),
        solver::SolverParameters = SolverParameters()
        )
Initialize Huginn parameters

Keyword arguments
=================
    
"""

function Parameters(;
    physical::PhysicalParameters = PhysicalParameters(),
    simulation::SimulationParameters = SimulationParameters(),
    OGGM::OGGMparameters = OGGMparameters(),
    solver::SolverParameters = SolverParameters()
    ) 

    # Build the parameters based on all the subtypes of parameters
    parameters = Sleipnir.Parameters(physical, simulation, OGGM,
                                     nothing, solver, nothing)

    if parameters.simulation.multiprocessing
        enable_multiprocessing(parameters)
    end
    
    oggm_config(OGGM.working_dir; oggm_processes=OGGM.workers)

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