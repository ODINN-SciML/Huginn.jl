export run₀, run!, generate_ground_truth!, apply_MB_mask!

"""
    run!(simulation::Prediction)

In-place run of the model.
"""
function run!(simulation::Prediction)

    @info "Running forward in-place PDE ice flow model"
    results_list = @showprogress pmap((glacier_idx) -> batch_iceflow_PDE!(glacier_idx, simulation), 1:length(simulation.glaciers))

    Sleipnir.save_results_file!(results_list, simulation)

    @everywhere GC.gc() # run garbage collector

end

"""
    batch_iceflow_PDE!(glacier_idx::I, simulation::Prediction) where {I <: Integer}

Solve the Shallow Ice Approximation iceflow PDE in-place for a given set of laws prescribed in the simulation object.
It creates the iceflow problem, the necessary callbacks and solve the PDE.

# Arguments:
- `glacier_idx::I`: Integer ID of the glacier.
- `simulation::Prediction`: Simulation object that contains all the necessary information to solve the iceflow.

# Returns
- A `Results` instance that stores the iceflow solution.
"""
function batch_iceflow_PDE!(glacier_idx::I, simulation::Prediction) where {I <: Integer}

    model = simulation.model
    params = simulation.parameters
    glacier = simulation.glaciers[glacier_idx]

    glacier_id = isnothing(glacier.rgi_id) ? "unnamed" : glacier.rgi_id
    println("Processing glacier $(glacier_id) for PDE forward simulation")

    # Initialize iceflow and mb cache
    simulation.cache = init_cache(model, simulation, glacier_idx, nothing)
    cache = simulation.cache

    # Create mass balance callback
    mb_tstops = define_callback_steps(params.simulation.tspan, params.solver.step)
    params.solver.tstops = mb_tstops

    mb_action! = let model = model, cache = cache, glacier = glacier, step = params.solver.step
        function (integrator)
            if params.simulation.use_MB
                # Compute mass balance
                MB_timestep!(cache, model, glacier, step, integrator.t)
                apply_MB_mask!(integrator.u, cache.iceflow)
            end
        end
    end
    cb_MB = PeriodicCallback(mb_action!, params.solver.step; initial_affect=false)

    # Create iceflow law callback
    cb_iceflow = build_callback(model.iceflow, simulation.cache.iceflow, glacier_idx)

    cb = CallbackSet(cb_MB, cb_iceflow)

    # Run iceflow PDE for this glacier
    du = params.simulation.use_iceflow ? SIA2D_PDE! : noSIA2D!
    results = simulate_iceflow_PDE!(simulation, cb, du)

    return results
end

"""
    function simulate_iceflow_PDE!(
        simulation::SIM,
        cb::DiscreteCallback,
        du
    ) where {SIM <: Simulation}

Make forward simulation of the iceflow PDE determined in `du` in-place and create the results.
"""
function simulate_iceflow_PDE!(
    simulation::SIM,
    cb::SciMLBase.DECallback,
    du
) where {SIM <: Simulation}
    cache = simulation.cache
    params = simulation.parameters

    # Define problem to be solved
    iceflow_prob = ODEProblem{true,SciMLBase.FullSpecialize}(du, cache.iceflow.H, params.simulation.tspan, simulation; tstops=params.solver.tstops)

    iceflow_sol = solve(iceflow_prob,
                        params.solver.solver,
                        callback=cb,
                        reltol=params.solver.reltol,
                        save_everystep=params.solver.save_everystep,
                        progress=params.solver.progress,
                        progress_steps=params.solver.progress_steps,
                        maxiters=params.solver.maxiters)
    @assert iceflow_sol.retcode==ReturnCode.Success "There was an error in the iceflow solver. Returned code is \"$(iceflow_sol.retcode)\""

    # @show iceflow_sol.destats
    # Compute average ice surface velocities for the simulated period
    cache.iceflow.H .= iceflow_sol.u[end]
    map!(x -> ifelse(x>0.0,x,0.0), cache.iceflow.H, cache.iceflow.H)

    # Average surface velocity
    avg_surface_V!(simulation, iceflow_sol.t[end], nothing)

    glacier_idx = cache.iceflow.glacier_idx
    glacier::Sleipnir.Glacier2D = simulation.glaciers[glacier_idx]

    # Surface topography
    @. cache.iceflow.S = glacier.B + cache.iceflow.H

    # Update simulation results
    results = Sleipnir.create_results(simulation, glacier_idx, iceflow_sol, nothing; light=!params.solver.save_everystep, processVelocity=V_from_H)

    return results
end

function SIA2D_PDE!(_dH::Matrix{R}, _H::Matrix{R}, simulation::SIM, t::R) where {R <: Real, SIM <: Simulation}
    SIA2D!(_dH, _H, simulation, t, nothing)
    return nothing
end

########################################################
##############  Out-of-place functions  ################
########################################################

"""
    run₀(simulation::Prediction)

Out-of-place run of the model.
"""
function run₀(simulation::Prediction)

    @info "Running forward out-of-place PDE ice flow model"
    results_list = @showprogress pmap((glacier_idx) -> batch_iceflow_PDE(glacier_idx, simulation), 1:length(simulation.glaciers))

    Sleipnir.save_results_file!(results_list, simulation)

    @everywhere GC.gc() # run garbage collector

end

"""
    batch_iceflow_PDE(glacier_idx::I, simulation::Prediction) where {I <: Integer}

Solve the Shallow Ice Approximation iceflow PDE out-of-place for a given set of laws prescribed in the simulation object.
It creates the iceflow problem, the necessary callbacks and solve the PDE.

# Arguments:
- `glacier_idx::I`: Integer ID of the glacier.
- `simulation::Prediction`: Simulation object that contains all the necessary information to solve the iceflow.

# Returns
- A `Results` instance that stores the iceflow solution.
"""
function batch_iceflow_PDE(glacier_idx::I, simulation::Prediction) where {I <: Integer}
    model = simulation.model
    params = simulation.parameters
    glacier = simulation.glaciers[glacier_idx]

    glacier_id = isnothing(glacier.rgi_id) ? "unnamed" : glacier.rgi_id
    println("Processing glacier $(glacier_id) for PDE forward simulation")

    # Initialize iceflow and mb cache
    simulation.cache = init_cache(model, simulation, glacier_idx, nothing)
    cache = simulation.cache

    # Create mass balance callback
    mb_tstops = define_callback_steps(params.simulation.tspan, params.solver.step)
    mb_stop_condition(u,t,integrator) = Sleipnir.stop_condition_tstops(u,t,integrator, mb_tstops) #closure
    params.solver.tstops = mb_tstops

    mb_action! = let model = model, cache = cache, glacier = glacier, step = params.solver.step
        function (integrator)
            if params.simulation.use_MB
                # Compute mass balance
                MB_timestep!(cache, model, glacier, step, integrator.t)
                apply_MB_mask!(integrator.u, cache.iceflow)
            end
        end
    end
    cb_MB = DiscreteCallback(mb_stop_condition, mb_action!)

    # Create iceflow law callback
    cb_iceflow = build_callback(model.iceflow, simulation.cache.iceflow, glacier_idx)

    cb = CallbackSet(cb_MB, cb_iceflow)

    # Run iceflow PDE for this glacier
    du = params.simulation.use_iceflow ? SIA2D_PDE : noSIA2D
    results = simulate_iceflow_PDE(simulation, cb, du)

    return results
end

"""
    function simulate_iceflow_PDE(
        simulation::SIM,
        cb::DiscreteCallback,
        du,
    ) where {SIM <: Simulation}

Make forward simulation of the iceflow PDE determined in `du` out-of-place and create the results.
"""
function simulate_iceflow_PDE(
    simulation::SIM,
    cb::SciMLBase.DECallback,
    du,
) where {SIM <: Simulation}
    cache = simulation.cache
    params = simulation.parameters

    # Define problem to be solved
    iceflow_prob = ODEProblem{false,SciMLBase.FullSpecialize}(du, cache.iceflow.H, params.simulation.tspan, simulation; tstops=params.solver.tstops)

    iceflow_sol = solve(iceflow_prob,
                        params.solver.solver,
                        callback=cb,
                        reltol=params.solver.reltol,
                        save_everystep=params.solver.save_everystep,
                        progress=params.solver.progress,
                        progress_steps=params.solver.progress_steps,
                        maxiters=params.solver.maxiters)
    @assert iceflow_sol.retcode==ReturnCode.Success "There was an error in the iceflow solver. Returned code is \"$(iceflow_sol.retcode)\""

    # @show iceflow_sol.destats
    # Compute average ice surface velocities for the simulated period
    cache.iceflow.H .= iceflow_sol.u[end]
    map!(x -> ifelse(x>0.0,x,0.0), cache.iceflow.H, cache.iceflow.H)

    # Average surface velocity
    avg_surface_V!(simulation, iceflow_sol.t[end], nothing)

    glacier_idx = cache.iceflow.glacier_idx
    glacier::Sleipnir.Glacier2D = simulation.glaciers[glacier_idx]

    # Surface topography
    @. cache.iceflow.S = glacier.B + cache.iceflow.H

    # Update simulation results
    results = Sleipnir.create_results(simulation, glacier_idx, iceflow_sol, nothing; light=!params.solver.save_everystep, processVelocity=V_from_H)

    return results
end

function SIA2D_PDE(_H::Matrix{R}, simulation::SIM, t::R) where {R <: Real, SIM <: Simulation}
    return SIA2D(_H, simulation, t, nothing)
    return nothing
end

"""
    store_thickness_data!(prediction::Prediction, tstops::Vector{F}) where {F <: AbstractFloat}

Store the simulated thickness data in the corresponding glaciers within a `Prediction` object.

# Arguments
- `prediction::Prediction`: A `Prediction` object containing the simulation results and associated glaciers.
- `tstops::Vector{F}`: A vector of time steps (of type `F <: AbstractFloat`) at which the simulation was evaluated.

# Description
This function iterates over the glaciers in the `Prediction` object and stores the simulated thickness data (`H`) and corresponding time steps (`t`) in the `data` field of each glacier. If the `data` field is empty (`nothing`), it initializes it with the thickness data. Otherwise, it appends the new thickness data to the existing data.

# Notes
- The function asserts that the time steps (`ts`) in the simulation results match the provided `tstops`. If they do not match, an error is raised.
"""
function store_thickness_data!(prediction::Prediction, tstops::Vector{F}) where {F <: AbstractFloat}
    # Store the thickness data in the glacier
    for i in 1:length(prediction.glaciers)
        ts = prediction.results[i].t
        Hs = prediction.results[i].H
        @assert ts ≈ tstops "Timestops of simulated PDE solution and the provided tstops do not match."
        prediction.glaciers[i].thicknessData = Sleipnir.ThicknessData(ts, Hs)
    end
end

"""
    generate_ground_truth!(
        glaciers::Vector{G},
        params::Sleipnir.Parameters,
        model::Sleipnir.Model,
        tstops::Vector{F},
    ) where {G <: Sleipnir.AbstractGlacier, F <: AbstractFloat}

Generate ground truth data for a glacier simulation by using the laws specified in the model and running a forward model.

# Arguments
- `glaciers::Vector{G}`: A vector of glacier objects of type `G`, where `G` is a subtype of `Sleipnir.AbstractGlacier`.
- `params::Sleipnir.Parameters`: Simulation parameters.
- `model::Sleipnir.Model`: The model to use for the simulation.
- `tstops::Vector{F}`: A vector of time steps at which the simulation will be evaluated.

# Description
1. Runs a forward model simulation for the glaciers using the provided laws, parameters, model, and time steps.
2. Store the simulation results as ground truth in the `glaciers` struct. For each glacier it populates the `thicknessData` field.

# Example
```julia
glaciers = [glacier1, glacier2] # dummy example
params = Sleipnir.Parameters(...) # to be filled
model = Sleipnir.Model(...) # to be filled
tstops = 0.0:1.0:10.0

generate_ground_truth!(glaciers, params, model, tstops)
```
"""
function generate_ground_truth!(
    glaciers::Vector{G},
    params::Sleipnir.Parameters,
    model::Sleipnir.Model,
    tstops::Vector{F},
) where {G <: Sleipnir.AbstractGlacier, F <: AbstractFloat}
    # Generate timespan from simulation
    t₀, t₁ = params.simulation.tspan
    @assert t₀ <= minimum(tstops)
    @assert t₁ >= maximum(tstops)

    prediction = Huginn.Prediction(model, glaciers, params)
    Huginn.run!(prediction)

    # Store the thickness data in the glacier
    store_thickness_data!(prediction, tstops)
end

"""
    apply_MB_mask!(H, ifm::SIA2DCache)

Apply the mass balance (MB) mask to the iceflow model in-place.
This function ensures that no MB is applied on the borders of the glacier to prevent overflow.

# Arguments:
- `H`: Ice thickness.
- `ifm::SIA2DCache`: Iceflow cache of the SIA2D that provides the mass balance information and that is modified in-place.
"""
function apply_MB_mask!(H, ifm::SIA2DCache)
    # Appy MB only over ice, and avoid applying it to the borders in the accummulation area to avoid overflow
    MB, MB_mask, MB_total = ifm.MB, ifm.MB_mask, ifm.MB_total
    MB_mask .= ((H .> 0.0) .&& (MB .< 0.0)) .|| ((H .> 10.0) .&& (MB .>= 0.0))
    H[MB_mask] .+= MB[MB_mask]
    MB_total[MB_mask] .+= MB[MB_mask]
    return nothing # For type stability
end
