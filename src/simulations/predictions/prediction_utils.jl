export run!, generate_ground_truth, generate_ground_truth_prediction, apply_MB_mask!

"""
    run!(simulation::Prediction)

In-place run of the model.
"""
function run!(simulation::Prediction)
    @info "Running forward in-place PDE ice flow model"
    results_list = @showprogress pmap(
        (glacier_idx) -> batch_iceflow_PDE!(glacier_idx, simulation),
        1:length(simulation.glaciers))

    # We don't save results files as this is not required and can crash multiple simulations
    # Sleipnir.save_results_file!(results_list, simulation)

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
    step = params.solver.step
    step_MB = params.simulation.step_MB

    glacier_id = isnothing(glacier.rgi_id) ? "unnamed" : glacier.rgi_id
    println("Processing glacier $(glacier_id) for PDE forward simulation")

    # Initialize iceflow and mb cache
    simulation.cache = init_cache(model, simulation, glacier_idx, nothing)
    cache = simulation.cache

    # Define tstops
    tstops = define_callback_steps(params.simulation.tspan, step)
    tstops = unique(vcat(tstops, params.solver.tstops)) # Merge time steps controlled by `step` with the user provided time steps

    # Create mass balance callback
    mb_action! = let model = model, cache = cache, glacier = glacier, step_MB = step_MB
        function (integrator)
            if params.simulation.use_MB
                # Compute mass balance
                glacier.S .= glacier.B .+ integrator.u
                MB_timestep!(cache, model, glacier, step_MB, integrator.t)
                apply_MB_mask!(integrator.u, cache.iceflow)
            end
        end
    end
    # A simulation period is sliced in time windows that are separated by `step_MB`
    # The mass balance is applied at the end of each of the windows
    cb_MB = PeriodicCallback(mb_action!, step_MB; initial_affect = false)

    # Create iceflow law callback
    cb_iceflow = build_callback(
        model.iceflow,
        simulation.cache.iceflow,
        glacier_idx,
        params.simulation.tspan
    )

    cb = CallbackSet(cb_MB, cb_iceflow)

    # Run iceflow PDE for this glacier
    du = params.simulation.use_iceflow ? SIA2D_PDE! : noSIA2D!
    results = simulate_iceflow_PDE!(simulation, cb, du, tstops)

    return results
end

"""
    simulate_iceflow_PDE!(
        simulation::SIM,
        cb::SciMLBase.DECallback,
        du,
        tstops::Vector{F},
    ) where {SIM <: Simulation, F <: AbstractFloat}

Make forward simulation of the iceflow PDE determined in `du` in-place and create the results.
"""
function simulate_iceflow_PDE!(
        simulation::SIM,
        cb::SciMLBase.DECallback,
        du,
        tstops::Vector{F}
) where {SIM <: Simulation, F <: AbstractFloat}
    cache = simulation.cache
    params = simulation.parameters

    # Define problem to be solved
    iceflow_prob = ODEProblem{true, SciMLBase.FullSpecialize}(
        du, cache.iceflow.H, params.simulation.tspan, simulation; tstops = tstops)

    iceflow_sol = solve(iceflow_prob,
        params.solver.solver,
        callback = cb,
        reltol = params.solver.reltol,
        save_everystep = params.solver.save_everystep,
        progress = params.solver.progress,
        progress_steps = params.solver.progress_steps,
        maxiters = params.solver.maxiters)
    @assert iceflow_sol.retcode==ReturnCode.Success "There was an error in the iceflow solver. Returned code is \"$(iceflow_sol.retcode)\""

    # @show iceflow_sol.destats
    # Compute average ice surface velocities for the simulated period
    cache.iceflow.H .= iceflow_sol.u[end]
    map!(x -> ifelse(x>0.0, x, 0.0), cache.iceflow.H, cache.iceflow.H)

    # Average surface velocity
    avg_surface_V!(simulation, iceflow_sol.t[end], nothing)

    glacier_idx = cache.iceflow.glacier_idx
    glacier::Sleipnir.Glacier2D = simulation.glaciers[glacier_idx]

    # Surface topography
    @. cache.iceflow.S = glacier.B + cache.iceflow.H

    # Update simulation results
    results = Sleipnir.create_results(
        simulation, glacier_idx, iceflow_sol, tstops; processVelocity = V_from_H)

    return results
end

function SIA2D_PDE!(_dH::Matrix{R}, _H::Matrix{R}, simulation::SIM,
        t::R) where {R <: Real, SIM <: Simulation}
    SIA2D!(_dH, _H, simulation, t, nothing)
    return nothing
end

"""
    thickness_velocity_data(
        prediction::Prediction,
        tstops::Vector{F};
        store::Tuple=(:H, :V),
    ) where {F <: AbstractFloat}

Return a new vector of glaciers with the simulated thickness and ice velocity data for each of the glaciers.

# Arguments

  - `prediction::Prediction`: A `Prediction` object containing the simulation results and associated glaciers.
  - `tstops::Vector{F}`: A vector of time steps (of type `F <: AbstractFloat`) at which the simulation was evaluated.
  - `store::Tuple`: Which generated simulation products to store. It can include `:H` and/or `:V`.

# Description

This function iterates over the glaciers in the `Prediction` object and generates the simulated data based on the
`store` argument at corresponding time steps (`t`).
If `store` includes `:H`, then the ice thickness is stored.
If `store` includes `:V`, then it computes the surface ice velocity data and store it.
A new vector of glaciers is created and each glacier is a copy with an updated `thicknessData` and `velocityData` fields.

# Notes

  - The function asserts that the time steps (`ts`) in the simulation results match the provided `tstops`. If they do not match, an error is raised.

# Returns

A new vector of glaciers where each glacier is a copy of the original one with the updated `thicknessData` and `velocityData` based on the values provided in `store`.
"""
function thickness_velocity_data(
        prediction::Prediction,
        tstops::Vector{F};
        store::Tuple = (:H, :V)
) where {F <: AbstractFloat}
    # Store the thickness data in the glacier
    glaciers = map(1:length(prediction.glaciers)) do i
        prediction.cache = init_cache(prediction.model, prediction, i, nothing)
        ts = prediction.results[i].t
        Hs = prediction.results[i].H
        @assert ts ≈ tstops "Timestops of simulated PDE solution and the provided tstops do not match."

        thicknessData = :H in store ? Sleipnir.ThicknessData(ts, Hs) : nothing

        Vx = Array{Matrix{F}, 1}()
        Vy = Array{Matrix{F}, 1}()
        Vabs = Array{Matrix{F}, 1}()
        for j in 1:length(ts)
            vx, vy, vabs = Huginn.V_from_H(prediction, Hs[j], ts[j], nothing)
            push!(Vx, vx)
            push!(Vy, vy)
            push!(Vabs, vabs)
        end
        velocityData = :V in store ?
                       SurfaceVelocityData(
            date = Sleipnir.Dates.DateTime.(Sleipnir.partial_year(Sleipnir.Dates.Day, ts)),
            vx = Vx,
            vy = Vy,
            vabs = Vabs
        ) : nothing

        Glacier2D(
            prediction.glaciers[i],
            thicknessData = thicknessData,
            velocityData = velocityData
        ) # Rebuild glacier since we cannot change type of `glacier.thicknessData` and `glacier.velocityData`
    end
    return glaciers
end

"""
    generate_ground_truth(
        glaciers::Vector{G},
        params::Sleipnir.Parameters,
        model::Sleipnir.Model,
        tstops::Vector{F};
        store::Tuple=(:H, :V),
    ) where {G <: Sleipnir.AbstractGlacier, F <: AbstractFloat}

Generate ground truth data for a glacier simulation by using the laws specified in the model and running a forward model.
It returns a new vector of glaciers with updated `thicknessData` and `velocityData` fields based on the `store` argument.

# Arguments

  - `glaciers::Vector{G}`: A vector of glacier objects of type `G`, where `G` is a subtype of `Sleipnir.AbstractGlacier`.
  - `params::Sleipnir.Parameters`: Simulation parameters.
  - `model::Sleipnir.Model`: The model to use for the simulation.
  - `tstops::Vector{F}`: A vector of time steps at which the simulation will be evaluated.
  - `store::Tuple`: Which generated simulation products to store. It can include `:H` and/or `:V`.

# Description

 1. Runs a forward model simulation for the glaciers using the provided laws, parameters, model, and time steps.
 2. Build a new vector of glaciers and store the simulation results as ground truth in the `glaciers` struct.
    For each glacier it populates the `thicknessData` field if `store` contains `:H` and it populates `velocityData` if `store` contains `:V`.

# Example

```julia
glaciers = [glacier1, glacier2] # dummy example
params = Huginn.Parameters() # to be filled
model = Huginn.Model() # to be filled
tstops = 0.0:1.0:10.0

glaciers = generate_ground_truth(glaciers, params, model, tstops)
```
"""
function generate_ground_truth(
        glaciers::Vector{G},
        params::Sleipnir.Parameters,
        model::Sleipnir.Model,
        tstops::Vector{F};
        store::Tuple = (:H, :V)
) where {G <: Sleipnir.AbstractGlacier, F <: AbstractFloat}
    # Generate timespan from simulation
    t₀, t₁ = params.simulation.tspan
    @assert t₀ <= minimum(tstops)
    @assert t₁ >= maximum(tstops)

    prediction = Prediction(model, glaciers, params)
    run!(prediction)

    # Create new glaciers with the thickness and velocity data
    return thickness_velocity_data(prediction, tstops; store = store)
end

"""
    generate_ground_truth_prediction(
        glaciers::Vector{G},
        params::Sleipnir.Parameters,
        model::Sleipnir.Model,
        tstops::Vector{F},
    ) where {G <: Sleipnir.AbstractGlacier, F <: AbstractFloat}

Wrapper for `generate_ground_truth` that also updates the `glaciers` field of the `Prediction` object.

# Arguments

  - `glaciers::Vector{G}`: A vector of glacier objects of type `G`, where `G` is a subtype of `Sleipnir.AbstractGlacier`.
  - `params::Sleipnir.Parameters`: Simulation parameters.
  - `model::Sleipnir.Model`: The model to use for the simulation.
  - `tstops::Vector{F}`: A vector of time steps at which the simulation will be evaluated.

# Description

This function calls `generate_ground_truth` to generate ground truth data for the glaciers using the provided laws, parameters, model, and time steps. In addition, it updates the `glaciers` field of the `Prediction` object with the newly generated glaciers containing the ground truth data.

# Example

```julia
glaciers = [glacier1, glacier2] # dummy example
params = Huginn.Parameters() # to be filled
model = Huginn.Model() # to be filled
tstops = 0.0:1.0:10.0

prediction = generate_ground_truth_prediction(glaciers, params, model, tstops)
```
"""
function generate_ground_truth_prediction(
        glaciers::Vector{G},
        params::Sleipnir.Parameters,
        model::Sleipnir.Model,
        tstops::Vector{F}
) where {G <: Sleipnir.AbstractGlacier, F <: AbstractFloat}

    # We update the current prediction to include the newly generated glaciers
    glaciers = generate_ground_truth(glaciers, params, model, tstops)
    prediction = Prediction(model, glaciers, params)

    # We return the prediction object so that it can be used later
    return prediction
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
