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
    step = params.solver.step

    glacier_id = isnothing(glacier.rgi_id) ? "unnamed" : glacier.rgi_id
    println("Processing glacier $(glacier_id) for PDE forward simulation")

    # Initialize iceflow and mb cache
    simulation.cache = init_cache(model, simulation, glacier_idx, nothing)

    # Define tstops
    tstops = define_callback_steps(params.simulation.tspan, step)
    tstops = unique(vcat(tstops, params.solver.tstops)) # Merge time steps controlled by `step` with the user provided time steps

    # Create iceflow law callback
    cb_iceflow = build_callback(
        model.iceflow,
        simulation.cache.iceflow,
        glacier_idx,
        params.simulation.tspan
    )

    # Mass balance is a source term of the ice flow RHS, so there is no callback for it
    cb = CallbackSet(cb_iceflow)

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

The results are sampled on `tstops`. When the mass balance is evaluated in the right hand
side, the solver is additionally forced to stop on the mass balance window edges, which are
discontinuities of the right hand side but not result time steps.
"""
function simulate_iceflow_PDE!(
        simulation::SIM,
        cb::SciMLBase.DECallback,
        du,
        tstops::Vector{F}
) where {SIM <: Simulation, F <: AbstractFloat}
    cache = simulation.cache
    params = simulation.parameters

    solver_tstops, saveat = MB_solver_stops(simulation, tstops)
    # `dt` is passed only when stepping is fixed. In adaptive mode the solver picks its own
    # initial step, and supplying one overrides that choice; supplying zero aborts the solve
    # outright.
    step_kw = params.solver.adaptive ? NamedTuple() : (; dt = params.solver.dt)

    # Define problem to be solved
    iceflow_prob = ODEProblem{true, SciMLBase.FullSpecialize}(
        du, cache.iceflow.H, params.simulation.tspan, simulation; tstops = solver_tstops)

    iceflow_sol = solve(iceflow_prob,
        with_eigen_est(params.solver.solver, simulation);
        callback = cb,
        reltol = params.solver.reltol,
        abstol = params.solver.scale_abstol ?
                 effective_abstol(params.solver.abstol, params.simulation.tspan) :
                 params.solver.abstol,
        adaptive = params.solver.adaptive,
        saveat = saveat,
        save_everystep = params.solver.save_everystep,
        progress = params.solver.progress,
        progress_steps = params.solver.progress_steps,
        maxiters = params.solver.maxiters,
        step_kw...)
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

    MB, t_MB = MB_diagnostics(simulation, iceflow_sol)

    # Update simulation results
    results = Sleipnir.create_results(
        simulation,
        glacier_idx,
        iceflow_sol,
        tstops;
        processVelocity = V_from_H,
        MB = MB,
        t_MB = t_MB
    )

    return results
end

"""
    spectral_radius(simulation)

Bound on the spectral radius of the ice flow right hand side, in yr⁻¹.

Two terms contribute. The flux divergence is a diffusion with coefficient `D`, whose discrete
Laplacian is bounded by `4 D (1/Δx² + 1/Δy²)`. The mass balance source adds its own diagonal
Jacobian `∂ṁ/∂H`.

The second term is the one that is easy to forget, and it is not a correction: where the ice
is thin the ramp makes `∂ṁ/∂H` large, and where `A` is small `D` is negligible, so the mass
balance can be the *only* term. Dropping it there underestimates the spectral radius and the
stabilised solver picks too few stages.

The mass balance term is a bound precomputed from the lookup table rather than the exact
maximum at the current state, because this is called from inside the solver: the adjoint hands
back an augmented `[H; θ]` vector, so anything that reads the state cannot be evaluated there.
"""
function spectral_radius(simulation)
    cache = simulation.cache
    glacier = simulation.glaciers[cache.iceflow.glacier_idx]
    λ = 4 * maximum(cache.iceflow.D) * (1 / glacier.Δx^2 + 1 / glacier.Δy^2)
    mb_cache = cache.mass_balance
    return mb_cache_active(mb_cache) ? λ + mb_cache.∂ṁ_max : λ
end

"""
    with_eigen_est(alg, simulation)

Give a stabilised solver the spectral radius instead of letting it estimate one.

`ROCK2` otherwise runs an internal power iteration, which costs right hand side evaluations
and, because the iteration itself depends on the state, makes the solution jitter with the
parameters. That jitter is invisible to a forward run but it puts a noise floor under the
loss, which is fatal for finite differences. Every other algorithm is returned unchanged.
"""
function with_eigen_est(alg::ROCK2, simulation)
    return ROCK2(min_stages = alg.min_stages, max_stages = alg.max_stages,
        eigen_est = (integrator) -> integrator.eigen_est = spectral_radius(simulation))
end
function with_eigen_est(alg::ROCK4, simulation)
    return ROCK4(min_stages = alg.min_stages, max_stages = alg.max_stages,
        eigen_est = (integrator) -> integrator.eigen_est = spectral_radius(simulation))
end
with_eigen_est(alg, simulation) = alg

"""
    ABSTOL_REFERENCE_YEARS

Run length the default `abstol` was chosen for. Longer runs are tightened relative to it by
[`effective_abstol`](@ref).
"""
const ABSTOL_REFERENCE_YEARS = 5.0

"""
    effective_abstol(abstol, tspan; verbose = true)

Absolute tolerance actually handed to the solver, tightened in proportion to the run length.

Solver error accumulates: on a test glacier it grew as roughly `t^1.4`, so a tolerance that is
appropriate for a few years is too loose for several decades. At the default `abstol` a 30 year
run reached an RMS error of 0.22 m and a local error of 3.8 m, against 0.02 m and 0.16 m over
five years.

The scaling is linear in the run length, which lands near the tolerance that measurement
recommends for a multi-decade run at roughly twice the cost. It is deliberately *not* the
scaling that would hold the error strictly constant: the error responds weakly to the tolerance
(about `abstol^0.4`), so holding it fixed would demand a tolerance some hundreds of times
tighter and a cost to match. This trades a little accuracy for a run that finishes.

The tolerance is only ever tightened, never loosened, and the adjustment is logged. Pass
`verbose = false` to silence it.

!!! note

    The exponents behind this rule were measured on a single glacier over three run lengths.
    They set the shape of the rule, not a guarantee, and a run that needs a specific accuracy
    should set `abstol` explicitly rather than rely on it.
"""
function effective_abstol(abstol::F, tspan; verbose::Bool = true) where {F}
    years = tspan[2] - tspan[1]
    years > ABSTOL_REFERENCE_YEARS || return abstol
    scaled = abstol * F(ABSTOL_REFERENCE_YEARS / years)
    verbose &&
        @info "Tightening abstol for a $(round(years; digits = 1)) year run" abstol scaled
    return scaled
end

"""
    MB_solver_stops(simulation, tstops::Vector{F}) where {F <: AbstractFloat}

Times the solver must stop at, and times it must save at, for a given result grid `tstops`.

Saving has to be asked for explicitly. No callback runs on the result grid any more, and a
callback used to be what put those states in the solution: a `PeriodicCallback` saves either
side of itself by default, so the mass balance one did it as a side effect even on a run
without mass balance. `create_results` then reads the result grid out of the solution and
fails on the states that are missing.

With mass balance in the right hand side there is more to do: `ṁ` is piecewise constant in
time and the right hand side jumps at every window edge, so the solver also has to stop
there, and the states on those edges are saved for [`MB_diagnostics`](@ref) to read.
"""
function MB_solver_stops(simulation, tstops::Vector{F}) where {F <: AbstractFloat}
    mb_cache_active(simulation.cache.mass_balance) || return tstops, tstops
    params = simulation.parameters
    edges = define_callback_steps(params.simulation.tspan, params.simulation.step_MB)
    solver_tstops = sort(unique(vcat(tstops, edges)))
    return solver_tstops, solver_tstops
end

"""
    MB_diagnostics(simulation, iceflow_sol)

Mass balance accumulated over each mass balance window, and the times it is reported at.

These populate the `MB` and `t_MB` fields of `Results`. When the mass balance is evaluated
inside the ice flow right hand side, no snapshot is recorded during the solve and the
accumulation is rebuilt from the states saved on the window edges with a midpoint rule,

```math
MB_k ≈ ṁ\\left(\\frac{H_{k-1} + H_k}{2}, \\frac{t_{k-1} + t_k}{2}\\right) (t_k - t_{k-1})
```

The grid is the window grid, not the result grid, so `MB` and `t_MB` do not depend on `step`.
Otherwise the snapshots recorded during the solve are returned unchanged.
"""
function MB_diagnostics(simulation, iceflow_sol)
    cache = simulation.cache
    mb_cache = cache.mass_balance
    mb_cache_active(mb_cache) || return cache.iceflow.MB_history, cache.iceflow.MB_times

    glacier_idx = cache.iceflow.glacier_idx
    glacier = simulation.glaciers[glacier_idx]
    mb_model = get_mb_model(simulation.model.mass_balance, glacier_idx)

    params = simulation.parameters
    tspan = params.simulation.tspan
    edges = define_callback_steps(tspan, params.simulation.step_MB)
    H = iceflow_sol.u[Sleipnir.indFromT(tspan, edges, iceflow_sol.t)]

    F = eltype(edges)
    n = length(edges) - 1
    MB = Vector{Matrix{F}}(undef, max(n, 0))
    t_MB = Vector{F}(undef, max(n, 0))
    for k in 1:n
        MB_rate!(mb_cache.ṁ, 0.5 .* (H[k] .+ H[k + 1]), mb_cache, mb_model, glacier,
            0.5 * (edges[k] + edges[k + 1]))
        MB[k] = mb_cache.ṁ .* (edges[k + 1] - edges[k])
        t_MB[k] = edges[k + 1]
    end
    return MB, t_MB
end

function SIA2D_PDE!(_dH::Matrix{R}, _H::Matrix{R}, simulation::SIM,
        t::Real) where {R <: Real, SIM <: Simulation}
    SIA2D!(_dH, _H, simulation, t, nothing)
    return nothing
end

"""
    thickness_velocity_data(
        prediction::Prediction,
        tstops::Vector{F};
        store::Tuple=(:H, :V, :dhdt),
    ) where {F <: AbstractFloat}

Return a new vector of glaciers with the simulated thickness ice velocity and dhdt data for each of the glaciers.

# Arguments

  - `prediction::Prediction`: A `Prediction` object containing the simulation results and associated glaciers.
  - `tstops::Vector{F}`: A vector of time steps (of type `F <: AbstractFloat`) at which the simulation was evaluated.
  - `store::Tuple`: Which generated simulation products to store. It can include `:H`, `:V`, `:avgV` and/or `:dhdt`.

# Description

This function iterates over the glaciers in the `Prediction` object and generates the simulated data based on the
`store` argument at corresponding time steps (`t`).
If `store` includes `:H`, then the ice thickness is stored.
If `store` includes `:V` or `:avgV`, then it computes the surface ice velocity data and stores it. These two options are mutually exclusive. When `:avgV` is provided, only one snapshot of ice surface velocity is computed and it corresponds to the ice surface velocity at the closest time to (tspan[1]+tspan[2])/2.
If `store` includes `:dhdt`, then it computes the mean surface elevation change and stores it.
A new vector of glaciers is created and each glacier is a copy with an updated `thicknessData`, `velocityData` and `dhdtData` fields.

# Notes

  - The function asserts that the time steps (`ts`) in the simulation results match the provided `tstops`. If they do not match, an error is raised.

# Returns

A new vector of glaciers where each glacier is a copy of the original one with the updated `thicknessData`, `velocityData` and `dhdtData` based on the values provided in `store`.
"""
function thickness_velocity_data(
        prediction::Prediction,
        tstops::Vector{F};
        store::Tuple = (:H, :V, :dhdt)
) where {F <: AbstractFloat}
    # Store the thickness data in the glacier
    glaciers = map(1:length(prediction.glaciers)) do i
        prediction.cache = init_cache(prediction.model, prediction, i, nothing)
        ts = prediction.results[i].t
        Hs = prediction.results[i].H
        @assert ts ≈ tstops "Timestops of simulated PDE solution and the provided tstops do not match."
        if :V in store || :avgV in store
            @assert (:V in store) != (:avgV in store) "Cannot store :V and :avgV at the same time."
        end

        thicknessData = :H in store ? Sleipnir.ThicknessData(ts, Hs) : nothing

        velocityData = if :V in store
            Vx = Array{Matrix{F}, 1}()
            Vy = Array{Matrix{F}, 1}()
            Vabs = Array{Matrix{F}, 1}()
            for j in 1:length(ts)
                apply_all_callback_laws!(
                    prediction.model.iceflow, prediction.cache.iceflow,
                    prediction, i, ts[j], nothing)
                vx, vy, vabs = Huginn.V_from_H(prediction, Hs[j], ts[j], nothing)
                push!(Vx, vx)
                push!(Vy, vy)
                push!(Vabs, vabs)
            end
            if all(norm.(Vabs) .== 0)
                @warn "All velocities are null which is probably a bug."
            end
            SurfaceVelocityData(
                date = Sleipnir.Dates.DateTime.(Sleipnir.partial_year(Sleipnir.Dates.Day, ts)),
                vx = Vx,
                vy = Vy,
                vabs = Vabs
            )
        elseif :avgV in store
            Vx = Array{Matrix{F}, 1}()
            Vy = Array{Matrix{F}, 1}()
            Vabs = Array{Matrix{F}, 1}()

            tspan_velocity = prediction.parameters.simulation.tspan # Use the whole simulation to compute average velocity
            step_velocity = prediction.parameters.solver.step # Use the `step` parameter for average velocity computation
            avg_Vx_pred, avg_Vy_pred,
            avg_V_pred = Huginn.averageV(
                nothing, prediction, tspan_velocity, step_velocity, ts, Hs)
            if norm(avg_V_pred) == 0
                @warn "Velocity is null which is probably a bug."
            end
            SurfaceVelocityData(
                date = Sleipnir.Dates.DateTime.(Sleipnir.partial_year(
                    Sleipnir.Dates.Day, [(minimum(ts)+maximum(ts))/2])),
                date1 = Sleipnir.Dates.DateTime.(Sleipnir.partial_year(Sleipnir.Dates.Day, [minimum(ts)])),
                date2 = Sleipnir.Dates.DateTime.(Sleipnir.partial_year(Sleipnir.Dates.Day, [maximum(ts)])),
                vx = [avg_Vx_pred],
                vy = [avg_Vy_pred],
                vabs = [avg_V_pred]
            )
        else
            nothing
        end

        dhdtData = if :dhdt in store
            tdhdt = (minimum(ts), maximum(ts))
            ind = Sleipnir.indFromT(prediction.parameters.simulation.tspan, tdhdt, ts)
            H0 = Hs[ind[1]]
            H1 = Hs[ind[2]]
            mask = H0 .> 1e-2
            dhdt = mean(H1[mask] .- H0[mask])/(tdhdt[2]-tdhdt[1])
            DhdtData(tdhdt, dhdt)
        else
            nothing
        end

        Glacier2D(
            prediction.glaciers[i],
            thicknessData = thicknessData,
            velocityData = velocityData,
            dhdtData = dhdtData
        ) # Rebuild glacier since we cannot change type of `glacier.thicknessData`, `glacier.velocityData` and `glacier.dhdtData`
    end
    return glaciers
end

"""
    generate_ground_truth(
        glaciers::Vector{G},
        params::Sleipnir.Parameters,
        model::Sleipnir.Model,
        tstops::Vector{F};
        store::Tuple=(:H, :V, :dhdt),
    ) where {G <: Sleipnir.AbstractGlacier, F <: AbstractFloat}

Generate ground truth data for a glacier simulation by using the laws specified in the model and running a forward model.
It returns a new vector of glaciers with updated `thicknessData`, `velocityData` and `dhdtData` fields based on the `store` argument.

# Arguments

  - `glaciers::Vector{G}`: A vector of glacier objects of type `G`, where `G` is a subtype of `Sleipnir.AbstractGlacier`.
  - `params::Sleipnir.Parameters`: Simulation parameters.
  - `model::Sleipnir.Model`: The model to use for the simulation.
  - `tstops::Vector{F}`: A vector of time steps at which the simulation will be evaluated.
  - `store::Tuple`: Which generated simulation products to store. It can include `:H`, `:V`, `:avgV` and/or `:dhdt`.

# Description

 1. Runs a forward model simulation for the glaciers using the provided laws, parameters, model, and time steps.
 2. Build a new vector of glaciers and store the simulation results as ground truth in the `glaciers` struct.
    For each glacier it populates
      + `thicknessData` field if `store` contains `:H`,
      + `velocityData` if `store` contains `:V` or `:avgV`,
      + `dhdtData` if `store` contains `:dhdt`.

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
        store::Tuple = (:H, :V, :dhdt)
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
    # Set MB to zero outside of MB_mask
    MB[.!MB_mask] .= 0
    # Get the linear indices where MB_mask is true
    mask_indices = findall(MB_mask)
    # Among those, find where ice would disappear after MB application
    mask_ice_disappear = (H[mask_indices] .+ MB[mask_indices]) .< 0.0
    # Get the actual indices to modify
    disappear_indices = mask_indices[mask_ice_disappear]
    # Clip MB in-place at those indices
    MB[disappear_indices] .= .-H[disappear_indices]
    H[MB_mask] .+= MB[MB_mask]
    MB_total[MB_mask] .+= MB[MB_mask]
    return nothing # For type stability
end
