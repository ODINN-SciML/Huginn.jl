export run₀, run!, apply_MB_mask!

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

Solve the Shallow Ice Approximation iceflow PDE for a given temperature series batch in-place.
"""
function batch_iceflow_PDE!(glacier_idx::I, simulation::Prediction) where {I <: Integer}

    model = simulation.model
    params = simulation.parameters
    glacier = simulation.glaciers[glacier_idx]

    glacier_id = isnothing(glacier.rgi_id) ? "unnamed" : glacier.rgi_id
    println("Processing glacier $(glacier_id) for PDE forward simulation")

    # Initialize iceflow and mb cache
    simulation.cache = init_cache(model, simulation, glacier_idx)
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
                apply_MB_mask!(integrator.u, glacier, cache.iceflow)
            end
        end
    end
    cb_MB = DiscreteCallback(mb_stop_condition, mb_action!)

    # Create iceflow law callback
    cb_iceflow = build_callback(model.iceflow, simulation.cache.iceflow, glacier_idx)

    cb = CallbackSet(cb_MB, cb_iceflow)

    # Run iceflow PDE for this glacier
    du = params.simulation.use_iceflow ? SIA2D! : noSIA2D!
    results = simulate_iceflow_PDE!(simulation, cb, du)

    return results
end

"""
    function simulate_iceflow_PDE!(
        simulation::SIM,
        cb::DiscreteCallback,
        du) where {SIM <: Simulation}

Make forward simulation of the iceflow PDE determined in `du`.
"""
function simulate_iceflow_PDE!(
    simulation::SIM,
    cb::SciMLBase.DECallback,
    du) where {SIM <: Simulation}
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
    @assert params.simulation.test_mode || iceflow_sol.retcode==ReturnCode.Success "There was an error in the iceflow solver. Returned code is \"$(iceflow_sol.retcode)\""

    # @show iceflow_sol.destats
    # Compute average ice surface velocities for the simulated period
    cache.iceflow.H .= iceflow_sol.u[end]
    map!(x -> ifelse(x>0.0,x,0.0), cache.iceflow.H, cache.iceflow.H)

    # Average surface velocity
    avg_surface_V!(simulation, iceflow_sol.t[end])

    glacier_idx = cache.iceflow.glacier_idx
    glacier::Sleipnir.Glacier2D = simulation.glaciers[glacier_idx[]]

    # Surface topography
    @. cache.iceflow.S = glacier.B + cache.iceflow.H

    # Update simulation results
    results = Sleipnir.create_results(simulation, glacier_idx[], iceflow_sol, nothing; light=!params.solver.save_everystep, processVelocity=V_from_H)

    return results
end

########################################################
##############  Out-of-place functions  ################
########################################################

"""
    run(simulation::Prediction)

Out-of-place run of the model. 
"""
function run₀(simulation::Prediction)

    @info "Running forward out-of-place PDE ice flow model"
    results_list = @showprogress pmap((glacier_idx) -> batch_iceflow_PDE(glacier_idx, simulation), 1:length(simulation.glaciers))

    Sleipnir.save_results_file!(results_list, simulation)

    @everywhere GC.gc() # run garbage collector

end

"""
    batch_iceflow_PDE(glacier_idx::I, simulation::Prediction) 

Solve the Shallow Ice Approximation iceflow PDE for a given temperature series batch out-of-place.
"""
function batch_iceflow_PDE(glacier_idx::I, simulation::Prediction) where {I <: Integer}
    model = simulation.model
    params = simulation.parameters
    glacier = simulation.glaciers[glacier_idx]

    glacier_id = isnothing(glacier.rgi_id) ? "unnamed" : glacier.rgi_id
    println("Processing glacier $(glacier_id) for PDE forward simulation")

    # Initialize iceflow and mb cache
    simulation.cache = init_cache(model, simulation, glacier_idx)
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
                apply_MB_mask!(integrator.u, glacier, cache.iceflow)
            end
        end
    end
    cb_MB = DiscreteCallback(mb_stop_condition, mb_action!)

    # Create iceflow law callback
    cb_iceflow = build_callback(model.iceflow, simulation.cache.iceflow, glacier_idx)

    cb = CallbackSet(cb_MB, cb_iceflow)

    # Run iceflow PDE for this glacier
    du = params.simulation.use_iceflow ? SIA2D : noSIA2D
    results = simulate_iceflow_PDE(simulation, cb; du = du)

    return results
end

"""
    function simulate_iceflow_PDE(
        simulation::SIM,
        cb::DiscreteCallback;
        du = SIA2D) where {SIM <: Simulation}

Make forward simulation of the iceflow PDE determined in `du`.
"""
function simulate_iceflow_PDE(
    simulation::SIM,
    cb::SciMLBase.DECallback;
    du = SIA2D) where {SIM <: Simulation}
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
    @assert params.simulation.test_mode || iceflow_sol.retcode==ReturnCode.Success "There was an error in the iceflow solver. Returned code is \"$(iceflow_sol.retcode)\""

    # @show iceflow_sol.destats
    # Compute average ice surface velocities for the simulated period
    cache.iceflow.H .= iceflow_sol.u[end]
    map!(x -> ifelse(x>0.0,x,0.0), cache.iceflow.H, cache.iceflow.H)

    # Average surface velocity
    avg_surface_V!(simulation, iceflow_sol.t[end])

    glacier_idx = cache.iceflow.glacier_idx
    glacier::Sleipnir.Glacier2D = simulation.glaciers[glacier_idx[]]

    # Surface topography
    @. cache.iceflow.S = glacier.B + cache.iceflow.H

    # Update simulation results
    results = Sleipnir.create_results(simulation, glacier_idx[], iceflow_sol, nothing; light=!params.solver.save_everystep, processVelocity=V_from_H)

    return results
end

function apply_MB_mask!(H, glacier::G, ifm) where {G <: Sleipnir.AbstractGlacier}
    # Appy MB only over ice, and avoid applying it to the borders in the accummulation area to avoid overflow
    MB, MB_mask, MB_total = ifm.MB, ifm.MB_mask, ifm.MB_total
    MB_mask .= ((H .> 0.0) .&& (MB .< 0.0)) .|| ((H .> 10.0) .&& (MB .>= 0.0))
    H[MB_mask] .+= MB[MB_mask]
    MB_total[MB_mask] .+= MB[MB_mask]
    return nothing # For type stability
end
