
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
    batch_iceflow_PDE!(glacier_idx::I, simulation::Prediction) 

Solve the Shallow Ice Approximation iceflow PDE for a given temperature series batch in-place.
"""
function batch_iceflow_PDE!(glacier_idx::I, simulation::Prediction) where {I <: Integer}

    model = simulation.model
    params = simulation.parameters
    glacier = simulation.glaciers[glacier_idx]

    glacier_id = isnothing(glacier.rgi_id) ? "unnamed" : glacier.rgi_id
    println("Processing glacier $(glacier_id) for PDE forward simulation")

    # Initialize glacier ice flow model
    initialize_iceflow_model!(model.iceflow, glacier_idx, glacier, params)
    params.solver.tstops = define_callback_steps(params.simulation.tspan, params.solver.step)
    stop_condition(u,t,integrator) = Sleipnir.stop_condition_tstops(u,t,integrator, params.solver.tstops) #closure
    function action!(integrator)
        if params.simulation.use_MB
            # Compute mass balance
            MB_timestep!(model, glacier, params.solver.step, integrator.t)
            apply_MB_mask!(integrator.u, glacier, model.iceflow)
        end
    end
    cb_MB = DiscreteCallback(stop_condition, action!)

    # Run iceflow PDE for this glacier
    du = params.simulation.use_iceflow ? SIA2D! : noSIA2D!
    results = simulate_iceflow_PDE!(simulation, cb_MB; du = du)

    return results
end

"""
    function simulate_iceflow_PDE!(
        simulation::SIM,
        cb::DiscreteCallback;
        du = SIA2D!) where {SIM <: Simulation}

Make forward simulation of the iceflow PDE determined in `du`.
"""
function simulate_iceflow_PDE!(
    simulation::SIM,
    cb::DiscreteCallback;
    du = SIA2D!) where {SIM <: Simulation}
    model = simulation.model
    params = simulation.parameters

    # Define problem to be solved
    iceflow_prob = ODEProblem{true,SciMLBase.FullSpecialize}(du, model.iceflow.H, params.simulation.tspan, simulation; tstops=params.solver.tstops)

    iceflow_sol = solve(iceflow_prob,
                        params.solver.solver,
                        callback=cb,
                        reltol=params.solver.reltol,
                        save_everystep=params.solver.save_everystep,
                        progress=params.solver.progress,
                        progress_steps=params.solver.progress_steps)

    # @show iceflow_sol.destats
    # Compute average ice surface velocities for the simulated period
    model.iceflow.H .= iceflow_sol.u[end]
    map!(x -> ifelse(x>0.0,x,0.0), model.iceflow.H, model.iceflow.H)

    # Average surface velocity
    avg_surface_V!(simulation)

    glacier_idx = simulation.model.iceflow.glacier_idx
    glacier::Sleipnir.Glacier2D = simulation.glaciers[glacier_idx[]]

    # Surface topography
    model.iceflow.S .= glacier.B .+ model.iceflow.H

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
    println("Processing glacier: ", glacier_id)
    
    # Initialize glacier ice flow model (don't needed for out-of-place? maybe a simplified version?)
    initialize_iceflow_model(model.iceflow, glacier_idx, glacier, params)

    params.solver.tstops = define_callback_steps(params.simulation.tspan, params.solver.step)
    stop_condition(u,t,integrator) = Sleipnir.stop_condition_tstops(u,t,integrator, params.solver.tstops) #closure
    function action!(integrator)
        if params.simulation.use_MB 
            # Compute mass balance
            MB_timestep!(model, glacier, params.solver.step, integrator.t)
            apply_MB_mask!(integrator.u, glacier, model.iceflow)
        end
    end
    
    cb_MB = DiscreteCallback(stop_condition, action!)

    # Run iceflow PDE for this glacier
    du = params.simulation.use_iceflow ? SIA2D : noSIA2D
    results = simulate_iceflow_PDE(simulation, cb_MB; du = du)

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
    cb::DiscreteCallback;
    du = SIA2D) where {SIM <: Simulation}
    model = simulation.model
    params = simulation.parameters

    # Define problem to be solved
    iceflow_prob = ODEProblem{false,SciMLBase.FullSpecialize}(du, model.iceflow.H, params.simulation.tspan, tstops=params.solver.tstops, simulation)
    iceflow_sol = solve(iceflow_prob, 
                        params.solver.solver, 
                        callback=cb, 
                        tstops=params.solver.tstops, 
                        reltol=params.solver.reltol, 
                        save_everystep=params.solver.save_everystep, 
                        progress=params.solver.progress, 
                        progress_steps=params.solver.progress_steps)
    # @show iceflow_sol.destats
    # Compute average ice surface velocities for the simulated period
    model.iceflow.H .= iceflow_sol.u[end]
    map!(x -> ifelse(x>0.0,x,0.0), model.iceflow.H, model.iceflow.H)

    # Average surface velocity
    Vx, Vy, V = avg_surface_V(simulation)

    # Since we are doing out-of-place, we need to add this to the result
    model.iceflow.Vx = Vx
    model.iceflow.Vy = Vy
    model.iceflow.V  = V

    glacier_idx = simulation.model.iceflow.glacier_idx
    glacier = simulation.glaciers[glacier_idx[]]

    # Surface topography
    model.iceflow.S = glacier.B .+ model.iceflow.H

    # Update simulation results
    results = Sleipnir.create_results(simulation, glacier_idx[], iceflow_sol, nothing; light=!params.solver.save_everystep, processVelocity=V_from_H)

    return results
end

function apply_MB_mask!(H, glacier::G, ifm::IceflowModel) where {G <: Sleipnir.AbstractGlacier}
    # Appy MB only over ice, and avoid applying it to the borders in the accummulation area to avoid overflow
    MB, MB_mask, MB_total = ifm.MB, ifm.MB_mask, ifm.MB_total
    MB_mask .= ((H .> 0.0) .&& (MB .< 0.0)) .|| ((H .> 10.0) .&& (MB .>= 0.0)) 
    H[MB_mask] .+= MB[MB_mask]
    MB_total[MB_mask] .+= MB[MB_mask]
end
