
function pde_solve_test(; rtol::F, atol::F, save_refs::Bool=false, MB::Bool=false, fast::Bool=true, laws = nothing, callback_laws = false) where {F <: AbstractFloat}

    println("PDE solving with MB = $MB, laws = $laws, callback_laws = $callback_laws")

    ## Retrieving gdirs and climate for the following glaciers
    ## Fast version includes less glacier to reduce computation time on GitHub CI
    if fast
        rgi_ids = ["RGI60-11.03638", "RGI60-11.01450"] #, "RGI60-08.00213", "RGI60-04.04351", "RGI60-01.02170"]
    else
        rgi_ids = ["RGI60-11.03638", "RGI60-11.01450", "RGI60-08.00213", "RGI60-04.04351", "RGI60-01.02170",
        "RGI60-02.05098", "RGI60-01.01104", "RGI60-01.09162", "RGI60-01.00570", "RGI60-04.07051",
        "RGI60-07.00274", "RGI60-07.01323",  "RGI60-01.17316"]
    end

    rgi_paths = get_rgi_paths()
    # Filter out glaciers that are not used to avoid having references that depend on all the glaciers processed in Gungnir
    rgi_paths = Dict(k => rgi_paths[k] for k in rgi_ids)

    params = Huginn.Parameters(
        simulation = SimulationParameters(
            use_MB = MB,
            use_velocities = false,
            tspan = (2010.0, 2015.0),
            working_dir = Huginn.root_dir,
            test_mode = true,
            rgi_paths = rgi_paths
        ),
        solver = SolverParameters(reltol=1e-12)
    )
    JET.@test_opt target_modules=(Sleipnir,Muninn,Huginn) Huginn.Parameters(
        simulation = SimulationParameters(
            use_MB = MB,
            use_velocities = false,
            tspan = (2010.0, 2015.0),
            working_dir = Huginn.root_dir,
            test_mode = true,
            rgi_paths = rgi_paths
        ),
        solver = SolverParameters(reltol=1e-12)
    )

    mass_balance = isnothing(MB) ? nothing : TImodel1(params)

    A_law = if isnothing(laws)
        nothing
    elseif laws == :scalar
        # dumb law that gives the default value of A as a 0-dimensional array
        Law{ScalarCacheNoVJP}(;
            f! = (A, sim, glacier_idx, t, θ) -> A.value .= sim.glaciers[glacier_idx].A,
            init_cache = (sim, glacier_idx, θ) -> ScalarCacheNoVJP(zeros()),
            callback_freq = callback_laws ? 1/12 : nothing
        )
    elseif laws == :matrix
        # dumb law that gives the default value of A as a constant matrix
        Law{MatrixCacheNoVJP}(;
            f! = (A, sim, glacier_idx, t, θ) -> A.value .= sim.glaciers[glacier_idx].A,
            init_cache = function (sim, glacier_idx, θ)
                (;nx, ny) = sim.glaciers[glacier_idx]
                return MatrixCacheNoVJP(zeros(nx - 1, ny - 1))
            end,
            callback_freq = callback_laws ? 1/12 : nothing
        )
    else
        throw("laws keyword should be either nothing, :scalar, or :matrix")
    end

    # for now C is not used in SIA2D
    C_law = nothing

    iceflow = SIA2Dmodel(params; A = A_law, C = C_law)
    JET.@test_opt SIA2Dmodel(params; A = A_law, C = C_law)

    model = Huginn.Model(;iceflow, mass_balance)
    JET.@test_opt Huginn.Model(;iceflow, mass_balance)

    # We retrieve some glaciers for the simulation
    glaciers = initialize_glaciers(rgi_ids, params)
    # JET.@test_opt broken=true target_modules=(Sleipnir,Muninn,Huginn) initialize_glaciers(rgi_ids, params) # For the moment this is not type stable because of the readings (type of CSV files and RasterStack cannot be determined at compilation time)

    # We create an ODINN prediction
    prediction = Prediction(model, glaciers, params)

    JET.@test_opt target_modules=(Sleipnir, Muninn, Huginn) Prediction(model, glaciers, params)

    # We run the simulation
    @time run!(prediction)

    # Test below is not ready yet
    JET.@test_opt broken=true target_modules=(Sleipnir,Muninn,Huginn) Huginn.batch_iceflow_PDE!(1, prediction) # Call only the core of run! because saving to JLD2 file is not type stable and GC interferes with JET

    file_name = @match (MB, laws, callback_laws) begin
        (false, nothing, false) => "PDE_refs_noMB"
        (true, nothing, false) => "PDE_refs_MB"
        (true, :scalar, false) => "PDE_refs_MB_law"
        (true, :scalar, true) => "PDE_refs_MB_law"
        (true, :matrix, false) => "PDE_refs_MB_law"
        (true, :matrix, true) => "PDE_refs_MB_law"
    end

    # /!\ Saves current run as reference values
    if save_refs
        jldsave(joinpath(Huginn.root_dir, "test/data/PDE/$(file_name).jld2"); prediction.results)
    end

    # Load reference values for the simulation
    PDE_refs = load(joinpath(Huginn.root_dir, "test/data/PDE/$(file_name).jld2"))["results"]

    let results=prediction.results

    for result in results
        let result=result, test_ref=nothing
        for PDE_ref in PDE_refs
            if result.rgi_id == PDE_ref.rgi_id
                test_ref = PDE_ref
            end
        end

        ##############################
        #### Make plots of errors ####
        ##############################
        test_plot_path = joinpath(Huginn.root_dir, "test/plots")
        if !isdir(test_plot_path)
            mkdir(test_plot_path)
        end
        MB ? vtol = 30.0*atol : vtol = 12.0*atol # a little extra tolerance for surface velocities

        ### PDE ###
        plot_test_error(result, test_ref, "H",  result.rgi_id, atol, MB)
        plot_test_error(result, test_ref, "Vx", result.rgi_id, vtol, MB)
        plot_test_error(result, test_ref, "Vy", result.rgi_id, vtol, MB)

        # Test that the PDE simulations are correct
        mask = test_ref.H[end] .> 0.0
        @test all(isapprox.(result.H[end][mask], test_ref.H[end][mask], rtol=rtol, atol=atol))
        @test all(isapprox.(result.Vx, test_ref.Vx, rtol=rtol, atol=vtol))
        @test all(isapprox.(result.Vy, test_ref.Vy, rtol=rtol, atol=vtol))

        end # let
    end
    end
end

function TI_run_test!(save_refs::Bool = false; rtol::F, atol::F) where {F <: AbstractFloat}

    rgi_ids = ["RGI60-11.03638"]

    rgi_paths = get_rgi_paths()
    # Filter out glaciers that are not used to avoid having references that depend on all the glaciers processed in Gungnir
    rgi_paths = Dict(k => rgi_paths[k] for k in rgi_ids)

    params = Huginn.Parameters(
        simulation = SimulationParameters(
            use_MB = true,
            use_velocities = false,
            tspan = (2010.0, 2015.0),
            working_dir = Huginn.root_dir,
            test_mode = true,
            rgi_paths = rgi_paths
        ),
        solver = SolverParameters(reltol=1e-8)
    )
    model = Huginn.Model(iceflow = SIA2Dmodel(params), mass_balance = TImodel1(params))
    JET.@test_opt Huginn.Model(iceflow = SIA2Dmodel(params), mass_balance = TImodel1(params))

    glacier_idx = 1

    glaciers = initialize_glaciers(rgi_ids, params)
    JET.@test_opt target_modules=(Sleipnir,Muninn,Huginn) initialize_glaciers(rgi_ids, params)[1]

    glacier = glaciers[glacier_idx]

    # fake simulation
    simulation = (;model, glaciers)

    cache = init_cache(model, simulation, glacier_idx, nothing)
    JET.@test_opt init_cache(model, simulation, glacier_idx, nothing)

    t = 2015.0

    MB_timestep!(cache, model, glacier, params.solver.step, t)
    JET.@test_opt target_modules=(Sleipnir,Muninn,Huginn) MB_timestep!(cache, model, glacier, params.solver.step, t) # RasterStack manipulation is type unstable, so for the moment this test is deactivated

    apply_MB_mask!(cache.iceflow.H, cache.iceflow)
    JET.@test_opt target_modules=(Sleipnir,Muninn,Huginn) apply_MB_mask!(cache.iceflow.H, cache.iceflow)

    # /!\ Saves current run as reference values
    if save_refs
        jldsave(joinpath(Huginn.root_dir, "test/data/PDE/MB_ref.jld2"); cache.iceflow.MB)
        jldsave(joinpath(Huginn.root_dir, "test/data/PDE/H_w_MB_ref.jld2"); cache.iceflow.H)
    end

    MB_ref = load(joinpath(Huginn.root_dir, "test/data/PDE/MB_ref.jld2"))["MB"]
    H_w_MB_ref = load(joinpath(Huginn.root_dir, "test/data/PDE/H_w_MB_ref.jld2"))["H"]

    @test all(isapprox.(MB_ref, cache.iceflow.MB, rtol=rtol, atol=atol))
    @test all(isapprox.(H_w_MB_ref, cache.iceflow.H, rtol=rtol, atol=atol))

end

function ground_truth_generation()
    rgi_ids = ["RGI60-11.03638", "RGI60-11.01450"]
    tspan = (2010.0, 2012.0)
    δt = 1/12
    params = Huginn.Parameters(
        simulation = SimulationParameters(
            use_MB = true,
            use_velocities = false,
            tspan = tspan,
            working_dir = Huginn.root_dir,
            test_mode = true,
            rgi_paths = get_rgi_paths()
        ),
        solver = SolverParameters(
            reltol=1e-8,
            save_everystep=true,
        ),
    )
    model = Huginn.Model(iceflow = SIA2Dmodel(params), mass_balance = TImodel1(params))
    glaciers = initialize_glaciers(rgi_ids, params)
    tstops = collect(tspan[1]:δt:tspan[2])
    generate_ground_truth(glaciers, params, model, tstops)
end
