function test_prediction_instantiation()
    rgi_ids = ["RGI60-11.03638", "RGI60-11.01450"]

    rgi_paths = get_rgi_paths()
    # Filter out glaciers that are not used to avoid having references that depend on all the glaciers processed in Gungnir
    rgi_paths = Dict(k => rgi_paths[k] for k in rgi_ids)

    params = Parameters(
        simulation = SimulationParameters(
            use_MB = true,
            use_velocities = false,
            tspan = (2010.0, 2015.0),
            working_dir = Huginn.root_dir,
            test_mode = true,
            rgi_paths = rgi_paths
        ),
        solver = SolverParameters(reltol = 1e-8)
    )
    JET.@test_opt target_modules=(Sleipnir, Muninn, Huginn) Parameters(
        simulation = SimulationParameters(
            use_MB = true,
            use_velocities = false,
            tspan = (2010.0, 2015.0),
            working_dir = Huginn.root_dir,
            test_mode = true,
            rgi_paths = rgi_paths
        ),
        solver = SolverParameters(reltol = 1e-8)
    )
    @test check_concrete_types(params; show = false)
    @test check_field_types(typeof(params); show = false)

    iceflow = SIA2Dmodel(params)
    JET.@test_opt SIA2Dmodel(params)
    @test check_concrete_types(iceflow; show = false)
    @test check_field_types(typeof(iceflow); show = false)

    model = Model(iceflow = iceflow, mass_balance = TImodel1(params))
    JET.@test_opt Model(iceflow = iceflow, mass_balance = TImodel1(params))
    @test check_concrete_types(model; show = false)
    @test check_field_types(typeof(model); show = false)

    glaciers = initialize_glaciers(rgi_ids, params)
    JET.@test_opt target_modules=(Sleipnir, Muninn, Huginn) initialize_glaciers(rgi_ids, params)[1]
    @test check_concrete_types(glaciers; show = false)
    @test check_field_types(typeof(glaciers); show = false)

    glacier_idx = 1

    simulation = Prediction(model, glaciers, params)
    # Not type-stable: calibrate_MB reads the climate RasterStack (same limitation as
    # initialize_glaciers / batch_iceflow_PDE! above — RasterStack eltype not known at compile time).
    JET.@test_opt broken=true target_modules=(Sleipnir, Muninn, Huginn) Prediction(model, glaciers, params)
    @test check_concrete_types(simulation; show = false)
    @test_broken check_field_types(typeof(simulation); show = false)

    cache = init_cache(model, simulation, glacier_idx, nothing)
    # Not type-stable: the mass balance cache precomputes the climate windows, which reads the
    # climate RasterStack (same limitation as Prediction above).
    JET.@test_opt broken=true init_cache(model, simulation, glacier_idx, nothing)
    @test check_concrete_types(cache; show = false)
    @test_broken check_field_types(typeof(cache); show = false)
end

function test_glacier_tspan()
    rgi_ids = ["RGI60-11.03638", "RGI60-11.01450"]
    rgi_paths = get_rgi_paths()
    rgi_paths = Dict(k => rgi_paths[k] for k in rgi_ids)
    # Not on the grid of the mass balance steps, so the last window ends after 2011.9
    window = (2010.55, 2011.9)

    # A time of the user that is outside the window of the second glacier
    t_user = 2010.2
    function make_params(tspan, glacier_tspans)
        Parameters(
            simulation = SimulationParameters(
                use_MB = true,
                calibrate_MB = false,
                use_velocities = false,
                tspan = tspan,
                glacier_tspans = glacier_tspans,
                working_dir = Huginn.root_dir,
                test_mode = true,
                rgi_paths = rgi_paths
            ),
            solver = SolverParameters(reltol = 1e-8, step = 1/12, tstops = [t_user])
        )
    end
    # The second glacier has its own window
    params = make_params((2010.0, 2012.0), Dict(rgi_ids[2] => window))
    # The same window for all glaciers
    params_ref = make_params(window, Dict{String, Tuple{Float64, Float64}}())

    glaciers = initialize_glaciers(rgi_ids, params)
    prediction = Prediction(
        Model(iceflow = SIA2Dmodel(params), mass_balance = TImodel1(params)),
        deepcopy(glaciers), params)
    prediction_ref = Prediction(
        Model(iceflow = SIA2Dmodel(params_ref), mass_balance = TImodel1(params_ref)),
        deepcopy(glaciers), params_ref)
    run!(prediction)
    run!(prediction_ref)

    @test prediction.results[1].t[[begin, end]] == [2010.0, 2012.0]
    @test t_user in prediction.results[1].t
    @test !(t_user in prediction.results[2].t)
    @test prediction.results[2].t == prediction_ref.results[2].t
    @test prediction.results[2].Vx ≈ prediction_ref.results[2].Vx
    @test prediction.results[2].t[[begin, end]] == [window...]
    @test prediction.results[2].H ≈ prediction_ref.results[2].H
    @test prediction.results[2].t_MB == prediction_ref.results[2].t_MB
    @test prediction.results[2].t_MB[end] == window[2]
    @test prediction.results[2].MB ≈ prediction_ref.results[2].MB

    # The ground truth uses the same tstops for all the glaciers, so it refuses windows
    model = Model(iceflow = SIA2Dmodel(params), mass_balance = TImodel1(params))
    @test_throws ArgumentError generate_ground_truth(
        deepcopy(glaciers), params, model, collect(2010.0:(1 / 12):2012.0))
    # A window outside tspan is refused when the simulation is built
    params.simulation.glacier_tspans[rgi_ids[2]] = (2009.0, 2011.0)
    @test_throws ArgumentError Prediction(model, deepcopy(glaciers), params)
end
