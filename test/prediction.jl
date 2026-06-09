function test_prediction()
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
    model = Model(iceflow = SIA2Dmodel(params), mass_balance = TImodel1(params))
    JET.@test_opt Model(iceflow = SIA2Dmodel(params), mass_balance = TImodel1(params))
    @test check_concrete_types(model; show = false)
    @test check_field_types(typeof(model); show = false)

    glacier_idx = 1

    glaciers = initialize_glaciers(rgi_ids, params)
    JET.@test_opt target_modules=(Sleipnir, Muninn, Huginn) initialize_glaciers(rgi_ids, params)[1]
    @test check_concrete_types(glaciers; show = false)
    @test check_field_types(typeof(glaciers); show = false)

    simulation = Prediction(model, glaciers, params)
    JET.@test_opt target_modules=(Sleipnir, Muninn, Huginn) Prediction(model, glaciers, params)
    @test check_concrete_types(simulation; show = false)
    @test_broken check_field_types(typeof(simulation); show = false)

    cache = init_cache(model, simulation, glacier_idx, nothing)
    JET.@test_opt init_cache(model, simulation, glacier_idx, nothing)
    @test check_concrete_types(cache; show = false)
    @test_broken check_field_types(typeof(cache); show = false)
end
