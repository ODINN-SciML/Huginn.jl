
function make_prediction_test()
    rgi_ids = ["RGI60-11.03646"]
    rgi_paths = get_rgi_paths()
    working_dir = joinpath(Huginn.root_dir, "test/data")

    params = Parameters(
        simulation = SimulationParameters(
            use_MB = true,
            use_iceflow = true,
            velocities = true,
            use_glathida_data = false,
            tspan = (2014.0, 2015.0),
            working_dir = working_dir,
            multiprocessing = true,
            workers = 1,
            rgi_paths = rgi_paths,
            ice_thickness_source = "Farinotti19",
        ),
        solver = SolverParameters(reltol = 1e-8, save_everystep = true)
    )

    model = Model(
        iceflow = SIA2Dmodel(params, C=0.),
        mass_balance = TImodel1(params)
    )

    glaciers = initialize_glaciers(rgi_ids, params)

    prediction = Prediction(model, glaciers, params)

    run!(prediction)

    jldsave(joinpath(Huginn.root_dir, "../Sleipnir/test/data/prediction/results2D_test.jld2"); prediction.results)
    jldsave(joinpath(Huginn.root_dir, "../Sleipnir/test/data/prediction/glaciers2D_test.jld2"); prediction.glaciers)
    jldsave(joinpath(Huginn.root_dir, "../Sleipnir/test/data/prediction/simuparams2D_test.jld2"); prediction.parameters.simulation)
end
