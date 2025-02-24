function plot_analysis_flow_parameters_test()

    working_dir = joinpath(dirname(Base.current_project()), "data")
    if !ispath(working_dir)
        mkdir("data")
    end

    rgi_paths = get_rgi_paths()

    params = Huginn.Parameters(simulation = SimulationParameters(use_MB=true,
                                                          tspan=(2010.0, 2015.0),
                                                          working_dir = working_dir,
                                                          test_mode = true,
                                                          multiprocessing=false,
                                                          workers=1,
                                                          rgi_paths=rgi_paths),
                        solver = SolverParameters(reltol=1e-8)
                        )
    model = Huginn.Model(iceflow = SIA2Dmodel(params), mass_balance = TImodel1(params))
    rgi_ids = ["RGI60-11.01450"]
    glaciers = initialize_glaciers(rgi_ids, params)

    # Test for valid input
    prediction = Prediction(model, glaciers, params)

    A_values = [8.5e-20]
    n_values = [3.0]


    try
        plot_analysis_flow_parameters(prediction, A_values, n_values)
        @test true  # Test passes if no error is thrown
    catch e
        println("Error occurred: ", e)
        @test false  # Test fails if any error is caught

    end

end


function make_thickness_video_test()
    rgi_ids = ["RGI60-11.03646"]
    rgi_paths = get_rgi_paths()
    working_dir = joinpath(Huginn.root_dir, "test/data")

    params = Parameters(
        simulation = SimulationParameters(
            use_MB = true,
            use_iceflow = true,
            velocities = true,
            use_glathida_data = false,
            tspan = (2014.0, 2017.0),
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

    tempPath = mktempdir()*".mp4"

    plot_glacier_vid("thickness", prediction.results[1], prediction.glaciers[1], prediction.parameters.simulation, tempPath; baseTitle="Bossons glacier")
end
