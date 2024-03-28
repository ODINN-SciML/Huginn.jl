function plot_analysis_flow_parameters_test()
    
    working_dir = joinpath(dirname(Base.current_project()), "data")
    if !ispath(working_dir)
        mkdir("data")
    end

    params = Parameters(OGGM = OGGMparameters(working_dir=working_dir,
                                              multiprocessing=false,
                                              workers=1,
                                              ice_thickness_source = "Farinotti19"),
                        simulation = SimulationParameters(use_MB=true,
                                                          tspan=(2010.0, 2015.0),
                                                          working_dir = working_dir,
                                                          multiprocessing=false,
                                                          workers=1),
                        solver = SolverParameters(reltol=1e-8)
                        ) 
    model = Model(iceflow = SIA2Dmodel(params), mass_balance = TImodel1(params))
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

