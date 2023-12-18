function plot_analysis_flow_parameters_test()
    
    working_dir = joinpath(dirname(Base.current_project()), "data")
    if !ispath(working_dir)
        mkdir("data")
    end

    params = Parameters(OGGM = OGGMparameters(working_dir=working_dir,
                                              multiprocessing=true,
                                              workers=2,
                                              ice_thickness_source = "Farinotti19"),
                        simulation = SimulationParameters(use_MB=true,
                                                          use_iceflow= true,
                                                          tspan=(2000.0, 2015.0),
                                                          working_dir = working_dir,
                                                          test_mode = true),
                        solver = SolverParameters(reltol=1e-8)
                        ) 
    
    
    # Test for valid input
    A_values = [8.5e-20]
    n_values = [3.0]
    rgi_ids = ["RGI60-11.01450"]
    
    try
        plot_analysis_flow_parameters(params, A_values, n_values, rgi_ids)
        @test true  # Test passes if no error is thrown
    catch e
        println("Error occurred: ", e)
        @test false  # Test fails if any error is caught
        
    end

end

