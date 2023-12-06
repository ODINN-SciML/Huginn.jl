function plot_analysis_flow_parameters_test()
    # Test for valid input
    tspan = (2000.0, 2005.0)
    A_values = [8.5e-20]
    n_values = [3.0]
    rgi_ids = ["RGI60-11.01450"]
    
    try
        plot_analysis_flow_parameters(tspan, A_values, n_values, rgi_ids)
        @test true  # Test passes if no error is thrown
    catch e
        println("Error occurred: ", e)
        @test false  # Test fails if any error is caught
        
    end

end

