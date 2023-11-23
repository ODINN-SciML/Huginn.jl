function glacier_analysis_test()
    # Add all individual analysis test functions
    plot_analysis_flow_parameters_test()
end

function plot_analysis_flow_parameters_test()
    # Test for valid input
    tspan = (2000.0, 2015.0)
    A_values = [8.5e-20, 8e-17]
    n_values = [3.0, 3.2]
    rgi_ids = ["RGI60-11.01450"]
    
  
    # Test for error with too many rows/cols
    A_values_large = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    n_values_large = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    @test_throws ErrorException plot_analysis_flow_parameters(tspan, A_values_large, n_values, rgi_ids)
    @test_throws ErrorException plot_analysis_flow_parameters(tspan, A_values, n_values_large, rgi_ids)

    # Test for error with too many rgi_ids
    rgi_ids_large = ["RGI60-11.01450", "RGI60-11.03638"]
    @test_throws ErrorException plot_analysis_flow_parameters(tspan, A_values, n_values, rgi_ids_large)
end

