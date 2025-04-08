

function pde_solve_test(; rtol::F, atol::F, save_refs::Bool=false, MB::Bool=false, fast::Bool=true) where {F <: AbstractFloat}

    println("PDE solving with MB = $MB")

    ## Retrieving gdirs and climate for the following glaciers
    ## Fast version includes less glacier to reduce the amount of downloaded files and computation time on GitHub CI
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

    params = Huginn.Parameters(simulation = SimulationParameters(use_MB=MB,
                                                          velocities=false,
                                                          tspan=(2010.0, 2015.0),
                                                          working_dir = Huginn.root_dir,
                                                          test_mode = true,
                                                          rgi_paths = rgi_paths),
                        solver = SolverParameters(reltol=1e-12)
                        )

    if MB
        model = Huginn.Model(iceflow = SIA2Dmodel(params), mass_balance = TImodel1(params))
    else
        model = Huginn.Model(iceflow = SIA2Dmodel(params), mass_balance = nothing)
    end

    # We retrieve some glaciers for the simulation
    glaciers = initialize_glaciers(rgi_ids, params)

    # We create an ODINN prediction
    prediction = Prediction(model, glaciers, params)

    # We run the simulation
    @time run!(prediction)

    # /!\ Saves current run as reference values
    if save_refs
        if MB
            jldsave(joinpath(Huginn.root_dir, "test/data/PDE/PDE_refs_MB.jld2"); prediction.results)
        else
            jldsave(joinpath(Huginn.root_dir, "test/data/PDE/PDE_refs_noMB.jld2"); prediction.results)
        end
    end

    # Load reference values for the simulation
    if MB
        PDE_refs = load(joinpath(Huginn.root_dir, "test/data/PDE/PDE_refs_MB.jld2"))["results"]
    else
        PDE_refs = load(joinpath(Huginn.root_dir, "test/data/PDE/PDE_refs_noMB.jld2"))["results"]
    end

    let results=prediction.results

    for result in results
        let result=result, test_ref=nothing, UDE_pred=nothing
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
        MB ? vtol = 30.0*atol : vtol = 12.0*atol # a little extra tolerance for UDE surface velocities

        ### PDE ###
        plot_test_error(result, test_ref, "H",  result.rgi_id, atol, MB)
        plot_test_error(result, test_ref, "Vx", result.rgi_id, vtol, MB)
        plot_test_error(result, test_ref, "Vy", result.rgi_id, vtol, MB)

        # Test that the PDE simulations are correct
        @test all(isapprox.(result.H[end], test_ref.H[end], rtol=rtol, atol=atol))
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

    params = Huginn.Parameters(simulation = SimulationParameters(use_MB=true,
                                                          velocities=false,
                                                          tspan=(2010.0, 2015.0),
                                                          working_dir = Huginn.root_dir,
                                                          test_mode = true,
                                                          rgi_paths = rgi_paths),
                        solver = SolverParameters(reltol=1e-8)
                        )
    model = Huginn.Model(iceflow = SIA2Dmodel(params), mass_balance = TImodel1(params))

    glacier = initialize_glaciers(rgi_ids, params)[1]
    initialize_iceflow_model!(model.iceflow, 1, glacier, params)
    t = 2015.0

    MB_timestep!(model, glacier, params.solver.step, t)
    apply_MB_mask!(model.iceflow.H, glacier, model.iceflow)

    # /!\ Saves current run as reference values
    if save_refs
        jldsave(joinpath(Huginn.root_dir, "test/data/PDE/MB_ref.jld2"); model.iceflow.MB)
        jldsave(joinpath(Huginn.root_dir, "test/data/PDE/H_w_MB_ref.jld2"); model.iceflow.H)
    end

    MB_ref = load(joinpath(Huginn.root_dir, "test/data/PDE/MB_ref.jld2"))["MB"]
    H_w_MB_ref = load(joinpath(Huginn.root_dir, "test/data/PDE/H_w_MB_ref.jld2"))["H"]

    @test all(isapprox.(MB_ref, model.iceflow.MB, rtol=rtol, atol=atol))
    @test all(isapprox.(H_w_MB_ref, model.iceflow.H, rtol=rtol, atol=atol))

end
