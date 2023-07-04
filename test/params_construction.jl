
function params_constructor_specified(save_refs::Bool = false)

    solver_params = SolverParameters(solver = Ralston(),
                                    reltol = 1e-8,
                                    step= 1.0/12.0,
                                    save_everystep = false,
                                    tstops = nothing,
                                    progress = true,
                                    progress_steps = 10)

    if save_refs
        jldsave(joinpath(Sleipnir.root_dir, "test/data/params/solver_params.jld2"); solver_params)
    end
                    

end

function params_constructor_default()


end