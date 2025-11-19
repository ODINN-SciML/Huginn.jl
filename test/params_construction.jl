
function params_constructor_specified(save_refs::Bool = false)

    solver_params = SolverParameters(
        solver = Ralston(),
        reltol = 1e-8,
        step= 1.0/12.0,
        save_everystep = false,
        tstops = Vector{Float64}(),
        progress = true,
        progress_steps = 10
    )
    JET.@test_opt SolverParameters(
        solver = Ralston(),
        reltol = 1e-8,
        step= 1.0/12.0,
        save_everystep = false,
        tstops = Vector{Float64}(),
        progress = true,
        progress_steps = 10
    )

    if save_refs
        jldsave(joinpath(Huginn.root_dir, "test/data/params/solver_params_specified.jld2"); solver_params)
    end

    solver_params_ref = load(joinpath(Huginn.root_dir, "test/data/params/solver_params_specified.jld2"))["solver_params"]

    @test solver_params == solver_params_ref

end

function params_constructor_default(save_refs::Bool = false)

    solver_params = SolverParameters()
    JET.@test_opt SolverParameters()

    if save_refs
        jldsave(joinpath(Huginn.root_dir, "test/data/params/solver_params_default.jld2"); solver_params)
    end

    solver_params_ref = load(joinpath(Huginn.root_dir, "test/data/params/solver_params_default.jld2"))["solver_params"]

    @test solver_params == solver_params_ref

end
