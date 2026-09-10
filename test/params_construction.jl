
function params_constructor_specified(save_refs::Bool = false)
    solver_params = SolverParameters(
        solver = Ralston(),
        reltol = 1e-8,
        step = 1.0/12.0,
        save_everystep = false,
        tstops = Vector{Float64}(),
        progress = true,
        progress_steps = 10
    )
    JET.@test_opt SolverParameters(
        solver = Ralston(),
        reltol = 1e-8,
        step = 1.0/12.0,
        save_everystep = false,
        tstops = Vector{Float64}(),
        progress = true,
        progress_steps = 10
    )
    @test check_concrete_types(solver_params; show = false)
    @test check_field_types(typeof(solver_params); show = false)

    # Test prints
    println(solver_params)

    if save_refs
        jldsave(joinpath(Huginn.root_dir, "test/data/params/solver_params_specified.jld2"); solver_params)
    end

    solver_params_ref = load(joinpath(Huginn.root_dir, "test/data/params/solver_params_specified.jld2"))["solver_params"]

    @test solver_params == solver_params_ref
end

function params_constructor_default(save_refs::Bool = false)
    solver_params = SolverParameters()
    JET.@test_opt SolverParameters()
    @test check_concrete_types(solver_params; show = false)
    @test check_field_types(typeof(solver_params); show = false)

    if save_refs
        jldsave(joinpath(Huginn.root_dir, "test/data/params/solver_params_default.jld2"); solver_params)
    end

    solver_params_ref = load(joinpath(Huginn.root_dir, "test/data/params/solver_params_default.jld2"))["solver_params"]

    @test solver_params == solver_params_ref
end

function effective_abstol_test()
    atol = 1e-3
    ref = Huginn.ABSTOL_REFERENCE_YEARS

    # Short runs keep the tolerance they were given, boundary included
    @test Huginn.effective_abstol(atol, (2010.0, 2011.0); verbose = false) == atol
    @test Huginn.effective_abstol(atol, (2010.0, 2010.0 + ref); verbose = false) == atol

    # Longer runs are tightened in proportion to the run length
    @test Huginn.effective_abstol(atol, (2010.0, 2010.0 + 2 * ref); verbose = false) ≈
          atol / 2
    @test Huginn.effective_abstol(atol, (1990.0, 2020.0); verbose = false) ≈ atol * ref / 30

    # Never loosened, and monotone in the run length
    prev = atol
    for years in (1.0, ref, 10.0, 30.0, 100.0)
        cur = Huginn.effective_abstol(atol, (2010.0, 2010.0 + years); verbose = false)
        @test cur <= atol
        @test cur <= prev
        prev = cur
    end

    # The float type of abstol is preserved
    @test Huginn.effective_abstol(Float32(atol), (1990.0, 2020.0); verbose = false) isa
          Float32

    # The flag round-trips and is respected by the constructor
    @test SolverParameters(scale_abstol = false).scale_abstol == false
    @test SolverParameters().scale_abstol == true
end
