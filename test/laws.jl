function laws_constructor_default()
    # Test default constructors for law inputs
    temp = iAvgScalarTemp()
    cpdd = iCPDD()
    h̄ = iH̄()
    ∇S = i∇S()
    topo_rough = iTopoRough()
    @test typeof(temp) == iAvgScalarTemp
    @test typeof(cpdd) == iCPDD{Week}
    @test typeof(h̄) == iH̄
    @test typeof(∇S) == i∇S
    @test typeof(topo_rough) == iTopoRough{Float64}
    # Test default law constructors
    A_cuffey = CuffeyPaterson(scalar = true)
    @test isdefined(A_cuffey, :f)
    # For SyntheticC, need dummy params
    params = Parameters(simulation = SimulationParameters(test_mode = true))
    C_syn = SyntheticC(params; inputs = (; CPDD = cpdd, topo_roughness = topo_rough))
    # Tests on fields of SyntheticC
    @test isdefined(C_syn, :f)
    @test hasfield(typeof(C_syn), :name)
    @test C_syn.name == :SyntheticC
    @test hasfield(typeof(C_syn), :min_value)
    @test hasfield(typeof(C_syn), :max_value)
    @test hasfield(typeof(C_syn), :callback_freq)
end

function laws_constructor_specified()
    # Test constructors with specified values for law inputs
    temp = iAvgScalarTemp()
    cpdd = iCPDD(window = Week(1))
    h̄ = iH̄()
    ∇S = i∇S()
    topo_rough = iTopoRough(window = 300.0)
    @test typeof(temp) == iAvgScalarTemp
    @test typeof(cpdd) == iCPDD{Week}
    @test typeof(h̄) == iH̄
    @test typeof(∇S) == i∇S
    @test typeof(topo_rough) == iTopoRough{Float64}
    # Test default law constructors
    A_cuffey = CuffeyPaterson(scalar = true)
    @test isdefined(A_cuffey, :f)
    # For SyntheticC, need dummy params
    params = Parameters(simulation = SimulationParameters(test_mode = true))
    C_syn = SyntheticC(params; inputs = (; CPDD = cpdd, topo_roughness = topo_rough))
    # Tests on fields of SyntheticC
    @test isdefined(C_syn, :f)
    @test hasfield(typeof(C_syn), :name)
    @test C_syn.name == :SyntheticC
    @test hasfield(typeof(C_syn), :min_value)
    @test hasfield(typeof(C_syn), :max_value)
    @test hasfield(typeof(C_syn), :callback_freq)
end

function test_SyntheticC()
    # Dummy structs for simulation and glacier
    rgi_ids = ["RGI60-11.03638"]
    rgi_paths = get_rgi_paths()
    params = Parameters(simulation = SimulationParameters(test_mode = true,
        use_velocities = false,
        rgi_paths = rgi_paths))
    model = Model(iceflow = SIA2Dmodel(params), mass_balance = nothing)
    glaciers = initialize_glaciers(rgi_ids, params)
    simulation = Prediction(model, glaciers, params)
    # Create dummy normalized inputs
    law_inputs = (; CPDD = iCPDD(), topo_roughness = iTopoRough())
    C_syn = SyntheticC(params; inputs = law_inputs)
    # Test init_cache
    cache = C_syn.init_cache(simulation, 1, nothing)
    @test size(cache.value) == size(glaciers[1].S) .- 1
    # Test f!
    apply_law!(C_syn, cache, simulation, 1, 2010.0, nothing)
    @test all(cache.value .>= params.physical.minC)
    @test all(cache.value .<= params.physical.maxC)
end

function test_sliding_split()
    # At C = sliding_scale, sliding and deformation give the same flux
    A, H = 5e-17, 300.0
    C = Huginn.sliding_scale(A, H)
    @test Huginn.sliding_fraction([C], [H], A; basis = :flux)[1] ≈ 0.5
    @test isnan(Huginn.sliding_fraction([C], [0.0], A)[1])
    @test_throws ArgumentError Huginn.sliding_scale(A, H; p = 2.0)

    # The two parts add up to the velocity of `surface_V`
    rgi_ids = ["RGI60-11.03638"]
    params = Parameters(simulation = SimulationParameters(test_mode = true,
        use_velocities = false, rgi_paths = get_rgi_paths()))
    model = Model(iceflow = SIA2Dmodel(params), mass_balance = nothing)
    glaciers = initialize_glaciers(rgi_ids, params)
    glacier = glaciers[1]
    # q ≠ 0, so that the exponents p - q of the sliding term are tested
    glacier.q = 1.0
    glacier.C = Huginn.sliding_scale(glacier.A, H; p = glacier.p, q = glacier.q)
    simulation = Prediction(model, glaciers, params)
    simulation.cache = init_cache(model, simulation, 1, nothing)
    Vx, Vy = Huginn.surface_V(glacier.H₀, simulation, 2010.0, nothing)
    (; V_slide, V_deform) = Huginn.velocity_split(glacier.H₀, simulation, 2010.0, nothing)
    @test any(V_slide .> 0)
    @test V_slide .+ V_deform≈sqrt.(Vx .^ 2 .+ Vy .^ 2) rtol=1e-5

    # Their depth average is the velocity of the flux of `SIA2D!`, D / H̄ * ∇S
    Huginn.SIA2D!(zero(glacier.H₀), glacier.H₀, simulation, 2010.0, nothing)
    D, H̄,
    ∇S = copy.((simulation.cache.iceflow.D, simulation.cache.iceflow.H̄,
        simulation.cache.iceflow.∇S))
    flux = Huginn.velocity_split(glacier.H₀, simulation, 2010.0, nothing; basis = :flux)
    ice = H̄ .> 0
    @test (flux.V_slide .+ flux.V_deform)[ice] ≈ (D ./ H̄ .* ∇S)[ice]
end

function test_velocity_Y_U_laws()
    params = Parameters(simulation = SimulationParameters(test_mode = true,
        use_velocities = false, rgi_paths = get_rgi_paths()))
    glaciers = initialize_glaciers(["RGI60-11.03638"], params)
    H = glaciers[1].H₀
    scalar_law(value) = Law{ScalarCacheNoVJP}(;
        f! = (cache, sim, glacier_idx, t, θ) -> cache.value .= value,
        init_cache = (sim, glacier_idx, θ) -> ScalarCacheNoVJP(zeros()),
        callback_freq = nothing)
    function velocity(; laws...)
        model = Model(iceflow = SIA2Dmodel(params; laws...), mass_balance = nothing)
        simulation = Prediction(model, deepcopy(glaciers), params)
        simulation.cache = init_cache(model, simulation, 1, nothing)
        # `surface_V_inplace!` has its own copy of the formulas
        Vx, Vy = zero(H), zero(H)
        Huginn.surface_V_inplace!(Vx, Vy, H, simulation, 2010.0, nothing)
        Vx_ref, Vy_ref, _ = Huginn.V_from_H(simulation, H, 2010.0, nothing)
        @test Vx ≈ Vx_ref
        @test Vy ≈ Vy_ref
        return simulation, Huginn.surface_V(H, simulation, 2010.0, nothing)
    end

    # A Y law with the value of A gives the depth averaged velocity of the A law
    simulation_A, _ = velocity()
    simulation_Y, (Vx, Vy) = velocity(Y = scalar_law(glaciers[1].A))
    flux = Huginn.velocity_split(H, simulation_A, 2010.0, nothing; basis = :flux)
    @test sqrt.(Vx .^ 2 .+ Vy .^ 2)≈flux.V_slide .+ flux.V_deform rtol=1e-5
    @test any(Vx .!= 0)

    # A U law gives the velocity U / f along the slope, and cannot be split
    U₀ = 100.0
    simulation_U, (Vx, Vy) = velocity(U = scalar_law(U₀))
    f = params.simulation.f_surface_velocity_factor
    @test Vx ≈ -(U₀ / f) .* simulation_U.cache.iceflow.∇Sx
    @test Vy ≈ -(U₀ / f) .* simulation_U.cache.iceflow.∇Sy
    @test_throws ArgumentError Huginn.velocity_split(H, simulation_U, 2010.0, nothing)
end

function test_iTopoRough()
    # Dummy structs for simulation and glacier
    rgi_ids = ["RGI60-11.03638"]
    rgi_paths = get_rgi_paths()
    params = Parameters(simulation = SimulationParameters(test_mode = true,
        use_velocities = false,
        rgi_paths = rgi_paths))
    model = Model(iceflow = SIA2Dmodel(params), mass_balance = nothing)
    glaciers = initialize_glaciers(rgi_ids, params)
    simulation = Prediction(model, glaciers, params)

    topo_rough = iTopoRough(window = 200.0)
    roughness = get_input(topo_rough, simulation, 1, 2010.0)
    zero_roughness = zero(topo_rough, simulation, 1)
    @test size(roughness) == size(glaciers[1].S)
    @test typeof(zero_roughness) == typeof(roughness)
    @test size(zero_roughness) == size(roughness)

    topo_rough = iTopoRough(window = 400.0, curvature_type = :variability, direction = :flow)
    roughness = get_input(topo_rough, simulation, 1, 2010.0)
    zero_roughness = zero(topo_rough, simulation, 1)
    @test size(roughness) == size(glaciers[1].S)
    @test typeof(zero_roughness) == typeof(roughness)
    @test size(zero_roughness) == size(roughness)
    @test all(roughness .>= 0)
end

function test_iCPDD()
    # Dummy structs for simulation and glacier
    rgi_ids = ["RGI60-11.03638"]
    rgi_paths = get_rgi_paths()
    params = Parameters(simulation = SimulationParameters(test_mode = true,
        use_velocities = false,
        rgi_paths = rgi_paths))
    model = Model(iceflow = SIA2Dmodel(params), mass_balance = nothing)
    glaciers = initialize_glaciers(rgi_ids, params)
    simulation = Prediction(model, glaciers, params)

    cpdd = iCPDD(window = Week(1))
    pdd = get_input(cpdd, simulation, 1, 2010.0)
    zero_pdd = zero(cpdd, simulation, 1)
    @test size(pdd) == size(glaciers[1].S)
    @test typeof(zero_pdd) == typeof(pdd)
    @test size(zero_pdd) == size(pdd)
    @test all(pdd .>= 0.0)
end
