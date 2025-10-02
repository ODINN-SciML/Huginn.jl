function laws_constructor_default()
    # Test default constructors for law inputs
    temp = iTemp()
    cpdd = iCPDD()
    h̄ = iH̄()
    ∇S = i∇S()
    topo_rough = iTopoRough()
    @test typeof(temp) == iTemp
    @test typeof(cpdd) == iCPDD{Week}
    @test typeof(h̄) == iH̄
    @test typeof(∇S) == i∇S
    @test typeof(topo_rough) == iTopoRough{Float64}
    # Test default law constructors
    A_cuffey = CuffeyPaterson()
    @test isdefined(A_cuffey, :f)
    # For SyntheticC, need dummy params
    params = Parameters(simulation=SimulationParameters(test_mode=true))
    C_syn = SyntheticC(params; inputs=(; CPDD=cpdd, topo_roughness=topo_rough))
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
    temp = iTemp()
    cpdd = iCPDD(window=Week(1))
    h̄ = iH̄()
    ∇S = i∇S()
    topo_rough = iTopoRough(window=300.0)
    @test typeof(temp) == iTemp
    @test typeof(cpdd) == iCPDD{Week}
    @test typeof(h̄) == iH̄
    @test typeof(∇S) == i∇S
    @test typeof(topo_rough) == iTopoRough{Float64}
    # Test default law constructors
    A_cuffey = CuffeyPaterson()
    @test isdefined(A_cuffey, :f)
    # For SyntheticC, need dummy params
    params = Parameters(simulation=SimulationParameters(test_mode=true))
    C_syn = SyntheticC(params; inputs=(; CPDD=cpdd, topo_roughness=topo_rough))
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
    # Filter out glaciers that are not used to avoid having references that depend on all the glaciers processed in Gungnir
    rgi_paths = Dict(k => rgi_paths[k] for k in rgi_ids)
    params = Parameters(simulation=SimulationParameters(test_mode=true, 
                                    use_velocities=false, 
                                    rgi_paths=rgi_paths))
    model = Model(iceflow=SIA2Dmodel(params), mass_balance=nothing)
    glaciers = initialize_glaciers(rgi_ids, params)
    simulation = Prediction(model, glaciers, params)
    # Create dummy normalized inputs
    law_inputs = (; CPDD=iCPDD(), topo_roughness=iTopoRough())
    C_syn = SyntheticC(params; inputs=law_inputs)
    # Test init_cache
    cache = C_syn.init_cache(simulation, 1, nothing)
    @test size(cache) == size(glaciers[1].S) .- 1
    # Test f!
    apply_law!(C_syn, cache, simulation, 1, 2010.0, nothing)
    @test all(cache .>= params.physical.minC)
    @test all(cache .<= params.physical.maxC)
end

function test_iTopoRough()
    # Dummy structs for simulation and glacier
    rgi_ids = ["RGI60-11.03638"]
    rgi_paths = get_rgi_paths()
    # Filter out glaciers that are not used to avoid having references that depend on all the glaciers processed in Gungnir
    rgi_paths = Dict(k => rgi_paths[k] for k in rgi_ids)
    params = Parameters(simulation=SimulationParameters(test_mode=true, 
                                    use_velocities=false, 
                                    rgi_paths=rgi_paths))
    model = Model(iceflow=SIA2Dmodel(params), mass_balance=nothing)
    glaciers = initialize_glaciers(rgi_ids, params)
    simulation = Prediction(model, glaciers, params)

    topo_rough = iTopoRough(window=200.0)
    roughness = get_input(topo_rough, simulation, 1, 2010.0)
    @test size(roughness) == size(glaciers[1].S)
    @test all(roughness .>= 0)
end

function test_iCPDD()
    # Dummy structs for simulation and glacier
    rgi_ids = ["RGI60-11.03638"]
    rgi_paths = get_rgi_paths()
    # Filter out glaciers that are not used to avoid having references that depend on all the glaciers processed in Gungnir
    rgi_paths = Dict(k => rgi_paths[k] for k in rgi_ids)
    params = Parameters(simulation=SimulationParameters(test_mode=true, 
                                    use_velocities=false, 
                                    rgi_paths=rgi_paths))
    model = Model(iceflow=SIA2Dmodel(params), mass_balance=nothing)
    glaciers = initialize_glaciers(rgi_ids, params)
    simulation = Prediction(model, glaciers, params)

    cpdd = iCPDD(window=Week(1))
    pdd = get_input(cpdd, simulation, 1, 2010.0)
    @test size(pdd) == size(glaciers[1].S)
    @test all(pdd .>= 0.0)
end