using Revise
using Huginn


rgi_ids = ["RGI60-11.03638"] #"RGI60-11.01450"] , "RGI60-08.00213", "RGI60-04.04351", "RGI60-01.02170"]

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
run!(prediction)