import Pkg
Pkg.activate(dirname(Base.current_project()))

using Huginn
using Sleipnir: DummyClimate2D
using BenchmarkTools
using Logging
Logging.disable_logging(Logging.Info)
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 10
using Random

println("# Performance benchmark")


Random.seed!(1234)

nx = 100
ny = 120
tspan = (2010.0, 2015.0)
A = 4e-17
n = 3.0
Δx = 1.
Δy = 1.3
params = Huginn.Parameters(
    simulation = SimulationParameters(
        use_MB=false,
        use_velocities=false,
        tspan=tspan,
        working_dir = Huginn.root_dir,
        test_mode = true,
    ),
    solver = SolverParameters(reltol=1e-12)
)
model = Huginn.Model(iceflow = SIA2Dmodel(params), mass_balance = nothing)

H₀ = abs.(randn(nx, ny))
B = abs.(randn(nx, ny))
S = H₀ + B
glacier = Glacier2D(rgi_id = "toy", climate = DummyClimate2D(), H₀ = H₀, S = S, B = B, A = A, n=n, Δx=Δx, Δy=Δy, nx=nx, ny=ny, C = 0.0)
glaciers = [glacier]

simulation = Prediction(model, glaciers, params)

glacier_idx = 1
simulation.cache = init_cache(model, simulation, glacier_idx, nothing)

H = H₀
t = tspan[1]

vecBackwardSIA2D = randn(size(H,1), size(H,2))

println("## Benchmark of SIA2D!")
dH = deepcopy(H)
trial = @benchmark Huginn.SIA2D!(dH, $H, simulation, $t, $nothing)
display(trial)
println("")

