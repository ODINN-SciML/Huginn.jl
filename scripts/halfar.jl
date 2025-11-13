# Minimum working example of how to run forward simulation and compared with Halfar solution

# We set default values as they are used in Bueler et al (2005).

using Revise
using Sleipnir: make_thickness_video, DummyClimate2D
using Huginn
using GLMakie, CairoMakie

Δt = 1000.0
nx = 240
ny = 240

halfar_params = HalfarParameters(
        λ = 0.0, # This value controls the intensity of MB
        R₀ = 750000.0,
        H₀ = 3600.0,
        A = 1e-16,
        n = 3.0
    )

# Obtain analytical solution and initial time for simulation
halfar, t₀ = Halfar(halfar_params)
n_time = 800
δt = Δt / n_time
t₁ = t₀ + Δt

parameters = Huginn.Parameters(
    simulation=SimulationParameters(
        tspan = (t₀, t₁),
        multiprocessing = false,
        use_MB = false,
        use_iceflow = true,
        working_dir = Huginn.root_dir
        ),
    physical = PhysicalParameters(
        ρ = halfar_params.ρ,
        g = halfar_params.g
        ),
    solver = SolverParameters(
        reltol = 1e-12,
        step = δt,
        )
    )

model = Huginn.Model(
    iceflow = SIA2Dmodel(parameters),
    mass_balance = nothing
)

# Construct a grid that includes the initial Dome
η = 0.66
Δx = halfar_params.R₀ / nx / (η / 2)
Δy = halfar_params.R₀ / ny / (η / 2)
xs = [(i - nx / 2) * Δx for i in 1:nx]
ys = [(j - ny / 2) * Δy for j in 1:ny]
ts = Huginn.define_callback_steps((t₀, t₁), δt) |> collect

# Bed (it has to be flat for the Halfar solution)
B = zeros((nx,ny))
# Computed Halfar solution
Hs = [[halfar(x, y, t) for x in xs, y in ys] for t in ts]

glacier = Glacier2D(rgi_id = "Halfar", climate = DummyClimate2D(), H₀ = Hs[begin], S = B + Hs[begin], B = B,
    A = halfar_params.A, n = halfar_params.n, C = 0.0,
    Δx = Δx, Δy = Δy, nx = nx, ny = ny,
    )
glaciers = Vector{Sleipnir.AbstractGlacier}([glacier])

prediction = Prediction(model, glaciers, parameters)

run!(prediction)

# Make video
make_thickness_video(
    prediction.results[1].H,
    glaciers[1],
    parameters,
    "./scripts/figures/Halfar_sol_video.mp4";
    framerate = 24,
    baseTitle = "Halfar solution"
    )

# Error plot

Err = (prediction.results[1].H .- Hs)

make_thickness_video(
    Err,
    glaciers[1],
    parameters,
    "./scripts/figures/Halfar_error_video.mp4";
    framerate = 24,
    baseTitle = "Halfar error",
    colormap = :bam,
    colorrange = (-20.0, 20.0)
    )


# Solutions are radially symmetric, so we can just look at the profile.

index = Observable(1)

xs = Δx .* collect(1:nx)
ys_1 = @lift(Hs[$index][120, :])
ys_2 = @lift(prediction.results[1].H[$index][120, :])
err = @lift((Hs[$index][120, :] .- prediction.results[1].H[$index][120, :]))
# err = @lift(abs.(Hs[$index][120, :] .- prediction.results[1].H[$index][120, :]))

fig = CairoMakie.Figure(; size = (1000, 600))
ax = CairoMakie.Axis(
    fig[1, 1],
    title = @lift("t = $(round(ts[$index] - t₀, digits = 4))"),
    xlabel = "Distance (m)",
    ylabel = "Ice Thickness (m)"
    )

lines!(
    xs, ys_1,
    color = :purple4, linewidth = 4,
    label = "Halfar solution"
    )
lines!(
    xs, ys_2,
    color = :orangered1, linewidth = 4,
    linestyle = :dash,
    label = "Numerical solution"
    )
ax2 = Axis(fig[1, 1], yticklabelcolor = :red, yaxisposition = :right, ylabel = "Error (m)")
ylims!(ax2, -5.0, 5.0)
lines!(
    ax2,
    xs, err,
    color = :red,
    linewidth = 1,
    label = "Error (m)"
    );
hlines!(ax2, [0.0], color = :black)

fig[1, 2] = Legend(fig, ax)

framerate = 24
timestamps = collect(1:n_time)

record(fig, "./scripts/figures/Halfar_evolution_video.mp4", timestamps;
        framerate = framerate) do t
    index[] = t
end
