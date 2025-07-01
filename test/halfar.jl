"""
Test that analytical Halfar solution satisfies the SIA equation
"""
function unit_halfar_is_solution(; rtol = 1e-6, atol = 1e-6)

    halfar_param = HalfarParameters()

    n = halfar_param.n
    Γ = 2 * halfar_param.A * (halfar_param.ρ * halfar_param.g)^n / (n + 2)

    halfar, t₀ = Halfar(halfar_param)

    # Selection of grid points used for evaluate Halfar solution
    xs = halfar.R₀ .* [0.01, 0.1, -0.5]
    ys = halfar.R₀ .* [0.1, 0.4, 0.6]
    ts = t₀ .* [1.1, 100]

    all_tests = Bool[]
    # for x in xs, y in ys, t in ts
    for (x, y, t) in Iterators.product(xs, ys, ts)
        # Ice thickness
        h = halfar(x, y, t)
        # Variation in ice thickness
        ∂h∂t = ForwardDiff.derivative(_t -> halfar(x, y, _t), t)
        # Mass balance
        b = halfar_param.λ * h / t
        # Flux
        qx = - ForwardDiff.derivative(
                _x -> (Γ * halfar(_x, y, t)^(n + 2) * (
                    ForwardDiff.derivative(__x -> halfar(__x, y, t), _x)^2.0 +
                    ForwardDiff.derivative(__y -> halfar(_x, __y, t), y)^2.0
                )^((n - 1) / 2) *
                ForwardDiff.derivative(__x -> halfar(__x, y, t), _x)),
            x)
        qy = - ForwardDiff.derivative(
                _y -> Γ * halfar(x, _y, t)^(n + 2) * (
                    ForwardDiff.derivative(__x -> halfar(__x, _y, t), x)^2 +
                    ForwardDiff.derivative(__y -> halfar(x, __y, t), _y)^2
                )^((n - 1) / 2) *
                ForwardDiff.derivative(__y -> halfar(x, __y, t), _y),
            y)
        q = qx + qy

        push!(all_tests, isapprox(∂h∂t, b - q, rtol = rtol, atol = atol))
    end
    @test all(all_tests)
end


"""
    unit_halfar_test(; kwargs...) -> Nothing

Runs a unit test comparing a numerical shallow ice model (SIA) to the analytic Halfar
similarity solution introduced in Halfar (1983) and Bueler (2005).

This function sets up and runs a 2D shallow ice simulation over a flat bed using the
`HalfarParameters` and compares the model results against the exact Halfar solution.
The test checks absolute and relative errors in ice thickness, total mass, and dome height.
Fails if any error exceeds tolerance thresholds.

# Keyword Arguments
- `A::Float64=1e-16`: Glen flow law rate factor [Pa⁻ⁿ yr⁻¹].
- `n::Float64=3.0`: Glen flow law exponent.
- `Δt::Float64=25000.0`: Time span of the simulation [years].
- `nx::Int=240`, `ny::Int=240`: Number of grid points in `x` and `y` directions.
- `H₀::Float64=3600.0`: Dome height at t₀ [m].
- `R₀::Float64=750000.0`: Dome radius at t₀ [m].
- `use_MB::Bool=false`: If `true`, includes a mass balance term.
- `reltol::Float64=0.007`: Maximum allowed relative error.
- `abstol::Float64=5.0`: Maximum allowed absolute thickness error [m].
- `masstol::Float64=2e-4`: Maximum allowed relative error in total ice mass.
- `dometol::Float64=0.20`: Maximum allowed dome height error [m].
- `distance_to_border::Int=3`: Distance (in grid cells) from ice margin to exclude from error computation.
- `save_plot::Bool=false`: If `true`, saves diagnostic plots comparing analytical and simulated solutions.
- `inplace::Bool=true`: If `true`, runs the model using in-place updates (`run!`), otherwise uses a non-mutating solver (`run₀`).

# Outputs
- Returns `nothing`, but throws `@test` failures if errors exceed tolerance.
- Optionally saves a diagnostic figure to `test/halfar_test.png`.

# Notes
- For a full explanation of this test please check Bueler (2005) "Exact solutions and
verification of numerical models for isothermalice sheets", experiments B and C.
- Ice flow numerical models experience artifacts near the margin. This is the reason why
errors are evaluated away from the margin. The choice of distance_to_border = 3 coincides
with the one found in Bueler (2005). For this same reason, we evaluate both the maximum
error and the error at the dome (maximum), which allows to evaluate the error away from the
margin.
- Error tolerances are selected based on Bueler (2005), and if well they seem large, they
are in aggreement with state of the art numerical models.

# Example
```julia
unit_halfar_test()

"""
function unit_halfar_test(;
    A = 1e-16,
    n = 3.0,
    Δt = 25000.0,
    nx = 240,
    ny = 240,
    H₀ = 3600.0,
    R₀ = 750000.0,
    use_MB = false,
    reltol = 0.007,
    abstol = 5.0,
    masstol = 2e-4,
    dometol = 0.20,
    distance_to_border = 3,
    save_plot = false,
    inplace = true
    )

    if use_MB
        λ = 5.0
    else
        λ = 0.0
    end

    halfar_params = HalfarParameters(
        λ = λ,
        R₀ = R₀,
        H₀ = H₀,
        A = A,
        n = n
    )

    halfar, t₀ = Halfar(halfar_params)

    t₁ = t₀ + Δt
    δt = Δt / 10

    # Get parameters for a simulation
    parameters = Huginn.Parameters(
        simulation=SimulationParameters(
            tspan = (t₀, t₁),
            multiprocessing = false,
            use_MB = use_MB,
            step = δt,
            use_iceflow = true,
            working_dir = Huginn.root_dir
            ),
        physical = PhysicalParameters(
            ρ = halfar_params.ρ,
            g = halfar_params.g
            ),
        solver = SolverParameters(
            reltol = 1e-12,
            # abstol = 1e-12,
            step = δt,
            save_everystep = true
            )
        )

    @assert !use_MB "Need to find way to pass MB"
    model = Huginn.Model(
        iceflow = SIA2Dmodel(parameters),
        mass_balance = nothing
    )

    # Construct a grid that includes the initial Dome
    η = 0.66
    Δx = R₀ / nx / (η / 2)
    Δy = R₀ / ny / (η / 2)
    xs = [(i - nx / 2) * Δx for i in 1:nx]
    ys = [(j - ny / 2) * Δy for j in 1:ny]
    ts = Huginn.define_callback_steps((t₀, t₁), δt) |> collect

    # Bed (it has to be flat for the Halfar solution)
    B = zeros((nx,ny))
    # Computed Halfar solution
    Hs = [[halfar(x, y, t) for x in xs, y in ys] for t in ts]
    H₀ = Hs[begin]
    H₁ = Hs[end]

    # In order for the tests to make sense, we want to be sure the glacier is actually
    # changing over time:
    @assert maximum(H₀ - H₁) > 5.0 "The glacier is not changing significativelly between initial and final state."

    # Define glacier object
    glacier = Glacier2D(
        rgi_id = "Halfar",
        climate = DummyClimate2D(),
        H₀ = H₀,
        S = B + H₀,
        B = B,
        A = A,
        n = n,
        Δx = Δx,
        Δy = Δy,
        nx = nx,
        ny = ny,
        C = 0.0
        )
    glaciers = Vector{Sleipnir.AbstractGlacier}([glacier])

    prediction = Prediction(model, glaciers, parameters)

    if inplace
        run!(prediction)
    else
        run₀(prediction)
    end

    abs_errors = Float64[]
    rel_errors = Float64[]
    mass_errors = Float64[]
    dome_errors = Float64[]

    Hs_preds = prediction.results[1].H

    for i in 1:length(Hs_preds)
        H_diff = Hs[i] - Hs_preds[i]
        push!(abs_errors, maximum(abs.(H_diff[is_in_glacier(Hs[i], distance_to_border)])))
        push!(rel_errors, maximum(abs.((H_diff ./ Hs[i])[is_in_glacier(Hs[i], distance_to_border)])))
        push!(mass_errors, sum(H_diff) / sum(H₁) )
        push!(dome_errors, abs(maximum(Hs[i]) - maximum(Hs_preds[i])))
    end

    absolute_error = maximum(abs_errors)
    relative_error = maximum(rel_errors)
    mass_error = maximum(mass_errors)
    dome_error = maximum(dome_errors)

    # Optional plot
    if save_plot
        fig = Figure(resolution = (800, 800))

        Axis(fig[1, 1], title = "Initial Condition")
        CairoMakie.heatmap!(H₀, colormap=:viridis, colorrange=(0, maximum(H₀))) # Specify CairoMakie to remove ambiguity with Plots.heatmap!

        Axis(fig[1, 2], title = "Final State")
        CairoMakie.heatmap!(Hs[end], colormap=:viridis, colorrange=(0, maximum(H₀)))

        Axis(fig[2,1], title="Prediction")
        CairoMakie.heatmap!(Hs_preds[end], colormap=:viridis, colorrange=(0, maximum(H₀)))

        Axis(fig[2,2], title="Difference")
        CairoMakie.heatmap!(Hs_preds[end] - Hs[end], colormap=Reverse(:balance), colorrange=(-10, 10))

        save("plots/halfar_test.png", fig)
    end

    @show absolute_error
    @show relative_error
    @show mass_error
    @show dome_error

    @test absolute_error < abstol
    @test relative_error < reltol
    @test mass_error < masstol
    @test dome_error < dometol
end

"""
    function halfar_test()

Multiple tests using Halfar solution.
"""
function halfar_test()#; rtol, atol, inplace, distance_to_border)
    # Default test as introduced in experiment B in Bueler et. al (2005)
    unit_halfar_test()
    unit_halfar_test(inplace = false)
    # Mass balance
    # unit_halfar_test(use_MB = true)
    # Changing A
    unit_halfar_test(A = 1e-17, reltol = 1e-2, distance_to_border = 5)
    # Smaller glacier in shorter timescale
    unit_halfar_test(Δt = 100.0, H₀ = 200.0, R₀ = 3000.0, reltol = 9e-3, masstol = 1e-3)
    unit_halfar_test(Δt = 100.0, H₀ = 200.0, R₀ = 3000.0, reltol = 9e-3, masstol = 1e-3, nx = 100, ny = 100)
end
