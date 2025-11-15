using DiffEqCallbacks: PeriodicCallback, PresetTimeCallback
import Sleipnir: init_cache, cache_type

export SIA2Dmodel, SIA2DCache

###############################################
###### SHALLOW ICE APPROXIMATION MODELS #######
###############################################

"""
    SIA2Dmodel(A, C, n, Y, U, n_H, n_∇S)
    SIA2Dmodel(params; A, C, n, Y, U, n_H, n_∇S)

Create a `SIA2Dmodel`, representing a two-dimensional Shallow Ice Approximation (SIA) model.

The SIA model describes glacier flow under the assumption that deformation and basal sliding dominate the ice dynamics. It relies on:
- Glen's flow law for internal deformation, with flow rate factor `A` and exponent `n`,
- A sliding law governed by coefficient `C`,
- Optionally the user can provide either:
    - A specific diffusive velocity `U` such that `D = U * H`
    - A modified creep coefficient `Y` that takes into account the ice thickness
        such that `D = (C + Y * 2/(n+2)) * (ρ*g)^n * H^(n_H+1) * |∇S|^(n_∇S-1)`
        where `n_H` and `n_∇S` are optional parameters that control if the SIA
        should use the `n` law or not.
        This formulation is denoted as the hybrid diffusivity in the code.

This struct stores the laws used to compute these three parameters during a simulation. If not provided, default constant laws are used based on glacier-specific values.

# Arguments
- `A`: Law for the flow rate factor. Defaults to a constant value from the glacier.
- `C`: Law for the sliding coefficient. Defaults similarly.
- `n`: Law for the flow law exponent. Defaults similarly.
- `Y`: Law for the hybrid diffusivity. Providing a law for `Y` discards the laws of `A`, `C` and `n`.
- `U`: Law for the diffusive velocity. Defaults behavior is to disable it and in such a case it is computed from `A`, `C` and `n`. Providing a law for `U` discards the laws of `A`, `C`, `n` and `Y`.
- `n_H::F`: The exponent to use for `H` in the SIA equation when using the Y law (hybrid diffusivity). It should be `nothing` when this law is not used.
- `n_∇S::F`: The exponent to use for `∇S` in the SIA equation when using the Y law (hybrid diffusivity). It should be `nothing` when this law is not used.
- `Y_is_provided::Bool`: Whether the diffusivity is provided by the user through the hybrid diffusivity `Y` or it has to be computed from the SIA formula from `A`, `C` and `n`.
- `U_is_provided::Bool`: Whether the diffusivity is provided by the user through the diffusive velocity `U` or it has to be computed from the SIA formula from `A`, `C` and `n`.
- `n_H_is_provided::Bool`: Whether the `H` exponent is prescribed by the user, or the one of the `n` law has to be used. This flag is used only when a law for `Y` is used.
- `n_∇S_is_provided::Bool`: Whether the `∇S` exponent is prescribed by the user, or the one of the `n` law has to be used. This flag is used only when a law for `Y` is used.
- `apply_A_in_SIA::Bool`: Whether the value of the `A` law should be computed each time the SIA is evaluated.
- `apply_C_in_SIA::Bool`: Whether the value of the `C` law should be computed each time the SIA is evaluated.
- `apply_n_in_SIA::Bool`: Whether the value of the `n` law should be computed each time the SIA is evaluated.
- `apply_Y_in_SIA::Bool`: Whether the value of the `Y` law should be computed each time the SIA is evaluated.
- `apply_U_in_SIA::Bool`: Whether the value of the `U` law should be computed each time the SIA is evaluated.
"""
@kwdef struct SIA2Dmodel{
        F,
        ALAW <: AbstractLaw,
        CLAW <: AbstractLaw,
        nLAW <: AbstractLaw,
        pLAW <: AbstractLaw,
        qLAW <: AbstractLaw,
        YLAW <: AbstractLaw,
        ULAW <: AbstractLaw
        } <: SIAmodel
    A::ALAW = nothing
    C::CLAW = nothing
    n::nLAW = nothing
    p::pLAW = nothing
    q::qLAW = nothing
    Y::YLAW = nothing
    U::ULAW = nothing
    n_H::Array{F, 0} = nothing
    n_∇S::Array{F, 0} = nothing
    Y_is_provided::Bool = false
    U_is_provided::Bool = false
    n_H_is_provided::Bool = false
    n_∇S_is_provided::Bool = false
    apply_A_in_SIA::Bool = false
    apply_C_in_SIA::Bool = false
    apply_n_in_SIA::Bool = false
    apply_p_in_SIA::Bool = false
    apply_q_in_SIA::Bool = false
    apply_Y_in_SIA::Bool = false
    apply_U_in_SIA::Bool = false

    function SIA2Dmodel(A, C, n, p, q, Y, U, n_H, n_∇S)
        Y_is_provided = !isnothing(Y)
        U_is_provided = !isnothing(U)
        n_H_is_provided = !isnothing(n_H)
        n_∇S_is_provided = !isnothing(n_∇S)

        if U_is_provided
            @assert isnothing(A) "When U law is provided, A should not be provided."
            @assert isnothing(C) "When U law is provided, C should not be provided."
            @assert isnothing(n) "When U law is provided, n should not be provided."
            @assert isnothing(Y) "When U law is provided, Y should not be provided."
            A = NullLaw()
            C = NullLaw()
            n = NullLaw()
            p = NullLaw()
            q = NullLaw()
            Y = NullLaw()
        elseif Y_is_provided
            @assert isnothing(A) "When Y law is provided, A should not be provided."
            @assert isnothing(U) "When Y law is provided, U should not be provided."
            A = NullLaw()
            C = something(C, _default_C_law)
            n = something(n, _default_n_law) # We need n with the hybrid diffusivity
            p = something(p, _default_p_law)
            q = something(q, _default_q_law)
            U = NullLaw()
        else
            @assert isnothing(Y) "When either A, C or n law are provided, Y should not be provided."
            @assert isnothing(U) "When either A, C or n law are provided, U should not be provided."
            A = something(A, _default_A_law)
            C = something(C, _default_C_law)
            n = something(n, _default_n_law)
            p = something(p, _default_p_law)
            q = something(q, _default_q_law)
            Y = NullLaw()
            U = NullLaw()
        end

        if !Y_is_provided
            @assert isnothing(n_H) "When Y law is not used, n_H must be set to nothing."
            @assert isnothing(n_∇S) "When Y law is not used, n_∇S must be set to nothing."
        end

        n_H = fill(something(n_H, 1.))
        n_∇S = fill(something(n_∇S, 1.))

        new{Sleipnir.Float, typeof(A), typeof(C), typeof(n), typeof(p), typeof(q), typeof(Y), typeof(U)}(
            A, C, n, p, q, Y, U, n_H, n_∇S,
            Y_is_provided,
            U_is_provided,
            n_H_is_provided,
            n_∇S_is_provided,
            apply_law_in_model(A),
            apply_law_in_model(C),
            apply_law_in_model(n),
            apply_law_in_model(p),
            apply_law_in_model(q),
            apply_law_in_model(Y),
            apply_law_in_model(U),
        )
    end
end

# Set default of laws to be the value of the physical parameter stored in the glacier struct
const _default_A_law = ConstantLaw{ScalarCacheNoVJP}(
    (simulation, glacier_idx, θ) -> ScalarCacheNoVJP(fill(simulation.glaciers[glacier_idx].A))
)

const _default_C_law = ConstantLaw{ScalarCacheNoVJP}(
    (simulation, glacier_idx, θ) -> ScalarCacheNoVJP(fill(simulation.glaciers[glacier_idx].C))
)

const _default_n_law = ConstantLaw{ScalarCacheNoVJP}(
    (simulation, glacier_idx, θ) -> ScalarCacheNoVJP(fill(simulation.glaciers[glacier_idx].n))
)

const _default_p_law = ConstantLaw{ScalarCacheNoVJP}(
    (simulation, glacier_idx, θ) -> ScalarCacheNoVJP(fill(simulation.glaciers[glacier_idx].p))
)

const _default_q_law = ConstantLaw{ScalarCacheNoVJP}(
    (simulation, glacier_idx, θ) -> ScalarCacheNoVJP(fill(simulation.glaciers[glacier_idx].q))
)

SIA2Dmodel(
    params::Sleipnir.Parameters;
    A = nothing,
    C = nothing,
    n = nothing,
    p = nothing,
    q = nothing,
    Y = nothing,
    U = nothing,
    n_H = nothing,
    n_∇S = nothing
    ) = SIA2Dmodel(A, C, n, p, q, Y, U, n_H, n_∇S)

"""
    SIA2DCache{
        R <: Real, I <: Integer, A_CACHE, C_CACHE, n_CACHE, n_H_CACHE, n_∇S_CACHE, Y_CACHE, U_CACHE
    } <: SIAmodel

Store and preallocated all variables needed for running the 2D Shallow Ice Approximation (SIA) model efficiently.

# Type Parameters
- `R`: Real number type used for physical fields.
- `I`: Integer type used for indexing glaciers.
- `A_CACHE`, `C_CACHE`, `n_CACHE`: Types used for caching `A`, `C`, and `n`, which can be scalars, vectors, or matrices.
- `Y_CACHE`: Type used for caching `Y` which is a matrix.
- `U_CACHE`: Type used for caching `U` which is a matrix.

# Fields
- `A::A_CACHE`: Flow rate factor.
- `n::n_CACHE`: Flow law exponent.
- `n_H::n_CACHE`: Exponent used for the power of `H` when using the `Y` law.
- `n_∇S::n_CACHE`: Exponent used for the power of `∇S` when using the `Y` law.
- `C::C_CACHE`: Sliding coefficient.
- `Y::Y_CACHE`: Hybrid diffusivity.
- `U::U_CACHE`: Diffusive velocity.
- `H₀::Matrix{R}`: Initial ice thickness.
- `H::Matrix{R}`: Ice thickness.
- `H̄::Matrix{R}`: Averaged ice thickness.
- `S::Matrix{R}`: Surface elevation.
- `dSdx::Matrix{R}`: Surface slope in the x-direction.
- `dSdy::Matrix{R}`: Surface slope in the y-direction.
- `D::Matrix{R}`: Diffusivity.
- `Dx::Matrix{R}`: Diffusivity in the x-direction.
- `Dy::Matrix{R}`: Diffusivity in the y-direction.
- `dSdx_edges::Matrix{R}`: Surface slope at edges in the x-direction.
- `dSdy_edges::Matrix{R}`: Surface slope at edges in the y-direction.
- `∇S::Matrix{R}`: Norm of the surface gradient.
- `∇Sy::Matrix{R}`: Surface gradient component in the y-direction.
- `∇Sx::Matrix{R}`: Surface gradient component in the x-direction.
- `Fx::Matrix{R}`: Flux in the x-direction.
- `Fy::Matrix{R}`: Flux in the y-direction.
- `Fxx::Matrix{R}`: Second derivative of flux in the x-direction.
- `Fyy::Matrix{R}`: Second derivative of flux in the y-direction.
- `V::Matrix{R}`: Velocity magnitude.
- `Vx::Matrix{R}`: Velocity in the x-direction.
- `Vy::Matrix{R}`: Velocity in the y-direction.
- `Γ::A_CACHE`: Basal shear stress.
- `MB::Matrix{R}`: Mass balance.
- `MB_mask::BitMatrix`: Boolean mask for applying the mass balance.
- `MB_total::Matrix{R}`: Total mass balance field.
- `glacier_idx::I`: Index of the glacier for use in simulations with multiple glaciers.
- `A_prep_vjps`, `C_prep_vjps`, `n_prep_vjps`, `Y_prep_vjps` and `U_prep_vjps`: Structs
    that contain the prepared VJP functions for the adjoint computation and for the
    different laws. Useful mainly when the user does not provide the VJPs and they are
    automatically inferred using DifferentiationInterface.jl which requires to store
    precompiled functions. When no gradient is computed, these structs are `nothing`.
"""
@kwdef struct SIA2DCache{
    R <: Real,
    I <: Integer,
    A_CACHE,
    C_CACHE,
    n_CACHE,
    p_CACHE,
    q_CACHE,
    n_H_CACHE,
    n_∇S_CACHE,
    Y_CACHE,
    U_CACHE
} <: SIAmodel
    A::A_CACHE
    n::n_CACHE
    n_H::n_H_CACHE
    n_∇S::n_∇S_CACHE
    C::C_CACHE
    p::p_CACHE
    q::q_CACHE
    Y::Y_CACHE
    U::U_CACHE
    H₀::Matrix{R}
    H::Matrix{R}
    H̄::Matrix{R}
    S::Matrix{R}
    dSdx::Matrix{R}
    dSdy::Matrix{R}
    D::Matrix{R}
    Dx::Matrix{R}
    Dy::Matrix{R}
    dSdx_edges::Matrix{R}
    dSdy_edges::Matrix{R}
    ∇S::Matrix{R}
    ∇Sy::Matrix{R}
    ∇Sx::Matrix{R}
    Fx::Matrix{R}
    Fy::Matrix{R}
    Fxx::Matrix{R}
    Fyy::Matrix{R}
    V::Matrix{R}
    Vx::Matrix{R}
    Vy::Matrix{R}
    Γ::A_CACHE
    MB::Matrix{R}
    MB_mask::BitMatrix
    MB_total::Matrix{R}
    glacier_idx::I
    A_prep_vjps
    C_prep_vjps
    n_prep_vjps
    Y_prep_vjps
    U_prep_vjps
end

function cache_type(sia2d_model::SIA2Dmodel)
    A_CACHE = cache_type(sia2d_model.A)
    Y_CACHE = cache_type(sia2d_model.Y)
    U_CACHE = cache_type(sia2d_model.U)
    return SIA2DCache{
        Sleipnir.Float,
        Sleipnir.Int,
        A_CACHE,
        cache_type(sia2d_model.C),
        cache_type(sia2d_model.n),
        cache_type(sia2d_model.p),
        cache_type(sia2d_model.q),
        typeof(sia2d_model.n_H),
        typeof(sia2d_model.n_∇S),
        Y_CACHE,
        U_CACHE,
    }
end

"""
function init_cache(
    iceflow_model::SIA2Dmodel,
    glacier::AbstractGlacier,
    glacier_idx::I,
    θ,
) where {IF <: IceflowModel, I <: Integer}

Initialize iceflow model data structures to enable in-place mutation.

Keyword arguments
- `iceflow_model`: Iceflow model used for simulation.
- `glacier_idx`: Index of glacier.
- `glacier`: `Glacier` to provide basic initial state of the ice flow model.
- `θ`: Optional parameters of the laws.
"""
function init_cache(
    model::SIA2Dmodel,
    simulation,
    glacier_idx::Int,
    θ,
)
    glacier = simulation.glaciers[glacier_idx]

    nx, ny = glacier.nx, glacier.ny
    F = Sleipnir.Float

    A = init_cache(model.A, simulation, glacier_idx, θ)
    C = init_cache(model.C, simulation, glacier_idx, θ)
    n = init_cache(model.n, simulation, glacier_idx, θ)
    p = init_cache(model.p, simulation, glacier_idx, θ)
    q = init_cache(model.q, simulation, glacier_idx, θ)
    Y = init_cache(model.Y, simulation, glacier_idx, θ)
    U = init_cache(model.U, simulation, glacier_idx, θ)

    n_H = model.n_H
    n_∇S = model.n_∇S
    Γ = similar(A)

    A_prep_vjps = prepare_vjp_law(simulation, model.A, A, θ, glacier_idx)
    C_prep_vjps = prepare_vjp_law(simulation, model.C, C, θ, glacier_idx)
    n_prep_vjps = prepare_vjp_law(simulation, model.n, n, θ, glacier_idx)
    Y_prep_vjps = prepare_vjp_law(simulation, model.Y, Y, θ, glacier_idx)
    U_prep_vjps = prepare_vjp_law(simulation, model.U, U, θ, glacier_idx)

    return SIA2DCache(;
        A,
        n,
        n_H,
        n_∇S,
        C,
        p,
        q,
        Y,
        U,
        Γ,
        H₀ = deepcopy(glacier.H₀),
        H = deepcopy(glacier.H₀),
        H̄ = zeros(F,nx-1,ny-1),
        S = deepcopy(glacier.S),
        dSdx = zeros(F,nx-1,ny),
        dSdy= zeros(F,nx,ny-1),
        D = zeros(F,nx-1,ny-1),
        Dx = zeros(F,nx-1,ny-2),
        Dy = zeros(F,nx-2,ny-1),
        dSdx_edges = zeros(F,nx-1,ny-2),
        dSdy_edges = zeros(F,nx-2,ny-1) ,
        ∇S = zeros(F,nx-1,ny-1),
        ∇Sx = zeros(F,nx-1,ny-1),
        ∇Sy = zeros(F,nx-1,ny-1),
        Fx = zeros(F,nx-1,ny-2),
        Fy = zeros(F,nx-2,ny-1),
        Fxx = zeros(F,nx-2,ny-2),
        Fyy = zeros(F,nx-2,ny-2),
        V = zeros(F,nx,ny),
        Vx = zeros(F,nx,ny),
        Vy = zeros(F,nx,ny),
        MB = zeros(F,nx,ny),
        MB_mask = falses(nx,ny),
        MB_total = zeros(F,nx,ny),
        glacier_idx = Sleipnir.Int(glacier_idx),
        A_prep_vjps,
        C_prep_vjps,
        n_prep_vjps,
        Y_prep_vjps,
        U_prep_vjps,
    )
end

"""
    build_callback(model::SIA2Dmodel, cache::SIA2DCache, glacier_idx::Real, θ, tspan) -> CallbackSet

Return a `CallbackSet` that updates the cached values of `A`, `C`, `n` and `U` at provided time intervals.

Each law can optionally specify a callback frequency via `callback_freq`.
- If `callback_freq > 0`, a `PeriodicCallback` is used to update the corresponding component at regular intervals.
- If `callback_freq == 0`, a `PresetTimeCallback` is used to trigger the update only at the initial time
(taken from `tspan[1]`).
- If no callback is specified for a component, a dummy `CallbackSet` is returned.

Arguments:
- `model::SIA2Dmodel`: The ice flow model definition.
- `cache::SIA2DCache`: Model cache for efficient component access and updates.
- `glacier_idx::Real`: Index of the glacier in the simulation.
- `θ`: Optional parameter for law evaluation.
- `tspan`: Tuple or floats specifying the simulation time span. Used to determine initial callback time when `freq == 0`.

Returns:
- A `CallbackSet` containing all the callbacks for periodic or preset updates of model components.
"""
function build_callback(model::SIA2Dmodel, cache::SIA2DCache, glacier_idx::Real, θ, tspan)
    tstopsPresetCb = [tspan[1]]
    A_cb = if is_callback_law(model.A)
        A_affect! = build_affect(model.A, cache.A, glacier_idx, θ)
        freq = callback_freq(model.A)

        if freq>0
            PeriodicCallback(A_affect!, freq; initial_affect = true)
        else
            PresetTimeCallback(tstopsPresetCb, A_affect!)
        end
    else
        CallbackSet()
    end

    C_cb = if is_callback_law(model.C)
        C_affect! = build_affect(model.C, cache.C, glacier_idx, θ)
        freq = callback_freq(model.C)

        if freq>0
            PeriodicCallback(C_affect!, freq; initial_affect = true)
        else
            PresetTimeCallback(tstopsPresetCb, C_affect!)
        end
    else
        CallbackSet()
    end

    n_cb = if is_callback_law(model.n)
        n_affect! = build_affect(model.n, cache.n, glacier_idx, θ)
        freq = callback_freq(model.n)

        if freq>0
            PeriodicCallback(n_affect!, freq; initial_affect = true)
        else
            PresetTimeCallback(tstopsPresetCb, n_affect!)
        end
    else
        CallbackSet()
    end

    Y_cb = if is_callback_law(model.Y)
        Y_affect! = build_affect(model.Y, cache.Y, glacier_idx, θ)
        freq = callback_freq(model.Y)

        if freq>0
            PeriodicCallback(Y_affect!, freq; initial_affect = true)
        else
            PresetTimeCallback(tstopsPresetCb, Y_affect!)
        end
    else
        CallbackSet()
    end

    U_cb = if is_callback_law(model.U)
        U_affect! = build_affect(model.U, cache.U, glacier_idx, θ)
        freq = callback_freq(model.U)

        if freq>0
            PeriodicCallback(U_affect!, freq; initial_affect = true)
        else
            PresetTimeCallback(tstopsPresetCb, U_affect!)
        end
    else
        CallbackSet()
    end

    return CallbackSet(A_cb, C_cb, n_cb, Y_cb, U_cb)
end

build_callback(model::SIA2Dmodel, cache::SIA2DCache, glacier_idx::Real, tspan) = build_callback(model::SIA2Dmodel, cache::SIA2DCache, glacier_idx, nothing, tspan)


# Display setup
function Base.show(io::IO, model::SIA2Dmodel)
    colorD = :green
    colorU = :red
    colorA = :blue
    colorC = :magenta
    colorn = :yellow
    colorp = :yellow
    colorq = :red
    colorY = :blue
    colorΓ = :cyan
    print(io, "SIA2D iceflow equation")
    print(io, "  = ∇("); printstyled(io, "D";color=colorD); print(io, " ∇S)")
    print(io, "  with "); printstyled(io, "D";color=colorD); print(io, " = "); printstyled(io, "U";color=colorU);println(io, " H̄")
    inp = []
    if model.U_is_provided
        print(io, "  and "); printstyled(io, "U";color=colorU); print(io, ": "); print(io, model.U)
        if inputs_defined(model.U)
            push!(inp, inputs(model.U))
        end
    elseif model.Y_is_provided
        print(io, "  and "); printstyled(io, "U";color=colorU); print(io, " = (")

        # Sliding part
        printstyled(io, "C";color=colorC);print(io, " (ρg)^");printstyled(io, "n";color=colorn);print(io, " + ")
        # Creeping part
        printstyled(io, "Y";color=colorY);print(io, " ")
        printstyled(io, "Γ";color=colorΓ);print(io, " H̄)")
        # Non linear part
        print(io, " H̄^")
        if isnothing(model.n_H)
            printstyled(io, "n";color=colorn)
        else
            print(io, "n_H")
        end
        print(io, " ∇S^(")
        if isnothing(model.n_∇S)
            printstyled(io, "n";color=colorn); print(io, "-1)")
        else
            print(io, "n_∇S-1)")
        end
        println(io)

        printstyled(io, "      Γ";color=colorΓ);print(io, " = 2");print(io, " (ρg)^"); printstyled(io, "n";color=colorn)
        print(io, " /(");printstyled(io, "n";color=colorn);println(io, "+2)")

        printstyled(io, "      Y: ";color=colorY); print(io, model.Y)
        printstyled(io, "      C: ";color=colorC); print(io, model.C)
        printstyled(io, "      n: ";color=colorn); print(io, model.n)
        printstyled(io, "      p: ";color=colorp); print(io, model.p)
        printstyled(io, "      q: ";color=colorq); print(io, model.q)
        if !isnothing(model.n_H)
            println(io, "      n_H = $(model.n_H)")
        end
        if !isnothing(model.n_∇S)
            println(io, "      n_∇S = $(model.n_∇S)")
        end
        if inputs_defined(model.Y)
            push!(inp, inputs(model.Y))
        end
        if inputs_defined(model.C)
            push!(inp, inputs(model.C))
        end
        if inputs_defined(model.n)
            push!(inp, inputs(model.n))
        end
    else
        print(io, "  and "); printstyled(io, "U";color=colorU); print(io, " = (")
        # Sliding part
        printstyled(io, "C";color=colorC);print(io, " (ρg)^");printstyled(io, "n";color=colorn);print(io, " + ")
        # Creeping part
        printstyled(io, "Γ";color=colorΓ);print(io, " H̄)")
        # Non linear part
        print(io, " H̄^");printstyled(io, "n";color=colorn);print(io, " ∇S^(");printstyled(io, "n";color=colorn);println(io, "-1)")

        printstyled(io, "      Γ";color=colorΓ);print(io, " = 2");printstyled(io, "A";color=colorA);print(io, " (ρg)^"); printstyled(io, "n";color=colorn)
        print(io, " /(");printstyled(io, "n";color=colorn);println(io, "+2)")

        printstyled(io, "      A: ";color=colorA); print(io, model.A)
        printstyled(io, "      C: ";color=colorC); print(io, model.C)
        printstyled(io, "      n: ";color=colorn); print(io, model.n)
        if inputs_defined(model.A)
            push!(inp, inputs(model.A))
        end
        if inputs_defined(model.C)
            push!(inp, inputs(model.C))
        end
        if inputs_defined(model.n)
            push!(inp, inputs(model.n))
        end
    end
    inp = merge(inp...)
    if length(inp)>0
        println(io, "  where")
        for (k,v) in pairs(inp)
            println(io, "      $(k) => $(default_name(v))")
        end
    end
end
