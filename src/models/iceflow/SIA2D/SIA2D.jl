using DiffEqCallbacks: PeriodicCallback
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
@kwdef struct SIA2Dmodel{F, ALAW <: AbstractLaw, CLAW <: AbstractLaw, nLAW <: AbstractLaw, YLAW <: AbstractLaw, ULAW <: AbstractLaw} <: SIAmodel
    A::ALAW = nothing
    C::CLAW = nothing
    n::nLAW = nothing
    Y::YLAW = nothing
    U::ULAW = nothing
    n_H::F = nothing
    n_∇S::F = nothing
    Y_is_provided::Bool = false
    U_is_provided::Bool = false
    n_H_is_provided::Bool = false
    n_∇S_is_provided::Bool = false
    apply_A_in_SIA::Bool = false
    apply_C_in_SIA::Bool = false
    apply_n_in_SIA::Bool = false
    apply_Y_in_SIA::Bool = false
    apply_U_in_SIA::Bool = false

    function SIA2Dmodel(A, C, n, Y, U, n_H, n_∇S)
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
            Y = NullLaw()
        elseif Y_is_provided
            @assert isnothing(A) "When Y law is provided, A should not be provided."
            @assert isnothing(U) "When Y law is provided, U should not be provided."
            A = NullLaw()
            C = something(C, _default_C_law)
            n = something(n, _default_n_law) # We need n with the hybrid diffusivity
            U = NullLaw()
        else
            @assert isnothing(Y) "When either A, C or n law are provided, Y should not be provided."
            @assert isnothing(U) "When either A, C or n law are provided, U should not be provided."
            A = something(A, _default_A_law)
            C = something(C, _default_C_law)
            n = something(n, _default_n_law)
            Y = NullLaw()
            U = NullLaw()
        end

        if !Y_is_provided
            @assert isnothing(n_H) "When Y law is not used, n_H must be set to nothing."
            @assert isnothing(n_∇S) "When Y law is not used, n_∇S must be set to nothing."
        end

        n_H = something(n_H, 1.)
        n_∇S = something(n_∇S, 1.)

        new{Sleipnir.Float, typeof(A), typeof(C), typeof(n), typeof(Y), typeof(U)}(
            A, C, n, Y, U, n_H, n_∇S,
            Y_is_provided,
            U_is_provided,
            n_H_is_provided,
            n_∇S_is_provided,
            !is_callback_law(A) && !(A isa ConstantLaw) && !(A isa NullLaw),
            !is_callback_law(C) && !(C isa ConstantLaw) && !(C isa NullLaw),
            !is_callback_law(n) && !(n isa ConstantLaw) && !(n isa NullLaw),
            !is_callback_law(Y) && !(Y isa ConstantLaw) && !(Y isa NullLaw),
            !is_callback_law(U) && !(U isa ConstantLaw) && !(U isa NullLaw),
        )
    end
end

const _default_A_law = ConstantLaw{Array{Sleipnir.Float, 0}}(
    (simulation, glacier_idx, θ) -> fill(simulation.glaciers[glacier_idx].A)
)

const _default_C_law = ConstantLaw{Array{Sleipnir.Float, 0}}(
    (simulation, glacier_idx, θ) -> fill(simulation.glaciers[glacier_idx].C)
)

const _default_n_law = ConstantLaw{Array{Sleipnir.Float, 0}}(
    (simulation, glacier_idx, θ) -> fill(simulation.glaciers[glacier_idx].n)
)

SIA2Dmodel(params::Sleipnir.Parameters; A = nothing, C = nothing, n = nothing, Y = nothing, U = nothing, n_H = nothing, n_∇S = nothing) = SIA2Dmodel(A, C, n, Y, U, n_H, n_∇S)

"""
    struct SIA2DCache{R <: Real, I <: Integer, A_CACHE, C_CACHE, n_CACHE, Y_CACHE, U_CACHE, ∂A∂θ_CACHE, ∂Y∂θ_CACHE, ∂U∂θ_CACHE} <: SIAmodel

Store and preallocated all variables needed for running the 2D Shallow Ice Approximation (SIA) model efficiently.

# Type Parameters
- `R`: Real number type used for physical fields.
- `I`: Integer type used for indexing glaciers.
- `A_CACHE`, `C_CACHE`, `n_CACHE`: Types used for caching `A`, `C`, and `n`, which can be scalars, vectors, or matrices.
- `Y_CACHE`: Type used for caching `Y` which is a matrix.
- `U_CACHE`: Type used for caching `U` which is a matrix.
- `∂A∂θ_CACHE`: Type used for caching `∂A∂θ` in the VJP computation which is a scalar.
- `∂Y∂θ_CACHE`: Type used for caching `∂Y∂θ` in the VJP computation which is a scalar.
- `∂U∂θ_CACHE`: Type used for caching `∂U∂θ` in the VJP computation which is a scalar.

# Fields
- `A::A_CACHE`: Flow rate factor.
- `n::n_CACHE`: Flow law exponent.
- `n_H::n_CACHE`: Exponent used for the power of `H` when using the `Y` law.
- `n_∇S::n_CACHE`: Exponent used for the power of `∇S` when using the `Y` law.
- `C::C_CACHE`: Sliding coefficient.
- `Y::Y_CACHE`: Hybrid diffusivity.
- `U::U_CACHE`: Diffusive velocity.
- `∂A∂H::A_CACHE`: Buffer for VJP computation.
- `∂A∂θ::∂A∂θ_CACHE`: Buffer for VJP computation.
- `∂Y∂H::Y_CACHE`: Buffer for VJP computation.
- `∂Y∂θ::∂Y∂θ_CACHE`: Buffer for VJP computation.
- `∂U∂H::U_CACHE`: Buffer for VJP computation.
- `∂U∂θ::∂U∂θ_CACHE`: Buffer for VJP computation.
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
"""
@kwdef struct SIA2DCache{R <: Real, I <: Integer, A_CACHE, C_CACHE, n_CACHE, Y_CACHE, U_CACHE, ∂A∂θ_CACHE, ∂Y∂θ_CACHE, ∂U∂θ_CACHE} <: SIAmodel
    A::A_CACHE
    n::n_CACHE
    n_H::n_CACHE
    n_∇S::n_CACHE
    C::C_CACHE
    Y::Y_CACHE
    U::U_CACHE
    ∂A∂H::A_CACHE
    ∂A∂θ::∂A∂θ_CACHE
    ∂Y∂H::Y_CACHE
    ∂Y∂θ::∂Y∂θ_CACHE
    ∂U∂H::U_CACHE
    ∂U∂θ::∂U∂θ_CACHE
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
        Y_CACHE,
        U_CACHE,
        Array{eltype(A_CACHE), 0},
        Array{eltype(Y_CACHE), 0},
        Array{eltype(U_CACHE), 0},
    }
end

"""
function init_cache(
    iceflow_model::SIA2Dmodel,
    glacier::AbstractGlacier,
    glacier_idx::I,
    θ
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
    θ
)
    glacier = simulation.glaciers[glacier_idx]

    nx, ny = glacier.nx, glacier.ny
    F = Sleipnir.Float

    A = init_cache(model.A, simulation, glacier_idx, θ)
    C = init_cache(model.C, simulation, glacier_idx, θ)
    n = init_cache(model.n, simulation, glacier_idx, θ)
    Y = init_cache(model.Y, simulation, glacier_idx, θ)
    U = init_cache(model.U, simulation, glacier_idx, θ)

    n_H = fill(model.n_H, size(n))
    n_∇S = fill(model.n_∇S, size(n))

    # Buffer for VJP computation, they are used when the law needs either to be evaluated or differentiated
    ∂A∂H = similar(A)
    ∂Y∂H = similar(Y)
    ∂U∂H = similar(U)
    # Needs to be a scalar as it may be used with a backward interpolation which evaluates the backward element wise
    ∂A∂θ = similar(A, ())
    ∂Y∂θ = similar(Y, ())
    ∂U∂θ = similar(U, ())

    Γ = similar(A)

    return SIA2DCache(;
        A,
        n,
        n_H,
        n_∇S,
        C,
        Y,
        U,
        ∂A∂H,
        ∂A∂θ,
        ∂Y∂H,
        ∂Y∂θ,
        ∂U∂H,
        ∂U∂θ,
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
    )
end

"""
    build_callback(model::SIA2Dmodel, cache::SIA2DCache, glacier_idx, θ) -> CallbackSet

Return a `CallbackSet` that updates the cached values of `A`, `C`, `n` and `U` at provided time intervals.

Each law can optionally specify a callback frequency. If such a frequency is set (via `callback_freq`),
the update is done using a `PeriodicCallback`. Otherwise, no callback is used for that component.
"""
function build_callback(model::SIA2Dmodel, cache::SIA2DCache, glacier_idx, θ)
    A_cb = if is_callback_law(model.A)
        A_affect! = build_affect(model.A, cache.A, glacier_idx, θ)
        freq = callback_freq(model.A)

        PeriodicCallback(A_affect!, freq; initial_affect = true)
    else
        CallbackSet()
    end

    C_cb = if is_callback_law(model.C)
        C_affect! = build_affect(model.C, cache.C, glacier_idx, θ)
        freq = callback_freq(model.C)

        PeriodicCallback(C_affect!, freq; initial_affect = true)
    else
        CallbackSet()
    end

    n_cb = if is_callback_law(model.n)
        n_affect! = build_affect(model.n, cache.n, glacier_idx, θ)
        freq = callback_freq(model.n)

        PeriodicCallback(n_affect!, freq; initial_affect = true)
    else
        CallbackSet()
    end

    Y_cb = if is_callback_law(model.Y)
        Y_affect! = build_affect(model.Y, cache.Y, glacier_idx, θ)
        freq = callback_freq(model.Y)

        PeriodicCallback(Y_affect!, freq; initial_affect = true)
    else
        CallbackSet()
    end

    U_cb = if is_callback_law(model.U)
        U_affect! = build_affect(model.U, cache.U, glacier_idx, θ)
        freq = callback_freq(model.U)

        PeriodicCallback(U_affect!, freq; initial_affect = true)
    else
        CallbackSet()
    end

    return CallbackSet(A_cb, C_cb, n_cb, Y_cb, U_cb)
end

build_callback(model::SIA2Dmodel, cache::SIA2DCache, glacier_idx) = build_callback(model::SIA2Dmodel, cache::SIA2DCache, glacier_idx, nothing)


# Display setup
function Base.show(io::IO, type::MIME"text/plain", model::SIA2Dmodel)
    Base.show(io, model)
end
function Base.show(io::IO, model::SIA2Dmodel)
    colorD = :green
    colorU = :red
    colorA = :blue
    colorC = :magenta
    colorn = :yellow
    colorY = :blue
    colorΓ = :cyan
    print("SIA2D iceflow equation")
    print("  = ∇("); printstyled("D";color=colorD); print(" ∇S)")
    print("  with "); printstyled("D";color=colorD); print(" = "); printstyled("U";color=colorU);println(" H̄")
    inp = []
    if model.U_is_provided
        print("  and "); printstyled("U";color=colorU); print(": "); println(model.U)
        push!(inp, inputs(model.U))
    elseif model.Y_is_provided
        print("  and "); printstyled("U";color=colorU); print(" = (")

        # Sliding part
        printstyled("C";color=colorC);print(" (ρg)^");printstyled("n";color=colorn);print(" + ")
        # Creeping part
        printstyled("Y";color=colorY);print(" ")
        printstyled("Γ";color=colorΓ);print(" H̄)")
        # Non linear part
        print(" H̄^")
        if isnothing(model.n_H)
            printstyled("n";color=colorn)
        else
            print("n_H")
        end
        print(" ∇S^(")
        if isnothing(model.n_∇S)
            printstyled("n";color=colorn); print("-1)")
        else
            print("n_∇S-1)")
        end
        println()

        printstyled("      Γ";color=colorΓ);print(" = 2");print(" (ρg)^"); printstyled("n";color=colorn)
        print(" /(");printstyled("n";color=colorn);println("+2)")

        printstyled("      Y: ";color=colorY); println(model.Y)
        printstyled("      C: ";color=colorC); println(model.C)
        printstyled("      n: ";color=colorn); println(model.n)
        if !isnothing(model.n_H)
            println("      n_H = $(model.n_H)")
        end
        if !isnothing(model.n_∇S)
            println("      n_∇S = $(model.n_∇S)")
        end
        push!(inp, inputs(model.Y))
        push!(inp, inputs(model.C))
        push!(inp, inputs(model.n))
    else
        print("  and "); printstyled("U";color=colorU); print(" = (")
        # Sliding part
        printstyled("C";color=colorC);print(" (ρg)^");printstyled("n";color=colorn);print(" + ")
        # Creeping part
        printstyled("Γ";color=colorΓ);print(" H̄)")
        # Non linear part
        print(" H̄^");printstyled("n";color=colorn);print(" ∇S^(");printstyled("n";color=colorn);println("-1)")

        printstyled("      Γ";color=colorΓ);print(" = 2");printstyled("A";color=colorA);print(" (ρg)^"); printstyled("n";color=colorn)
        print(" /(");printstyled("n";color=colorn);println("+2)")

        printstyled("      A: ";color=colorA); println(model.A)
        printstyled("      C: ";color=colorC); println(model.C)
        printstyled("      n: ";color=colorn); println(model.n)
        push!(inp, inputs(model.A))
        push!(inp, inputs(model.C))
        push!(inp, inputs(model.n))
    end
    inp = merge(inp...)
    if length(inp)>0
        println("  where")
        for (k,v) in pairs(inp)
            println("      $(k) => $(default_name(v))")
        end
    end
end


include("SIA2D_utils.jl")
