using DiffEqCallbacks: PeriodicCallback
import Sleipnir: init_cache, cache_type

export SIA2Dmodel, SIA2DCache

###############################################
###### SHALLOW ICE APPROXIMATION MODELS #######
###############################################

"""
    SIA2Dmodel(A, C, n, ϕ)
    SIA2Dmodel(;A, C, n, ϕ)

Create a `SIA2Dmodel`, representing a two-dimensional Shallow Ice Approximation (SIA) model.

The SIA model describes glacier flow under the assumption that deformation and basal sliding dominate the ice dynamics. It relies on:
- Glen's flow law for internal deformation, with flow rate factor `A` and exponent `n`,
- A sliding law governed by coefficient `C`,
- Optionally the user can provide a specific diffusivity flux `ϕ`.

This struct stores the laws used to compute these three parameters during a simulation. If not provided, default constant laws are used based on glacier-specific values.

# Arguments
- `A`: Law for the flow rate factor. Defaults to a constant value from the glacier.
- `C`: Law for the sliding coefficient. Defaults similarly.
- `n`: Law for the flow law exponent. Defaults similarly.
- `ϕ`: Law for the diffusivity flux. Defaults behavior is to disable it and in such a case it is computed from `A`, `C` and `n`. Providing a law for ϕ discards the laws of `A`, `C` and `n`.
"""
@kwdef struct SIA2Dmodel{ALAW <: AbstractLaw, CLAW <: AbstractLaw, nLAW <: AbstractLaw, ϕLAW <: AbstractLaw} <: SIAmodel
    A::ALAW = nothing
    C::CLAW = nothing
    n::nLAW = nothing
    ϕ::ϕLAW = nothing
    ϕ_is_provided::Bool = false # Whether the diffusivity is provided by the user through the diffusivity flux `ϕ` or it has to be computed from the SIA formula from `A`, `C` and `n`.

    function SIA2Dmodel(A, C, n, ϕ)
        ϕ_is_provided = !isnothing(ϕ)

        if ϕ_is_provided
            @assert isnothing(A) "When ϕ law is provided, A should not be provided."
            @assert isnothing(C) "When ϕ law is provided, C should not be provided."
            @assert isnothing(n) "When ϕ law is provided, n should not be provided."
            A = NullLaw()
            C = NullLaw()
            n = NullLaw()
        else
            @assert isnothing(ϕ) "When either A, C or n law are provided, ϕ should not be provided."
            A = something(A, _default_A_law)
            C = something(C, _default_C_law)
            n = something(n, _default_n_law)
            ϕ = NullLaw()
        end
        new{typeof(A), typeof(C), typeof(n), typeof(ϕ)}(A, C, n, ϕ, ϕ_is_provided)
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

SIA2Dmodel(params::Sleipnir.Parameters; A = nothing, C = nothing, n = nothing, ϕ = nothing) = SIA2Dmodel(A, C, n, ϕ)

"""
    struct SIA2DCache{R <: Real, I <: Integer, A_CACHE, C_CACHE, n_CACHE, ϕ_CACHE, ∂A∂θ_CACHE, ∂ϕ∂θ_CACHE} <: SIAmodel

Store and preallocated all variables needed for running the 2D Shallow Ice Approximation (SIA) model efficiently.

# Type Parameters
- `R`: Real number type used for physical fields.
- `I`: Integer type used for indexing glaciers.
- `A_CACHE`, `C_CACHE`, `n_CACHE`: Types used for caching `A`, `C`, and `n`, which can be scalars, vectors, or matrices.
- `ϕ_CACHE`: Type used for caching `ϕ` which is a matrix.
- `∂A∂θ_CACHE`: Type used for caching `∂A∂θ` in the VJP computation which is a scalar.
- `∂ϕ∂θ_CACHE`: Type used for caching `∂ϕ∂θ` in the VJP computation which is a scalar.

# Fields
- `A::A_CACHE`: Flow rate factor.
- `n::n_CACHE`: Flow law exponent.
- `C::C_CACHE`: Sliding coefficient.
- `ϕ::ϕ_CACHE`: Diffusivity flux.
- `∂A∂H::A_CACHE`: Buffer for VJP computation.
- `∂A∂θ::∂A∂θ_CACHE`: Buffer for VJP computation.
- `∂ϕ∂H::ϕ_CACHE`: Buffer for VJP computation.
- `∂ϕ∂θ::∂ϕ∂θ_CACHE`: Buffer for VJP computation.
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
@kwdef struct SIA2DCache{R <: Real, I <: Integer, A_CACHE, C_CACHE, n_CACHE, ϕ_CACHE, ∂A∂θ_CACHE, ∂ϕ∂θ_CACHE} <: SIAmodel
    A::A_CACHE
    n::n_CACHE
    C::C_CACHE
    ϕ::ϕ_CACHE
    ∂A∂H::A_CACHE
    ∂A∂θ::∂A∂θ_CACHE
    ∂ϕ∂H::ϕ_CACHE
    ∂ϕ∂θ::∂ϕ∂θ_CACHE
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
    ϕ_CACHE = cache_type(sia2d_model.ϕ)
    return SIA2DCache{
        Sleipnir.Float,
        Sleipnir.Int,
        A_CACHE,
        cache_type(sia2d_model.C),
        cache_type(sia2d_model.n),
        ϕ_CACHE,
        Array{eltype(A_CACHE), 0},
        Array{eltype(ϕ_CACHE), 0},
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
    ϕ = init_cache(model.ϕ, simulation, glacier_idx, θ)

    # Buffer for VJP computation, they are used when the law needs either to be evaluated or differentiated
    ∂A∂H = similar(A)
    ∂ϕ∂H = similar(ϕ)
    # Needs to be a scalar as it may be used with a backward interpolation which evaluates the backward element wise
    ∂A∂θ = similar(A, ())
    ∂ϕ∂θ = similar(ϕ, ())

    Γ = similar(A)

    return SIA2DCache(;
        A,
        n,
        C,
        ϕ,
        ∂A∂H,
        ∂A∂θ,
        ∂ϕ∂H,
        ∂ϕ∂θ,
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

Return a `CallbackSet` that updates the cached values of `A`, `C`, `n` and `ϕ` at provided time intervals.

Each law can optionally specify a callback frequency. If such a frequency is set (via `callback_freq`),
the update is done using a `PeriodicCallback`. Otherwise, no callback is used for that component.
"""
function build_callback(model::SIA2Dmodel, cache::SIA2DCache, glacier_idx, θ)
    A_cb = if !isnothing(model.A) && is_callback_law(model.A)
        A_affect! = build_affect(model.A, cache.A, glacier_idx, θ)
        freq = callback_freq(model.A)

        PeriodicCallback(A_affect!, freq; initial_affect = true)
    else
        CallbackSet()
    end

    C_cb = if !isnothing(model.C) && is_callback_law(model.C)
        C_affect! = build_affect(model.C, cache.C, glacier_idx, θ)
        freq = callback_freq(model.C)

        PeriodicCallback(C_affect!, freq; initial_affect = true)
    else
        CallbackSet()
    end

    n_cb = if !isnothing(model.n) && is_callback_law(model.n)
        n_affect! = build_affect(model.n, cache.n, glacier_idx, θ)
        freq = callback_freq(model.n)

        PeriodicCallback(n_affect!, freq; initial_affect = true)
    else
        CallbackSet()
    end

    ϕ_cb = if !isnothing(model.ϕ) && is_callback_law(model.ϕ)
        ϕ_affect! = build_affect(model.ϕ, cache.ϕ, glacier_idx, θ)
        freq = callback_freq(model.ϕ)

        PeriodicCallback(ϕ_affect!, freq; initial_affect = true)
    else
        CallbackSet()
    end

    return CallbackSet(A_cb, C_cb, n_cb, ϕ_cb)
end

build_callback(model::SIA2Dmodel, cache::SIA2DCache, glacier_idx) = build_callback(model::SIA2Dmodel, cache::SIA2DCache, glacier_idx, nothing)

include("SIA2D_utils.jl")
