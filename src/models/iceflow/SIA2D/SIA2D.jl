using DiffEqCallbacks: PeriodicCallback

using Sleipnir: AbstractLaw
import Sleipnir: Law, init_cache, cache_type, apply_law!, build_affect, is_callback_law, callback_freq

export SIA2Dmodel, SIA2DCache, initialize_iceflow_model!, initialize_iceflow_model

include("SIA2D_utils.jl")

###############################################
###### SHALLOW ICE APPROXIMATION MODELS #######
###############################################

"""
    SIA2Dmodel(A, C, n)
    SIA2Dmodel(;A, C, n)

Create a `SIA2Dmodel`, representing a two-dimensional Shallow Ice Approximation (SIA) model.

The SIA model describes glacier flow under the assumption that deformation and basal sliding dominate the ice dynamics. It relies on:
- Glen's flow law for internal deformation, with flow rate factor `A` and exponent `n`,
- A sliding law governed by coefficient `C`.

This struct stores the laws used to compute these three parameters during a simulation. If not provided, default constant laws are used based on glacier-specific values.

# Arguments
- `A`: Law for the flow rate factor. Defaults to a constant value from the glacier.
- `C`: Law for the sliding coefficient. Defaults similarly.
- `n`: Law for the flow law exponent. Defaults similarly.
"""
@kwdef struct SIA2Dmodel{ALAW <: AbstractLaw, CLAW <: AbstractLaw, nLAW <: AbstractLaw} <: SIAmodel
    A::ALAW = nothing
    C::CLAW = nothing
    n::nLAW = nothing

    function SIA2Dmodel(A, C, n)
        A = something(A, _default_A_law)
        C = something(C, _default_C_law)
        n = something(n, _default_n_law)
        
        new{typeof(A), typeof(C), typeof(n)}(A, C, n)
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

SIA2Dmodel(params::Sleipnir.Parameters; A = nothing, C = nothing, n = nothing) = SIA2Dmodel(A, C, n)

"""
    struct SIA2DCache{R <: Real, I <: Integer, A_CACHE, C_CACHE, n_CACHE}

Store and preallocated all variables needed for running the 2D Shallow Ice Approximation (SIA) model efficiently.

# Type Parameters
- `R`: Real number type used for physical fields.
- `I`: Integer type used for indexing glaciers.
- `A_CACHE`, `C_CACHE`, `n_CACHE`: Types used for caching `A`, `C`, and `n`, which can be scalars, vectors, or matrices.

# Fields
- `A::A_CACHE`: Flow rate factor.
- `n::n_CACHE`: Flow law exponent.
- `C::C_CACHE`: Sliding coefficient.
- `H₀::Matrix{R}`: Initial ice thickness.
- `H::Matrix{R}`: Ice thickness.
- `H̄::Matrix{R}`: Averaged ice thickness.
- `S::Matrix{R}`: Surface elevation.
- `dSdx::Matrix{R}`: Surface slope in the x-direction.
- `dSdy::Matrix{R}`: Surface slope in the y-direction.
- `D::Matrix{R}`: Diffusivity.
- `D_is_provided::Bool`: Whether the diffusivity is provided by the user.
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
- `glacier_idx::Ref{I}`: Index of the glacier for use in simulations with multiple glaciers.
"""
@kwdef struct SIA2DCache{R <: Real, I <: Integer, A_CACHE, C_CACHE, n_CACHE} <: SIAmodel
    A::A_CACHE
    n::n_CACHE
    C::C_CACHE
    H₀::Matrix{R}
    H::Matrix{R}
    H̄::Matrix{R}
    S::Matrix{R}
    dSdx::Matrix{R}
    dSdy::Matrix{R}
    D::Matrix{R}
    D_is_provided::Bool
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
    glacier_idx::Ref{I}
end

cache_type(sia2d_model::SIA2Dmodel) = SIA2DCache{
    Sleipnir.Float,
    Sleipnir.Int,
    cache_type(sia2d_model.A),
    cache_type(sia2d_model.C),
    cache_type(sia2d_model.n),
}

"""
function init_cache(
    iceflow_model::SIA2Dmodel,
    glacier::AbstractGlacier,
    glacier_idx::I,
    params::Sleipnir.Parameters
) where {IF <: IceflowModel, I <: Integer}

Initialize iceflow model data structures to enable in-place mutation.

Keyword arguments
- `iceflow_model`: Iceflow model used for simulation.
- `glacier_idx`: Index of glacier.
- `glacier`: `Glacier` to provide basic initial state of the ice flow model.
- `parameters`: `Parameters` to configure some physical variables.
"""
function init_cache(model::SIA2Dmodel, simulation, glacier_idx, θ)
    glacier = simulation.glaciers[glacier_idx]

    nx, ny = glacier.nx, glacier.ny
    F = Sleipnir.Float

    A = init_cache(model.A, simulation, glacier_idx, θ)
    C = init_cache(model.C, simulation, glacier_idx, θ)
    n = init_cache(model.n, simulation, glacier_idx, θ)

    Γ = similar(A)

    return SIA2DCache(;
        A,
        C,
        Γ,
        n, 
        H₀ = deepcopy(glacier.H₀),
        H = deepcopy(glacier.H₀),
        H̄ = zeros(F,nx-1,ny-1),
        S = deepcopy(glacier.S),
        dSdx = zeros(F,nx-1,ny),
        dSdy= zeros(F,nx,ny-1),
        D = zeros(F,nx-1,ny-1),
        D_is_provided = false,
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
        # this is a `Ref` ?
        glacier_idx = Ref{Sleipnir.Int}(glacier_idx),
    )
end

"""
    build_callback(model::SIA2Dmodel, cache::SIA2DCache, glacier_idx, θ) -> CallbackSet

Return a `CallbackSet` that updates the cached values of `A`, `C`, and `n` at provided time intervals.

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

    return CallbackSet(A_cb, C_cb, n_cb)
end
