using DiffEqCallbacks: PeriodicCallback

using Sleipnir: AbstractLaw
import Sleipnir: Law, init_cache, cache_type, apply_law!, is_callback_law, callback_freq

export SIA2Dmodel, SIA2DCache, initialize_iceflow_model!, initialize_iceflow_model

include("SIA2D_utils.jl")

###############################################
###### SHALLOW ICE APPROXIMATION MODELS #######
###############################################

@kwdef struct SIA2Dmodel{ALAW <: AbstractLaw, CLAW <: AbstractLaw} <: SIAmodel
    A::ALAW
    C::CLAW

    function SIA2Dmodel(A, C)
        A = something(A, _default_A_law)
        C = something(C, _default_C_law)

        new{typeof(A), typeof(C)}(A, C)
    end
end

const _default_A_law = Law{Array{Sleipnir.Float, 0}}(;
    f! = (cache, simulation, glacier_idx, t, θ) -> nothing,
    init_cache = (simulation, glacier_idx, θ) -> fill(simulation.glaciers[glacier_idx].A)
)

const _default_C_law = Law{Array{Sleipnir.Float, 0}}(;
    f! = (cache, simulation, glacier_idx, t, θ) -> nothing,
    init_cache = (simulation, glacier_idx, θ) -> fill(simulation.glaciers[glacier_idx].C)
)

SIA2Dmodel(; A = nothing, C = nothing) = SIA2Dmodel(A, C)
SIA2Dmodel(params::Sleipnir.Parameters; A = nothing, C = nothing) = SIA2Dmodel(A, C)

"""
    mutable struct SIA2Dmodel{R <: Real, I <: Integer} <: SIAmodel

A struct storing all variables needed to compute 2D Shallow Ice Approximation (SIA) model.

# Fields
- `A::Union{Ref{R}, Vector{R}, Matrix{R}, Nothing}`: Flow rate factor.
- `n::Union{Ref{R}, Vector{R}, Matrix{R}, Nothing}`: Flow law exponent.
- `C::Union{Ref{R}, Vector{R}, Matrix{R}, Nothing}`: Sliding coefficient.
- `H₀::Matrix{R}`: Initial ice thickness.
- `H::Union{Matrix{R}, Nothing}`: Ice thickness.
- `H̄::Union{Matrix{R}, Nothing}`: Averaged ice thickness.
- `S::Matrix{R}`: Surface elevation.
- `dSdx::Union{Matrix{R}, Nothing}`: Surface slope in the x-direction.
- `dSdy::Union{Matrix{R}, Nothing}`: Surface slope in the y-direction.
- `D::Union{Matrix{R}, Nothing}`: Diffusivity.
- `D_is_provided::Union{Bool, Nothing}`: Specifies if D is provided by user or computed.
- `Dx::Union{Matrix{R}, Nothing}`: Diffusivity in the x-direction.
- `Dy::Union{Matrix{R}, Nothing}`: Diffusivity in the y-direction.
- `dSdx_edges::Union{Matrix{R}, Nothing}`: Surface slope at edges in the x-direction.
- `dSdy_edges::Union{Matrix{R}, Nothing}`: Surface slope at edges in the y-direction.
- `∇S::Union{Matrix{R}, Nothing}`: Gradient of the surface elevation.
- `∇Sy::Union{Matrix{R}, Nothing}`: Gradient of the surface elevation in the y-direction.
- `∇Sx::Union{Matrix{R}, Nothing}`: Gradient of the surface elevation in the x-direction.
- `Fx::Union{Matrix{R}, Nothing}`: Flux in the x-direction.
- `Fy::Union{Matrix{R}, Nothing}`: Flux in the y-direction.
- `Fxx::Union{Matrix{R}, Nothing}`: Second derivative of flux in the x-direction.
- `Fyy::Union{Matrix{R}, Nothing}`: Second derivative of flux in the y-direction.
- `V::Union{Matrix{R}, Nothing}`: Velocity.
- `Vx::Union{Matrix{R}, Nothing}`: Velocity in the x-direction.
- `Vy::Union{Matrix{R}, Nothing}`: Velocity in the y-direction.
- `Γ::Union{Ref{R}, Vector{R}, Matrix{R}, Nothing}`: Basal shear stress.
- `MB::Union{Matrix{R}, Nothing}`: Mass balance.
- `MB_mask::Union{AbstractArray{Bool}, Nothing}`: Mask for mass balance.
- `MB_total::Union{Matrix{R}, Nothing}`: Total mass balance.
- `glacier_idx::Union{Ref{I}, Nothing}`: Index of the glacier.
"""
@kwdef struct SIA2DCache{R <: Real, I <: Integer, A_CACHE, C_CACHE} <: SIAmodel
    A::A_CACHE
    #n::Union{Ref{R}, Vector{R}, Matrix{R}, Nothing}
    n::Vector{R}
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
    #Γ::Union{Ref{R}, Vector{R}, Matrix{R}}
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
}

"""
function init_cache(
    iceflow_model::SIA2DModel,
    glacier::AbstractGlacier,
    glacier_idx::I,
    params::Sleipnir.Parameters
) where {IF <: IceflowModel, I <: Integer}

Initialize iceflow model data structures to enable in-place mutation.

Keyword arguments
=================
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

    Γ = similar(A)

    return SIA2DCache(;
        A,
        C,
        Γ, 
        #A = isnothing(iceflow_model.A) ? [glacier.A] : iceflow_model.A,
        #n = isnothing(iceflow_model.n) ? [glacier.n] : iceflow_model.n,
        n = [glacier.n],
        #C = isnothing(iceflow_model.C) ? [glacier.C] : iceflow_model.C,
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
        #Γ = isnothing(iceflow_model.Γ) ? [0.0] : iceflow_model.Γ,
        MB = zeros(F,nx,ny),
        MB_mask = falses(nx,ny),
        MB_total = zeros(F,nx,ny),
        # this is a `Ref` ?
        glacier_idx = Ref{Sleipnir.Int}(glacier_idx),
    )
end

# may be moved to Sleipnir
function build_affect(law::Law, cache, glacier_idx, θ)
    # The let block make sure that every variable are type stable
    return let law = law, cache = cache, glacier_idx = glacier_idx, θ = θ
        function affect!(integrator)
            simulation = integrator.p
            t = integrator.t

            apply_law!(law, cache, simulation, glacier_idx, t, θ)
        end
    end
end


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

    return CallbackSet(A_cb, C_cb)
end
