
export SIA2Dmodel, initialize_iceflow_model!, initialize_iceflow_model

include("SIA2D_utils.jl")
include("SIA2D_adjoint.jl")

###############################################
###### SHALLOW ICE APPROXIMATION MODELS #######
###############################################

"""
    mutable struct SIA2Dmodel{R <: Real, I <: Integer} <: SIAmodel

A mutable struct representing a 2D Shallow Ice Approximation (SIA) model.

# Fields
- `A::Union{Ref{R}, Nothing}`: Flow rate factor.
- `n::Union{Ref{R}, Nothing}`: Flow law exponent.
- `C::Union{Ref{R}, Matrix{R}, Nothing}`: Sliding coefficient.
- `H₀::Union{Matrix{R}, Nothing}`: Initial ice thickness.
- `H::Union{Matrix{R}, Nothing}`: Ice thickness.
- `H̄::Union{Matrix{R}, Nothing}`: Averaged ice thickness.
- `S::Union{Matrix{R}, Nothing}`: Surface elevation.
- `dSdx::Union{Matrix{R}, Nothing}`: Surface slope in the x-direction.
- `dSdy::Union{Matrix{R}, Nothing}`: Surface slope in the y-direction.
- `D::Union{Matrix{R}, Nothing}`: Diffusivity.
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
- `Γ::Union{Ref{R}, Nothing}`: Basal shear stress.
- `MB::Union{Matrix{R}, Nothing}`: Mass balance.
- `MB_mask::Union{AbstractArray{Bool}, Nothing}`: Mask for mass balance.
- `MB_total::Union{Matrix{R}, Nothing}`: Total mass balance.
- `glacier_idx::Union{Ref{I}, Nothing}`: Index of the glacier.
"""
mutable struct SIA2Dmodel{R <: Real, I <: Integer} <: SIAmodel
    A::Union{Ref{R}, Vector{R}, Matrix{R}, Nothing}
    n::Union{Ref{R}, Vector{R}, Matrix{R}, Nothing}
    C::Union{Ref{R}, Vector{R}, Matrix{R}, Nothing}
    H₀::Union{Matrix{R}, Nothing}
    H::Union{Matrix{R}, Nothing}
    H̄::Union{Matrix{R}, Nothing}
    S::Union{Matrix{R}, Nothing}
    dSdx::Union{Matrix{R}, Nothing}
    dSdy::Union{Matrix{R}, Nothing}
    D::Union{Matrix{R}, Nothing}
    Dx::Union{Matrix{R}, Nothing}
    Dy::Union{Matrix{R}, Nothing}
    dSdx_edges::Union{Matrix{R}, Nothing}
    dSdy_edges::Union{Matrix{R}, Nothing}
    ∇S::Union{Matrix{R}, Nothing}
    ∇Sy::Union{Matrix{R}, Nothing}
    ∇Sx::Union{Matrix{R}, Nothing}
    Fx::Union{Matrix{R}, Nothing}
    Fy::Union{Matrix{R}, Nothing}
    Fxx::Union{Matrix{R}, Nothing}
    Fyy::Union{Matrix{R}, Nothing}
    V::Union{Matrix{R}, Nothing}
    Vx::Union{Matrix{R}, Nothing}
    Vy::Union{Matrix{R}, Nothing}
    Γ::Union{Ref{R}, Vector{R}, Matrix{R}, Nothing}
    MB::Union{Matrix{R}, Nothing}
    MB_mask::Union{AbstractArray{Bool}, Nothing}
    MB_total::Union{Matrix{R}, Nothing}
    glacier_idx::Union{Ref{I}, Nothing}
end

"""
    SIA2Dmodel(params::Sleipnir.Parameters; 
               A::Union{R, Nothing} = nothing,
               n::Union{R, Nothing} = nothing,
               C::Union{R, Matrix{R}, Nothing} = nothing,
               H₀::Union{Matrix{R}, Nothing} = nothing,
               H::Union{Matrix{R}, Nothing} = nothing,
               H̄::Union{Matrix{R}, Nothing} = nothing,
               S::Union{Matrix{R}, Nothing} = nothing,
               dSdx::Union{Matrix{R}, Nothing} = nothing,
               dSdy::Union{Matrix{R}, Nothing} = nothing,
               D::Union{Matrix{R}, Nothing} = nothing,
               Dx::Union{Matrix{R}, Nothing} = nothing,
               Dy::Union{Matrix{R}, Nothing} = nothing,
               dSdx_edges::Union{Matrix{R}, Nothing} = nothing,
               dSdy_edges::Union{Matrix{R}, Nothing} = nothing,
               ∇S::Union{Matrix{R}, Nothing} = nothing,
               ∇Sy::Union{Matrix{R}, Nothing} = nothing,
               ∇Sx::Union{Matrix{R}, Nothing} = nothing,
               Fx::Union{Matrix{R}, Nothing} = nothing,
               Fy::Union{Matrix{R}, Nothing} = nothing,
               Fxx::Union{Matrix{R}, Nothing} = nothing,
               Fyy::Union{Matrix{R}, Nothing} = nothing,
               V::Union{Matrix{R}, Nothing} = nothing,
               Vx::Union{Matrix{R}, Nothing} = nothing,
               Vy::Union{Matrix{R}, Nothing} = nothing,
               Γ::Union{R, Nothing} = nothing,
               MB::Union{Matrix{R}, Nothing} = nothing,
               MB_mask::Union{BitMatrix, Nothing} = nothing,
               MB_total::Union{Matrix{R}, Nothing} = nothing,
               glacier_idx::Union{I, Nothing} = nothing) where {I <: Integer, R <: Real}

Constructs a new `SIA2Dmodel` object with the given parameters.

# Arguments
- `params::Sleipnir.Parameters`: Simulation parameters.
- `A::Union{R, Nothing}`: Flow law parameter (default: `nothing`).
- `n::Union{R, Nothing}`: Flow law exponent (default: `nothing`).
- `C::Union{R, Matrix{R}, Nothing}`: Basal sliding parameter (default: `nothing`).
- `H₀::Union{Matrix{R}, Nothing}`: Initial ice thickness (default: `nothing`).
- `H::Union{Matrix{R}, Nothing}`: Ice thickness (default: `nothing`).
- `H̄::Union{Matrix{R}, Nothing}`: Averaged ice thickness (default: `nothing`).
- `S::Union{Matrix{R}, Nothing}`: Surface elevation (default: `nothing`).
- `dSdx::Union{Matrix{R}, Nothing}`: Surface slope in x-direction (default: `nothing`).
- `dSdy::Union{Matrix{R}, Nothing}`: Surface slope in y-direction (default: `nothing`).
- `D::Union{Matrix{R}, Nothing}`: Diffusivity (default: `nothing`).
- `Dx::Union{Matrix{R}, Nothing}`: Diffusivity in x-direction (default: `nothing`).
- `Dy::Union{Matrix{R}, Nothing}`: Diffusivity in y-direction (default: `nothing`).
- `dSdx_edges::Union{Matrix{R}, Nothing}`: Surface slope at edges in x-direction (default: `nothing`).
- `dSdy_edges::Union{Matrix{R}, Nothing}`: Surface slope at edges in y-direction (default: `nothing`).
- `∇S::Union{Matrix{R}, Nothing}`: Gradient of surface elevation (default: `nothing`).
- `∇Sy::Union{Matrix{R}, Nothing}`: Gradient of surface elevation in y-direction (default: `nothing`).
- `∇Sx::Union{Matrix{R}, Nothing}`: Gradient of surface elevation in x-direction (default: `nothing`).
- `Fx::Union{Matrix{R}, Nothing}`: Flux in x-direction (default: `nothing`).
- `Fy::Union{Matrix{R}, Nothing}`: Flux in y-direction (default: `nothing`).
- `Fxx::Union{Matrix{R}, Nothing}`: Second derivative of flux in x-direction (default: `nothing`).
- `Fyy::Union{Matrix{R}, Nothing}`: Second derivative of flux in y-direction (default: `nothing`).
- `V::Union{Matrix{R}, Nothing}`: Velocity (default: `nothing`).
- `Vx::Union{Matrix{R}, Nothing}`: Velocity in x-direction (default: `nothing`).
- `Vy::Union{Matrix{R}, Nothing}`: Velocity in y-direction (default: `nothing`).
- `Γ::Union{R, Nothing}`: Auxiliary matrix (default: `nothing`).
- `MB::Union{Matrix{R}, Nothing}`: Mass balance (default: `nothing`).
- `MB_mask::Union{BitMatrix, Nothing}`: Mask for mass balance (default: `nothing`).
- `MB_total::Union{Matrix{R}, Nothing}`: Total mass balance (default: `nothing`).
- `glacier_idx::Union{I, Nothing}`: Index of the glacier (default: `nothing`).

# Returns
- `SIA2Dmodel`: A new `SIA2Dmodel` object.
"""
function SIA2Dmodel(params::Sleipnir.Parameters;
                    A::Union{R, Vector{R}, Matrix{R}, Nothing} = nothing,
                    n::Union{R, Vector{R}, Matrix{R}, Nothing} = nothing,
                    C::Union{R, Vector{R}, Matrix{R}, Nothing} = nothing,
                    H₀::Union{Matrix{R}, Nothing} = nothing,
                    H::Union{Matrix{R}, Nothing} = nothing,
                    H̄::Union{Matrix{R}, Nothing} = nothing,
                    S::Union{Matrix{R}, Nothing} = nothing,
                    dSdx::Union{Matrix{R}, Nothing} = nothing,
                    dSdy::Union{Matrix{R}, Nothing} = nothing,
                    D::Union{Matrix{R}, Nothing} = nothing,
                    Dx::Union{Matrix{R}, Nothing} = nothing,
                    Dy::Union{Matrix{R}, Nothing} = nothing,
                    dSdx_edges::Union{Matrix{R}, Nothing} = nothing,
                    dSdy_edges::Union{Matrix{R}, Nothing} = nothing,
                    ∇S::Union{Matrix{R}, Nothing} = nothing,
                    ∇Sy::Union{Matrix{R}, Nothing} = nothing,
                    ∇Sx::Union{Matrix{R}, Nothing} = nothing,
                    Fx::Union{Matrix{R}, Nothing} = nothing,
                    Fy::Union{Matrix{R}, Nothing} = nothing,
                    Fxx::Union{Matrix{R}, Nothing} = nothing,
                    Fyy::Union{Matrix{R}, Nothing} = nothing,
                    V::Union{Matrix{R}, Nothing} = nothing,
                    Vx::Union{Matrix{R}, Nothing} = nothing,
                    Vy::Union{Matrix{R}, Nothing} = nothing,
                    Γ::Union{R, Vector{R}, Matrix{R}, Nothing} = nothing,
                    MB::Union{Matrix{R}, Nothing} = nothing,
                    MB_mask::Union{BitMatrix, Nothing} = nothing,
                    MB_total::Union{Matrix{R}, Nothing} = nothing,
                    glacier_idx::Union{I, Nothing} = nothing) where {I <: Integer, R <: Real}

    ft = Sleipnir.Float
    it = Sleipnir.Int
    if !isnothing(A)
        A = [A]
    end
    if !isnothing(n)
        n = [n]
    end
    if !isnothing(C)
        C = [C]
    end
    if !isnothing(Γ)
        Γ = [Γ]
    end
    if !isnothing(glacier_idx)
        glacier_idx = Ref{I}(glacier_idx)
    end

    SIA2D_model = SIA2Dmodel{ft, it}(A, n, C, H₀, H, H̄, S, dSdx, dSdy, D, Dx, Dy, dSdx_edges, dSdy_edges,
                            ∇S, ∇Sx, ∇Sy, Fx, Fy, Fxx, Fyy, V, Vx, Vy, Γ, MB, MB_mask, MB_total, glacier_idx)

    return SIA2D_model
end

"""
function initialize_iceflow_model!(iceflow_model::IF,  
    glacier_idx::I,
    glacier::AbstractGlacier,
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
function initialize_iceflow_model!(iceflow_model::IF,
                                   glacier_idx::I,
                                   glacier::G,
                                   params::Sleipnir.Parameters
                                   ) where {IF <: IceflowModel, I <: Integer, G <: Sleipnir.AbstractGlacier}
    nx, ny = glacier.nx, glacier.ny
    F = Sleipnir.Float
    iceflow_model.A = isnothing(iceflow_model.A) ? [glacier.A] : iceflow_model.A
    iceflow_model.n = isnothing(iceflow_model.n) ? [glacier.n] : iceflow_model.n
    iceflow_model.C = isnothing(iceflow_model.C) ? [glacier.C] : iceflow_model.C
    iceflow_model.H₀ = deepcopy(glacier.H₀)
    iceflow_model.H = deepcopy(glacier.H₀)
    iceflow_model.H̄ = zeros(F,nx-1,ny-1)
    iceflow_model.S = deepcopy(glacier.S)
    iceflow_model.dSdx = zeros(F,nx-1,ny)
    iceflow_model.dSdy= zeros(F,nx,ny-1)
    iceflow_model.D = zeros(F,nx-1,ny-1)
    iceflow_model.Dx = zeros(F,nx-1,ny-2)
    iceflow_model.Dy = zeros(F,nx-2,ny-1)
    iceflow_model.dSdx_edges = zeros(F,nx-1,ny-2)
    iceflow_model.dSdy_edges = zeros(F,nx-2,ny-1) 
    iceflow_model.∇S = zeros(F,nx-1,ny-1)
    iceflow_model.∇Sx = zeros(F,nx-1,ny-1)
    iceflow_model.∇Sy = zeros(F,nx-1,ny-1)
    iceflow_model.Fx = zeros(F,nx-1,ny-2)
    iceflow_model.Fy = zeros(F,nx-2,ny-1)
    iceflow_model.Fxx = zeros(F,nx-2,ny-2)
    iceflow_model.Fyy = zeros(F,nx-2,ny-2)
    iceflow_model.V = zeros(F,nx,ny)
    iceflow_model.Vx = zeros(F,nx,ny)
    iceflow_model.Vy = zeros(F,nx,ny)
    iceflow_model.Γ = isnothing(iceflow_model.Γ) ? [0.0] : iceflow_model.Γ
    iceflow_model.MB = zeros(F,nx,ny)
    iceflow_model.MB_mask= zeros(F,nx,ny)
    iceflow_model.MB_total = zeros(F,nx,ny)
    iceflow_model.glacier_idx = Ref{Sleipnir.Int}(glacier_idx)
end

"""
function initialize_iceflow_model(iceflow_model::IF,  
    glacier_idx::I,
    glacier::AbstractGlacier,
    params::Sleipnir.Parameters
    ) where {IF <: IceflowModel, I <: Integer}

Initialize iceflow model data structures to enable out-of-place mutation.

Keyword arguments
=================
    - `iceflow_model`: Iceflow model used for simulation. 
    - `glacier_idx`: Index of glacier.
    - `glacier`: `Glacier` to provide basic initial state of the ice flow model.
    - `parameters`: `Parameters` to configure some physical variables.
"""
function initialize_iceflow_model(iceflow_model::IF,  
                                   glacier_idx::I,
                                   glacier::Sleipnir.AbstractGlacier,
                                   params::Sleipnir.Parameters
                                   ) where {IF <: IceflowModel, I <: Integer}
    F = Sleipnir.Float
    nx, ny = glacier.nx, glacier.ny
    iceflow_model.A = [glacier.A]
    iceflow_model.n = [glacier.n]
    iceflow_model.glacier_idx = Ref{I}(glacier_idx)
    iceflow_model.H₀ = deepcopy(glacier.H₀)
    iceflow_model.H  = deepcopy(glacier.H₀)
    iceflow_model.S = deepcopy(glacier.S)
    iceflow_model.V = zeros(F,nx,ny)
    iceflow_model.Vx = zeros(F,nx,ny)
    iceflow_model.Vy = zeros(F,nx,ny)
    # Initialize MB matrices for in-place MB operations
    iceflow_model.MB = zeros(F,nx,ny)
    iceflow_model.MB_mask = zeros(I,nx,ny)
    iceflow_model.MB_total = zeros(F,nx,ny)
end

