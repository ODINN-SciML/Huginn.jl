
export SIA2Dmodel, initialize_iceflow_model!

include("SIA2D_utils.jl")

###############################################
###### SHALLOW ICE APPROXIMATION MODELS #######
###############################################

mutable struct SIA2Dmodel{F <: AbstractFloat, I <: Integer} <: SIAmodel
    A::Union{Ref{Real}, Nothing}
    n::Union{Ref{Real}, Nothing}
    C::Union{Ref{Real}, Nothing}
    H₀::Union{Matrix{F}, Nothing}
    H::Union{Matrix{F}, Nothing}
    H̄::Union{Matrix{F}, Nothing}
    S::Union{Matrix{F}, Nothing}
    dSdx::Union{Matrix{F}, Nothing}
    dSdy::Union{Matrix{F}, Nothing}
    D::Union{Matrix{F}, Nothing}
    Dx::Union{Matrix{F}, Nothing}
    Dy::Union{Matrix{F}, Nothing}
    dSdx_edges::Union{Matrix{F}, Nothing}
    dSdy_edges::Union{Matrix{F}, Nothing}
    ∇S::Union{Matrix{F}, Nothing}
    ∇Sy::Union{Matrix{F}, Nothing}
    ∇Sx::Union{Matrix{F}, Nothing}
    Fx::Union{Matrix{F}, Nothing}
    Fy::Union{Matrix{F}, Nothing}
    Fxx::Union{Matrix{F}, Nothing}
    Fyy::Union{Matrix{F}, Nothing}
    V::Union{Matrix{F}, Nothing}
    Vx::Union{Matrix{F}, Nothing}
    Vy::Union{Matrix{F}, Nothing}
    Γ::Union{Ref{Real}, Nothing}
    MB::Union{Matrix{F}, Nothing}
    MB_mask::Union{BitMatrix, Nothing}
    MB_total::Union{Matrix{F}, Nothing}
    glacier_idx::Union{Ref{I}, Nothing}
end

function SIA2Dmodel(params::Sleipnir.Parameters;
                    A::Union{Real, Nothing} = nothing,
                    n::Union{Real, Nothing} = nothing,
                    C::Union{Real, Nothing} = nothing,
                    H₀::Union{Matrix{F}, Nothing} = nothing,
                    H::Union{Matrix{F}, Nothing} = nothing,
                    H̄::Union{Matrix{F}, Nothing} = nothing,
                    S::Union{Matrix{F}, Nothing} = nothing,
                    dSdx::Union{Matrix{F}, Nothing} = nothing,
                    dSdy::Union{Matrix{F}, Nothing} = nothing,
                    D::Union{Matrix{F}, Nothing} = nothing,
                    Dx::Union{Matrix{F}, Nothing} = nothing,
                    Dy::Union{Matrix{F}, Nothing} = nothing,
                    dSdx_edges::Union{Matrix{F}, Nothing} = nothing,
                    dSdy_edges::Union{Matrix{F}, Nothing} = nothing,
                    ∇S::Union{Matrix{F}, Nothing} = nothing,
                    ∇Sy::Union{Matrix{F}, Nothing} = nothing,
                    ∇Sx::Union{Matrix{F}, Nothing} = nothing,
                    Fx::Union{Matrix{F}, Nothing} = nothing,
                    Fy::Union{Matrix{F}, Nothing} = nothing,
                    Fxx::Union{Matrix{F}, Nothing} = nothing,
                    Fyy::Union{Matrix{F}, Nothing} = nothing,
                    V::Union{Matrix{F}, Nothing} = nothing,
                    Vx::Union{Matrix{F}, Nothing} = nothing,
                    Vy::Union{Matrix{F}, Nothing} = nothing,
                    Γ::Union{F, Nothing} = nothing,
                    MB::Union{Matrix{F}, Nothing} = nothing,
                    MB_mask::Union{BitMatrix, Nothing} = nothing,
                    MB_total::Union{Matrix{F}, Nothing} = nothing,
                    glacier_idx::Union{I, Nothing} = nothing) where {F <: AbstractFloat, I <: Integer}
    
    ft = params.simulation.float_type
    it = params.simulation.int_type
    if !isnothing(A)
        A = Ref{F}(A)
    end
    if !isnothing(n)
        n = Ref{F}(n)
    end
    if !isnothing(C)
        A = Ref{F}(C)
    end
    if !isnothing(Γ)
        Γ = Ref{F}(Γ)
    end
    if !isnothing(glacier_idx)
        glacier_idx = Ref{I}(glacier_idx)
    end

    SIA2D_model = SIA2Dmodel{ft,it}(A, n, C, H₀, H, H̄, S, dSdx, dSdy, D, Dx, Dy, dSdx_edges, dSdy_edges,
                            ∇S, ∇Sx, ∇Sy, Fx, Fy, Fxx, Fyy, V, Vx, Vy, Γ, MB, MB_mask, MB_total, glacier_idx)

    return SIA2D_model
end

"""
function initialize_iceflow_model!(iceflow_model::IF,  
    glacier_idx::I,
    glacier::AbstractGlacier,
    params::Sleipnir.Parameters
    ) where {IF <: IceflowModel, I <: Int}

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
                                   ) where {IF <: IceflowModel, I <: Int, G <: Sleipnir.AbstractGlacier}
    nx, ny = glacier.nx, glacier.ny
    F = params.simulation.float_type
    iceflow_model.A = isnothing(iceflow_model.A) ? Ref{Real}(glacier.A) : iceflow_model.A
    iceflow_model.n = isnothing(iceflow_model.n) ? Ref{Real}(glacier.n) : iceflow_model.n
    iceflow_model.C = isnothing(iceflow_model.C) ? Ref{Real}(glacier.C) : iceflow_model.C
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
    iceflow_model.V = zeros(F,nx-1,ny-1)
    iceflow_model.Vx = zeros(F,nx-1,ny-1)
    iceflow_model.Vy = zeros(F,nx-1,ny-1)
    iceflow_model.Γ = isnothing(iceflow_model.Γ) ? Ref{Real}(0.0) : iceflow_model.Γ
    iceflow_model.MB = zeros(F,nx,ny)
    iceflow_model.MB_mask= zeros(F,nx,ny)
    iceflow_model.MB_total = zeros(F,nx,ny)
    iceflow_model.glacier_idx = Ref{I}(glacier_idx)
end

"""
function initialize_iceflow_model(iceflow_model::IF,  
    glacier_idx::I,
    glacier::AbstractGlacier,
    params::Sleipnir.Parameters
    ) where {IF <: IceflowModel, I <: Int}

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
                                   ) where {IF <: IceflowModel, I <: Int}
    nx, ny = glacier.nx, glacier.ny
    F = params.simulation.float_type
    iceflow_model.A = glacier.A
    iceflow_model.n = glacier.n
    iceflow_model.C = glacier.C
    iceflow_model.glacier_idx = glacier_idx
    # We just need initial condition to run out-of-place forward model
    iceflow_model.H₀ = deepcopy(glacier.H₀)
    iceflow_model.H  = deepcopy(glacier.H₀)
    
    iceflow_model.MB = zeros(F,nx,ny)
    iceflow_model.MB_mask = zeros(F,nx,ny)
    iceflow_model.MB_total = zeros(F,nx,ny)


end

