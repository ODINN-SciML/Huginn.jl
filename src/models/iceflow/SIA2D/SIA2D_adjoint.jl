
function diff_x_adjoint(I, Δx)
    O = zeros(Sleipnir.Float, (size(I,1)+1,size(I,2)))
    O[begin+1:end,:] += I
    O[1:end-1,:] -= I
    return O / Δx
end
function diff_y_adjoint(I, Δy)
    O = zeros(Sleipnir.Float, (size(I,1),size(I,2)+1))
    O[:,begin+1:end] += I
    O[:,1:end - 1] -= I
    return O / Δy
end
function clamp_borders_dx(dS, H, η₀, Δx)
    return max.(min.(dS,  η₀ * H[2:end, 2:end-1]/Δx), -η₀ * H[1:end-1, 2:end-1]/Δx)
end
function clamp_borders_dx_adjoint!(∂dS, ∂H, ∂C, η₀, Δx, H, dS)
    # Note: this implementation doesn't hold if H is negative
    ∂dS .= ∂C .* ((dS .< η₀ * H[2:end, 2:end-1]/Δx) .& (dS .> -η₀ * H[1:end-1, 2:end-1]/Δx))
    ∂H[1:end-1, 2:end-1] .= - (η₀ * ∂C / Δx) .* (dS .< -η₀ * H[1:end-1, 2:end-1]/Δx)
    ∂H[2:end, 2:end-1] += (η₀ * ∂C / Δx) .* (dS .> η₀ * H[2:end, 2:end-1]/Δx)
end
function clamp_borders_dy(dS, H, η₀, Δy)
    return max.(min.(dS,  η₀ * H[2:end-1, 2:end]/Δy), -η₀ * H[2:end-1, 1:end-1]/Δy)
end
function clamp_borders_dy_adjoint!(∂dS, ∂H, ∂C, η₀, Δy, H, dS)
    # Note: this implementation doesn't hold if H is negative
    ∂dS .= ∂C .* ((dS .< η₀ * H[2:end-1, 2:end]/Δy) .& (dS .> -η₀ * H[2:end-1, 1:end-1]/Δy))
    ∂H[2:end-1, 1:end-1] .= - (η₀ * ∂C / Δy) .* (dS .< -η₀ * H[2:end-1, 1:end-1]/Δy)
    ∂H[2:end-1, 2:end] += (η₀ * ∂C / Δy) .* (dS .> η₀ * H[2:end-1, 2:end]/Δy)
end
function avg_adjoint(I)
    O = zeros(Sleipnir.Float, (size(I,1)+1,size(I,2)+1))
    O[1:end-1,1:end-1] += I
    O[2:end,1:end-1] += I
    O[1:end-1,2:end] += I
    O[2:end,2:end] += I
    return 0.25*O
end
function avg_x_adjoint(I)
    O = zeros(Sleipnir.Float, (size(I, 1)+1, size(I, 2)))
    O[1:end-1,:] += I
    O[2:end,:] += I
    return 0.5*O
end
function avg_y_adjoint(I)
    O = zeros(Sleipnir.Float, (size(I, 1), size(I, 2)+1))
    O[:,1:end-1] += I
    O[:,2:end] += I
    return 0.5*O
end


"""
    SIA2D_discrete_adjoint(
        ∂dH::Matrix{R},
        H::Matrix{R},
        simulation::SIM,
        t::R;
        batch_id::Union{Nothing, I} = nothing
    )

Compute an out-of-place adjoint step of the Shallow Ice Approximation PDE.
Given an output gradient, it backpropagates the gradient to the inputs H and A.
To some extent, this function is equivalent to VJP_λ_∂SIA∂H_continuous and
VJP_λ_∂SIA∂θ_continuous.

Arguments:
- `∂dH::Matrix{R}`: Output gradient to backpropagate.
- `H::Matrix{R}`: Ice thickness which corresponds to the input state of the SIA2D.
- `simulation::SIM`: Simulation parameters.
- `t::R`: Time value, not used as SIA2D is time independent.
- `batch_id::Union{Nothing, I}`: Batch index.

Returns:
- `∂H::Matrix{R}`: Input gradient wrt H.
- `∂A::F`: Input gradient wrt A.
"""
function SIA2D_discrete_adjoint(
    ∂dH::Matrix{R},
    H::Matrix{R},
    simulation::SIM,
    t::R;
    batch_id::Union{Nothing, I} = nothing
) where {R <:Real, I <: Integer, SIM <: Simulation}

    if isnothing(batch_id)
        SIA2D_model = simulation.model.iceflow
        glacier = simulation.glaciers[SIA2D_model.glacier_idx[]]
    else
        SIA2D_model = simulation.model.iceflow[batch_id] # We pick the right iceflow model for this glacier
        glacier = simulation.glaciers[batch_id]
    end

    # Retrieve parameters
    params = simulation.parameters
    A = SIA2D_model.A
    n = SIA2D_model.n
    B = glacier.B
    Δx = glacier.Δx
    Δy = glacier.Δy
    Γ = SIA2D_model.Γ
    ρ = simulation.parameters.physical.ρ
    g = simulation.parameters.physical.g

    # First, enforce values to be positive
    map!(x -> ifelse(x>0.0,x,0.0), H, H)
    # Update glacier surface altimetry
    S = B .+ H

    # All grid variables computed in a staggered grid
    # Compute surface gradients on edges
    dSdx = diff_x(S)/Δx
    dSdy = diff_y(S)/Δy
    ∇Sx = avg_y(dSdx)
    ∇Sy = avg_x(dSdy)
    ∇S = (∇Sx.^2 .+ ∇Sy.^2).^((n[] - 1)/2)

    H̄ = avg(H)
    Γ = 2.0 * A[] * (ρ * g)^n[] / (n[]+2) # 1 / m^3 s
    D = Γ .* H̄.^(n[] + 2) .* ∇S

    # Compute flux components
    @views dSdx_edges = diff_x(S[:,2:end - 1]) / Δx
    @views dSdy_edges = diff_y(S[2:end - 1,:]) / Δy

    # Cap surface elevaton differences with the upstream ice thickness to
    # impose boundary condition of the SIA equation
    η₀ = params.physical.η₀
    dSdx_edges_clamp = clamp_borders_dx(dSdx_edges, H, η₀, Δx)
    dSdy_edges_clamp = clamp_borders_dy(dSdy_edges, H, η₀, Δy)

    Dx = avg_y(D)
    Dy = avg_x(D)

    ∂dH_inn = ∂dH[2:end-1,2:end-1]
    Fx_adjoint = diff_x_adjoint(-∂dH_inn, Δx)
    Fy_adjoint = diff_y_adjoint(-∂dH_inn, Δy)
    Dx_adjoint = avg_y_adjoint(-Fx_adjoint .* dSdx_edges_clamp)
    Dy_adjoint = avg_x_adjoint(-Fy_adjoint .* dSdy_edges_clamp)
    D_adjoint = Dx_adjoint + Dy_adjoint

    # First term
    α = (n[]+2) .* H̄.^(n[]+1) .* ∇S
    β = (n[]-1) .* H̄.^(n[]+2) .* (∇Sx.^2 .+ ∇Sy.^2).^((n[] - 3)/2)
    βx = β .* ∇Sx
    βy = β .* ∇Sy
    ∂D∂H_adj = avg_adjoint(Γ .* α .* D_adjoint) + diff_x_adjoint(avg_y_adjoint(Γ .* βx .* D_adjoint), Δx) + diff_y_adjoint(avg_x_adjoint(Γ .* βy .* D_adjoint), Δy)

    # Second term
    ∂Cx = - Fx_adjoint .* Dx
    ∂Cy = - Fy_adjoint .* Dy
    ∂dSx = zeros(Sleipnir.Float, size(dSdx_edges))
    ∂dSy = zeros(Sleipnir.Float, size(dSdy_edges))
    ∂Hlocx = zeros(Sleipnir.Float, size(H))
    ∂Hlocy = zeros(Sleipnir.Float, size(H))
    clamp_borders_dx_adjoint!(∂dSx, ∂Hlocx, ∂Cx, η₀, Δx, H, dSdx_edges)
    clamp_borders_dy_adjoint!(∂dSy, ∂Hlocy, ∂Cy, η₀, Δy, H, dSdy_edges)
    ∇adj∂dSx = zero(S); ∇adj∂dSx[:,2:end - 1] .= diff_x_adjoint(∂dSx, Δx)
    ∂C∂H_adj_x = ∇adj∂dSx + ∂Hlocx
    ∇adj∂dSy = zero(S); ∇adj∂dSy[2:end - 1,:] .= diff_y_adjoint(∂dSy, Δy)
    ∂C∂H_adj_y = ∇adj∂dSy + ∂Hlocy
    ∂C∂H_adj = ∂C∂H_adj_x + ∂C∂H_adj_y

    # Sum contributions of diffusivity and clipping
    ∂H = ∂D∂H_adj + ∂C∂H_adj
    ∂H .= ∂H.*(H.>0)

    # Gradient wrt A
    fac = 2.0 * (ρ * g)^n[] / (n[]+2)
    ∂A_spatial = fac .* avg(H).^(n[] + 2) .* ∇S .* D_adjoint
    ∂A = sum(∂A_spatial)

    return ∂H, ∂A
end
