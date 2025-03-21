
function diff_x_adjoint(I, Δx)
    # TODO: improve implementation
    O = zeros(Sleipnir.Float, (size(I,1)+1,size(I,2)))
    O[begin,:] = -I[begin,:]/Δx # O[0] = -I[0] /Δx
    O[end,:] = I[end,:]/Δx # O[n-1] = I[n-2] /Δx
    O[begin+1:end-1,:] = (I[begin:end-1,:]-I[begin+1:end,:])/Δx # O[1:n-2] = (I[0:n-3]-I[1:n-2]) /Δx
    return O
end
function diff_y_adjoint(I, Δy)
    # TODO: improve implementation
    O = zeros(Sleipnir.Float, (size(I,1),size(I,2)+1))
    O[:,begin] = -I[:,begin]/Δy # O[0] = -I[0] /Δy
    O[:,end] = I[:,end]/Δy # O[n-1] = I[n-2] /Δy
    O[:,begin+1:end-1] = (I[:,begin:end-1]-I[:,begin+1:end])/Δy # O[1:n-2] = (I[0:n-3]-I[1:n-2]) /Δy
    return O
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
    # avg!(O, I) = @views @. O = (I[1:end-1,1:end-1] + I[2:end,1:end-1] + I[1:end-1,2:end] + I[2:end,2:end]) * 0.25
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
SIA2D_adjoint!(∂H, ∂dH, H, SIA2Dmodel, t)

Compute an in-place adjoint step of the Shallow Ice Approximation PDE
"""
function SIA2D_adjoint(∂dH::Matrix{R}, H::Matrix{R}, simulation::SIM, t::R) where {R <:Real, SIM <: Simulation}
    # Retrieve parameters
    SIA2D_model::SIA2Dmodel = simulation.model.iceflow
    glacier::Sleipnir.Glacier2D = simulation.glaciers[simulation.model.iceflow.glacier_idx[]]
    params::Sleipnir.Parameters = simulation.parameters
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
    # imporse boundary condition of the SIA equation
    η₀ = params.physical.η₀
    dSdx_edges_clamp = clamp_borders_dx(dSdx_edges, H, η₀, Δx)
    dSdy_edges_clamp = clamp_borders_dy(dSdy_edges, H, η₀, Δy)

    # D: nx-1, ny-1
    Dx = avg_y(D) # nx-1, ny-2
    Dy = avg_x(D) # nx-2, ny-1
    # Fx .= .-Dx .* dSdx_edges_clamp
    # Fy .= .-Dy .* dSdy_edges_clamp

    #  Flux divergence
    # diff_x!(Fxx, Fx, Δx) # nx-2, ny-2
    # diff_y!(Fyy, Fy, Δy)
    # inn(dH) .= .-(Fxx .+ Fyy)


    ########
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

    # Sum everything
    ∂H= ∂D∂H_adj + ∂C∂H_adj
    ∂H .= ∂H.*(H.>0)
    return ∂H
end
