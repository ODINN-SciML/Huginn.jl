import Sleipnir: apply_all_non_callback_laws!

"""
    SIA2D!(
        dH::Matrix{R},
        H::Matrix{R},
        simulation::SIM,
        t::R,
        θ,
    ) where {R <:Real, SIM <: Simulation}

Simulates the evolution of ice thickness in a 2D shallow ice approximation (SIA) model. Works in-place.

# Arguments
- `dH::Matrix{R}`: Matrix to store the rate of change of ice thickness.
- `H::Matrix{R}`: Matrix representing the ice thickness.
- `simulation::SIM`: Simulation object containing model parameters and state.
- `t::R`: Current simulation time.
- `θ`: Parameters of the laws to be used in the SIA. Can be `nothing` when no learnable laws are used.

# Details
This function updates the ice thickness `H` and computes the rate of change `dH` using the shallow ice approximation in 2D.
It retrieves necessary parameters from the `simulation` object, enforces positive ice thickness values, updates glacier surface altimetry and computes surface gradients.
It then applies the necessary laws that are not updated via callbacks (`A`, `C`, `n` or `U` depending on the use-case) and computes the flux components, and flux divergence.

# Notes
- The function operates on a staggered grid for computing gradients and fluxes.
- Surface elevation differences are capped using upstream ice thickness to impose boundary conditions.
- The function modifies the input matrices `dH` and `H` in-place.

See also `SIA2D`
"""
function SIA2D!(
    dH::Matrix{R},
    H::Matrix{R},
    simulation::SIM,
    t::R,
    θ,
) where {R <:Real, SIM <: Simulation}
    SIA2D_model = simulation.model.iceflow
    SIA2D_cache = simulation.cache.iceflow
    glacier_idx = SIA2D_cache.glacier_idx
    glacier = simulation.glaciers[glacier_idx]

    params = simulation.parameters

    (;
        H̄, S, dSdx, dSdy,
        D, Dx, Dy,
        dSdx_edges, dSdy_edges,
        ∇S, ∇Sx, ∇Sy,
        Fx, Fy, Fxx, Fyy, Γ,
    ) = SIA2D_cache

    (;Δx, Δy, B) = glacier

    (;ρ, g, ϵ) = params.physical

    # First, enforce values to be positive
    Hclip = map(x -> ifelse(x > 0.0, x, 0.0), H) # We cannot change H otherwise Enzyme cannot differentiate it using Const (which is the case in SciMLSensitivity)
    # Update glacier surface altimetry
    S .= B .+ Hclip

    # All grid variables computed in a staggered grid
    # Compute surface gradients on edges
    diff_x!(dSdx, S, Δx)
    diff_y!(dSdy, S, Δy)
    avg_y!(∇Sx, dSdx)
    avg_x!(∇Sy, dSdy)
    @. ∇S = (∇Sx^2 + ∇Sy^2 .+ ϵ)^(1/2) # Add a very small constant for numerical stability of AD
    avg!(H̄, Hclip)

    apply_all_non_callback_laws!(SIA2D_model, SIA2D_cache, simulation, glacier_idx, t, θ)
    (; A, C, n, Y, U) = SIA2D_cache

    if SIA2D_model.U_is_provided
        # Compute D from U
        D .= U.value .* H̄
    elseif SIA2D_model.Y_is_provided
        # Compute D from Y, H and the exponent defined in target
        n_H = SIA2D_model.n_H_is_provided ? SIA2D_cache.n_H : n.value
        n_∇S = SIA2D_model.n_∇S_is_provided ? SIA2D_cache.n_∇S : n.value
        gravity_term = (ρ * g).^n.value
        Γ_no_A = @. 2.0 * gravity_term / (n.value + 2)
        D .= (C.value .* gravity_term .+ Y.value .* Γ_no_A .* H̄) .* H̄.^(n_H .+ 1) .* ∇S.^(n_∇S .- 1)
    else
        # Compute D from A, C and n
        gravity_term = (ρ * g).^n.value
        @. Γ.value = 2.0 * A.value * gravity_term / (n.value + 2) # 1 / m^3 s
        @. D = (C.value * gravity_term + Γ.value * H̄) * H̄^(n.value + 1) * ∇S ^ (n.value - 1)
    end

    # Compute flux components
    @views diff_x!(dSdx_edges, S[:,2:end - 1], Δx)
    @views diff_y!(dSdy_edges, S[2:end - 1,:], Δy)

    # Cap surface elevaton differences with the upstream ice thickness to
    # imporse boundary condition of the SIA equation
    η₀ = params.physical.η₀
    dSdx_edges .= @views @. min(dSdx_edges,  η₀ * Hclip[2:end, 2:end-1] / Δx)
    dSdx_edges .= @views @. max(dSdx_edges, -η₀ * Hclip[1:end-1, 2:end-1] / Δx)
    dSdy_edges .= @views @. min(dSdy_edges,  η₀ * Hclip[2:end-1, 2:end] / Δy)
    dSdy_edges .= @views @. max(dSdy_edges, -η₀ * Hclip[2:end-1, 1:end-1] / Δy)

    avg_y!(Dx, D)
    avg_x!(Dy, D)
    Fx .= .-Dx .* dSdx_edges
    Fy .= .-Dy .* dSdy_edges

    # Flux divergence
    diff_x!(Fxx, Fx, Δx)
    diff_y!(Fyy, Fy, Δy)
    inn(dH) .= .-(Fxx .+ Fyy)
end

# Dummy function to bypass ice flow
function noSIA2D!(dH::Matrix{R}, H::Matrix{R}, simulation::SIM, t::R) where {R <: Real,SIM <: Simulation}
    return nothing
end

"""
    function apply_all_non_callback_laws!(
        SIA2D_model::SIA2Dmodel,
        SIA2D_cache::SIA2DCache,
        simulation,
        glacier_idx::Integer,
        t::Real,
        θ,
    )

Applies the different laws required by the SIA2D glacier model for a given glacier and simulation state.
If `U_is_provided` is `true` in `SIA2D_model` and `U` is not a callback law, it applies the law for `U` only.
Otherwise if `Y_is_provided` and `Y` is not a callback law, it applies the law for `Y` only.
Finally, if `U_is_provided` and `Y_is_provided` are false, the function checks and applies the laws for `A`, `C`, and `n`, unless they are defined as "callback" laws (i.e., handled as callbacks by the ODE solver).
Results are written in-place to the cache for subsequent use in the simulation step.

# Arguments
- `SIA2D_model`: The model object containing the laws (`A`, `C`, `n`, `Y` and `U`).
- `SIA2D_cache`: A cache object to store the evaluated values of the laws (`A`, `C`, `n`, `Y` and `U`) for the current step.
- `simulation`: The simulation object.
- `glacier_idx::Integer`: Index of the glacier being simulated, used to select data for multi-glacier simulations.
- `t::Real`: Current simulation time.
- `θ`: Parameters of the laws to be used in the SIA. Can be `nothing` when no learnable laws are used.

# Notes
- The function mutates the contents of `SIA2D_cache`.
- "Callback" laws are skipped, as they are expected to be handled outside this function.
- This function is typically called at each simulation time step for each glacier.
"""
function apply_all_non_callback_laws!(
    SIA2D_model::SIA2Dmodel,
    SIA2D_cache::SIA2DCache,
    simulation,
    glacier_idx::Integer,
    t::Real,
    θ,
)
    # Compute A, C, n, Y or U
    if SIA2D_model.U_is_provided
        if SIA2D_model.apply_U_in_SIA
            apply_law!(SIA2D_model.U, SIA2D_cache.U, simulation, glacier_idx, t, θ)
        end
    elseif SIA2D_model.Y_is_provided
        if SIA2D_model.apply_Y_in_SIA
            apply_law!(SIA2D_model.Y, SIA2D_cache.Y, simulation, glacier_idx, t, θ)
        end
    else
        if SIA2D_model.apply_A_in_SIA
            apply_law!(SIA2D_model.A, SIA2D_cache.A, simulation, glacier_idx, t, θ)
        end
        if SIA2D_model.apply_C_in_SIA
            apply_law!(SIA2D_model.C, SIA2D_cache.C, simulation, glacier_idx, t, θ)
        end
        if SIA2D_model.apply_n_in_SIA
            apply_law!(SIA2D_model.n, SIA2D_cache.n, simulation, glacier_idx, t, θ)
        end
    end
end

"""
    precompute_all_VJPs_laws!(
        SIA2D_model::SIA2Dmodel,
        SIA2D_cache::SIA2DCache,
        simulation::Prediction,
        glacier_idx::Integer,
        t::Real,
        θ,
    )

Function that does nothing and its existence is just to support
multiple dispatch. The implementation that is useful is available
in ODINN when simulation is a `FunctionalInversion` object.
"""
precompute_all_VJPs_laws!(
    SIA2D_model::SIA2Dmodel,
    SIA2D_cache::SIA2DCache,
    simulation::Prediction,
    glacier_idx::Integer,
    t::Real,
    θ,
) = nothing

"""
    avg_surface_V!(simulation::SIM, t::R, θ) where {SIM <: Simulation, R <: Real}

Calculate the average surface velocity for a given simulation.

# Arguments
- `simulation::SIM`: A simulation object of type `SIM` which is a subtype of `Simulation`.
- `t::R`: Current simulation time.
- `θ`: Parameters of the laws to be used in the SIA. Can be `nothing` when no learnable laws are used.

# Description
This function computes the average surface velocity components (`Vx` and `Vy`) and the resultant velocity (`V`)
for the ice flow model within the given simulation. It first calculates the surface velocities at the initial
and current states, then averages these velocities and updates the ice flow model's velocity fields.

# Notes
- The function currently uses a simple averaging method and may need more datapoints for better interpolation.
"""
function avg_surface_V!(simulation::SIM, t::R, θ) where {SIM <: Simulation, R <: Real}
    # TODO: Add more datapoints to better interpolate this
    iceflow_cache = simulation.cache.iceflow

    Vx₀, Vy₀ = surface_V!(iceflow_cache.H₀, simulation, t, θ)
    Vx,  Vy  = surface_V!(iceflow_cache.H,  simulation, t, θ)

    inn1(iceflow_cache.Vx) .= (Vx₀ .+ Vx)./2.0
    inn1(iceflow_cache.Vy) .= (Vy₀ .+ Vy)./2.0
    iceflow_cache.V .= (iceflow_cache.Vx.^2 .+ iceflow_cache.Vy.^2).^(1/2)
end

"""
    surface_V!(H::Matrix{<:Real}, simulation::SIM, t::R, θ) where {SIM <: Simulation, R <: Real}

Compute the surface velocities of a glacier using the Shallow Ice Approximation (SIA) in 2D.

# Arguments
- `H::Matrix{<:Real}`: The ice thickness matrix.
- `simulation::SIM`: The simulation object containing parameters and model information.
- `t::R`: Current simulation time.
- `θ`: Parameters of the laws to be used in the SIA. Can be `nothing` when no learnable laws are used.

# Returns
- `Vx`: The x-component of the surface velocity.
- `Vy`: The y-component of the surface velocity.

# Description
This function updates the glacier surface altimetry and computes the surface gradients on edges using a staggered grid. It then calculates the surface velocities based on the Shallow Ice Approximation (SIA) model.

# Details
- `params`: The simulation parameters.
- `iceflow_model`: The ice flow model from the simulation.
- `glacier`: The glacier object from the simulation.
- `B`: The bedrock elevation matrix.
- `H̄`: The average ice thickness matrix.
- `dSdx`, `dSdy`: The surface gradient matrices in x and y directions.
- `∇S`, `∇Sx`, `∇Sy`: The gradient magnitude and its components.
- `Γꜛ`: The surface stress.
- `D`: The diffusivity matrix.
- `A`: The flow rate factor.
- `n`: The flow law exponent.
- `C`: The sliding coefficient.
- `Δx`, `Δy`: The grid spacing in x and y directions.
- `ρ`: The ice density.
- `g`: The gravitational acceleration.

The function computes the surface gradients, averages the ice thickness, and calculates the surface stress and diffusivity. Finally, it computes the surface velocities `Vx` and `Vy` based on the gradients and diffusivity.
"""
function surface_V!(H::Matrix{<:Real}, simulation::SIM, t::R, θ) where {SIM <: Simulation, R <: Real}
    params::Sleipnir.Parameters = simulation.parameters
    iceflow_model = simulation.model.iceflow
    iceflow_cache = simulation.cache.iceflow
    glacier_idx = iceflow_cache.glacier_idx
    glacier = simulation.glaciers[glacier_idx]
    B = glacier.B
    H̄ = iceflow_cache.H̄
    dSdx = iceflow_cache.dSdx
    dSdy = iceflow_cache.dSdy
    ∇S = iceflow_cache.∇S
    ∇Sx = iceflow_cache.∇Sx
    ∇Sy = iceflow_cache.∇Sy
    Γꜛ = iceflow_cache.Γ
    Δx = glacier.Δx
    Δy = glacier.Δy
    (; ρ, g) = params.physical

    # Update glacier surface altimetry
    S = B .+ H

    # All grid variables computed in a staggered grid
    # Compute surface gradients on edges
    diff_x!(dSdx, S, Δx)
    diff_y!(dSdy, S, Δy)
    avg_y!(∇Sx, dSdx)
    avg_x!(∇Sy, dSdy)
    ∇S .= (∇Sx.^2 .+ ∇Sy.^2).^(1/2)
    avg!(H̄, H)

    apply_all_non_callback_laws!(iceflow_model, iceflow_cache, simulation, glacier_idx, t, θ)
    (; A, C, n, Y, U) = iceflow_cache

    D = if iceflow_model.U_is_provided
        # With a U law we can only compute the surface velocity with an approximation as it would require to integrate the diffusivity wrt H
        U.value
    elseif iceflow_model.Y_is_provided
        # With a Y law we can only compute the surface velocity with an approximation as it would require to integrate the diffusivity wrt H
        n_H = iceflow_model.n_H_is_provided ? iceflow_cache.n_H : n.value
        n_∇S = iceflow_model.n_∇S_is_provided ? iceflow_cache.n_∇S : n.value
        gravity_term = (ρ * g).^n.value
        Γ_no_A = @. 2.0 * gravity_term / (n.value + 2)
        (C.value .* gravity_term .+ Y.value .* Γ_no_A .* H̄) .* H̄.^n_H .* ∇S.^(n_∇S .- 1)
    else
        gravity_term = (ρ * g).^n.value
        @. Γꜛ.value = 2.0 * A.value * gravity_term / (n.value+1) # surface stress (not average)  # 1 / m^3 s
        @. (C.value * (n.value+2) * gravity_term + Γꜛ.value) * H̄^(n.value + 1) * ∇S .^ (n.value - 1)
    end

    # Compute averaged surface velocities
    Vx = .-D .* ∇Sx
    Vy = .-D .* ∇Sy

    return Vx, Vy
end

"""
    surface_V(
        H::Matrix{R},
        simulation::SIM,
        t::Real,
        θ,
    ) where {R <: Real, SIM <: Simulation}

Compute the surface velocities of a glacier using the Shallow Ice Approximation (SIA) in 2D.

# Arguments
- `H::Matrix{R}`: Ice thickness matrix.
- `simulation::SIM`: Simulation object containing parameters and model information.
- `t::R`: Current simulation time.
- `θ`: Parameters of the laws to be used in the SIA. Can be `nothing` when no learnable laws are used.

# Returns
- `Vx`: Matrix of surface velocities in the x-direction.
- `Vy`: Matrix of surface velocities in the y-direction.

# Details
This function computes the surface velocities of a glacier by updating the glacier surface altimetry and calculating the surface gradients on the edges. It uses a staggered grid approach to compute the gradients and velocities.

# Notes
- The function assumes that the `simulation` object contains the necessary parameters and model information.
"""
function surface_V(
    H::Matrix{R},
    simulation::SIM,
    t::Real,
    θ,
) where {R <: Real, SIM <: Simulation}
    params::Sleipnir.Parameters = simulation.parameters
    iceflow_model = simulation.model.iceflow
    iceflow_cache = simulation.cache.iceflow
    glacier_idx = iceflow_cache.glacier_idx
    glacier = simulation.glaciers[glacier_idx]
    B = glacier.B
    Δx = glacier.Δx
    Δy = glacier.Δy
    (; ρ, g) = params.physical

    # Update glacier surface altimetry
    S = B .+ H

    # All grid variables computed in a staggered grid
    # Compute surface gradients on edges
    dSdx = diff_x(S) / Δx
    dSdy = diff_y(S) / Δy
    ∇S = (avg_y(dSdx).^2 .+ avg_x(dSdy).^2).^(1/2)
    H̄ = avg(H)

    # Store temporary variables for use with the laws
    iceflow_cache.∇S .= ∇S
    iceflow_cache.H̄ .= H̄

    apply_all_non_callback_laws!(iceflow_model, iceflow_cache, simulation, glacier_idx, t, θ)
    (; A, C, n, Y, U) = iceflow_cache

    D = if iceflow_model.U_is_provided
        # With a U law we can only compute the surface velocity with an approximation as it would require to integrate the diffusivity wrt H
        U.value
    elseif iceflow_model.Y_is_provided
        # With a Y law we can only compute the surface velocity with an approximation as it would require to integrate the diffusivity wrt H
        n_H = iceflow_model.n_H_is_provided ? iceflow_cache.n_H : n.value
        n_∇S = iceflow_model.n_∇S_is_provided ? iceflow_cache.n_∇S : n.value
        gravity_term = (ρ * g).^n.value
        Γ_no_A = @. 2.0 * gravity_term / (n.value + 2)
        (C.value .* gravity_term .+ Y.value .* Γ_no_A .* H̄) .* H̄.^n_H .* ∇S.^(n_∇S .- 1)
    else
        gravity_term = (ρ * g).^n.value
        Γꜛ = @. 2.0 * A.value * gravity_term / (n.value+1) # surface stress (not average)  # 1 / m^3 s
        (C.value .* (n.value.+2) .* gravity_term .+ Γꜛ) .* H̄.^(n.value .+ 1) .* ∇S .^ (n.value .- 1)
    end

    # Compute averaged surface velocities
    Vx = - D .* avg_y(dSdx)
    Vy = - D .* avg_x(dSdy)

    return Vx, Vy
end

"""
    V_from_H(
        simulation::SIM,
        H::Matrix{F},
        t::Real,
        θ,
    ) where {F <: AbstractFloat, SIM <: Simulation}

Compute surface velocity from ice thickness using the SIA model.
It relies on `surface_V` to compute `Vx` and `Vy` and it additionally computes the
magnitude of the velocity `V`.

Arguments:
- `simulation::SIM`: The simulation structure used to retrieve the physical
    parameters.
- `H::Matrix{F}`: The ice thickness matrix.
- `t::R`: Current simulation time.
- `θ`: Parameters of the laws to be used in the SIA. Can be `nothing` when no learnable laws are used.

Returns:
- `Vx`: x axis component of the surface velocity.
- `Vy`: y axis component of the surface velocity.
- `V`: Magnitude velocity.
"""
function V_from_H(
    simulation::SIM,
    H::Matrix{F},
    t::Real,
    θ,
) where {F <: AbstractFloat, SIM <: Simulation}
    Vx_in, Vy_in = surface_V(H, simulation, t, θ)
    Vx = zero(H)
    Vy = zero(H)
    inn1(Vx) .= Vx_in
    inn1(Vy) .= Vy_in
    V = (Vx.^2 .+ Vy.^2).^(1/2)
    return Vx, Vy, V
end
