export velocity_split, sliding_fraction

import Sleipnir: apply_all_non_callback_laws!, apply_all_callback_laws!

"""
    SIA2D!(
        dH::Matrix{R},
        H::Matrix{R},
        simulation::SIM,
        t::Real,
        θ,
    ) where {R <:Real, SIM <: Simulation}

Simulates the evolution of ice thickness in a 2D shallow ice approximation (SIA) model. Works in-place.

# Arguments

  - `dH::Matrix{R}`: Matrix to store the rate of change of ice thickness.
  - `H::Matrix{R}`: Matrix representing the ice thickness.
  - `simulation::SIM`: Simulation object containing model parameters and state.
  - `t::Real`: Current simulation time. Deliberately not tied to the element type of the state, so that the state can be dual or tracked while time stays a plain float.
  - `θ`: Parameters of the laws to be used in the SIA. Can be `nothing` when no learnable laws are used.

# Details

This function updates the ice thickness `H` and computes the rate of change `dH` using the shallow ice approximation in 2D.
It retrieves necessary parameters from the `simulation` object, enforces positive ice thickness values, updates glacier surface altimetry and computes surface gradients.
It then applies the necessary laws that are not updated via callbacks (`A`, `C`, `n` or `U` depending on the use-case) and computes the flux components, and flux divergence.
Finally it adds the surface mass balance source term (see [`add_MB!`](@ref)).

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
        t::Real,
        θ
) where {R <: Real, SIM <: Simulation}
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
        Fx, Fy, Fxx, Fyy, Γ
    ) = SIA2D_cache

    (; Δx, Δy, B) = glacier

    (; ρ, g, ϵ) = params.physical

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
    (; A, n, C, p, q, Y, U) = SIA2D_cache

    if SIA2D_model.U_is_provided
        # Compute D from U
        @. D = U.value * H̄
    elseif SIA2D_model.Y_is_provided
        # Compute D from Y, H and the exponent defined in target
        n_H = SIA2D_model.n_H_is_provided ? SIA2D_cache.n_H : n.value
        n_∇S = SIA2D_model.n_∇S_is_provided ? SIA2D_cache.n_∇S : n.value
        gravity_term = ρ * g
        sliding_term = if any(C.value .> 0)
            @. C.value * gravity_term^(p.value - q.value) * H̄^(p.value - q.value + 1) *
               ∇S^(p.value - 1)
        else
            zero(H̄)
        end
        rheology_term = @. 2.0 * gravity_term^n.value * Y.value * H̄^(n_H + 2) *
                           ∇S^(n_∇S - 1) / (n.value + 2)
        @. D = sliding_term + rheology_term
    else
        # Compute D from A, C and n
        gravity_term = ρ * g
        sliding_term = if any(C.value .> 0)
            @. C.value .* gravity_term^(p.value - q.value) * H̄^(p.value - q.value + 1) *
               ∇S^(p.value - 1)
        else
            zero(H̄)
        end
        rheology_term = @. 2.0 * A.value * gravity_term^n.value * H̄^(n.value + 2) *
                           ∇S^(n.value - 1) / (n.value + 2)
        @. D = sliding_term + rheology_term
    end

    # Compute flux components
    @views diff_x!(dSdx_edges, S[:, 2:(end - 1)], Δx)
    @views diff_y!(dSdy_edges, S[2:(end - 1), :], Δy)

    # Cap surface elevaton differences with the upstream ice thickness to
    # imporse boundary condition of the SIA equation
    η₀ = params.physical.η₀
    dSdx_edges .= @views @. min(dSdx_edges, η₀ * Hclip[2:end, 2:(end - 1)] / Δx)
    dSdx_edges .= @views @. max(dSdx_edges, -η₀ * Hclip[1:(end - 1), 2:(end - 1)] / Δx)
    dSdy_edges .= @views @. min(dSdy_edges, η₀ * Hclip[2:(end - 1), 2:end] / Δy)
    dSdy_edges .= @views @. max(dSdy_edges, -η₀ * Hclip[2:(end - 1), 1:(end - 1)] / Δy)

    avg_y!(Dx, D)
    avg_x!(Dy, D)
    Fx .= .-Dx .* dSdx_edges
    Fy .= .-Dy .* dSdy_edges

    # Flux divergence
    diff_x!(Fxx, Fx, Δx)
    diff_y!(Fyy, Fy, Δy)
    inn(dH) .= .-(Fxx .+ Fyy)

    add_MB!(dH, H, simulation, t)
    return nothing
end

# Dummy function to bypass ice flow. Mass balance still drives the surface on its own.
function noSIA2D!(dH::Matrix{R}, H::Matrix{R}, simulation::SIM,
        t::Real) where {R <: Real, SIM <: Simulation}
    # `dH` is a solver work array reused across steps, and nothing else writes it here
    dH .= zero(R)
    add_MB!(dH, H, simulation, t)
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
        θ
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
        if SIA2D_model.apply_p_in_SIA
            apply_law!(SIA2D_model.p, SIA2D_cache.p, simulation, glacier_idx, t, θ)
        end
        if SIA2D_model.apply_q_in_SIA
            apply_law!(SIA2D_model.q, SIA2D_cache.q, simulation, glacier_idx, t, θ)
        end
    end
end

"""
    apply_all_callback_laws!(
        SIA2D_model::SIA2Dmodel,
        SIA2D_cache::SIA2DCache,
        simulation,
        glacier_idx::Integer,
        t::Real,
        θ,
    )

Applies the different laws required by the SIA2D glacier model for a given glacier and simulation state.
If `U_is_provided` is `true` in `SIA2D_model` and `U` is a callback law, it applies the law for `U` only.
Otherwise if `Y_is_provided` and `Y` is a callback law, it applies the law for `Y` only.
Finally, if `U_is_provided` and `Y_is_provided` are false, the function checks and applies the laws for `A`, `C`, and `n`, if they are defined as "callback" laws (i.e., handled as callbacks by the ODE solver).
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
  - Only "callback" laws are applied.
  - This function is typically used in the manual adjoint and in the tests where only portions of the code are applied and we need to apply all the laws used in the iceflow model.
"""
function apply_all_callback_laws!(
        SIA2D_model::SIA2Dmodel,
        SIA2D_cache::SIA2DCache,
        simulation,
        glacier_idx::Integer,
        t::Real,
        θ
)
    # Compute A, C, n, Y or U
    if SIA2D_model.U_is_provided
        if !SIA2D_model.apply_U_in_SIA
            apply_law!(SIA2D_model.U, SIA2D_cache.U, simulation, glacier_idx, t, θ)
        end
    elseif SIA2D_model.Y_is_provided
        if !SIA2D_model.apply_Y_in_SIA
            apply_law!(SIA2D_model.Y, SIA2D_cache.Y, simulation, glacier_idx, t, θ)
        end
    else
        if !SIA2D_model.apply_A_in_SIA
            apply_law!(SIA2D_model.A, SIA2D_cache.A, simulation, glacier_idx, t, θ)
        end
        if !SIA2D_model.apply_C_in_SIA
            apply_law!(SIA2D_model.C, SIA2D_cache.C, simulation, glacier_idx, t, θ)
        end
        if !SIA2D_model.apply_n_in_SIA
            apply_law!(SIA2D_model.n, SIA2D_cache.n, simulation, glacier_idx, t, θ)
        end
        if !SIA2D_model.apply_p_in_SIA
            apply_law!(SIA2D_model.p, SIA2D_cache.p, simulation, glacier_idx, t, θ)
        end
        if !SIA2D_model.apply_q_in_SIA
            apply_law!(SIA2D_model.q, SIA2D_cache.q, simulation, glacier_idx, t, θ)
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
in ODINN when simulation is a `Inversion` object.
"""
function precompute_all_VJPs_laws!(
        SIA2D_model::SIA2Dmodel,
        SIA2D_cache::SIA2DCache,
        simulation::Prediction,
        glacier_idx::Integer,
        t::Real,
        θ
)
    nothing
end

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
The velocities are placed on the grid of `H` (centred on the cells), as in [`V_from_H`](@ref).

# Notes

  - The function currently uses a simple averaging method and may need more datapoints for better interpolation.    # TODO: Add more datapoints to better interpolate this
"""
function avg_surface_V!(simulation::SIM, t::R, θ) where {SIM <: Simulation, R <: Real}
    # TODO: Add more datapoints to better interpolate this
    iceflow_cache = simulation.cache.iceflow

    Vx₀, Vy₀ = surface_V(iceflow_cache.H₀, simulation, t, θ)
    Vx, Vy = surface_V(iceflow_cache.H, simulation, t, θ)

    # Same H-grid placement as the default of `V_from_H`
    inn(iceflow_cache.Vx) .= avg((Vx₀ .+ Vx) ./ 2.0)
    inn(iceflow_cache.Vy) .= avg((Vy₀ .+ Vy) ./ 2.0)
    iceflow_cache.V .= (iceflow_cache.Vx .^ 2 .+ iceflow_cache.Vy .^ 2) .^ (1/2)
end

"""
    surface_V_diffusivity!(D_slide, D_deform, H, simulation, t, θ; basis = :surface)

Diffusivity of the surface velocity, split in its sliding and deformation parts: the surface
velocity on the staggered grid is `-(D_slide + D_deform) .* (∇Sx, ∇Sy)`. It also computes
the geometry of `H` (`∇Sx`, `∇Sy`, `∇S` and `H̄`) in the ice flow cache and applies the laws.
It is used by [`surface_V`](@ref) and [`velocity_split`](@ref).
[`surface_V_inplace!`](@ref) has its own copy of the formulas, since Enzyme fails in reverse
mode on this function: a change here must be made there too.

`D_slide` and `D_deform` are staggered buffers. `basis = :flux` gives the depth average of the
deformation `(n+2)` instead of its surface value `(n+1)`. A `U` law gives the total, in
`D_deform`. The sliding term needs `p > q`, otherwise the cells without ice have a velocity.
It writes the geometry in the ice flow cache, and it does not clip `H` to zero, as the old
`surface_V` did not.
"""
function surface_V_diffusivity!(
        D_slide, D_deform, H, simulation, t, θ; basis::Symbol = :surface)
    params::Sleipnir.Parameters = simulation.parameters
    iceflow_model = simulation.model.iceflow
    iceflow_cache = simulation.cache.iceflow
    glacier_idx = iceflow_cache.glacier_idx
    glacier = simulation.glaciers[glacier_idx]
    (; H̄, dSdx, dSdy, ∇S, ∇Sx, ∇Sy) = iceflow_cache
    (; ρ, g, ϵ) = params.physical

    # Geometry on the staggered grid, with ϵ as in SIA2D!
    S = glacier.B .+ H
    diff_x!(dSdx, S, glacier.Δx)
    diff_y!(dSdy, S, glacier.Δy)
    avg_y!(∇Sx, dSdx)
    avg_x!(∇Sy, dSdy)
    ∇S .= (∇Sx .^ 2 .+ ∇Sy .^ 2 .+ ϵ) .^ (1/2)
    avg!(H̄, H)

    apply_all_non_callback_laws!(
        iceflow_model, iceflow_cache, simulation, glacier_idx, t, θ)
    (; A, n, C, p, q, Y, U) = iceflow_cache

    fill!(D_slide, zero(eltype(D_slide)))
    if iceflow_model.U_is_provided
        # U is depth averaged: the surface velocity is approximated with a shape factor
        @. D_deform = U.value / params.simulation.f_surface_velocity_factor
        return nothing
    end
    gravity_term = ρ * g
    # Plug flow: sliding velocity = sliding flux of SIA2D! / H̄ = C τ^p / N^q
    if any(C.value .> 0)
        @. D_slide = C.value * gravity_term^(p.value - q.value) *
                     H̄^(p.value - q.value) * ∇S ^ (p.value - 1)
    end
    if iceflow_model.Y_is_provided
        # Y cannot be integrated over the depth, so this is always the depth average (n+2)
        n_H = iceflow_model.n_H_is_provided ? iceflow_cache.n_H : n.value
        n_∇S = iceflow_model.n_∇S_is_provided ? iceflow_cache.n_∇S : n.value
        @. D_deform = 2.0 * gravity_term^n.value * Y.value * H̄^(n_H + 1) *
                      ∇S^(n_∇S - 1) / (n.value + 2)
    else
        # (n+1) is the surface deformation velocity, (n+2) its depth average
        shape = basis === :surface ? 1 : 2
        @. D_deform = (2.0 * A.value * gravity_term^n.value / (n.value + shape)) *
                      H̄^(n.value + 1) * ∇S ^ (n.value - 1)
    end
    return nothing
end

"""
    surface_V_inplace!(Vx, Vy, H, simulation, t, θ)

In-place variant of [`surface_V`](@ref): it writes the surface velocity components in the
H-sized buffers `Vx`, `Vy`, with the same placement as [`V_from_H`](@ref), and returns
`nothing`. Only the inner cells are written, so the buffers must start at zero. Enzyme
differentiates it in the `V_from_H` rrule of ODINN; a function that returns a tuple fails
in Enzyme in reverse mode. It does not call [`surface_V_diffusivity!`](@ref), which also fails
in Enzyme in reverse mode, so the formulas of both must be kept the same.
"""
function surface_V_inplace!(
        Vx, Vy, H::Matrix{<:Real}, simulation::SIM,
        t::R, θ) where {SIM <: Simulation, R <: Real}
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
    Δx = glacier.Δx
    Δy = glacier.Δy
    (; ρ, g, ϵ) = params.physical

    # Update glacier surface altimetry
    S = B .+ H

    # All grid variables computed in a staggered grid
    diff_x!(dSdx, S, Δx)
    diff_y!(dSdy, S, Δy)
    avg_y!(∇Sx, dSdx)
    avg_x!(∇Sy, dSdy)
    # ϵ regularizes the sqrt for AD stability (∂∇S/∂∇Sx = ∇Sx/∇S → 0/0 at flat cells),
    # matching SIA2D!; without it the reverse-mode velocity VJP is corrupted.
    ∇S .= (∇Sx .^ 2 .+ ∇Sy .^ 2 .+ ϵ) .^ (1/2)
    avg!(H̄, H)

    apply_all_non_callback_laws!(
        iceflow_model, iceflow_cache, simulation, glacier_idx, t, θ)
    (; A, n, C, p, q, Y, U) = iceflow_cache

    # Typed buffer for D: a `D = if ... end` Union breaks Enzyme
    D = zero(H̄)
    if iceflow_model.U_is_provided
        f = simulation.parameters.simulation.f_surface_velocity_factor
        @. D = U.value / f
    elseif iceflow_model.Y_is_provided
        n_H = iceflow_model.n_H_is_provided ? iceflow_cache.n_H : n.value
        n_∇S = iceflow_model.n_∇S_is_provided ? iceflow_cache.n_∇S : n.value
        gravity_term = ρ * g
        rheology_term = @. 2.0 * gravity_term^n.value * Y.value * H̄^(n_H + 1) *
                           ∇S^(n_∇S - 1) / (n.value + 2)
        # Plug flow: the surface sliding velocity is the depth average, i.e. the flux term
        # of SIA2D! divided by H̄, which recovers u_b = C τ^p / N^q
        sliding_term = zero(H̄)
        if any(C.value .> 0)
            @. sliding_term = C.value * gravity_term^(p.value - q.value) *
                              H̄^(p.value - q.value) * ∇S ^ (p.value - 1)
        end
        @. D = sliding_term + rheology_term
    else
        gravity_term = ρ * g
        rheology_term = @. (2.0 * A.value * gravity_term^n.value / (n.value+1)) *
                           H̄^(n.value + 1) * ∇S ^ (n.value - 1)
        # Plug flow: the surface sliding velocity is the depth average, i.e. the flux term
        # of SIA2D! divided by H̄, which recovers u_b = C τ^p / N^q
        sliding_term = zero(H̄)
        if any(C.value .> 0)
            @. sliding_term = C.value * gravity_term^(p.value - q.value) *
                              H̄^(p.value - q.value) * ∇S ^ (p.value - 1)
        end
        @. D = sliding_term + rheology_term
    end

    # Average back to the H grid, as the default (:centered) placement of `V_from_H` does:
    # the rrule takes its value from there, so the two have to match.
    inn(Vx) .= avg(.-D .* ∇Sx)
    inn(Vy) .= avg(.-D .* ∇Sy)
    return nothing
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
  - With `A`, the deformation is the velocity at the surface, `(n+1)`. With a `Y` law it
    is the depth average, `(n+2)`, since `Y` cannot be integrated over the depth.
  - `∇S` has a small `ϵ` inside, as in `SIA2D!`. The geometry is written in the ice flow cache.
  - The velocities are on the corners of the cells of `H`, with size `(nx-1, ny-1)`. Use
    [`V_from_H`](@ref) to get them on the grid of `H`.
"""
function surface_V(
        H::Matrix{R},
        simulation::SIM,
        t::Real,
        θ
) where {R <: Real, SIM <: Simulation}
    (; H̄, ∇Sx, ∇Sy) = simulation.cache.iceflow
    D_slide, D_deform = zero(H̄), zero(H̄)
    surface_V_diffusivity!(D_slide, D_deform, H, simulation, t, θ)
    D = D_slide .+ D_deform
    return .-D .* ∇Sx, .-D .* ∇Sy
end

"""
    velocity_split(
        H::Matrix{R},
        simulation::SIM,
        t::Real,
        θ;
        basis::Symbol = :surface,
    ) where {R <: Real, SIM <: Simulation}

Split the SIA velocity magnitude into its sliding and deformation contributions.

It computes the same geometry and laws as [`surface_V`](@ref), and returns the two terms
of the diffusivity separately. Use it to see if an inverted `C` really changes the flow.

# Arguments

  - `H::Matrix{R}`: Ice thickness matrix.
  - `simulation::SIM`: Simulation object containing parameters and model information.
  - `t::Real`: Current simulation time.
  - `θ`: Parameters of the laws. Can be `nothing` when no learnable laws are used.
  - `basis::Symbol`: `:surface` (default) compares the two velocities at the ice surface,
    as seen by velocity observations. `:flux` compares their depth averages, that is their
    part of the flux. The sliding velocity does not depend on depth, so only the
    deformation term changes, with `(n+1)` or `(n+2)`.

# Returns

A named tuple `(; V_slide, V_deform)` of staggered matrices of size `(nx-1, ny-1)`,
the velocity magnitudes in m/yr. Their sum is `‖(Vx, Vy)‖` of `surface_V` when
`basis = :surface` (up to the small `ϵ` added to `∇S`).

# Notes

  - A `U` law gives the total velocity, which cannot be split, so it throws.
  - With a `Y` law the deformation term always uses the depth average `(n+2)`, so `basis`
    has no effect.
"""
function velocity_split(
        H::Matrix{R},
        simulation::SIM,
        t::Real,
        θ;
        basis::Symbol = :surface
) where {R <: Real, SIM <: Simulation}
    basis in (:surface, :flux) ||
        throw(ArgumentError("basis must be :surface or :flux, got $(basis)"))
    simulation.model.iceflow.U_is_provided && throw(ArgumentError(
        "velocity_split: a U law provides the total velocity, it cannot be split into \
         sliding and deformation."))
    (; H̄, ∇S) = simulation.cache.iceflow
    D_slide, D_deform = zero(H̄), zero(H̄)
    surface_V_diffusivity!(D_slide, D_deform, H, simulation, t, θ; basis = basis)
    return (; V_slide = D_slide .* ∇S, V_deform = D_deform .* ∇S)
end

"""
    sliding_fraction(
        H::Matrix{R},
        simulation::SIM,
        t::Real,
        θ;
        basis::Symbol = :surface,
    ) where {R <: Real, SIM <: Simulation}

Share of the SIA velocity due to sliding rather than deformation, on the staggered grid.

Uses [`velocity_split`](@ref), see it for the meaning of `basis`. Cells with no ice return
`NaN`, since the ratio is `0/0` there. Mask them before averaging.
"""
function sliding_fraction(
        H::Matrix{R},
        simulation::SIM,
        t::Real,
        θ;
        basis::Symbol = :surface
) where {R <: Real, SIM <: Simulation}
    (; V_slide, V_deform) = velocity_split(H, simulation, t, θ; basis = basis)
    total = V_slide .+ V_deform
    return ifelse.(avg(H) .> 0, V_slide ./ total, NaN)
end

"""
    sliding_fraction(
        C::AbstractArray, H::AbstractArray, A::Real;
        p = 3.0, q = 0.0, n = 3.0, ρg = 900.0 * 9.81, basis = :surface,
    )

Closed-form sliding fraction from `C`, `H` and `A` alone, with no simulation.

Same quantity as the `(H, simulation, t, θ)` method, for saved fields (for example the `C`
of several inversions). It needs `p == n`, so that the surface slope cancels. Otherwise the
split depends on the slope, and only the method with a simulation is correct.

`H` must already be on the staggered grid (`avg` of the H-grid field, as `H̄` in `SIA2D!`),
matching the shape of a gridded `C`. Cells with no ice return `NaN`, as in the other method.
"""
function sliding_fraction(
        C::AbstractArray, H::AbstractArray, A::Real;
        p::Real = 3.0, q::Real = 0.0, n::Real = 3.0, ρg::Real = 900.0 * 9.81,
        basis::Symbol = :surface
)
    basis in (:surface, :flux) ||
        throw(ArgumentError("basis must be :surface or :flux, got $(basis)"))
    p == n || throw(ArgumentError("sliding_fraction: the closed form needs p == n so ∇S \
        cancels (got p = $(p), n = $(n)). Use the simulation-based method instead."))
    shape = basis === :surface ? 1 : 2
    slide = @. C * ρg^(p - q) * H^(p - q)
    deform = @. 2 * A * ρg^n * H^(n + 1) / (n + shape)
    return @. ifelse(H > 0, slide / (slide + deform), NaN)
end

"""
    V_from_H(
        simulation::SIM,
        H::Matrix{F},
        t::Real,
        θ;
        padding_method = :centered,
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
  - `padding_method`: How the velocities, computed on the cell corners, are put on the grid of
    `H`. `:centered` (default) averages the four corners of each cell. `:bottom_right` puts
    them half a cell off; it is not supported by the automatic adjoint in ODINN.

Returns:

  - `Vx`: x axis component of the surface velocity.
  - `Vy`: y axis component of the surface velocity.
  - `V`: Magnitude velocity.
"""
function V_from_H(
        simulation::SIM,
        H::Matrix{F},
        t::Real,
        θ;
        padding_method = :centered
) where {F <: AbstractFloat, SIM <: Simulation}
    Vx_in, Vy_in = surface_V(H, simulation, t, θ)
    Vx = zero(H)
    Vy = zero(H)
    # The velocities on the cell corners are put on the larger grid of H, with zero borders
    if padding_method == :centered
        inn(Vx) .= avg(Vx_in)
        inn(Vy) .= avg(Vy_in)
    elseif padding_method == :bottom_right
        inn1(Vx) .= Vx_in
        inn1(Vy) .= Vy_in
    else
        throw(ArgumentError("padding_method must be :centered or :bottom_right, got $(padding_method)."))
    end
    V = (Vx .^ 2 .+ Vy .^ 2) .^ (1/2)
    return Vx, Vy, V
end

"""
    averageV(
        θ,
        simulation,
        tspan_velocity::Tuple{F,F},
        step::F,
        ts::Vector{F},
        Hs::Vector{Matrix{F}},
    ) where {F <: AbstractFloat}

Compute average surface velocity from ice thickness over a given time window and using a constant time stepping.
This function uses the SIA model to map the ice thickness to instant surface velocity.

Arguments:

  - `θ`: Parameters of the laws to be used in the SIA. Can be `nothing` when no learnable laws are used.
  - `simulation::SIM`: The simulation structure used to retrieve the physical
    parameters.
  - `tspan_velocity::Tuple{F,F}`: Time window over which the average surface velocity is computed. It can be different than the simulation period.
  - `step::F`: Constant time step used to compute the average surface velocity. If the
    window is not a whole number of steps, the last partial step is not in the average.
  - `ts::Vector{F}`: Vector representing time associated to each ice thickness matrix.
  - `Hs::Vector{Matrix{F}}`: Vector ice thickness matrices.
  - `H::Matrix{F}`: The ice thickness matrix.
  - `t::R`: Current simulation time.

Returns:

  - `avg_Vx`: x axis component of the average surface velocity.
  - `avg_Vy`: y axis component of the average surface velocity.
  - `avg_V`: Magnitude velocity.
"""
function averageV(
        θ,
        simulation::SIM,
        tspan_velocity::Tuple{F, F},
        step::F,
        ts::Vector{F},
        Hs::Vector{Matrix{F}}
) where {F <: AbstractFloat, SIM <: Simulation}
    # 1. Determine indices in prediction that will be used for average velocity estimation
    tLoss = collect(tspan_velocity[1]:step:tspan_velocity[2])
    dt = diff(tLoss)
    tLoss = tLoss[begin:(end - 1)] # Discard last point t=tspan_velocity[2]
    ind_pred = Sleipnir.indFromT(
        glacier_tspan(simulation, simulation.cache.iceflow.glacier_idx), tLoss, ts)
    T = sum(dt)

    # 2. Compute the predicted velocity Vx_pred, Vy_pred, V_pred
    if !isnothing(simulation.model.trainable_components)
        simulation.model.trainable_components.θ = θ
    end
    res = map(i -> Huginn.V_from_H(simulation, Hs[ind_pred[i]], tLoss[i], θ),
        1:length(dt)
    )
    Vx_pred = first.(res)
    Vy_pred = getindex.(res, 2)

    # 3. Aggregate the velocities
    avg_Vx = sum((Vx_pred .* dt)/T)
    avg_Vy = sum((Vy_pred .* dt)/T)
    avg_V = (avg_Vx .^ 2 .+ avg_Vy .^ 2) .^ (1/2)

    return avg_Vx, avg_Vy, avg_V
end
