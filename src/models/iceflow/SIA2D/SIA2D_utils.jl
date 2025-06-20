
"""
    SIA2D!(dH::Matrix{R}, H::Matrix{R}, simulation::SIM, t::R) where {R <:Real, SIM <: Simulation}

Simulates the evolution of ice thickness in a 2D shallow ice approximation (SIA) model. Works in-place.

# Arguments
- `dH::Matrix{R}`: Matrix to store the rate of change of ice thickness.
- `H::Matrix{R}`: Matrix representing the ice thickness.
- `simulation::SIM`: Simulation object containing model parameters and state.
- `t::R`: Current simulation time.

# Details
This function updates the ice thickness `H` and computes the rate of change `dH` using the shallow ice approximation in 2D. It retrieves necessary parameters from the `simulation` object, enforces positive ice thickness values, updates glacier surface altimetry, computes surface gradients, flux components, and flux divergence.

# Notes
- The function operates on a staggered grid for computing gradients and fluxes.
- Surface elevation differences are capped using upstream ice thickness to impose boundary conditions.
- The function modifies the input matrices `dH` and `H` in-place.

See also [`SIA2D`](@ref), [`SIA2D_with_laws!`](@ref), [`SIA2D_with_laws`](@ref)
"""
function SIA2D!(
    dH::Matrix{R},
    H::Matrix{R},
    simulation::SIM,
    t::R;
    batch_id::Union{Nothing, I} = nothing
) where {R <:Real, I <: Integer, SIM <: Simulation}
    # For simulations using Reverse Diff, an iceflow model per glacier is needed
    if isnothing(batch_id)
        SIA2D_cache = simulation.cache.iceflow
        glacier = simulation.glaciers[SIA2D_cache.glacier_idx[]]
    else
        # SIA2D_cache = simulation.cache.iceflow[batch_id]
        glacier = simulation.glaciers[batch_id]
    end

    params = simulation.parameters
    
    (;
        H̄, A, C, n, S, dSdx, dSdy,
        D, D_is_provided, Dx, Dy, 
        dSdx_edges, dSdy_edges, 
        ∇S, ∇Sx, ∇Sy, 
        Fx, Fy, Fxx, Fyy, Γ, 
    ) = SIA2D_cache

    (;Δx, Δy, B) = glacier

    (;ρ, g) = simulation.parameters.physical
    
    # First, enforce values to be positive
    map!(x -> ifelse(x > 0.0, x, 0.0), H, H)
    # Update glacier surface altimetry
    S .= B .+ H

    # Compute D in case is not provided in the simulation
    if !D_is_provided
        # All grid variables computed in a staggered grid
        # Compute surface gradients on edges
        diff_x!(dSdx, S, Δx)
        diff_y!(dSdy, S, Δy)
        avg_y!(∇Sx, dSdx)
        avg_x!(∇Sy, dSdy)

        @. ∇S = (∇Sx^2 + ∇Sy^2)^((n - 1) / 2)

        avg!(H̄, H)
        gravity_term = (ρ * g).^n
        @. Γ = 2.0 * A * gravity_term / (n + 2) # 1 / m^3 s
        @. D = (C * gravity_term + Γ * H̄) * H̄^(n + 1) * ∇S
    end

    # Compute flux components
    @views diff_x!(dSdx_edges, S[:,2:end - 1], Δx)
    @views diff_y!(dSdy_edges, S[2:end - 1,:], Δy)

    # Cap surface elevaton differences with the upstream ice thickness to
    # imporse boundary condition of the SIA equation
    η₀ = params.physical.η₀
    dSdx_edges .= @views @. min(dSdx_edges,  η₀ * H[2:end, 2:end-1] / Δx)
    dSdx_edges .= @views @. max(dSdx_edges, -η₀ * H[1:end-1, 2:end-1] / Δx)
    dSdy_edges .= @views @. min(dSdy_edges,  η₀ * H[2:end-1, 2:end] / Δy)
    dSdy_edges .= @views @. max(dSdy_edges, -η₀ * H[2:end-1, 1:end-1] / Δy)

    avg_y!(Dx, D)
    avg_x!(Dy, D)
    Fx .= .-Dx .* dSdx_edges
    Fy .= .-Dy .* dSdy_edges

    #  Flux divergence
    diff_x!(Fxx, Fx, Δx)
    diff_y!(Fyy, Fy, Δy)
    inn(dH) .= .-(Fxx .+ Fyy)
end

"""
    SIA2D_with_laws!(dH::Matrix{R}, H::Matrix{R}, simulation::SIM, t::R; batch_id::Union{Nothing, I} = nothing)

Compute the in-place 2D shallow ice approximation update while applying laws (`A`, `C`, `n`) 
that are not handled via callbacks.

This function ensures that time-dependent laws are applied before calling the core `SIA2D!` function.

# Arguments
- `dH`: Matrix where the change in ice thickness is stored.
- `H`: Current ice thickness.
- `simulation`: Simulation object containing model state and parameters.
- `t`: Current simulation time.
- `batch_id`: Optional glacier index for batch simulations. Defaults to `nothing`.

See also [`SIA2D`](@ref), [`SIA2D!`](@ref), [`SIA2D_with_laws`](@ref)
"""
function SIA2D_with_laws!(
    dH::Matrix{R},
    H::Matrix{R},
    simulation::SIM,
    t::R;
    batch_id::Union{Nothing, I} = nothing
) where {R <:Real, I <: Integer, SIM <: Simulation}
    SIA2D_model = simulation.model.iceflow
    SIA2D_cache = simulation.cache.iceflow

    glacier_idx = SIA2D_cache.glacier_idx[]

    # for now θ is ignored
    # in the future it should looks like `θ = simulation.params.θ.iceflow`
    θ = nothing 

    if !is_callback_law(SIA2D_model.A)
        apply_law!(SIA2D_model.A, SIA2D_cache.A, simulation, glacier_idx, t, θ)
    end

    if !is_callback_law(SIA2D_model.C)
        apply_law!(SIA2D_model.C, SIA2D_cache.C, simulation, glacier_idx, t, θ)
    end

    if !is_callback_law(SIA2D_model.n)
        apply_law!(SIA2D_model.n, SIA2D_cache.n, simulation, glacier_idx, t, θ)
    end

    SIA2D!(dH, H, simulation, t; batch_id)
end

# Dummy function to bypass ice flow
function noSIA2D!(dH::Matrix{R}, H::Matrix{R}, simulation::SIM, t::R) where {R <: Real,SIM <: Simulation}
    return nothing
end


"""
    SIA2D(H::Matrix{R}, simulation::SIM, t::R; batch_id::Union{Nothing, I} = nothing) where {R <: Real, I <: Integer, SIM <: Simulation}

Compute the change in ice thickness (`dH`) for a 2D Shallow Ice Approximation (SIA) model. Works out-of-place.

# Arguments
- `H::Matrix{R}`: Ice thickness matrix.
- `simulation::SIM`: Simulation object containing model parameters and glacier data.
- `t::R`: Current time step.
- `batch_id::Union{Nothing, I}`: Optional batch ID to select a specific glacier model. Defaults to `nothing`.

# Returns
- `dH::Matrix{R}`: Matrix representing the change in ice thickness.

# Details
This function performs the following steps:
1. Retrieves the appropriate iceflow model and glacier data based on `batch_id`.
2. Retrieves physical parameters from the simulation object.
3. Ensures that ice thickness values are non-negative.
4. Updates the glacier surface altimetry.
5. Computes surface gradients on the edges of the grid.
6. Calculates the diffusivity `D` based on the surface gradients and ice thickness.
7. Computes the flux components `Fx` and `Fy`.
8. Calculates the flux divergence to determine the change in ice thickness `dH`.

# Notes
- The function uses `@views` to avoid unnecessary array allocations.
- The `@tullio` macro is used for efficient tensor operations.

See also [`SIA2D!`](@ref), [`SIA2D_with_laws!`](@ref), [`SIA2D_with_laws`](@ref)
"""
function SIA2D(
    H::Matrix{R},
    simulation::SIM,
    t::R;
    batch_id::Union{Nothing, I} = nothing,
    # diffusivity_provided::Bool = false
    ) where {R <: Real, I <: Integer, SIM <: Simulation}

    # Retrieve parameters
    # For simulations using Reverse Diff, an iceflow model per glacier is needed
    if isnothing(batch_id)
        SIA2D_model = simulation.model.iceflow
        SIA2D_cache = simulation.cache.iceflow
        glacier = simulation.glaciers[SIA2D_cache.glacier_idx[]]
    else
        SIA2D_model = simulation.model.iceflow[batch_id] # We pick the right iceflow model for this glacier
        SIA2D_cache = simulation.model.cache[batch_id]
        glacier = simulation.glaciers[batch_id]
    end

    params = simulation.parameters
    
    # Retrieve parameters
    B = glacier.B
    Δx = glacier.Δx
    Δy = glacier.Δy
    A = SIA2D_cache.A
    C = SIA2D_cache.C
    n = SIA2D_cache.n
    ρ = params.physical.ρ
    g = params.physical.g

    @views H = ifelse.(H .< 0.0, 0.0, H) # prevent values from going negative

    # First, enforce values to be positive
     ## Uncomment this line!!!! Why this is neccesary if we have the previous function???
    # map!(x -> ifelse(x>0.0,x,0.0), H, H)
    @assert sum(H .< 0.0) == 0 "Ice thickness values are below zero."

    # Update glacier surface altimetry
    S = B .+ H

    # Compute D in case is not provided in the simulation
    if SIA2D_cache.D_is_provided
        D = SIA2D_cache.D
    else
        # All grid variables computed in a staggered grid
        # Compute surface gradients on edges
        dSdx = diff_x(S) ./ Δx
        dSdy = diff_y(S) ./ Δy
        ∇S = (avg_y(dSdx).^2 .+ avg_x(dSdy).^2).^((n[] - 1) / 2)
        H̄ = avg(H)
        gravity_term = (ρ * g).^n
        Γ = @. 2.0 * A * gravity_term / (n + 2) # 1 / m^3 s
        D = @. (C * gravity_term + Γ * H̄) * H̄^(n + 1) * ∇S
    end

    # Compute flux components
    @views dSdx_edges = diff_x(S[:,2:end - 1]) ./ Δx
    @views dSdy_edges = diff_y(S[2:end - 1,:]) ./ Δy

    # Cap surface elevaton differences with the upstream ice thickness to
    # impose boundary condition of the SIA equation
    # We need to do this with Tullio or something else that allow us to set indices.
    η₀ = 1.0
    dSdx_edges = @views @. min(dSdx_edges,  η₀ * H[2:end, 2:end-1] / Δx)
    dSdx_edges = @views @. max(dSdx_edges, -η₀ * H[1:end-1, 2:end-1] / Δx)
    dSdy_edges = @views @. min(dSdy_edges,  η₀ * H[2:end-1, 2:end] / Δy)
    dSdy_edges = @views @. max(dSdy_edges, -η₀ * H[2:end-1, 1:end-1] / Δy)

    Fx = .-avg_y(D) .* dSdx_edges
    Fy = .-avg_x(D) .* dSdy_edges

    Fxx = diff_x(Fx) / Δx
    Fyy = diff_y(Fy) / Δy

    #  Flux divergence
    # @tullio dH[i,j] := -(diff_x(Fx)[pad(i-1,1,1),pad(j-1,1,1)] / Δx + diff_y(Fy)[pad(i-1,1,1),pad(j-1,1,1)] / Δy) 

    # return dH
    dH = zero(H)
    inn(dH) .= .-(Fxx .+ Fyy)
    return dH
end

"""
    SIA2D_with_laws(H::Matrix{R}, simulation::SIM, t::R; batch_id::Union{Nothing, I} = nothing)

Compute the out-of-place 2D shallow ice approximation (`dH`) while applying the laws (`A`, `C`, `n`)
that are not updated via callbacks.

This function evaluates the necessary laws based on the current time and simulation context before
calling the function `SIA2D` function.

# Arguments
- `H`: Current ice thickness.
- `simulation`: Simulation object containing model state and parameters.
- `t`: Current simulation time.
- `batch_id`: Optional glacier index for batch simulations. Defaults to `nothing`.

# Returns
- `dH`: Matrix of the computed change in ice thickness.

See also [`SIA2D`](@ref), [`SIA2D!`](@ref), [`SIA2D_with_laws!`](@ref)
"""
function SIA2D_with_laws(
    H::Matrix{R},
    simulation::SIM,
    t::R;
    batch_id::Union{Nothing, I} = nothing
) where {R <:Real, I <: Integer, SIM <: Simulation}
    SIA2D_model = simulation.model.iceflow
    SIA2D_cache = simulation.cache.iceflow

    glacier_idx = SIA2D_cache.glacier_idx[]

    # for now θ is ignored
    # in the future it should looks like `θ = simulation.params.θ.iceflow`
    θ = nothing 

    # Compute A, C and n if needed
    if !is_callback_law(SIA2D_model.A)
        apply_law!(SIA2D_model.A, SIA2D_cache.A, simulation, glacier_idx, t, θ)
    end

    if !is_callback_law(SIA2D_model.C)
        apply_law!(SIA2D_model.C, SIA2D_cache.C, simulation, glacier_idx, t, θ)
    end

    if !is_callback_law(SIA2D_model.n)
        apply_law!(SIA2D_model.n, SIA2D_cache.n, simulation, glacier_idx, t, θ)
    end

    return SIA2D(H, simulation, t; batch_id)
end

"""
    avg_surface_V!(simulation::SIM) where {SIM <: Simulation}

Calculate the average surface velocity for a given simulation.

# Arguments
- `simulation::SIM`: A simulation object of type `SIM` which is a subtype of `Simulation`.

# Description
This function computes the average surface velocity components (`Vx` and `Vy`) and the resultant velocity (`V`)
for the ice flow model within the given simulation. It first calculates the surface velocities at the initial
and current states, then averages these velocities and updates the ice flow model's velocity fields.

# Notes
- The function currently uses a simple averaging method and may need more datapoints for better interpolation.
"""
function avg_surface_V!(simulation::SIM) where {SIM <: Simulation}
    # TODO: Add more datapoints to better interpolate this
    iceflow_cache = simulation.cache.iceflow

    Vx₀, Vy₀ = surface_V!(iceflow_cache.H₀, simulation)
    Vx,  Vy  = surface_V!(iceflow_cache.H,  simulation)

    inn1(iceflow_cache.Vx) .= (Vx₀ .+ Vx)./2.0
    inn1(iceflow_cache.Vy) .= (Vy₀ .+ Vy)./2.0
    iceflow_cache.V .= (iceflow_cache.Vx.^2 .+ iceflow_cache.Vy.^2).^(1/2)
end

"""
    avg_surface_V(simulation::SIM; batch_id::Union{Nothing, I} = nothing) where {I <: Integer, SIM <: Simulation}

Compute the average surface velocity for a given simulation.

# Arguments
- `simulation::SIM`: The simulation object containing the model and other relevant data.
- `batch_id::Union{Nothing, I}`: An optional batch identifier. If provided, it specifies which batch of the iceflow model to use. Defaults to `nothing`.

# Returns
- `V̄x`: The average surface velocity in the x-direction.
- `V̄y`: The average surface velocity in the y-direction.
- `V`: The magnitude of the average surface velocity.

# Details
This function computes the initial and final surface velocities and averages them to obtain the average surface velocity. It handles simulations that use reverse differentiation by selecting the appropriate iceflow model for each glacier.
"""
function avg_surface_V(simulation::SIM; batch_id::Union{Nothing, I} = nothing) where {I <: Integer, SIM <: Simulation}
    # Simulations using Reverse Diff require an iceflow model for each glacier
    if isnothing(batch_id)
        iceflow_cache = simulation.cache.iceflow
    else
        iceflow_cache = simulation.cache.iceflow[batch_id]
    end

    # We compute the initial and final surface velocity and average them
    Vx₀, Vy₀ = surface_V(iceflow_cache.H₀, simulation; batch_id=batch_id)
    Vx,  Vy  = surface_V(iceflow_cache.H,  simulation; batch_id=batch_id)

    V̄x = (Vx₀ .+ Vx)./2.0
    V̄y = (Vy₀ .+ Vy)./2.0
    V  = (V̄x.^2 .+ V̄y.^2).^(1/2)

    return V̄x, V̄y, V
end

"""
    surface_V!(H::Matrix{<:Real}, simulation::SIM) where {SIM <: Simulation}

Compute the surface velocities of a glacier using the Shallow Ice Approximation (SIA) in 2D.

# Arguments
- `H::Matrix{<:Real}`: The ice thickness matrix.
- `simulation::SIM`: The simulation object containing parameters and model information.

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
function surface_V!(H::Matrix{<:Real}, simulation::SIM) where {SIM <: Simulation}
    params::Sleipnir.Parameters = simulation.parameters
    iceflow_cache = simulation.cache.iceflow
    glacier = simulation.glaciers[iceflow_cache.glacier_idx[]]
    B = glacier.B
    H̄ = iceflow_cache.H̄
    dSdx = iceflow_cache.dSdx
    dSdy = iceflow_cache.dSdy
    ∇S = iceflow_cache.∇S
    ∇Sx = iceflow_cache.∇Sx
    ∇Sy = iceflow_cache.∇Sy
    Γꜛ = iceflow_cache.Γ
    D = iceflow_cache.D
    # Dx iceflow_cache.Dx
    # Dy iceflow_cache.Dy
    A = iceflow_cache.A
    C = iceflow_cache.C
    n = iceflow_cache.n
    Δx = glacier.Δx
    Δy = glacier.Δy
    ρ = params.physical.ρ
    g = params.physical.g

    # Update glacier surface altimetry
    S = B .+ H

    # All grid variables computed in a staggered grid
    # Compute surface gradients on edges
    diff_x!(dSdx, S, Δx)
    diff_y!(dSdy, S, Δy)
    avg_y!(∇Sx, dSdx)
    avg_x!(∇Sy, dSdy)
    ∇S .= (∇Sx.^2 .+ ∇Sy.^2).^((n[] - 1)/2)

    avg!(H̄, H)
    gravity_term = (ρ * g).^n[]
    @. Γꜛ = 2.0 * A * gravity_term / (n+1) # surface stress (not average)  # 1 / m^3 s
    D = @. (C * (n+2) * gravity_term + Γꜛ) * H̄^(n + 1) * ∇S

    # Compute averaged surface velocities
    Vx = .-D .* ∇Sx
    Vy = .-D .* ∇Sy

    return Vx, Vy
end

"""
    surface_V(H::Matrix{R}, simulation::SIM; batch_id::Union{Nothing, I} = nothing) where {I <: Integer, R <: Real, SIM <: Simulation}

Compute the surface velocities of a glacier using the Shallow Ice Approximation (SIA) in 2D.

# Arguments
- `H::Matrix{R}`: Ice thickness matrix.
- `simulation::SIM`: Simulation object containing parameters and model information.
- `batch_id::Union{Nothing, I}`: Optional batch identifier for simulations using reverse differentiation. Defaults to `nothing`.

# Returns
- `Vx`: Matrix of surface velocities in the x-direction.
- `Vy`: Matrix of surface velocities in the y-direction.

# Details
This function computes the surface velocities of a glacier by updating the glacier surface altimetry and calculating the surface gradients on the edges. It uses a staggered grid approach to compute the gradients and velocities.

# Notes
- The function assumes that the `simulation` object contains the necessary parameters and model information.
- The `batch_id` is used to handle simulations that require an iceflow model per glacier.
"""
function surface_V(H::Matrix{R}, simulation::SIM; batch_id::Union{Nothing, I} = nothing) where {I <: Integer, R <: Real, SIM <: Simulation}
    params::Sleipnir.Parameters = simulation.parameters
    # Simulations using Reverse Diff require an iceflow model per glacier
    if isnothing(batch_id)
        iceflow_cache = simulation.cache.iceflow
        glacier = simulation.glaciers[iceflow_cache.glacier_idx[]]
    else
        iceflow_cache = simulation.cache.iceflow[batch_id]
        glacier = simulation.glaciers[batch_id]
    end
    B = glacier.B
    Δx = glacier.Δx
    Δy = glacier.Δy
    A = iceflow_cache.A
    C = iceflow_cache.C
    n = iceflow_cache.n

    ρ = params.physical.ρ
    g = params.physical.g

    # Update glacier surface altimetry
    S = B .+ H

    # All grid variables computed in a staggered grid
    # Compute surface gradients on edges
    dSdx = diff_x(S) / Δx
    dSdy = diff_y(S) / Δy
    ∇S = (avg_y(dSdx).^2 .+ avg_x(dSdy).^2).^((n .- 1)/2)

    gravity_term = (ρ * g).^n
    Γꜛ = @. 2.0 * A * gravity_term / (n+1) # surface stress (not average)  # 1 / m^3 s
    D = (C .* (n.+2) .* gravity_term .+ Γꜛ) .* avg(H).^(n .+ 1) .* ∇S

    # Compute averaged surface velocities
    Vx = - D .* avg_y(dSdx)
    Vy = - D .* avg_x(dSdy)

    return Vx, Vy
end

"""
    V_from_H(
        simulation::SIM,
        H::Matrix{F};
        batch_id::Union{Nothing, I}=nothing
    ) where {I <: Integer, F <: AbstractFloat, SIM <: Simulation}

Compute surface velocity from ice thickness using the SIA model.
It relies on `surface_V` to compute `Vx` and `Vy` and it additionally computes the
magnitude of the velocity `V`.

Arguments:
- `simulation::SIM`: The simulation structure used to retrieve the physical
    parameters.
- `H::Matrix{F}`: The ice thickness matrix.
- `batch_id::Union{Nothing, I}=nothing`: The batch ID that is used to retrieve the
    iceflow model in `surface_V`.

Returns:
- `Vx`: x axis component of the surface velocity.
- `Vy`: y axis component of the surface velocity.
- `V`: Magnitude velocity.
"""
function V_from_H(
    simulation::SIM,
    H::Matrix{F};
    batch_id::Union{Nothing, I}=nothing
) where {I <: Integer, F <: AbstractFloat, SIM <: Simulation}
    Vx_in, Vy_in = surface_V(H, simulation; batch_id=batch_id)
    Vx = zero(H)
    Vy = zero(H)
    inn1(Vx) .= Vx_in
    inn1(Vy) .= Vy_in
    V = (Vx.^2 .+ Vy.^2).^(1/2)
    return Vx, Vy, V
end
