
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
- The function modifies the input matrices `dH` and `H` in place.
"""
function SIA2D!(dH::Matrix{R}, H::Matrix{R}, simulation::SIM, t::R) where {R <:Real, SIM <: Simulation}
    # Retrieve parameters
    SIA2D_model::SIA2Dmodel = simulation.model.iceflow
    glacier::Sleipnir.Glacier2D = simulation.glaciers[simulation.model.iceflow.glacier_idx[]]
    params::Sleipnir.Parameters = simulation.parameters
    H̄ = SIA2D_model.H̄
    A = SIA2D_model.A
    n = SIA2D_model.n
    B = glacier.B
    S = SIA2D_model.S
    dSdx = SIA2D_model.dSdx
    dSdy = SIA2D_model.dSdy
    D = SIA2D_model.D
    Dx = SIA2D_model.Dx
    Dy = SIA2D_model.Dy
    dSdx_edges = SIA2D_model.dSdx_edges
    dSdy_edges = SIA2D_model.dSdy_edges
    ∇S = SIA2D_model.∇S
    ∇Sx = SIA2D_model.∇Sx
    ∇Sy = SIA2D_model.∇Sy
    Fx = SIA2D_model.Fx
    Fy = SIA2D_model.Fy
    Fxx = SIA2D_model.Fxx
    Fyy = SIA2D_model.Fyy
    Δx = glacier.Δx
    Δy = glacier.Δy
    Γ = SIA2D_model.Γ
    ρ = simulation.parameters.physical.ρ
    g = simulation.parameters.physical.g

    # First, enforce values to be positive
    map!(x -> ifelse(x>0.0,x,0.0), H, H)
    # Update glacier surface altimetry
    S .= B .+ H

    # All grid variables computed in a staggered grid
    # Compute surface gradients on edges
    diff_x!(dSdx, S, Δx)  
    diff_y!(dSdy, S, Δy) 
    avg_y!(∇Sx, dSdx)
    avg_x!(∇Sy, dSdy)
    ∇S .= (∇Sx.^2 .+ ∇Sy.^2).^((n[] - 1)/2) 

    avg!(H̄, H)
    Γ[] = 2.0 * A[] * (ρ * g)^n[] / (n[]+2) # 1 / m^3 s 
    D .= Γ[] .* H̄.^(n[] + 2) .* ∇S

    # Compute flux components
    @views diff_x!(dSdx_edges, S[:,2:end - 1], Δx)
    @views diff_y!(dSdy_edges, S[2:end - 1,:], Δy)

    # Cap surface elevaton differences with the upstream ice thickness to 
    # imporse boundary condition of the SIA equation
    η₀ = params.physical.η₀
    dSdx_edges .= @views @. min(dSdx_edges,  η₀ * H[2:end, 2:end-1]/Δx)
    dSdx_edges .= @views @. max(dSdx_edges, -η₀ * H[1:end-1, 2:end-1]/Δx)
    dSdy_edges .= @views @. min(dSdy_edges,  η₀ * H[2:end-1, 2:end]/Δy)
    dSdy_edges .= @views @. max(dSdy_edges, -η₀ * H[2:end-1, 1:end-1]/Δy)

    avg_y!(Dx, D)
    avg_x!(Dy, D)
    Fx .= .-Dx .* dSdx_edges
    Fy .= .-Dy .* dSdy_edges 

    #  Flux divergence
    diff_x!(Fxx, Fx, Δx)
    diff_y!(Fyy, Fy, Δy)
    inn(dH) .= .-(Fxx .+ Fyy) 
end

# Dummy function to bypass ice flow
function noSIA2D!(dH::Matrix{R}, H::Matrix{R}, simulation::SIM, t::R) where {R <: Real,SIM <: Simulation}
   
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
"""
function SIA2D(H::Matrix{R}, simulation::SIM, t::R; batch_id::Union{Nothing, I} = nothing) where {R <: Real, I <: Integer, SIM <: Simulation}
    # Retrieve parameters
    # For simulations using Reverse Diff, an iceflow model per glacier is needed
    if isnothing(batch_id)
        SIA2D_model = simulation.model.iceflow
        glacier = simulation.glaciers[SIA2D_model.glacier_idx[]]
    else
        SIA2D_model = simulation.model.iceflow[batch_id] # We pick the right iceflow model for this glacier
        glacier = simulation.glaciers[batch_id]
    end
    
    params = simulation.parameters
    # Retrieve parameters
    B = glacier.B
    Δx = glacier.Δx
    Δy = glacier.Δy
    A = SIA2D_model.A
    n = SIA2D_model.n
    ρ = params.physical.ρ
    g = params.physical.g

    @views H = ifelse.(H.<0.0, 0.0, H) # prevent values from going negative

    # First, enforce values to be positive
    map!(x -> ifelse(x>0.0,x,0.0), H, H)
    
    # Update glacier surface altimetry
    S = B .+ H

    # All grid variables computed in a staggered grid
    # Compute surface gradients on edges
    dSdx = diff_x(S) ./ Δx
    dSdy = diff_y(S) ./ Δy
    ∇S = (avg_y(dSdx).^2 .+ avg_x(dSdy).^2).^((n[] - 1)/2) 

    Γ = 2.0 * A[] * (ρ * g)^n[] / (n[]+2) # 1 / m^3 s 
    D = Γ .* avg(H).^(n[] + 2) .* ∇S

    # Compute flux components
    @views dSdx_edges = diff_x(S[:,2:end - 1]) ./ Δx
    @views dSdy_edges = diff_y(S[2:end - 1,:]) ./ Δy

    # Cap surface elevaton differences with the upstream ice thickness to
    # impose boundary condition of the SIA equation
    # We need to do this with Tullio or something else that allow us to set indices.
    η₀ = 1.0
    dSdx_edges = @views @. min(dSdx_edges,  η₀ * H[2:end, 2:end-1]/Δx)
    dSdx_edges = @views @. max(dSdx_edges, -η₀ * H[1:end-1, 2:end-1]/Δx)
    dSdy_edges = @views @. min(dSdy_edges,  η₀ * H[2:end-1, 2:end]/Δy)
    dSdy_edges = @views @. max(dSdy_edges, -η₀ * H[2:end-1, 1:end-1]/Δy)

    Fx = .-avg_y(D) .* dSdx_edges
    Fy = .-avg_x(D) .* dSdy_edges 

    #  Flux divergence
    @tullio dH[i,j] := -(diff_x(Fx)[pad(i-1,1,1),pad(j-1,1,1)] / Δx + diff_y(Fy)[pad(i-1,1,1),pad(j-1,1,1)] / Δy) 

    return dH
end

"""
    avg_surface_V!(simulation::SIM) where {SIM <: Simulation}

Computes the average ice surface velocity for a given glacier evolution period
based on the initial and final ice thickness states.
"""
function avg_surface_V!(simulation::SIM) where {SIM <: Simulation}
    # TODO: Add more datapoints to better interpolate this
    iceflow_model = simulation.model.iceflow

    Vx₀, Vy₀ = surface_V!(iceflow_model.H₀, simulation)
    Vx,  Vy  = surface_V!(iceflow_model.H,  simulation)

    inn1(iceflow_model.Vx) .= (Vx₀ .+ Vx)./2.0
    inn1(iceflow_model.Vy) .= (Vy₀ .+ Vy)./2.0
    iceflow_model.V .= (iceflow_model.Vx.^2 .+ iceflow_model.Vy.^2).^(1/2)
end

"""
    avg_surface_V(simulation::SIM; batch_id::Union{Nothing, I} = nothing) where {I <: Integer, SIM <: Simulation}

Computes the average ice surface velocity for a given glacier evolution period
based on the initial and final ice thickness states.
"""
function avg_surface_V(simulation::SIM; batch_id::Union{Nothing, I} = nothing) where {I <: Integer, SIM <: Simulation}
    # Simulations using Reverse Diff require an iceflow model for each glacier
    if isnothing(batch_id)
        iceflow_model = simulation.model.iceflow
    else
        iceflow_model = simulation.model.iceflow[batch_id]
    end

    # We compute the initial and final surface velocity and average them
    Vx₀, Vy₀ = surface_V(iceflow_model.H₀, simulation; batch_id=batch_id)
    Vx,  Vy  = surface_V(iceflow_model.H,  simulation; batch_id=batch_id)

    V̄x = (Vx₀ .+ Vx)./2.0
    V̄y = (Vy₀ .+ Vy)./2.0
    V  = (V̄x.^2 .+ V̄y.^2).^(1/2)

    return V̄x, V̄y, V
end

"""
    surface_V!(H, B, Δx, Δy, temp, sim, A_noise, θ=[], UA_f=[])

Computes the ice surface velocity for a given glacier state
"""
function surface_V!(H::Matrix{<:Real}, simulation::SIM) where {SIM <: Simulation}
    params::Sleipnir.Parameters = simulation.parameters
    iceflow_model = simulation.model.iceflow
    glacier = simulation.glaciers[iceflow_model.glacier_idx[]]
    B = glacier.B
    H̄ = iceflow_model.H̄
    dSdx = iceflow_model.dSdx
    dSdy = iceflow_model.dSdy
    ∇S = iceflow_model.∇S
    ∇Sx = iceflow_model.∇Sx
    ∇Sy = iceflow_model.∇Sy
    Γꜛ = iceflow_model.Γ
    D = iceflow_model.D
    # Dx = iceflow_model.Dx
    # Dy = iceflow_model.Dy
    A = iceflow_model.A
    n = iceflow_model.n
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
    Γꜛ[] = 2.0 * A[] * (ρ * g)^n[] / (n[]+1) # surface stress (not average)  # 1 / m^3 s 
    D = Γꜛ[] .* H̄.^(n[] + 1) .* ∇S
    
    # Compute averaged surface velocities
    Vx = .-D .* ∇Sx
    Vy = .-D .* ∇Sy 

    return Vx, Vy
end

"""
    surface_V(H, B, Δx, Δy, temp, sim, A_noise, θ=[], UA_f=[])

Computes the ice surface velocity for a given glacier state
"""
function surface_V(H::Matrix{R}, simulation::SIM; batch_id::Union{Nothing, I} = nothing) where {I <: Integer, R <: Real, SIM <: Simulation}
    params::Sleipnir.Parameters = simulation.parameters
    # Simulations using Reverse Diff require an iceflow model per glacier
    if isnothing(batch_id)
        iceflow_model = simulation.model.iceflow
        glacier = simulation.glaciers[iceflow_model.glacier_idx[]]
    else
        iceflow_model = simulation.model.iceflow[batch_id]
        glacier = simulation.glaciers[batch_id]
    end
    B = glacier.B
    Δx = glacier.Δx
    Δy = glacier.Δy
    A = iceflow_model.A
    n = iceflow_model.n
    ρ = params.physical.ρ
    g = params.physical.g
    
    # Update glacier surface altimetry
    S = B .+ H

    # All grid variables computed in a staggered grid
    # Compute surface gradients on edges
    dSdx = diff_x(S) / Δx
    dSdy = diff_y(S) / Δy
    ∇S = (avg_y(dSdx).^2 .+ avg_x(dSdy).^2).^((n[] - 1)/2) 
    
    Γꜛ = 2.0 * A[] * (ρ * g)^n[] / (n[]+1) # surface stress (not average)  # 1 / m^3 s 
    D = Γꜛ .* avg(H).^(n[] + 1) .* ∇S
    
    # Compute averaged surface velocities
    Vx = - D .* avg_y(dSdx)
    Vy = - D .* avg_x(dSdy)

    return Vx, Vy
end

function H_from_V(V::Matrix{<:Real}, simulation::SIM) where {SIM <: Simulation}
    params::Sleipnir.Parameters = simulation.parameters

    iceflow_model = simulation.model.iceflow
    glacier = simulation.glaciers[iceflow_model.glacier_idx[]]
    B = glacier.B
    Δx = glacier.Δx
    Δy = glacier.Δy
    A = iceflow_model.A
    n = iceflow_model.n
    C = iceflow_model.C
    ρ = params.physical.ρ
    g = params.physical.g
    H₀ = glacier.H₀

    # Update glacier surface altimetry
    S = iceflow_model.S  
    V = Huginn.avg(V)
    
    # All grid variables computed in a staggered grid
    # Compute surface gradients on edges
    dSdx = Huginn.diff_x(S) / Δx
    dSdy = Huginn.diff_y(S) / Δy
    ∇S = (Huginn.avg_y(dSdx).^2 .+ Huginn.avg_x(dSdy).^2).^(1/2)
   

    Γꜛ = (2.0 * A[] * (ρ * g)^n[]) / (n[]+1) # surface stress (not average)  # 1 / m^3 s 
    
    H = ( V + C[] ./ (Γꜛ .*(∇S .^ n[]))) .^ (1 / (n[] + 1)) 
    
    replace!(H, NaN=>0.0)
    replace!(H, Inf=>0.0)
    return H   
end

