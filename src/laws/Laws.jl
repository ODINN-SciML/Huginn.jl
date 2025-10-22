import Sleipnir: get_input, default_name

export iTemp, iH̄, i∇S, iCPDD, iTopoRough
export ConstantA, CuffeyPaterson, SyntheticC

########################
###### LAW INPUTS ######
########################

"""
    iTemp <: AbstractInput

Input that represents the long term air temperature of a glacier.
It is computed using the OGGM data over a period predefined in Gungnir.
"""
struct iTemp <: AbstractInput end
default_name(::iTemp) = :long_term_temperature
function get_input(::iTemp, simulation, glacier_idx, t)
    glacier = simulation.glaciers[glacier_idx]
    return mean(glacier.climate.longterm_temps)
end
function Base.zero(::iTemp, simulation, glacier_idx)
    glacier = simulation.glaciers[glacier_idx]
    return zero(glacier.climate.longterm_temps)
end

"""
    iCPDD <: AbstractInput

Input that represents the cumulative positive degree days (PDD) over the last time window `window`.
It is computed by summing the daily PDD values from `t - window` to `t` using the glacier's climate data.
"""
struct iCPDD{P<:Period} <: AbstractInput
    window::P
    function iCPDD(; window::P = Week(1)) where {P<:Period}
        new{typeof(window)}(window)
    end
end
default_name(::iCPDD) = :CPDD

function get_input(cpdd::iCPDD, simulation, glacier_idx, t)  
    window = cpdd.window  
    glacier = simulation.glaciers[glacier_idx]  
    # We trim only the time period between `t` and `t - x`, where `x` is the PDD time window defined in the input attributes. 
    period = (partial_year(Day, t) - window):Day(1):partial_year(Day, t)  
    get_cumulative_climate!(glacier.climate, period)  
    # Convert climate dataset to 2D based on the glacier's DEM  
    climate_2D_step = downscale_2D_climate(glacier.climate.climate_step, glacier.S, glacier.Coords)  

    return climate_2D_step.PDD
end
function Base.zero(::iCPDD, simulation, glacier_idx)
    (; nx, ny) = simulation.glaciers[glacier_idx]
    return zeros(nx, ny)
end

"""
    iH̄ <: AbstractInput

Input that represents the ice thickness in the SIA.
It is the averaged ice thickness computed on the dual grid, that is `H̄ = avg(H)`
which is different from the ice thickness solution H.
"""
struct iH̄ <: AbstractInput end
default_name(::iH̄) = :H_dual_grid
function get_input(::iH̄, simulation, glacier_idx, t)
    return simulation.cache.iceflow.H̄
end
function Base.zero(::iH̄, simulation, glacier_idx)
    (; nx, ny) = simulation.glaciers[glacier_idx]
    return zeros(nx-1, ny-1)
end

"""
Input that represents the surface slope in the SIA.
It is computed using the bedrock elevation and the ice thickness solution H. The
spatial differences are averaged over the opposite axis:
S = B + H
∇S = (avg_y(diff_x(S) / Δx).^2 .+ avg_x(diff_y(S) / Δy).^2).^(1/2)
"""
struct i∇S <: AbstractInput end
default_name(::i∇S) = :∇S
function get_input(::i∇S, simulation, glacier_idx, t)
    return simulation.cache.iceflow.∇S
end
function Base.zero(::i∇S, simulation, glacier_idx)
    (; nx, ny) = simulation.glaciers[glacier_idx]
    return zeros(nx-1, ny-1)
end

"""
Input that represents the topographic roughness of the glacier.
It is computed as the curvature of the glacier bed (or surface) over a specified window size. The curvature can be calculated in different directions (flow, cross-flow, or both)
and using different curvature types (scalar or variability).
"""
struct iTopoRough{F<:AbstractFloat} <: AbstractInput
    window::F
    curvature_type::Symbol
    direction::Symbol
    position::Symbol
    function iTopoRough(; window::F = 200.0, curvature_type::Symbol = :scalar, direction::Symbol = :flow, position::Symbol = :bed) where {F<:AbstractFloat}
        valid_directions = (:flow, :cross_flow, :both)
        valid_curvature_types = (:scalar, :variability)
        valid_positions = (:bed, :surface)
        if !(curvature_type in valid_curvature_types)
            error("Invalid curvature_type: $curvature_type. Must be one of $(valid_curvature_types).")
        end
        if !(position in valid_positions)
            error("Invalid position: $position. Must be one of $(valid_positions).")
        end
        if !(direction in valid_directions)
            error("Invalid direction: $direction. Must be one of $(valid_directions).")
        end
        new{F}(window, curvature_type, direction, position)
    end
end
default_name(::iTopoRough) = :topographic_roughness

function get_input(inp_topo_rough::iTopoRough, simulation, glacier_idx, t)
    window = inp_topo_rough.window
    glacier = simulation.glaciers[glacier_idx]
    # Select DEM based on position attribute
    if inp_topo_rough.position == :bed
        dem = glacier.B
    elseif inp_topo_rough.position == :surface
        dem = glacier.B .+ glacier.H
    end
    window_size = max(4, Int(round(window / glacier.Δx)))  # At least 4 for second derivative
    half_window = max(1, div(window_size, 2))
    rows, cols = size(dem)
    roughness = zeros(eltype(dem), size(dem))

    for i in 1:rows, j in 1:cols
        rmin = max(1, i - half_window)
        rmax = min(rows, i + half_window)
        cmin = max(1, j - half_window)
        cmax = min(cols, j + half_window)
        window_dem = dem[rmin:rmax, cmin:cmax]

        if inp_topo_rough.curvature_type == :variability
            # Slope direction at central point
            dx_c = diff_x(window_dem)[div(end,2), div(end,2)] / glacier.Δx  
            dy_c = diff_y(window_dem)[div(end,2), div(end,2)] / glacier.Δy 
            slope_vec = [dx_c, dy_c]
            nrm = norm(slope_vec)
            if nrm ≈ 0
                eₚ = [1.0, 0.0]   # arbitrary downslope direction
            else
                eₚ = slope_vec / nrm   # downslope unit vector
            end
            eₛ = [-eₚ[2], eₚ[1]]      # cross-slope unit vector

            # Compute curvature field inside the window 
            Kₚ = Float64[]
            Kₛ = Float64[]
            wrows, wcols = size(window_dem)
            for wi in 2:(wrows-1), wj in 2:(wcols-1)   # avoid borders

                # second derivatives using central difference utils
                dxx = d2dx(window_dem, wi, wj, glacier.Δx)
                dyy = d2dy(window_dem, wi, wj, glacier.Δy)
                dxy = d2dxy(window_dem, wi, wj, glacier.Δx, glacier.Δy)

                # Hessian
                H = [dxx dxy; dxy dyy]

                # project Hessian along slope directions
                Kₚᵢ, Kₛᵢ = project_curvatures(H, eₚ, eₛ)
                push!(Kₚ, Kₚᵢ)   # curvature parallel to slope
                push!(Kₛ, Kₛᵢ)   # curvature cross-slope
            end

            # Define roughness as variability depending on the directions
            if inp_topo_rough.direction == :flow
                val = std(Kₚ)
            elseif inp_topo_rough.direction == :cross_flow
                val = std(Kₛ)
            elseif inp_topo_rough.direction == :both
                val = sqrt(std(Kₚ)^2 + std(Kₛ)^2)
            end
            roughness[i,j] = isnan(val) ? 0.0 : val

        elseif inp_topo_rough.curvature_type == :scalar
            # Gradient (slope direction)
            dx = diff_x(window_dem) / glacier.Δx
            dy = diff_y(window_dem) / glacier.Δy
            gx, gy = mean(dx), mean(dy)   # average slope in window
            gvec = [gx, gy]
            gmag = norm(gvec) + eps()     # avoid div0
            ŝ = gvec / gmag               # upslope unit vector
            n̂ = [-ŝ[2], ŝ[1]]            # cross-slope unit vector

            # Hessian
            dxx = diff_x(dx) / glacier.Δx
            dyy = diff_y(dy) / glacier.Δy
            dxy = diff_y(dx) / glacier.Δy
            H = [mean(dxx) mean(dxy); mean(dxy) mean(dyy)]

            # Projected curvatures
            Kₚ, Kₛ = project_curvatures(H, ŝ, n̂)

            if inp_topo_rough.direction == :flow
                val = Kₚ   # or std over window
            elseif inp_topo_rough.direction == :cross_flow
                val = Kₛ   # or std over window
            elseif inp_topo_rough.direction == :both
                val = sqrt(Kₚ^2 + Kₛ^2)
            end
            roughness[i,j] = isnan(val) ? 0 : val
        end
    end

    return roughness
end
function Base.zero(::iTopoRough, simulation, glacier_idx)
    (; nx, ny) = simulation.glaciers[glacier_idx]
    return zeros(nx, ny)
end

"""
    ConstantA(A::F) where {F <: AbstractFloat}

Law that represents a constant A in the SIA.

# Arguments:
- `A::F`: Rheology factor A.
"""
function ConstantA(A::F) where {F <: AbstractFloat}
    return ConstantLaw{ScalarCacheNoVJP}(function (simulation, glacier_idx, θ)
            return ScalarCacheNoVJP(fill(A))
        end,
    )
end


"""
    polyA_PatersonCuffey()

Returns a function of the coefficient A as a polynomial of the temperature.
The values used to fit the polynomial come from Cuffey & Peterson.
"""
function polyA_PatersonCuffey()
    # Parameterization of A(T) from Cuffey & Peterson
    A_values_sec = ([0.0 -2.0 -5.0 -10.0 -15.0 -20.0 -25.0 -30.0 -35.0 -40.0 -45.0 -50.0;
                                2.4e-24 1.7e-24 9.3e-25 3.5e-25 2.1e-25 1.2e-25 6.8e-26 3.7e-26 2.0e-26 1.0e-26 5.2e-27 2.6e-27]) # s⁻¹Pa⁻³
    A_values = hcat(A_values_sec[1,:], A_values_sec[2,:].*60.0*60.0*24.0*365.25)'
    return Polynomials.fit(A_values[1,:], A_values[2,:])
end


"""
    CuffeyPaterson()

Create a rheology law for the flow rate factor `A`.
The created law maps the long term air temperature to `A` using the values from
Cuffey & Peterson through `polyA_PatersonCuffey` that returns a polynomial which is
then evaluated at a given temperature in the law.
"""
function CuffeyPaterson()
    A = polyA_PatersonCuffey()
    A_law = let A = A
        Law{ScalarCacheNoVJP}(;
            name = :CuffeyPaterson,
            inputs = (; T=iTemp()),
            f! = function (cache, inp, θ)
                cache.value .= A.(inp.T)
            end,
            init_cache = function (simulation, glacier_idx, θ)
                return ScalarCacheNoVJP(zeros())
            end,
        )
    end
    return A_law
end


"""
    SyntheticC(params::Sleipnir.Parameters)

Creates a synthetic law for calculating the parameter `C` using a nonlinear sigmoid transformation
based on the ratio of `CPDD` (cumulative positive degree days) to `topo_roughness` (topographic roughness).
The law is parameterized by minimum and maximum values (`Cmin`, `Cmax`) from `params.physical`, and
applies a sigmoid scaling to smoothly interpolate between these bounds.
"""
function SyntheticC(params::Sleipnir.Parameters; inputs = (; CPDD=iCPDD(), topo_roughness=iTopoRough()))
    C_synth_law = Law{MatrixCacheNoVJP}(;
        name = :SyntheticC,
        inputs = inputs,
        max_value = params.physical.maxC,
        min_value = params.physical.minC,
        f! = function (cache, inp, θ)
            required_fields = (:CPDD, :topo_roughness)
            missing = filter(f -> !haskey(inp, f), required_fields)
            if !isempty(missing)
                error("SyntheticC: Missing required input fields: $(missing)")
            end
            # Nonlinear scaling using a sigmoid transformation
            # β controls the steepness of the sigmoid, ϵ avoids division by zero
            Cmin = params.physical.minC
            Cmax = params.physical.maxC
            # Parameters
            α = 1.0   # CPDD weight
            γ = 10.0   # topo weight
            β = 4.0   # steepness
            norm_CPDD = normalize(inp.CPDD)
            norm_topo = normalize(inp.topo_roughness)
            # 2D logistic surface
            x = α .* norm_CPDD .- γ .* norm_topo
            sigmoid = @. 1.0 / (1.0 + exp(-β * (x - 1.0)))  # Center sigmoid at x=1 for flexibility
            # If the provided C values are a matrix, reduce matrix size to match operations
            cache.value .= Cmin .+ (Cmax - Cmin) .* inn1(sigmoid)
        end,
        init_cache = function (simulation, glacier_idx, θ)
            MatrixCacheNoVJP(zeros(size(simulation.glaciers[glacier_idx].S) .- 1))
        end,
        callback_freq = 1/52,  # TODO: modify depending on freq from params
    )
    return C_synth_law
end