import Sleipnir: get_input, default_name

export iAvgScalarTemp, iAvgGriddedTemp, iH̄, i∇S, iCPDD, iTopoRough

"""
    iAvgScalarTemp <: AbstractInput

Input that represents the long term air temperature over the whole glacier.
It is computed using the OGGM climate data over a period predefined in Gungnir (i.e. around 30 years).
"""
struct iAvgScalarTemp <: AbstractInput end

default_name(::iAvgScalarTemp) = :averaged_long_term_temperature
function get_input(temp::iAvgScalarTemp, simulation, glacier_idx, t)
    glacier = simulation.glaciers[glacier_idx]
    return mean(glacier.climate.longterm_temps_scalar)
end
function Base.zero(temp::iAvgScalarTemp, simulation, glacier_idx)
    glacier = simulation.glaciers[glacier_idx]
    return zero(glacier.climate.longterm_temps_scalar)
end

"""
    iAvgGriddedTemp <: AbstractInput

Input that represents the long term air temperature over the glacier grid.
It is computed using the OGGM climate data over a period predefined in Gungnir (i.e. around 30 years).
"""
struct iAvgGriddedTemp <: AbstractInput end

default_name(::iAvgGriddedTemp) = :gridded_long_term_temperature
function get_input(temp::iAvgGriddedTemp, simulation, glacier_idx, t)
    glacier = simulation.glaciers[glacier_idx]
    return glacier.climate.longterm_temps_gridded
end
function Base.zero(temp::iAvgGriddedTemp, simulation, glacier_idx)
    glacier = simulation.glaciers[glacier_idx]
    return zero(glacier.climate.longterm_temps_gridded)
end

"""
    iCPDD <: AbstractInput

Input that represents the cumulative positive degree days (PDD) over the last time window `window`.
It is computed by summing the daily PDD values from `t - window` to `t` using the glacier's climate data.
"""
struct iCPDD{P <: Period} <: AbstractInput
    window::P
    function iCPDD(; window::P = Week(1)) where {P <: Period}
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
    climate_2D_step = downscale_2D_climate(
        glacier.climate.climate_step, glacier.S, glacier.Coords)

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
    i∇S <: AbstractInput

Input that represents the surface slope in the SIA.
It is computed using the bedrock elevation and the ice thickness solution H. The
spatial differences are averaged over the opposite axis:

```julia
S = B + H
∇S = (avg_y(diff_x(S) / Δx) .^ 2 .+ avg_x(diff_y(S) / Δy) .^ 2) .^ (1/2)
```
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
    iTopoRough{F<:AbstractFloat} <: AbstractInput

Input that represents the topographic roughness of the glacier.
It is computed as the curvature of the glacier bed (or surface) over a specified window size. The curvature can be calculated in different directions (flow, cross-flow, or both)
and using different curvature types (scalar or variability).
"""
struct iTopoRough{F <: AbstractFloat} <: AbstractInput
    window::F
    curvature_type::Symbol
    direction::Symbol
    position::Symbol
    function iTopoRough(; window::F = 200.0, curvature_type::Symbol = :scalar,
            direction::Symbol = :flow, position::Symbol = :bed) where {F <: AbstractFloat}
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
            dx_c = diff_x(window_dem)[div(end, 2), div(end, 2)] / glacier.Δx
            dy_c = diff_y(window_dem)[div(end, 2), div(end, 2)] / glacier.Δy
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
            for wi in 2:(wrows - 1), wj in 2:(wcols - 1)   # avoid borders

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
            roughness[i, j] = isnan(val) ? 0.0 : val

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
            roughness[i, j] = isnan(val) ? 0 : val
        end
    end

    return roughness
end
function Base.zero(::iTopoRough, simulation, glacier_idx)
    (; nx, ny) = simulation.glaciers[glacier_idx]
    return zeros(nx, ny)
end
