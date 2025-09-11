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

"""
    iCPDD <: AbstractInput

Input that represents the cumulative positive degree days (PDD) over the last time window `window`.
It is computed by summing the daily PDD values from `t - window` to `t` using the glacier's climate data.
"""
struct iCPDD{I<:Integer} <: AbstractInput
    window::I
    iCPDD{I}(window::I = 7) where {I<:Integer} = new{I}(window)
end

iCPDD(; window::I = 7) where {I<:Integer} = iCPDD{typeof(window)}(window)

default_name(::iCPDD) = :CPDD  

function get_input(cpdd::iCPDD, simulation, glacier_idx, t)  
    window = cpdd.window  
    glacier = simulation.glaciers[glacier_idx]  
    # We trim only the time period between `t` and `t - x`, where `x` is the PDD time window defined in the physical parameters.  
    period = (partial_year(Day, t) - Day(window)):Day(1):partial_year(Day, t)  
    get_cumulative_climate!(glacier.climate, period)  
    # Convert climate dataset to 2D based on the glacier's DEM  
    climate_2D_step = downscale_2D_climate(glacier.climate.climate_step, glacier.S, glacier.Coords)  

    return climate_2D_step.PDD  
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

"""
Input that represents the topographic roughness of the glacier.
It is computed as the standard deviation of the elevation of the glacier's DEM.
"""
struct iTopoRough{F<:AbstractFloat} <: AbstractInput 
    window::F
    iTopoRough{F}(window::F = 200.0) where {F<:AbstractFloat} = new{F}(window)
end

iTopoRough(; window::F = 200.0) where {F<:AbstractFloat} = iTopoRough{F}(window)

default_name(::iTopoRough) = :topographic_roughness  

function get_input(inp_topo_rough::iTopoRough, simulation, glacier_idx, t)
    window = inp_topo_rough.window
    glacier = simulation.glaciers[glacier_idx]
    dem = glacier.S
    window_size = max(3, Int(round(window / glacier.Δx)))  # At least 3 for second derivative
    half_window = max(1, div(window_size, 2))
    rows, cols = size(dem)
    roughness = zeros(eltype(dem), size(dem))

    for i in 1:rows, j in 1:cols
        rmin = max(1, i - half_window)
        rmax = min(rows, i + half_window)
        cmin = max(1, j - half_window)
        cmax = min(cols, j + half_window)
        window_dem = dem[rmin:rmax, cmin:cmax]

        # Compute local slope (first derivative)
        dx = diff_x(window_dem) / glacier.Δx
        dy = diff_y(window_dem) / glacier.Δy

        # Compute local curvature (second derivative)
        dxx = diff_x(dx) / glacier.Δx
        dyy = diff_y(dy) / glacier.Δy

        # Ensure dxx and dyy have the same size for addition
        minlen = min(length(dxx), length(dyy))
        curvature = dxx[1:minlen] .+ dyy[1:minlen]
        val = std(curvature)
        # Double check that no NaNs are added due to border effects
        roughness[i, j] = isnan(val) ? zero(eltype(dem)) : val 
    end
    
    return roughness
end

"""
    ConstantA(A::F) where {F <: AbstractFloat}

Law that represents a constant A in the SIA.

# Arguments:
- `A::F`: Rheology factor A.
"""
function ConstantA(A::F) where {F <: AbstractFloat}
    return ConstantLaw{Array{Float64, 0}}(function (simulation, glacier_idx, θ)
            return fill(A)
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
        Law{Array{Float64, 0}}(;
            name = :CuffeyPaterson,
            inputs = (; T=iTemp()),
            f! = function (cache, inp, θ)
                cache .= A.(inp.T)
            end,
            init_cache = function (simulation, glacier_idx, θ; scalar::Bool = true)
                return zeros()
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
    C_synth_law = Law{Array{Float64, 2}}(;
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
            β = 1.0      # Steepness parameter for sigmoid
            ϵ = 1e-6     # Small value to avoid division by zeros
            norm_CPDD = normalize(inp.CPDD)
            norm_topo = normalize(inp.topo_roughness)
            # Predict value of C based a sigmoid function
            x = @. norm_CPDD / (norm_topo + ϵ)
            sigmoid = @. 1.0 / (1.0 + exp(-β * (x - 1.0)))  # Center sigmoid at x=1 for flexibility
            # If the provided C values are a matrix, reduce matrix size to match operations
            cache .= Cmin .+ (Cmax - Cmin) .* inn1(sigmoid)
        end,
        init_cache = function (simulation, glacier_idx, θ; scalar::Bool = false)
            # Initialize cache as a scalar or vector depending on the required output
            scalar ? zeros() : zeros(size(simulation.glaciers[glacier_idx].S) .- 1)
        end,
        callback_freq = 1/52,  # TODO: modify depending on freq from params
    )
    return C_synth_law
end