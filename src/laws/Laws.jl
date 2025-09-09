import Sleipnir: get_input, default_name

export InpTemp, InpH̄, Inp∇S, InpCPDD, InpTopoRough
export ConstantA, CuffeyPaterson, SyntheticC

########################
###### LAW INPUTS ######
########################

"""
    InpTemp <: AbstractInput

Input that represents the long term air temperature of a glacier.
It is computed using the OGGM data over a period predefined in Gungnir.
"""
struct InpTemp <: AbstractInput end
default_name(::InpTemp) = :long_term_temperature
function get_input(::InpTemp, simulation, glacier_idx, t)
    glacier = simulation.glaciers[glacier_idx]
    return mean(glacier.climate.longterm_temps)
end

"""
    InpCPDD <: AbstractInput

Input that represents the cumulative positive degree days (PDD) over the last week.
It is computed by summing the daily PDD values from `t - 1 week` to `t` using the glacier's climate data.
"""
struct InpCPDD{I<:Integer} <: AbstractInput
    window::I
    InpCPDD{I}(; window::I = 7) where {I<:Integer} = new{I}(window)
end

default_name(::InpCPDD) = :CPDD  

function get_input(cpdd::InpCPDD, simulation, glacier_idx, t)  
    window = cpdd.window  
    glacier = simulation.glaciers[glacier_idx]  
    # We trim only the time period between `t` and `t - x`, where `x` is the PDD time window defined in the physical parameters.  
    period = (partial_year(Day, t) - Day(window)):Day(1):partial_year(Day, t)  
    get_cumulative_climate!(glacier.climate, period)  
    # Convert climate dataset to 2D based on the glacier's DEM  
    climate_2D_step = downscale_2D_climate(glacier.climate.climate_step, glacier)  

    return climate_2D_step.PDD  
end  

"""
    InpH̄ <: AbstractInput

Input that represents the ice thickness in the SIA.
It is the averaged ice thickness computed on the dual grid, that is `H̄ = avg(H)`
which is different from the ice thickness solution H.
"""
struct InpH̄ <: AbstractInput end
default_name(::InpH̄) = :H_dual_grid
function get_input(::InpH̄, simulation, glacier_idx, t)
    return simulation.cache.iceflow.H̄
end

"""
Input that represents the surface slope in the SIA.
It is computed using the bedrock elevation and the ice thickness solution H. The
spatial differences are averaged over the opposite axis:
S = B + H
∇S = (avg_y(diff_x(S) / Δx).^2 .+ avg_x(diff_y(S) / Δy).^2).^(1/2)
"""
struct Inp∇S <: AbstractInput end
default_name(::Inp∇S) = :∇S
function get_input(::Inp∇S, simulation, glacier_idx, t)
    return simulation.cache.iceflow.∇S
end

"""
Input that represents the topographic roughness of the glacier.
It is computed as the standard deviation of the elevation of the glacier's DEM.
"""
struct InpTopoRough{F<:AbstractFloat} <: AbstractInput 
    window::F
    InpTopoRough{F}(; window::F = 200.0) where {F<:AbstractFloat} = new{F}(window)
end

default_name(::InpTopoRough) = :topographic_roughness  

function get_input(inp_topo_rough::InpTopoRough, simulation, glacier_idx, t)  
    window = inp_topo_rough.window  
    glacier = simulation.glaciers[glacier_idx]  
    # Compute the topographic roughness as the standard deviation of the elevation in a window of around 200 meters  
    # around each pixel of the glacier's DEM.  
    dem = glacier.S  
    window_size = max(1, Int(round(window / glacier.Δx)))  
    half_window = max(1, div(window_size, 2))  # Ensure at least 1 pixel  
    rows, cols = size(dem)  
    roughness = similar(dem)  
    for i in 1:rows, j in 1:cols  
        rmin = max(1, i - half_window)  
        rmax = min(rows, i + half_window)  
        cmin = max(1, j - half_window)  
        cmax = min(cols, j + half_window)  
        window = dem[rmin:rmax, cmin:cmax]  
        roughness[i, j] = std(window)  
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
            inputs = (; T=InpTemp()),
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
function SyntheticC(params::Sleipnir.Parameters; inputs = (; CPDD=InpCPDD(), topo_roughness=InpTopoRough()))
    C_synth_law = Law{Array{Float64, 2}}(;
        name = :SyntheticC,
        inputs = inputs,
        max_value = params.physical.maxC,
        min_value = params.physical.minC,
        f! = function (cache, inp, θ)
            # Nonlinear scaling using a sigmoid transformation
            # C = Cmin + (Cmax - Cmin) * sigmoid(β * (inp.CPDD / (inp.topo_roughness + ϵ)))
            # β controls the steepness of the sigmoid, ϵ avoids division by zero
            Cmin = params.physical.minC
            Cmax = params.physical.maxC
            β = 1.0      # Steepness parameter for sigmoid
            ϵ = 1e-6     # Small value to avoid division by zeros
            x = @. inp.CPDD / (inp.topo_roughness + ϵ)
            sigmoid = @. 1.0 / (1.0 + exp(-β * (x - 1.0)))  # Center sigmoid at x=1 for flexibility
            # If the provided C values are a matrix, reduce matrix size to match operations
            cache .= Cmin .+ (Cmax - Cmin) .* (isa(sigmoid, Matrix) ? inn1(sigmoid) : sigmoid)
        end,
        init_cache = function (simulation, glacier_idx, θ; scalar::Bool = false)
            # Initialize cache as a scalar or vector depending on the required output
            scalar ? zeros() : zeros(size(simulation.glaciers[glacier_idx].S) .- 1)
        end,
        callback_freq = 1/52,  # TODO: modify depending on freq from params
    )
    return C_synth_law
end