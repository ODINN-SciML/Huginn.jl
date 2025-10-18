import Sleipnir: get_input, default_name

export InpTemp, InpH̄, Inp∇S
export ConstantA, CuffeyPaterson

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
function Base.zero(::InpTemp, simulation, glacier_idx)
    glacier = simulation.glaciers[glacier_idx]
    return zero(glacier.climate.longterm_temps)
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
function Base.zero(::InpH̄, simulation, glacier_idx)
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
struct Inp∇S <: AbstractInput end
default_name(::Inp∇S) = :∇S
function get_input(::Inp∇S, simulation, glacier_idx, t)
    return simulation.cache.iceflow.∇S
end
function Base.zero(::Inp∇S, simulation, glacier_idx)
    (; nx, ny) = simulation.glaciers[glacier_idx]
    return zeros(nx-1, ny-1)
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
            inputs = (; T=InpTemp()),
            f! = function (cache, inp, θ)
                cache.value .= A.(inp.T)
            end,
            init_cache = function (simulation, glacier_idx, θ; scalar=false)
                return ScalarCacheNoVJP(zeros())
            end,
        )
    end
    return A_law
end
