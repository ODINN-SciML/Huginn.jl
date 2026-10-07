export ConstantA, CuffeyPaterson, SyntheticC, TemperateA

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
    TemperateA()

Law that represents a constant A in the SIA for temperate ice (0°C).

The value of A is set to 2.4e-24 s⁻¹Pa⁻³ which come from Cuffey & Peterson.
"""
function TemperateA()
    A_value = polyA_PatersonCuffey()(0.0) # year⁻¹Pa⁻³
    return ConstantA(A_value)
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
    A_values = hcat(A_values_sec[1, :], A_values_sec[2, :] .* 60.0*60.0*24.0*365.25)'
    return Polynomials.fit(A_values[1, :], A_values[2, :])
end

"""
    sliding_scale(A, H; p = 3.0, q = 0.0, n = 3.0, ρg = 900.0 * 9.81)

Value of `C` at which sliding and deformation contribute equally to the SIA diffusivity.

`SIA2D!` splits the diffusivity as `C(ρg)^(p-q) H^(p-q+1) ∇S^(p-1)` against
`2A(ρg)ⁿHⁿ⁺²∇Sⁿ⁻¹/(n+2)`, so the crossover depends on the sliding law. It sets both a
sensible bound on `C` and the scale of its regularization, which is why it is needed before
any `C` inversion: `C` spans orders of magnitude between sliding laws and a λ chosen for one
is meaningless for another. The `∇S` powers cancel when `p-1 = n-1`, so the scale depends
only on the thickness `H` at which the comparison is made.

Defaults are Weertman sliding (`q = 0`) in Glen's flow law with `n = 3`.
"""
function sliding_scale(
        A::Real, H::Real;
        p::Real = 3.0, q::Real = 0.0, n::Real = 3.0, ρg::Real = 900.0 * 9.81
)
    @assert p==n "sliding_scale: the ∇S factors only cancel when p == n (got p = $(p), n = $(n))."
    deformation = 2 * A * ρg^n * H^(n + 2) / (n + 2)
    sliding_unit_C = ρg^(p - q) * H^(p - q + 1)
    return deformation / sliding_unit_C
end

"""
    CuffeyPaterson(; scalar::Bool = true)

Create a rheology law for the flow rate factor `A`.
The created law maps the long term air temperature to `A` using the values from
Cuffey & Peterson through `polyA_PatersonCuffey` that returns a polynomial which is
then evaluated at a given temperature in the law.
"""
function CuffeyPaterson(; scalar::Bool = true)
    A = polyA_PatersonCuffey()
    A_law = let A = A
        if scalar
            Law{ScalarCacheNoVJP}(;
                name = :CuffeyPaterson,
                inputs = (; T = iAvgScalarTemp()),
                f! = function (cache, inp, θ)
                    cache.value .= A.(inp.T)
                end,
                init_cache = function (simulation, glacier_idx, θ)
                    return ScalarCacheNoVJP(zeros())
                end,
                callback_freq = 0
            )
        else
            Law{MatrixCacheNoVJP}(;
                name = :CuffeyPaterson,
                inputs = (; T = iAvgGriddedTemp()),
                f! = function (cache, inp, θ)
                    cache.value .= A.(inn1(inp.T))
                end,
                init_cache = function (simulation, glacier_idx, θ)
                    MatrixCacheNoVJP(zeros(size(simulation.glaciers[glacier_idx].S) .- 1))
                end,
                callback_freq = 0
            )
        end
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
function SyntheticC(params::Sleipnir.Parameters;
        inputs = (; CPDD = iCPDD(), topo_roughness = iTopoRough()))
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
        callback_freq = Week(1)  # TODO: modify depending on freq from params
    )
    return C_synth_law
end
