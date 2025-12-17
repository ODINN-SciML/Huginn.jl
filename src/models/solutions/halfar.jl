export HalfarParameters, Halfar

"""
    HalfarParameters(; λ=0.0, H₀=3600.0, R₀=750000.0, n=3.0, A=1e-16, f=0.0, ρ=910.0, g=9.81)

Holds parameters for the Halfar similarity solution of the shallow ice approximation (SIA).

# Parameters

  - `λ::AbstractFloat=0.0`: Mass balance coefficient (used to model accumulation/ablation).
  - `H₀::AbstractFloat=3600.0`: Dome height at initial time `t₀` [m].
  - `R₀::AbstractFloat=750000.0`: Ice sheet margin radius at `t₀` [m].
  - `n::AbstractFloat=3.0`: Glen flow law exponent.
  - `A::AbstractFloat=1e-16`: Flow rate factor in Glen's law [Pa⁻ⁿ yr⁻¹].
  - `f::AbstractFloat=0.0`: Fraction of isostatic bed depression (0 for fully grounded ice).
  - `ρ::AbstractFloat=910.0`: Ice density [kg/m³].
  - `g::AbstractFloat=9.81`: Gravitational acceleration [m/s²].

# Notes

Default parameters set as in Bueler (2005) "Exact solutions and verification of numerical
models for isothermalice sheets", experiment B.
"""
@kwdef struct HalfarParameters{F <: AbstractFloat}
    # Mass balance coefficient
    λ::F = 0.0
    # Dome height at t = t₀
    H₀::F = 3600.0 # [m]
    # Margin radius at t = t₀
    R₀::F = 750000.0 # [m]
    # Glen exponent
    n::F = 3.0
    A::F = 1e-16 # 3.16880e-24 # = 1e-16 Pa^{-3} yr^{-1}
    # Fraction for isostatic bed depression (equals zero for grounded ice)
    f::F = 0.0
    ρ::F = 910.0
    g::F = 9.81
end

"""
    Halfar(halfar_params::HalfarParameters) -> (_halfar::Function, t₀_years::Float64)

Constructs the Halfar similarity solution to the SIA for a radially symmetric ice sheet
dome following Bueler (2005) "Exact solutions and verification of numerical models for
isothermalice sheets"

# Arguments

  - `halfar_params::HalfarParameters`: A struct containing physical and geometric parameters for the Halfar solution, including dome height, margin radius, Glen exponent, and other constants.

# Returns

  - `_halfar::Function`: A function `(x, y, t) -> H` that evaluates the ice thickness `H` at position `(x, y)` and time `t` (in **years**).
  - `t₀_years::Float64`: The characteristic timescale `t₀` (in **years**) of the solution, based on the specified parameters.

# Description

The solution has the form:

```math
H(r, t) = H₀ (t / t₀)^(-α) [1 - ((t / t₀)^(-β) (r / R₀))^((n+1)/n)]^{n / (2n + 1)}
```
"""
function Halfar(
        halfar_params::HalfarParameters
)
    # Retrieve parameters
    (; λ, n, A, ρ, g, f) = halfar_params
    A /= 365.25 * 24 * 60 * 60

    Γ = 2 * A * (ρ * g)^n / (n + 2)

    H₀ = halfar_params.H₀
    R₀ = halfar_params.R₀

    α = (2 - (n - 1) * λ) / (5 * n + 3)
    β = (1 + (2 * n + 1) * λ) / (5 * n + 3)

    # Determines natural scale of solution
    t₀ = (β / Γ)
    t₀ *= ((2 * n + 1) / ((1 - f) * (n + 1)))^n
    t₀ *= R₀^(n + 1) / H₀^(2 * n + 1)
    t₀_years = t₀ / (365.25 * 24 * 60 * 60) # convert to seconds

    function _halfar(x, y, t)
        # convert years to seconds
        t *= (365.25 * 24 * 60 * 60)
        r = (x^2 + y^2)^0.5
        if r < R₀ * (t / t₀)^β
            return H₀ * (t / t₀)^(-α) *
                   (1 - ((t / t₀)^(-β) * (r / R₀))^((n + 1) / n))^(n / (2 * n + 1))
        else
            return 0.0
        end
    end

    return _halfar, t₀_years
end

"""
    Halfar_velocity(halfar_params::HalfarParameters)

Same as Halfar(halfar_params), but instead of returning a function that gives the ice
thickness as a function of space and time, this returns the ice surface velocity according
to the Shallow Ice Approximation.
"""
function Halfar_velocity(
        halfar_params::HalfarParameters
)
    H, t₀ = Halfar(halfar_params)
    # Convert to seconds
    t₀ *= 365.25 * 24 * 60 * 60

    (; λ, n, A, ρ, g, f, H₀, R₀) = halfar_params

    α = (2 - (n - 1) * λ) / (5 * n + 3)
    β = (1 + (2 * n + 1) * λ) / (5 * n + 3)

    # Compute the slope
    # This is just the analytical derivative of H(r, t) with respect to r
    function _halfar_slope(x, y, t)
        t *= (365.25 * 24 * 60 * 60)
        r = (x^2 + y^2)^0.5
        if r < 1e-2 * R₀ * (t / t₀)^β
            return 0.0
        elseif r < R₀ * (t / t₀)^β
            return H₀ * (t / t₀)^(- α - β) * ((n + 1) / (2 * n + 1)) *
                   ((t / t₀)^(-β) * (r / R₀))^(1 / n) / (R₀ *
                    (1 - ((t / t₀)^(-β) * (r / R₀))^((n + 1) / n))^((n + 1) / (2 * n + 1)) +
                    1e-10)
        else
            return 0.0
        end
    end

    function _halfar_velocity(x, y, t)
        r = (x^2 + y^2)^0.5
        vabs = 2 * A * (ρ * g)^n * H(x, y, t)^(n + 1) * _halfar_slope(x, y, t)^n / (n + 1)
        return vabs .* [x / r, y / r]
    end

    return _halfar_velocity
end
