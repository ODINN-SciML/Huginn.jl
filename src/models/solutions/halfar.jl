export halfar_solution

"""
    halfar_solution(R, t, h₀, r₀, A, n, physical_parameters::PhysicalParameters)

Return the evaluation of the Halfar solution for the SIA equation.

Arguments:
- `R`: Radial distance. The solution has polar symmetry around the center of origin.
- `t`: Time.
- `h₀` and `r₀`: Parameters of the Halfar solution.
- `A`: Glen's law parameter.
- `n`: Creep exponent.
- `physical_parameters::PhysicalParameters`: Physical parameters that allow
    retrieving the ice density and the gravity constant.
"""
function halfar_solution(R, t, h₀, r₀, A, n, physical_parameters::PhysicalParameters)

    # parameters of Halfar solutions
    ρ = physical_parameters.ρ
    g = physical_parameters.g

    Γ = 2 * A * (ρ * g)^n / (n+2)
    # Characteristic time
    τ₀ = (7/4)^3 * r₀^4 / ( 18 * Γ * h₀^7 )

    mask = R .<= r₀ .* (t./τ₀).^(1/18)
    # @infiltrate
    return mask .* real.( h₀ .* (τ₀./t).^(1/9) .* ( 1 .- ( (τ₀./t).^(1/18) * (R./r₀) ).^(4/3) .+ 0im ).^(3/7) ) .+ (.!mask) .* 0.0
    # return [r <= r₀ * (t/τ₀)^(1/18) ? 
    #         h₀ * (τ₀/t)^(1/9) * ( 1 - ( (τ₀/t)^(1/18) * (r/r₀) )^(4/3) )^(3/7) :
    #         0.0 
    #         for r in R]
end
