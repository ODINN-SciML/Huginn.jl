export halfar_solution

"""
    halfar_solution(t, r, θ)

Returns the evaluation of the Halfar solutions for the SIA equation. 

Arguments:
    - r: radial distance. The solutions have polar symmetry around the center of origin.
    - t: time
    - ν = (A, H₀, R₀) 
"""
function halfar_solution(R, t, h₀, r₀, A, n)#, physical_parameters::PhysicalParameters)

    # parameters of Halfar solutions
    # ρ = physical_parameters.ρ
    # g = physical_parameters.g
    ρ = 900.0
    g = 9.81

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
