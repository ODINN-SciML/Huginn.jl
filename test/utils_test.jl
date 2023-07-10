# -*- coding: utf-8 -*-
"""
    is_border(A::Matrix, distance::Int)

Return a matrix with booleans indicating if a given pixel is at distance at most `distance` of the end of the 
matrix, which is indicated by the first pixel value to reach zero. 

Arguments:
    - A: Array
    - distance: distance to the border, computed as the number of pixels we need to move to find a pixel with value zero
"""
function is_border(A::Matrix{F}, distance::Int) where {F <: AbstractFloat}
    B = copy(A) 
    for i in 1:distance
        B .= min.(B, circshift(B, (1,0)), circshift(B, (-1,0)), circshift(B, (0,1)), circshift(B, (0,-1)))
    end
    return B .> 0.001 
end

"""
    halfar_solution(t, r, θ)

Returns the evaluation of the Halfar solutions for the SIA equation. 

Arguments:
    - r: radial distance. The solutions have polar symmetry around the center of origin.
    - t: time
    - ν = (A, H₀, R₀) 
"""
function halfar_solution(R, t, h₀, r₀, physical_parameters::PhysicalParameters)

    # parameters of Halfar solutions
    ρ = physical_parameters.ρ
    g = physical_parameters.g
    n = physical_parameters.n
    A = physical_parameters.A

    Γ = 2 * A * (ρ * g)^n / (n+2)
    # Characteristic time
    τ₀ = (7/4)^3 * r₀^4 / ( 18 * Γ * h₀^7 )   

    return [r <= r₀ * (t/τ₀)^(1/18) ? 
            h₀ * (τ₀/t)^(1/9) * ( 1 - ( (τ₀/t)^(1/18) * (r/r₀) )^(4/3) )^(3/7) :
            0.0 
            for r in R]
end
